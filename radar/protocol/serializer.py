# serializer.py - 消息序列化
"""
本模块提供消息的序列化和反序列化功能。

支持两种序列化格式：
- JSON: 文本格式，易于调试，使用orjson实现高性能
- MessagePack: 二进制格式，更高效
"""

import json
import orjson
from typing import Type, TypeVar, Any, Dict
from dataclasses import asdict, fields, is_dataclass
import numpy as np
from datetime import datetime

from radar.protocol.messages import BaseMessage

T = TypeVar('T', bound=BaseMessage)


class MessageSerializer:
    """
    消息序列化器

    提供高效的消息序列化和反序列化功能。
    支持 numpy 数组的自动转换。
    """

    def __init__(self, format: str = 'json'):
        """
        初始化序列化器

        Args:
            format: 序列化格式 ('json' 或 'msgpack')
        """
        if format not in ['json', 'msgpack']:
            raise ValueError(f"不支持的序列化格式: {format}")

        self.format = format

    def serialize(self, message: BaseMessage) -> bytes:
        """
        序列化消息为字节

        Args:
            message: 要序列化的消息

        Returns:
            序列化后的字节数据

        Raises:
            ValueError: 如果序列化失败
        """
        try:
            # 转换为字典
            data_dict = self._dataclass_to_dict(message)

            # 根据格式序列化
            if self.format == 'json':
                return orjson.dumps(data_dict)
            else:  # msgpack
                import msgpack
                return msgpack.packb(data_dict, use_bin_type=True)

        except Exception as e:
            raise ValueError(f"序列化失败: {e}")

    def deserialize(self, data: bytes, message_class: Type[T]) -> T:
        """
        反序列化字节为消息对象

        Args:
            data: 要反序列化的字节数据
            message_class: 目标消息类

        Returns:
            反序列化后的消息对象

        Raises:
            ValueError: 如果反序列化失败
        """
        try:
            # 根据格式反序列化
            if self.format == 'json':
                data_dict = orjson.loads(data)
            else:  # msgpack
                import msgpack
                data_dict = msgpack.unpackb(data, raw=False)

            # 转换为消息对象
            return self._dict_to_dataclass(data_dict, message_class)

        except Exception as e:
            raise ValueError(f"反序列化失败: {e}")

    def _dataclass_to_dict(self, obj: Any) -> Any:
        """
        递归转换dataclass为字典

        处理特殊类型如numpy数组、datetime等。

        Args:
            obj: 要转换的对象

        Returns:
            转换后的字典或基本类型
        """
        if is_dataclass(obj):
            result = {}
            for field_info in fields(obj):
                value = getattr(obj, field_info.name)
                result[field_info.name] = self._dataclass_to_dict(value)
            return result

        elif isinstance(obj, np.ndarray):
            # numpy数组转换为列表
            return obj.tolist()

        elif isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (list, tuple)):
            return [self._dataclass_to_dict(item) for item in obj]

        elif isinstance(obj, dict):
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}

        else:
            return obj

    def _dict_to_dataclass(self, data: Dict, cls: Type[T]) -> T:
        """
        从字典创建dataclass对象

        Args:
            data: 源字典
            cls: 目标dataclass类

        Returns:
            dataclass实例
        """
        if not is_dataclass(cls):
            return data

        # 获取字段信息
        field_infos = {f.name: f for f in fields(cls)}
        kwargs = {}

        for key, value in data.items():
            if key not in field_infos:
                continue

            field_info = field_infos[key]
            field_type = field_info.type

            # 递归处理嵌套的dataclass
            kwargs[key] = self._dict_to_dataclass_type(value, field_type)

        return cls(**kwargs)

    def _dict_to_dataclass_type(self, value: Any, field_type: Type) -> Any:
        """
        根据字段类型转换值

        Args:
            value: 要转换的值
            field_type: 目标类型

        Returns:
            转换后的值
        """
        # 处理Optional类型
        if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
            # Optional[T] = Union[T, None]
            args = field_type.__args__
            if value is None:
                return None
            # 使用非None的类型
            for arg in args:
                if arg is not type(None):
                    return self._dict_to_dataclass_type(value, arg)

        # 处理List类型
        elif hasattr(field_type, '__origin__') and field_type.__origin__ is list:
            if not isinstance(value, list):
                return value
            # 获取元素类型
            if hasattr(field_type, '__args__') and len(field_type.__args__) > 0:
                elem_type = field_type.__args__[0]
                return [self._dict_to_dataclass_type(v, elem_type) for v in value]
            return value

        # 处理dataclass
        elif is_dataclass(field_type):
            if isinstance(value, dict):
                return self._dict_to_dataclass(value, field_type)
            return value

        # 处理numpy数组
        elif field_type == np.ndarray:
            if isinstance(value, (list, tuple)):
                return np.array(value, dtype=np.float64)
            return value

        # 基本类型直接返回
        else:
            return value


class MessageEncoder(json.JSONEncoder):
    """
    JSON编码器扩展

    支持numpy数组和datetime等特殊类型的编码。
    """

    def default(self, obj: Any) -> Any:
        """
        编码特殊对象

        Args:
            obj: 要编码的对象

        Returns:
            可JSON序列化的对象
        """
        # numpy数组
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # numpy标量
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)

        # datetime
        elif isinstance(obj, datetime):
            return obj.isoformat()

        # dataclass
        elif is_dataclass(obj):
            return asdict(obj)

        # 其他情况尝试默认处理
        return super().default(obj)


def serialize_message(message: BaseMessage, format: str = 'json') -> bytes:
    """
    便捷函数：序列化消息

    Args:
        message: 要序列化的消息
        format: 序列化格式 ('json' 或 'msgpack')

    Returns:
        序列化后的字节数据
    """
    serializer = MessageSerializer(format)
    return serializer.serialize(message)


def deserialize_message(data: bytes, message_class: Type[T],
                     format: str = 'json') -> T:
    """
    便捷函数：反序列化消息

    Args:
        data: 要反序列化的字节数据
        message_class: 目标消息类
        format: 序列化格式 ('json' 或 'msgpack')

    Returns:
        反序列化后的消息对象
    """
    serializer = MessageSerializer(format)
    return serializer.deserialize(data, message_class)


__all__ = [
    "MessageSerializer",
    "MessageEncoder",
    "serialize_message",
    "deserialize_message",
]
