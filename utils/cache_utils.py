import redis
import json
import pickle
import hashlib
import time
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
import logging
from functools import wraps
import threading
from collections import OrderedDict
import gzip
import sys

logger = logging.getLogger(__name__)


class CacheUtils:
    """缓存工具类 - 支持Redis和内存缓存"""

    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379,
                 redis_db: int = 0, use_redis: bool = True,
                 memory_cache_size: int = 1000, default_ttl: int = 3600):
        self.use_redis = use_redis
        self.redis_client = None
        self.memory_cache = OrderedDict()
        self.memory_cache_size = memory_cache_size
        self.default_ttl = default_ttl
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'memory_size': 0,
            'redis_size': 0
        }
        self._lock = threading.RLock()

        # 尝试连接Redis
        if self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=5
                )
                # 测试连接
                self.redis_client.ping()
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {str(e)}, using memory cache only")
                self.use_redis = False
                self.redis_client = None

        logger.info(f"CacheUtils initialized - Redis: {self.use_redis}, Memory cache size: {memory_cache_size}")

    def generate_key(self, prefix: str, *args, **kwargs) -> str:
        """生成缓存键"""
        key_parts = [prefix]

        # 添加位置参数
        for arg in args:
            if isinstance(arg, (dict, list)):
                key_parts.append(hashlib.md5(json.dumps(arg, sort_keys=True).encode()).hexdigest()[:8])
            else:
                key_parts.append(str(arg))

        # 添加关键字参数
        if kwargs:
            sorted_kwargs = sorted(kwargs.items())
            kwargs_str = json.dumps(sorted_kwargs, sort_keys=True)
            key_parts.append(hashlib.md5(kwargs_str.encode()).hexdigest()[:8])

        return ':'.join(key_parts)

    def set(self, key: str, value: Any, ttl: Optional[int] = None,
            compress: bool = False, serialize_method: str = 'pickle') -> bool:
        """设置缓存值"""
        try:
            if ttl is None:
                ttl = self.default_ttl

            # 序列化数据
            serialized_data = self._serialize_data(value, serialize_method, compress)

            success = False

            # Redis缓存
            if self.use_redis and self.redis_client:
                try:
                    cache_value = {
                        'data': serialized_data,
                        'timestamp': time.time(),
                        'ttl': ttl,
                        'compressed': compress,
                        'serialize_method': serialize_method
                    }

                    result = self.redis_client.setex(
                        key,
                        ttl,
                        pickle.dumps(cache_value)
                    )
                    success = bool(result)

                    if success:
                        self.cache_stats['redis_size'] = self._get_redis_size()

                except Exception as e:
                    logger.error(f"Redis set error for key {key}: {str(e)}")

            # 内存缓存（作为备份或主要缓存）
            with self._lock:
                # 如果内存缓存已满，删除最旧的项
                while len(self.memory_cache) >= self.memory_cache_size:
                    self.memory_cache.popitem(last=False)

                expire_time = time.time() + ttl
                self.memory_cache[key] = {
                    'data': serialized_data,
                    'expire_time': expire_time,
                    'timestamp': time.time(),
                    'compressed': compress,
                    'serialize_method': serialize_method
                }

                # 移动到末尾（最近使用）
                self.memory_cache.move_to_end(key)
                success = True

                self.cache_stats['memory_size'] = len(self.memory_cache)

            if success:
                self.cache_stats['sets'] += 1
                logger.debug(f"Cache set successful for key: {key}")

            return success

        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """获取缓存值"""
        try:
            # 首先尝试Redis
            if self.use_redis and self.redis_client:
                try:
                    cached_data = self.redis_client.get(key)
                    if cached_data:
                        cache_value = pickle.loads(cached_data)

                        # 检查是否过期
                        if time.time() - cache_value['timestamp'] < cache_value['ttl']:
                            deserialized_data = self._deserialize_data(
                                cache_value['data'],
                                cache_value['serialize_method'],
                                cache_value['compressed']
                            )
                            self.cache_stats['hits'] += 1
                            logger.debug(f"Redis cache hit for key: {key}")
                            return deserialized_data
                        else:
                            # 过期，删除
                            self.redis_client.delete(key)

                except Exception as e:
                    logger.error(f"Redis get error for key {key}: {str(e)}")

            # 尝试内存缓存
            with self._lock:
                if key in self.memory_cache:
                    cache_item = self.memory_cache[key]

                    # 检查是否过期
                    if time.time() < cache_item['expire_time']:
                        # 移动到末尾（最近使用）
                        self.memory_cache.move_to_end(key)

                        deserialized_data = self._deserialize_data(
                            cache_item['data'],
                            cache_item['serialize_method'],
                            cache_item['compressed']
                        )
                        self.cache_stats['hits'] += 1
                        logger.debug(f"Memory cache hit for key: {key}")
                        return deserialized_data
                    else:
                        # 过期，删除
                        del self.memory_cache[key]
                        self.cache_stats['memory_size'] = len(self.memory_cache)

            # 缓存未命中
            self.cache_stats['misses'] += 1
            logger.debug(f"Cache miss for key: {key}")
            return default

        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return default

    def delete(self, key: str) -> bool:
        """删除缓存"""
        try:
            success = False

            # 删除Redis缓存
            if self.use_redis and self.redis_client:
                try:
                    result = self.redis_client.delete(key)
                    success = bool(result)
                except Exception as e:
                    logger.error(f"Redis delete error for key {key}: {str(e)}")

            # 删除内存缓存
            with self._lock:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    success = True
                    self.cache_stats['memory_size'] = len(self.memory_cache)

            if success:
                self.cache_stats['deletes'] += 1
                logger.debug(f"Cache delete successful for key: {key}")

            return success

        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False

    def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            # 检查Redis
            if self.use_redis and self.redis_client:
                try:
                    if self.redis_client.exists(key):
                        return True
                except Exception as e:
                    logger.error(f"Redis exists error for key {key}: {str(e)}")

            # 检查内存缓存
            with self._lock:
                if key in self.memory_cache:
                    cache_item = self.memory_cache[key]
                    if time.time() < cache_item['expire_time']:
                        return True
                    else:
                        # 过期，删除
                        del self.memory_cache[key]
                        self.cache_stats['memory_size'] = len(self.memory_cache)

            return False

        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {str(e)}")
            return False

    def clear(self, pattern: Optional[str] = None) -> int:
        """清除缓存"""
        try:
            deleted_count = 0

            # 清除Redis缓存
            if self.use_redis and self.redis_client:
                try:
                    if pattern:
                        keys = self.redis_client.keys(pattern)
                        if keys:
                            deleted_count += self.redis_client.delete(*keys)
                    else:
                        deleted_count += len(self.redis_client.keys())
                        self.redis_client.flushdb()
                except Exception as e:
                    logger.error(f"Redis clear error: {str(e)}")

            # 清除内存缓存
            with self._lock:
                if pattern:
                    import fnmatch
                    keys_to_delete = [k for k in self.memory_cache.keys()
                                      if fnmatch.fnmatch(k, pattern)]
                    for key in keys_to_delete:
                        del self.memory_cache[key]
                        deleted_count += 1
                else:
                    deleted_count += len(self.memory_cache)
                    self.memory_cache.clear()

                self.cache_stats['memory_size'] = len(self.memory_cache)

            logger.info(f"Cleared {deleted_count} cache entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Cache clear error: {str(e)}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = self.cache_stats.copy()

        # 计算命中率
        total_requests = stats['hits'] + stats['misses']
        stats['hit_rate'] = stats['hits'] / total_requests if total_requests > 0 else 0

        # 获取Redis信息
        if self.use_redis and self.redis_client:
            try:
                redis_info = self.redis_client.info()
                stats['redis_memory_used'] = redis_info.get('used_memory_human', 'N/A')
                stats['redis_connected'] = True
            except:
                stats['redis_connected'] = False
        else:
            stats['redis_connected'] = False

        # 内存缓存信息
        stats['memory_cache_size'] = len(self.memory_cache)
        stats['memory_cache_max_size'] = self.memory_cache_size

        return stats

    def cleanup_expired(self) -> int:
        """清理过期的缓存项"""
        try:
            current_time = time.time()
            deleted_count = 0

            # 清理内存缓存中的过期项
            with self._lock:
                expired_keys = []
                for key, cache_item in self.memory_cache.items():
                    if current_time >= cache_item['expire_time']:
                        expired_keys.append(key)

                for key in expired_keys:
                    del self.memory_cache[key]
                    deleted_count += 1

                self.cache_stats['memory_size'] = len(self.memory_cache)

            logger.debug(f"Cleaned up {deleted_count} expired cache entries")
            return deleted_count

        except Exception as e:
            logger.error(f"Cache cleanup error: {str(e)}")
            return 0

    def _serialize_data(self, data: Any, method: str, compress: bool) -> bytes:
        """序列化数据"""
        if method == 'json':
            serialized = json.dumps(data).encode('utf-8')
        elif method == 'pickle':
            serialized = pickle.dumps(data)
        else:
            raise ValueError(f"Unsupported serialize method: {method}")

        if compress:
            serialized = gzip.compress(serialized)

        return serialized

    def _deserialize_data(self, data: bytes, method: str, compressed: bool) -> Any:
        """反序列化数据"""
        if compressed:
            data = gzip.decompress(data)

        if method == 'json':
            return json.loads(data.decode('utf-8'))
        elif method == 'pickle':
            return pickle.loads(data)
        else:
            raise ValueError(f"Unsupported serialize method: {method}")

    def _get_redis_size(self) -> int:
        """获取Redis数据库大小"""
        try:
            if self.redis_client:
                return self.redis_client.dbsize()
        except:
            pass
        return 0

    def cache_decorator(self, ttl: Optional[int] = None,
                        key_prefix: str = 'func',
                        compress: bool = False,
                        serialize_method: str = 'pickle'):
        """缓存装饰器"""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self.generate_key(
                    f"{key_prefix}:{func.__name__}",
                    *args,
                    **kwargs
                )

                # 尝试从缓存获取
                cached_result = self.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.set(cache_key, result, ttl, compress, serialize_method)

                return result

            return wrapper

        return decorator

    def get_or_set(self, key: str, callback: Callable,
                   ttl: Optional[int] = None,
                   compress: bool = False,
                   serialize_method: str = 'pickle') -> Any:
        """获取缓存，如果不存在则调用回调函数设置"""
        try:
            # 尝试获取缓存
            cached_value = self.get(key)
            if cached_value is not None:
                return cached_value

            # 缓存不存在，调用回调函数
            value = callback()

            # 设置缓存
            self.set(key, value, ttl, compress, serialize_method)

            return value

        except Exception as e:
            logger.error(f"get_or_set error for key {key}: {str(e)}")
            # 如果出错，直接调用回调函数
            return callback()

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存"""
        results = {}

        # Redis批量获取
        if self.use_redis and self.redis_client:
            try:
                redis_values = self.redis_client.mget(keys)
                for i, key in enumerate(keys):
                    if redis_values[i]:
                        try:
                            cache_value = pickle.loads(redis_values[i])
                            if time.time() - cache_value['timestamp'] < cache_value['ttl']:
                                results[key] = self._deserialize_data(
                                    cache_value['data'],
                                    cache_value['serialize_method'],
                                    cache_value['compressed']
                                )
                        except Exception as e:
                            logger.error(f"Error deserializing Redis value for key {key}: {str(e)}")
            except Exception as e:
                logger.error(f"Redis batch get error: {str(e)}")

        # 内存缓存补充
        current_time = time.time()
        with self._lock:
            for key in keys:
                if key not in results and key in self.memory_cache:
                    cache_item = self.memory_cache[key]
                    if current_time < cache_item['expire_time']:
                        try:
                            results[key] = self._deserialize_data(
                                cache_item['data'],
                                cache_item['serialize_method'],
                                cache_item['compressed']
                            )
                            # 更新LRU顺序
                            self.memory_cache.move_to_end(key)
                        except Exception as e:
                            logger.error(f"Error deserializing memory value for key {key}: {str(e)}")

        return results

    def batch_set(self, items: Dict[str, Any], ttl: Optional[int] = None,
                  compress: bool = False, serialize_method: str = 'pickle') -> Dict[str, bool]:
        """批量设置缓存"""
        results = {}

        for key, value in items.items():
            results[key] = self.set(key, value, ttl, compress, serialize_method)

        return results

    def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """递增缓存值"""
        try:
            # Redis递增
            if self.use_redis and self.redis_client:
                try:
                    result = self.redis_client.incr(key, amount)
                    if ttl:
                        self.redis_client.expire(key, ttl)
                    return result
                except Exception as e:
                    logger.error(f"Redis increment error for key {key}: {str(e)}")

            # 内存缓存递增
            with self._lock:
                current_value = self.get(key, 0)
                if isinstance(current_value, (int, float)):
                    new_value = current_value + amount
                    self.set(key, new_value, ttl)
                    return new_value
                else:
                    # 如果当前值不是数字，设置为amount
                    self.set(key, amount, ttl)
                    return amount

        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {str(e)}")
            return None

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        try:
            memory_usage = 0
            item_count = 0

            with self._lock:
                for key, cache_item in self.memory_cache.items():
                    try:
                        memory_usage += sys.getsizeof(key)
                        memory_usage += sys.getsizeof(cache_item['data'])
                        memory_usage += sys.getsizeof(cache_item)
                        item_count += 1
                    except:
                        continue

            return {
                'memory_usage_bytes': memory_usage,
                'memory_usage_mb': memory_usage / (1024 * 1024),
                'item_count': item_count,
                'average_item_size': memory_usage / item_count if item_count > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error getting memory usage: {str(e)}")
            return {}

    def close(self):
        """关闭缓存连接"""
        try:
            if self.redis_client:
                self.redis_client.close()
                logger.info("Redis connection closed")

            with self._lock:
                self.memory_cache.clear()
                self.cache_stats['memory_size'] = 0

            logger.info("Cache connections closed")

        except Exception as e:
            logger.error(f"Error closing cache connections: {str(e)}")


# 全局缓存实例
_cache_instance = None
_cache_lock = threading.Lock()


def get_cache_instance(**kwargs) -> CacheUtils:
    """获取全局缓存实例（单例模式）"""
    global _cache_instance

    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = CacheUtils(**kwargs)

    return _cache_instance


def cache_function(ttl: Optional[int] = None,
                   key_prefix: str = 'func',
                   compress: bool = False,
                   serialize_method: str = 'pickle'):
    """函数缓存装饰器的便捷方法"""
    cache = get_cache_instance()
    return cache.cache_decorator(ttl, key_prefix, compress, serialize_method)


# 便捷函数
def cache_set(key: str, value: Any, ttl: Optional[int] = None, **kwargs) -> bool:
    """设置缓存的便捷函数"""
    cache = get_cache_instance()
    return cache.set(key, value, ttl, **kwargs)


def cache_get(key: str, default: Any = None) -> Any:
    """获取缓存的便捷函数"""
    cache = get_cache_instance()
    return cache.get(key, default)


def cache_delete(key: str) -> bool:
    """删除缓存的便捷函数"""
    cache = get_cache_instance()
    return cache.delete(key)


def cache_clear(pattern: Optional[str] = None) -> int:
    """清除缓存的便捷函数"""
    cache = get_cache_instance()
    return cache.clear(pattern)


def cache_stats() -> Dict[str, Any]:
    """获取缓存统计的便捷函数"""
    cache = get_cache_instance()
    return cache.get_stats()
