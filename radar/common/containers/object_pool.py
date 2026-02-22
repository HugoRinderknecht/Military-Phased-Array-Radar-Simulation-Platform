# 对象池模块 (Object Pool Module)

import queue
import threading
from typing import TypeVar, Generic, Type, Optional, Callable, Any
from abc import ABC, abstractmethod
import numpy as np


T = TypeVar('T')


class PoolableObject(ABC):
    """可池化对象基类"""
    
    @abstractmethod
    def reset(self) -> None:
        """重置对象到初始状态"""
        pass


class ObjectPool(Generic[T]):
    """通用对象池类"""
    
    def __init__(
        self,
        object_class: Type[T],
        initial_size: int = 10,
        max_size: int = 100,
        factory: Optional[Callable[[], T]] = None
    ):
        """初始化对象池"""
        if initial_size < 0:
            raise ValueError(f'初始大小不能为负数, 得到: {initial_size}')
        if max_size < 0:
            raise ValueError(f'最大大小不能为负数, 得到: {max_size}')
        if max_size > 0 and initial_size > max_size:
            raise ValueError(f'初始大小不能超过最大大小')
        
        self.object_class = object_class
        self.factory = factory or (lambda: object_class())
        self.initial_size = initial_size
        self.max_size = max_size
        
        self.pool: queue.Queue[T] = queue.Queue()
        self.lock = threading.Lock()
        self.created_count = 0
        self.acquired_count = 0
        self.released_count = 0
        
        # 预创建初始对象
        for _ in range(initial_size):
            obj = self.factory()
            self.pool.put(obj)
            self.created_count += 1
    
    def acquire(self, timeout: Optional[float] = None) -> Optional[T]:
        """从池中获取一个对象"""
        try:
            obj = self.pool.get(block=timeout is not None, timeout=timeout)
            with self.lock:
                self.acquired_count += 1
            return obj
        except queue.Empty:
            with self.lock:
                if self.max_size == 0 or self.created_count < self.max_size:
                    obj = self.factory()
                    self.created_count += 1
                    self.acquired_count += 1
                    return obj
            return None
    
    def release(self, obj: T) -> bool:
        """将对象归还到池中"""
        if isinstance(obj, PoolableObject):
            obj.reset()
        
        with self.lock:
            if self.max_size > 0:
                current_size = self.pool.qsize()
                if current_size >= self.max_size:
                    return False
            
            self.pool.put(obj)
            self.released_count += 1
            return True
    
    def acquire_multiple(self, count: int, timeout: Optional[float] = None) -> list:
        """批量获取对象"""
        objects = []
        for _ in range(count):
            obj = self.acquire(timeout=timeout)
            if obj is not None:
                objects.append(obj)
            else:
                break
        return objects
    
    def release_multiple(self, objects: list) -> int:
        """批量归还对象"""
        count = 0
        for obj in objects:
            if self.release(obj):
                count += 1
        return count
    
    def clear(self) -> int:
        """清空对象池"""
        count = 0
        while not self.pool.empty():
            try:
                self.pool.get_nowait()
                count += 1
            except queue.Empty:
                break
        return count
    
    def shrink(self, target_size: int) -> int:
        """缩小对象池到指定大小"""
        if target_size < 0:
            target_size = 0
        
        removed = 0
        current_size = self.pool.qsize()
        
        while current_size > target_size:
            try:
                self.pool.get_nowait()
                removed += 1
                current_size -= 1
            except queue.Empty:
                break
        
        return removed
    
    def get_stats(self) -> dict:
        """获取对象池统计信息"""
        with self.lock:
            return {
                'pool_size': self.pool.qsize(),
                'created_count': self.created_count,
                'acquired_count': self.acquired_count,
                'released_count': self.released_count,
                'active_count': self.created_count - self.released_count + self.pool.qsize(),
                'initial_size': self.initial_size,
                'max_size': self.max_size,
            }
    
    def __len__(self) -> int:
        return self.pool.qsize()


class NumpyArrayPool:
    """NumPy数组对象池"""

    def __init__(
        self,
        shape: tuple,
        dtype: type = np.float32,
        initial_size: int = 10,
        max_size: int = 100
    ):
        """初始化NumPy数组池"""
        self.shape = shape
        self.dtype = dtype
        self.max_size = max_size
        
        self.pool: list = []
        self.lock = threading.Lock()
        self.created_count = 0
        
        for _ in range(initial_size):
            self.pool.append(np.zeros(shape, dtype=dtype))
            self.created_count += 1
    
    def acquire(self) -> np.ndarray:
        """获取一个数组"""
        with self.lock:
            if self.pool:
                arr = self.pool.pop()
                arr.fill(0)
                return arr
            else:
                self.created_count += 1
                return np.zeros(self.shape, dtype=self.dtype)
    
    def release(self, arr: np.ndarray) -> bool:
        """归还数组到池中"""
        if arr.shape != self.shape or arr.dtype != self.dtype:
            return False
        
        with self.lock:
            if len(self.pool) < self.max_size:
                arr.fill(0)
                self.pool.append(arr)
                return True
            return False
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self.lock:
            return {
                'pool_size': len(self.pool),
                'created_count': self.created_count,
                'shape': self.shape,
                'dtype': str(self.dtype),
            }


__all__ = ['PoolableObject', 'ObjectPool', 'NumpyArrayPool', 'PooledObject']

# 别名
PooledObject = PoolableObject
