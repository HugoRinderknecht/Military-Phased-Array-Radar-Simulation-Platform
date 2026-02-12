# 环形缓冲区模块 (Ring Buffer Module)
# 本模块提供高效的环形缓冲区实现

import numpy as np
from typing import TypeVar, Generic, List, Optional, Any
from collections import deque
import threading


T = TypeVar('T')


class RingBuffer(Generic[T]):
    """
    环形缓冲区类 (Ring Buffer Class)
    
    环形缓冲区是一种固定大小的数据结构，当缓冲区满时，新数据会覆盖最旧的数据。
    这在需要维护固定长度历史数据的场景中非常有用。
    
    Attributes:
        capacity: 缓冲区容量
        buffer: 数据存储数组
        write_index: 写入索引
        read_index: 读取索引
        size: 当前元素数量
        lock: 线程锁（用于线程安全操作）
    
    Examples:
        >>> rb = RingBuffer(capacity=5)
        >>> rb.write(1)
        >>> rb.write(2)
        >>> rb.write(3)
        >>> rb.read()
        1
        >>> rb.write(4)
        >>> rb.write(5)
        >>> rb.write(6)  # 覆盖最旧的元素1
        >>> rb.get_all()
        [2, 3, 4, 5, 6]
    """
    
    def __init__(self, capacity: int, dtype: type = object):
        """
        初始化环形缓冲区
        
        Args:
            capacity: 缓冲区容量（必须大于0）
            dtype: 数据类型（用于NumPy数组）
        
        Raises:
            ValueError: 如果capacity小于等于0
        """
        if capacity <= 0:
            raise ValueError(f'容量必须大于0, 得到: {capacity}')
        
        self.capacity = capacity
        self.dtype = dtype
        
        if dtype == object:
            self.buffer: List[Any] = [None] * capacity
        else:
            self.buffer = np.zeros(capacity, dtype=dtype)
        
        self.write_index = 0
        self.read_index = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def write(self, item: T) -> bool:
        """
        写入单个元素
        
        Args:
            item: 要写入的元素
        
        Returns:
            True表示写入成功（可能会覆盖旧数据）
        """
        with self.lock:
            self.buffer[self.write_index] = item
            self.write_index = (self.write_index + 1) % self.capacity
            
            if self.size < self.capacity:
                self.size += 1
            else:
                # 缓冲区已满，移动读取指针（覆盖最旧数据）
                self.read_index = (self.read_index + 1) % self.capacity
            
            return True
    
    def read(self) -> Optional[T]:
        """
        读取单个元素（FIFO顺序）
        
        Returns:
            读取的元素，如果缓冲区为空则返回None
        """
        with self.lock:
            if self.size == 0:
                return None
            
            item = self.buffer[self.read_index]
            self.read_index = (self.read_index + 1) % self.capacity
            self.size -= 1
            
            return item
    
    def write_batch(self, items: List[T]) -> int:
        """
        批量写入元素
        
        Args:
            items: 要写入的元素列表
        
        Returns:
            实际写入的元素数量
        """
        count = 0
        for item in items:
            if self.write(item):
                count += 1
        return count
    
    def read_batch(self, count: int) -> List[T]:
        """
        批量读取元素
        
        Args:
            count: 要读取的元素数量
        
        Returns:
            读取的元素列表
        """
        result = []
        for _ in range(min(count, self.size)):
            item = self.read()
            if item is not None:
                result.append(item)
        return result
    
    def peek(self, offset: int = 0) -> Optional[T]:
        """
        查看指定偏移位置的元素（不移动读取指针）
        
        Args:
            offset: 偏移量（0表示当前读取指针位置）
        
        Returns:
            查看的元素，如果偏移超出范围则返回None
        """
        with self.lock:
            if offset >= self.size:
                return None
            
            index = (self.read_index + offset) % self.capacity
            return self.buffer[index]
    
    def get_latest(self, n: int) -> List[T]:
        """
        获取最新的n个元素（不移动指针）
        
        Args:
            n: 要获取的元素数量
        
        Returns:
            最新的n个元素列表
        """
        with self.lock:
            n = min(n, self.size)
            result = []
            
            for i in range(n):
                # 从write_index倒推
                index = (self.write_index - 1 - i) % self.capacity
                result.insert(0, self.buffer[index])
            
            return result
    
    def get_all(self) -> List[T]:
        """
        获取所有元素（不移动指针，按写入顺序）
        
        Returns:
            所有元素的列表
        """
        with self.lock:
            result = []
            for i in range(self.size):
                index = (self.read_index + i) % self.capacity
                result.append(self.buffer[index])
            return result
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self.lock:
            self.write_index = 0
            self.read_index = 0
            self.size = 0
            
            if self.dtype == object:
                self.buffer = [None] * self.capacity
            else:
                self.buffer = np.zeros(self.capacity, dtype=self.dtype)
    
    def is_empty(self) -> bool:
        """检查缓冲区是否为空"""
        return self.size == 0
    
    def is_full(self) -> bool:
        """检查缓冲区是否已满"""
        return self.size == self.capacity
    
    def __len__(self) -> int:
        """返回当前元素数量"""
        return self.size
    
    def __repr__(self) -> str:
        return f'RingBuffer(capacity={self.capacity}, size={self.size})'


class NumpyRingBuffer:
    """
    NumPy优化的环形缓冲区 (Numpy-optimized Ring Buffer)
    
    专门用于NumPy数组的环形缓冲区，提供更高的性能。
    
    Attributes:
        capacity: 缓冲区容量
        shape: 数组形状
        dtype: 数据类型
        buffer: 数据存储（二维数组）
        index: 当前写入索引
        count: 当前元素数量
    """
    
    def __init__(self, capacity: int, shape: tuple, dtype: type = np.float32):
        """
        初始化NumPy环形缓冲区
        
        Args:
            capacity: 缓冲区容量
            shape: 单个数组的形状
            dtype: 数据类型
        """
        if capacity <= 0:
            raise ValueError(f'容量必须大于0, 得到: {capacity}')
        
        self.capacity = capacity
        self.shape = shape
        self.dtype = dtype
        
        # 预分配二维数组 [capacity, *shape]
        self.buffer = np.zeros((capacity,) + shape, dtype=dtype)
        self.index = 0
        self.count = 0
        self.lock = threading.Lock()
    
    def write(self, data: np.ndarray) -> None:
        """
        写入NumPy数组
        
        Args:
            data: 要写入的数组
        """
        with self.lock:
            self.buffer[self.index] = data
            self.index = (self.index + 1) % self.capacity
            
            if self.count < self.capacity:
                self.count += 1
    
    def get_latest(self, n: int) -> np.ndarray:
        """
        获取最新的n个数组
        
        Args:
            n: 要获取的数量
        
        Returns:
            形状为(n, *shape)的数组
        """
        with self.lock:
            n = min(n, self.count)
            if n == 0:
                return np.zeros((0,) + self.shape, dtype=self.dtype)
            
            result = np.zeros((n,) + self.shape, dtype=self.dtype)
            for i in range(n):
                idx = (self.index - 1 - i) % self.capacity
                result[n - 1 - i] = self.buffer[idx]
            
            return result
    
    def get_all(self) -> np.ndarray:
        """
        获取所有数组
        
        Returns:
            形状为(count, *shape)的数组
        """
        return self.get_latest(self.count)
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self.lock:
            self.buffer.fill(0)
            self.index = 0
            self.count = 0
    
    def __len__(self) -> int:
        return self.count


__all__ = ['RingBuffer', 'NumpyRingBuffer']
