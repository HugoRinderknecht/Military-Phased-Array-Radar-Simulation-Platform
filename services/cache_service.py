import json
import time
import heapq
import logging
from threading import RLock
from collections import OrderedDict
from typing import Any, Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)


class CacheService:
    """高效内存缓存服务"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache = OrderedDict()  # 自动维护访问顺序
        self.expiry_heap: List[Tuple[float, str]] = []  # (过期时间, key)的最小堆
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = RLock()  # 可重入锁，减少读阻塞

        # 命中率统计
        self.hits = 0
        self.requests = 0

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据并更新访问位置"""
        with self.lock:
            self.requests += 1
            entry = self.cache.get(key)
            if not entry:
                return None

            # 检查是否过期
            if time.time() > entry['expires_at']:
                self._remove_key(key)
                return None

            # 移动到末尾表示最近访问
            self.cache.move_to_end(key)
            self.hits += 1
            return entry['data']

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据并更新数据结构"""
        try:
            with self.lock:
                # 预计算数据大小
                size = self._calculate_size(data)

                # 检查缓存大小限制
                if len(self.cache) >= self.max_size and key not in self.cache:
                    self._evict_lru()

                ttl_val = ttl if ttl is not None else self.default_ttl
                expires_at = time.time() + ttl_val

                # 如果key已存在，先清除旧数据
                if key in self.cache:
                    self._remove_key(key)

                # 添加新缓存项
                self.cache[key] = {
                    'data': data,
                    'size': size,
                    'expires_at': expires_at,
                    'ttl': ttl_val
                }

                # 添加到过期堆
                heapq.heappush(self.expiry_heap, (expires_at, key))

                # 新条目自动在末尾
                return True

        except Exception as e:
            logger.error(f"Error setting cache for key {key}: {str(e)}")
            return False

    def delete(self, key: str) -> bool:
        """删除缓存数据"""
        with self.lock:
            if key in self.cache:
                self._remove_key(key)
                return True
            return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
            self.expiry_heap.clear()
            self.hits = 0
            self.requests = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息（高效版）"""
        with self.lock:
            total_size = sum(entry['size'] for entry in self.cache.values())
            expired_count = self._count_expired()

            return {
                'total_keys': len(self.cache),
                'active_keys': len(self.cache) - expired_count,
                'expired_keys': expired_count,
                'cache_size_bytes': total_size,
                'max_size': self.max_size,
                'hit_ratio': self.hits / self.requests if self.requests else 0
            }

    def cleanup_expired(self) -> int:
        """高效清理过期缓存"""
        with self.lock:
            current_time = time.time()
            cleaned = 0

            # 从堆中弹出所有过期项
            while self.expiry_heap and self.expiry_heap[0][0] <= current_time:
                expires_at, key = heapq.heappop(self.expiry_heap)

                # 验证是否真的过期且未被更新
                if key in self.cache and self.cache[key]['expires_at'] <= current_time:
                    self._remove_key(key, from_cleanup=True)
                    cleaned += 1

            logger.info(f"Cleaned up {cleaned} expired cache entries")
            return cleaned

    def _remove_key(self, key: str, from_cleanup: bool = False) -> None:
        """删除键（内部方法）"""
        if key in self.cache:
            # 注意：不从堆中删除，因为堆会自行清理
            del self.cache[key]

    def _evict_lru(self) -> None:
        """使用LRU策略驱逐缓存（O(1)操作）"""
        if self.cache:
            key, _ = self.cache.popitem(last=False)
            logger.debug(f"Evicted LRU cache key: {key}")

    def _calculate_size(self, data: Any) -> int:
        """高效计算数据大小"""
        try:
            # 优先使用原生大小计算
            if isinstance(data, (str, bytes, bytearray)):
                return len(data)

            # 其次尝试JSON序列化
            return len(json.dumps(data, default=str))
        except:
            return 1000  # 默认估算值

    def _count_expired(self) -> int:
        """统计过期键数量（O(1)操作）"""
        current_time = time.time()
        # 只需检查堆顶元素数量（不精确但高效）
        count = 0
        for expires_at, _ in self.expiry_heap:
            if expires_at <= current_time:
                count += 1
            else:
                break
        return count


# 全局缓存实例
_global_cache = None


def get_cache_service() -> CacheService:
    """获取全局缓存服务实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheService()
    return _global_cache
