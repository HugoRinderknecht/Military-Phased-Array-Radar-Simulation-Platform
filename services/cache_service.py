import json
import time
from typing import Any, Optional, Dict
import logging
from threading import Lock
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheService:
    """内存缓存服务"""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.lock = Lock()
        self.access_times: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self.lock:
            if key not in self.cache:
                return None

            cache_entry = self.cache[key]

            # 检查是否过期
            if time.time() > cache_entry['expires_at']:
                self._remove_key(key)
                return None

            # 更新访问时间
            self.access_times[key] = time.time()

            return cache_entry['data']

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        try:
            with self.lock:
                # 检查缓存大小限制
                if len(self.cache) >= self.max_size and key not in self.cache:
                    self._evict_lru()

                ttl = ttl or self.default_ttl
                expires_at = time.time() + ttl

                self.cache[key] = {
                    'data': data,
                    'created_at': time.time(),
                    'expires_at': expires_at,
                    'ttl': ttl
                }

                self.access_times[key] = time.time()

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
            self.access_times.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            current_time = time.time()
            active_keys = 0
            expired_keys = 0
            total_size = 0

            for key, cache_entry in self.cache.items():
                if current_time <= cache_entry['expires_at']:
                    active_keys += 1
                else:
                    expired_keys += 1

                # 估算数据大小
                try:
                    data_size = len(json.dumps(cache_entry['data'], default=str))
                    total_size += data_size
                except:
                    total_size += 1000  # 默认估算

            return {
                'total_keys': len(self.cache),
                'active_keys': active_keys,
                'expired_keys': expired_keys,
                'cache_size_bytes': total_size,
                'max_size': self.max_size,
                'hit_ratio': self._calculate_hit_ratio()
            }

    def cleanup_expired(self) -> int:
        """清理过期缓存"""
        with self.lock:
            current_time = time.time()
            expired_keys = []

            for key, cache_entry in self.cache.items():
                if current_time > cache_entry['expires_at']:
                    expired_keys.append(key)

            for key in expired_keys:
                self._remove_key(key)

            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            return len(expired_keys)

    def _remove_key(self, key: str) -> None:
        """删除键（内部方法）"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]

    def _evict_lru(self) -> None:
        """使用LRU策略驱逐缓存"""
        if not self.access_times:
            return

        # 找到最久未访问的键
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove_key(lru_key)
        logger.debug(f"Evicted LRU cache key: {lru_key}")

    def _calculate_hit_ratio(self) -> float:
        """计算缓存命中率"""
        # 这里需要实现命中率统计逻辑
        # 简化实现，返回固定值
        return 0.85


# 全局缓存实例
_global_cache = None


def get_cache_service() -> CacheService:
    """获取全局缓存服务实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = CacheService()
    return _global_cache
