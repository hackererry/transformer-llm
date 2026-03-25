# -*- coding: utf-8 -*-
"""
Redis 缓存模块
Redis Cache Module
"""
import json
import hashlib
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
from datetime import timedelta

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    ttl: Optional[int] = None
    created_at: Optional[float] = None

    def to_bytes(self) -> bytes:
        """序列化为字节"""
        data = {
            'key': self.key,
            'value': self.value,
            'ttl': self.ttl,
            'created_at': self.created_at,
        }
        return json.dumps(data, ensure_ascii=False).encode('utf-8')

    @classmethod
    def from_bytes(cls, data: bytes) -> 'CacheEntry':
        """从字节反序列化"""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            key=obj['key'],
            value=obj['value'],
            ttl=obj.get('ttl'),
            created_at=obj.get('created_at'),
        )


class RedisCache:
    """Redis 缓存"""

    # 默认配置
    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 6379
    DEFAULT_DB = 0
    DEFAULT_PASSWORD = None

    # Key 前缀
    PREFIX_URL = "crawler:url:"
    PREFIX_PAGE = "crawler:page:"
    PREFIX_RATE = "crawler:rate:"
    PREFIX_LOCK = "crawler:lock:"

    # 默认 TTL
    DEFAULT_TTL = 3600 * 24  # 1天
    PAGE_TTL = 3600 * 24 * 7  # 7天
    RATE_TTL = 60  # 1分钟

    def __init__(
        self,
        host: str = None,
        port: int = None,
        db: int = None,
        password: str = None,
        decode_responses: bool = False,
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")

        import redis as redis_module
        self._redis_module = redis_module

        self.host = host or self.DEFAULT_HOST
        self.port = port or self.DEFAULT_PORT
        self.db = db if db is not None else self.DEFAULT_DB
        self.password = password or self.DEFAULT_PASSWORD
        self.decode_responses = decode_responses

        self._client: Optional[Any] = None

    @property
    def client(self):
        """获取 Redis 客户端"""
        if self._client is None:
            self._client = self._redis_module.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=self.decode_responses,
            )
        return self._client

    def close(self):
        """关闭连接"""
        if self._client:
            self._client.close()
            self._client = None

    # ========== URL 缓存 ==========

    def is_url_crawled(self, url: str) -> bool:
        """检查 URL 是否已爬取"""
        key = f"{self.PREFIX_URL}{self._hash_key(url)}"
        return self.client.exists(key) > 0

    def mark_url_crawled(self, url: str, ttl: int = None) -> bool:
        """标记 URL 已爬取"""
        key = f"{self.PREFIX_URL}{self._hash_key(url)}"
        ttl = ttl or self.DEFAULT_TTL
        return self.client.setex(key, ttl, "1") is True

    def get_crawled_urls(self, pattern: str = "*", limit: int = 1000) -> List[str]:
        """获取已爬取的 URL 列表"""
        keys = self.client.keys(f"{self.PREFIX_URL}{pattern}")
        return [k.replace(f"{self.PREFIX_URL}", "") for k in keys[:limit]]

    # ========== 页面缓存 ==========

    def cache_page(self, url: str, content: str, metadata: Dict = None, ttl: int = None) -> bool:
        """缓存页面内容"""
        key = f"{self.PREFIX_PAGE}{self._hash_key(url)}"
        ttl = ttl or self.PAGE_TTL

        data = {
            'url': url,
            'content': content,
            'metadata': metadata or {},
        }

        return self.client.setex(key, ttl, json.dumps(data, ensure_ascii=False)) is True

    def get_cached_page(self, url: str) -> Optional[Dict]:
        """获取缓存的页面"""
        key = f"{self.PREFIX_PAGE}{self._hash_key(url)}"
        data = self.client.get(key)

        if data:
            if self.decode_responses:
                return json.loads(data)
            return json.loads(data.decode('utf-8'))
        return None

    def delete_cached_page(self, url: str) -> bool:
        """删除缓存的页面"""
        key = f"{self.PREFIX_PAGE}{self._hash_key(url)}"
        return self.client.delete(key) > 0

    # ========== 速率限制 ==========

    def check_rate_limit(self, domain: str, max_requests: int, window: int = 60) -> bool:
        """
        检查速率限制

        Args:
            domain: 域名
            max_requests: 时间窗口内最大请求数
            window: 时间窗口（秒）

        Returns:
            是否允许请求
        """
        key = f"{self.PREFIX_RATE}{domain}"

        # 使用滑动窗口计数器
        now = self.client.time()[0]
        window_start = now - window

        # 移除旧记录
        self.client.zremrangebyscore(key, 0, window_start)

        # 获取当前计数
        current_count = self.client.zcard(key)

        if current_count >= max_requests:
            return False

        # 添加新记录
        self.client.zadd(key, {str(now): now})
        self.client.expire(key, window)

        return True

    def get_rate_limit_remaining(self, domain: str, max_requests: int, window: int = 60) -> int:
        """获取剩余请求次数"""
        key = f"{self.PREFIX_RATE}{domain}"

        now = self.client.time()[0]
        window_start = now - window

        self.client.zremrangebyscore(key, 0, window_start)
        current_count = self.client.zcard(key)

        return max(0, max_requests - current_count)

    # ========== 分布式锁 ==========

    def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 10,
        blocking: bool = True,
        blocking_timeout: int = 5,
    ) -> bool:
        """
        获取分布式锁

        Args:
            lock_name: 锁名称
            timeout: 锁超时时间（秒）
            blocking: 是否阻塞等待
            blocking_timeout: 阻塞超时时间

        Returns:
            是否获取成功
        """
        key = f"{self.PREFIX_LOCK}{lock_name}"

        if blocking:
            end_time = self.client.time()[0] + blocking_timeout
            while self.client.time()[0] < end_time:
                if self.client.set(key, "1", nx=True, ex=timeout):
                    return True
                import time
                time.sleep(0.1)
            return False
        else:
            return self.client.set(key, "1", nx=True, ex=timeout) is True

    def release_lock(self, lock_name: str) -> bool:
        """释放分布式锁"""
        key = f"{self.PREFIX_LOCK}{lock_name}"
        return self.client.delete(key) > 0

    # ========== 通用操作 ==========

    def get(self, key: str) -> Optional[str]:
        """获取值"""
        return self.client.get(key)

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        """设置值"""
        if ttl:
            return self.client.setex(key, ttl, value) is True
        return self.client.set(key, value) is True

    def delete(self, key: str) -> bool:
        """删除键"""
        return self.client.delete(key) > 0

    def exists(self, key: str) -> bool:
        """检查键是否存在"""
        return self.client.exists(key) > 0

    def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        return self.client.expire(key, ttl) is True

    def ttl(self, key: str) -> int:
        """获取剩余 TTL"""
        return self.client.ttl(key)

    def keys(self, pattern: str = "*") -> List[str]:
        """获取匹配的键"""
        return [k.decode() if isinstance(k, bytes) else k for k in self.client.keys(pattern)]

    def flush_pattern(self, pattern: str) -> int:
        """删除匹配的所有键"""
        keys = self.keys(pattern)
        if keys:
            return self.client.delete(*keys)
        return 0

    # ========== 辅助方法 ==========

    def _hash_key(self, key: str) -> str:
        """对 key 进行哈希"""
        return hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]

    def ping(self) -> bool:
        """检查连接"""
        try:
            return self.client.ping()
        except Exception:
            return False

    def info(self) -> Dict:
        """获取 Redis 信息"""
        return self.client.info()


# ========== 便捷函数 ==========

def create_redis_cache(
    url: str = None,
    host: str = None,
    port: int = None,
    db: int = None,
    password: str = None,
) -> RedisCache:
    """
    创建 Redis 缓存

    Args:
        url: Redis URL (优先使用)
        host: 主机名
        port: 端口
        db: 数据库编号
        password: 密码

    Returns:
        RedisCache 实例
    """
    if url:
        # 从 URL 解析
        import re
        match = re.match(r'redis://(:[^@]+@)?([^:]+):(\d+)/(\d+)', url)
        if match:
            password = match.group(1).strip(':@') if match.group(1) else None
            host = match.group(2)
            port = int(match.group(3))
            db = int(match.group(4))

    return RedisCache(
        host=host or RedisCache.DEFAULT_HOST,
        port=port or RedisCache.DEFAULT_PORT,
        db=db or RedisCache.DEFAULT_DB,
        password=password,
    )
