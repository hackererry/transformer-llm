# -*- coding: utf-8 -*-
"""
代理池模块
Proxy Pool Module
"""
import asyncio
import random
import time
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from urllib.parse import urlparse

import httpx


@dataclass
class Proxy:
    """代理信息"""
    url: str
    proxy_type: str = "http"
    host: str = ""
    port: int = 0
    username: Optional[str] = None
    password: Optional[str] = None

    # 健康检查
    is_alive: bool = True
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    # 统计
    success_count: int = 0
    failure_count: int = 0
    total_response_time: float = 0.0

    @property
    def avg_response_time(self) -> float:
        """平均响应时间"""
        if self.success_count == 0:
            return float('inf')
        return self.total_response_time / self.success_count

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        return self.success_count / total

    @classmethod
    def from_url(cls, url: str) -> "Proxy":
        """从 URL 创建代理"""
        parsed = urlparse(url)
        proxy_type = parsed.scheme or "http"
        host = parsed.hostname or ""
        port = parsed.port or 8080
        username = parsed.username
        password = parsed.password

        return cls(
            url=url,
            proxy_type=proxy_type,
            host=host,
            port=port,
            username=username,
            password=password,
        )


class ProxyPool:
    """代理池"""

    def __init__(
        self,
        proxies: Optional[List[str]] = None,
        check_url: str = "https://httpbin.org/ip",
        check_timeout: float = 10.0,
        check_interval: float = 300.0,
        auto_check: bool = True,
        min_success_rate: float = 0.5,
    ):
        """
        初始化代理池

        Args:
            proxies: 代理 URL 列表
            check_url: 健康检查 URL
            check_timeout: 检查超时时间
            check_interval: 检查间隔（秒）
            auto_check: 是否自动健康检查
            min_success_rate: 最小成功率，低于此值则标记为不健康
        """
        self.proxies: Dict[str, Proxy] = {}
        self.check_url = check_url
        self.check_timeout = check_timeout
        self.check_interval = check_interval
        self.auto_check = auto_check
        self.min_success_rate = min_success_rate

        self._last_check = datetime.min
        self._check_task: Optional[asyncio.Task] = None

        if proxies:
            for url in proxies:
                self.add(url)

    def add(self, proxy_url: str) -> bool:
        """添加代理"""
        if proxy_url in self.proxies:
            return False

        proxy = Proxy.from_url(proxy_url)
        self.proxies[proxy_url] = proxy
        return True

    def remove(self, proxy_url: str) -> bool:
        """移除代理"""
        if proxy_url in self.proxies:
            del self.proxies[proxy_url]
            return True
        return False

    def get(self, strategy: str = "random") -> Optional[str]:
        """
        获取一个可用代理

        Args:
            strategy: 选择策略 ("random", "least_used", "fastest", "round_robin")

        Returns:
            代理 URL，未找到可用代理则返回 None
        """
        alive_proxies = [p for p in self.proxies.values() if p.is_alive]

        if not alive_proxies:
            return None

        if strategy == "random":
            return random.choice(alive_proxies).url
        elif strategy == "least_used":
            sorted_proxies = sorted(alive_proxies, key=lambda p: p.failure_count)
            return sorted_proxies[0].url
        elif strategy == "fastest":
            sorted_proxies = sorted(alive_proxies, key=lambda p: p.avg_response_time)
            return sorted_proxies[0].url
        elif strategy == "round_robin":
            for proxy in alive_proxies:
                if not hasattr(proxy, '_last_used') or not proxy._last_used:
                    proxy._last_used = datetime.min
                    return proxy.url
            sorted_proxies = sorted(alive_proxies, key=lambda p: p._last_used)
            return sorted_proxies[0].url
        else:
            return random.choice(alive_proxies).url

    def release(self, proxy_url: str, success: bool = True, response_time: float = 0.0):
        """
        释放代理（更新统计）

        Args:
            proxy_url: 代理 URL
            success: 请求是否成功
            response_time: 响应时间（秒）
        """
        if proxy_url not in self.proxies:
            return

        proxy = self.proxies[proxy_url]
        proxy.last_check = datetime.now()

        if success:
            proxy.success_count += 1
            proxy.last_success = datetime.now()
            proxy.total_response_time += response_time
            proxy.is_alive = proxy.success_rate >= self.min_success_rate
        else:
            proxy.failure_count += 1
            proxy.last_failure = datetime.now()
            proxy.is_alive = proxy.success_rate >= self.min_success_rate

    async def check_proxy(self, proxy_url: str) -> bool:
        """
        检查单个代理是否可用

        Returns:
            是否可用
        """
        if proxy_url not in self.proxies:
            return False

        proxy = self.proxies[proxy_url]
        start_time = time.time()

        try:
            async with httpx.AsyncClient(timeout=self.check_timeout) as client:
                response = await client.get(self.check_url, proxy=proxy_url)
                response_time = time.time() - start_time

                if response.status_code == 200:
                    self.release(proxy_url, success=True, response_time=response_time)
                    return True
                else:
                    self.release(proxy_url, success=False)
                    return False

        except Exception:
            self.release(proxy_url, success=False)
            return False

    async def check_all(self) -> Dict[str, bool]:
        """检查所有代理"""
        tasks = [self.check_proxy(url) for url in self.proxies.keys()]
        results = await asyncio.gather(*tasks)
        return dict(zip(self.proxies.keys(), results))

    async def start_auto_check(self):
        """启动自动健康检查"""
        if self._check_task is not None:
            return

        async def _check_loop():
            while True:
                try:
                    await asyncio.sleep(self.check_interval)
                    await self.check_all()
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass

        self._check_task = asyncio.create_task(_check_loop())

    async def stop_auto_check(self):
        """停止自动健康检查"""
        if self._check_task is not None:
            self._check_task.cancel()
            self._check_task = None

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.proxies)
        alive = sum(1 for p in self.proxies.values() if p.is_alive)

        success_rates = [p.success_rate for p in self.proxies.values() if p.success_count > 0]
        avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.0

        response_times = [p.avg_response_time for p in self.proxies.values() if p.success_count > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0

        return {
            'total': total,
            'alive': alive,
            'dead': total - alive,
            'avg_success_rate': avg_success_rate,
            'avg_response_time': avg_response_time,
        }

    def get_all_alive(self) -> List[str]:
        """获取所有存活的代理"""
        return [p.url for p in self.proxies.values() if p.is_alive]

    def filter_by_country(self, country: str) -> List[str]:
        """按国家过滤代理（需要外部IP数据库支持）"""
        return self.get_all_alive()


class ProxyMiddleware:
    """代理中间件，用于 httpx"""

    def __init__(self, pool: ProxyPool, strategy: str = "random"):
        self.pool = pool
        self.strategy = strategy
        self._current_proxy: Optional[str] = None

    async def __call__(self, request: httpx.Request) -> None:
        """修改请求使用代理"""
        proxy_url = self.pool.get(self.strategy)
        if proxy_url:
            self._current_proxy = proxy_url
            request.extensions["proxy"] = proxy_url

    def release(self, success: bool, response_time: float = 0.0):
        """释放当前代理"""
        if self._current_proxy:
            self.pool.release(self._current_proxy, success, response_time)
            self._current_proxy = None
