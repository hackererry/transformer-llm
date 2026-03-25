# -*- coding: utf-8 -*-
"""
User-Agent 轮换池
User-Agent Rotation Pool
"""
import random
from typing import List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime


@dataclass
class UserAgent:
    """User-Agent 信息"""
    ua: str
    browser: str
    os: str
    version: str
    last_used: Optional[datetime] = None
    use_count: int = 0


class UserAgentPool:
    """User-Agent 轮换池"""

    # 常用 User-Agent 列表
    DEFAULT_USER_AGENTS = [
        # Chrome on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",

        # Chrome on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",

        # Firefox on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",

        # Firefox on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",

        # Safari on macOS
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15",

        # Edge on Windows
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0",

        # Chrome on Linux
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",

        # Mobile Chrome on Android
        "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",

        # Safari on iOS
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
    ]

    def __init__(
        self,
        user_agents: Optional[List[str]] = None,
        rotate_strategy: str = "random",
        min_interval: float = 0.0,
    ):
        """
        初始化 UA 池

        Args:
            user_agents: UA 列表，为 None 则使用默认列表
            rotate_strategy: 轮换策略 ("random", "round_robin", "least_used")
            min_interval: 同一 UA 最小使用间隔（秒）
        """
        self.user_agents = user_agents or self.DEFAULT_USER_AGENTS
        self.rotate_strategy = rotate_strategy
        self.min_interval = min_interval

        self._ua_list: List[UserAgent] = [
            UserAgent(
                ua=ua,
                browser=self._detect_browser(ua),
                os=self._detect_os(ua),
                version=self._detect_version(ua),
            )
            for ua in self.user_agents
        ]

        self._index = 0
        self._last_rotation = datetime.min

    def _detect_browser(self, ua: str) -> str:
        """检测浏览器类型"""
        ua_lower = ua.lower()
        if 'edg/' in ua_lower or 'edge/' in ua_lower:
            return 'edge'
        if 'chrome/' in ua_lower and 'chromium/' not in ua_lower:
            return 'chrome'
        if 'safari/' in ua_lower and 'chrome/' not in ua_lower:
            return 'safari'
        if 'firefox/' in ua_lower:
            return 'firefox'
        if 'opera/' in ua_lower or 'opr/' in ua_lower:
            return 'opera'
        return 'unknown'

    def _detect_os(self, ua: str) -> str:
        """检测操作系统"""
        ua_lower = ua.lower()
        if 'windows nt 10' in ua_lower:
            return 'windows_10'
        if 'windows nt 6.3' in ua_lower:
            return 'windows_8.1'
        if 'windows' in ua_lower:
            return 'windows'
        if 'mac os x' in ua_lower:
            return 'macos'
        if 'linux' in ua_lower and 'android' not in ua_lower:
            return 'linux'
        if 'android' in ua_lower:
            return 'android'
        if 'iphone' in ua_lower or 'ipad' in ua_lower:
            return 'ios'
        return 'unknown'

    def _detect_version(self, ua: str) -> str:
        """检测浏览器版本"""
        import re
        patterns = [
            (r'chrome/(\d+)', 'Chrome'),
            (r'firefox/(\d+)', 'Firefox'),
            (r'safari/(\d+)', 'Safari'),
            (r'edg/(\d+)', 'Edge'),
        ]
        ua_lower = ua.lower()
        for pattern, browser in patterns:
            match = re.search(pattern, ua_lower)
            if match:
                return match.group(1)
        return '0'

    def get(self) -> str:
        """获取一个 User-Agent"""
        if not self.user_agents:
            return self.DEFAULT_USER_AGENTS[0]

        now = datetime.now()

        if self.rotate_strategy == "random":
            return self._get_random()
        elif self.rotate_strategy == "round_robin":
            return self._get_round_robin()
        elif self.rotate_strategy == "least_used":
            return self._get_least_used()
        else:
            return self._get_random()

    def _get_random(self) -> str:
        """随机获取"""
        return random.choice(self.user_agents)

    def _get_round_robin(self) -> str:
        """轮询获取"""
        ua = self.user_agents[self._index]
        self._index = (self._index + 1) % len(self.user_agents)
        return ua

    def _get_least_used(self) -> str:
        """获取使用次数最少的"""
        sorted_uas = sorted(
            self._ua_list,
            key=lambda x: (x.use_count, x.last_used or datetime.min)
        )
        return sorted_uas[0].ua

    def release(self, ua: str):
        """释放一个 User-Agent（更新使用统计）"""
        for ua_obj in self._ua_list:
            if ua_obj.ua == ua:
                ua_obj.use_count += 1
                ua_obj.last_used = datetime.now()
                break

    def get_for_browser(self, browser: str) -> Optional[str]:
        """获取指定浏览器的 UA"""
        matching = [ua for ua in self._ua_list if ua.browser == browser]
        if matching:
            return random.choice(matching).ua
        return None

    def get_for_os(self, os: str) -> Optional[str]:
        """获取指定操作系统的 UA"""
        matching = [ua for ua in self._ua_list if ua.os == os]
        if matching:
            return random.choice(matching).ua
        return None

    def get_desktop(self) -> str:
        """获取桌面浏览器的 UA"""
        desktop_uas = [ua for ua in self._ua_list if ua.os in ['windows', 'macos', 'linux']]
        if desktop_uas:
            return random.choice(desktop_uas).ua
        return self.user_agents[0]

    def get_mobile(self) -> str:
        """获取移动浏览器的 UA"""
        mobile_uas = [ua for ua in self._ua_list if ua.os in ['android', 'ios']]
        if mobile_uas:
            return random.choice(mobile_uas).ua
        return self.user_agents[0]

    def stats(self) -> dict:
        """获取统计信息"""
        return {
            'total': len(self.user_agents),
            'by_browser': {
                browser: len([ua for ua in self._ua_list if ua.browser == browser])
                for browser in set(ua.browser for ua in self._ua_list)
            },
            'by_os': {
                os: len([ua for ua in self._ua_list if ua.os == os])
                for os in set(ua.os for ua in self._ua_list)
            },
        }


class UserAgentMiddleware:
    """User-Agent 中间件，用于 httpx"""

    def __init__(self, pool: UserAgentPool):
        self.pool = pool

    async def __call__(self, request) -> None:
        """修改请求头"""
        import httpx
        request.headers['User-Agent'] = self.pool.get()
        self.pool.release(request.headers['User-Agent'])
