# -*- coding: utf-8 -*-
"""
等待策略模块
Wait Strategies Module
"""
from abc import ABC, abstractmethod
from typing import Optional, Callable

try:
    from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


class WaitStrategy(ABC):
    """等待策略基类"""

    @abstractmethod
    async def wait(self, page_or_locator):
        """执行等待"""
        pass


class NetworkIdle(WaitStrategy):
    """网络空闲等待策略"""

    def __init__(self, timeout: int = 30000):
        self.timeout = timeout

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            if isinstance(page_or_locator, Page):
                await page_or_locator.wait_for_load_state("networkidle", timeout=self.timeout)


class SelectorVisible(WaitStrategy):
    """选择器可见等待策略"""

    def __init__(self, selector: str, timeout: int = 30000, state: str = "visible"):
        self.selector = selector
        self.timeout = timeout
        self.state = state

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            if isinstance(page_or_locator, Page):
                await page_or_locator.wait_for_selector(self.selector, timeout=self.timeout, state=self.state)
            elif isinstance(page_or_locator, Locator):
                await page_or_locator.wait_for(timeout=self.timeout, state=self.state)


class SelectorHidden(WaitStrategy):
    """选择器隐藏等待策略"""

    def __init__(self, selector: str, timeout: int = 30000):
        self.selector = selector
        self.timeout = timeout

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            if isinstance(page_or_locator, Page):
                await page_or_locator.wait_for_selector(self.selector, timeout=self.timeout, state="hidden")
            elif isinstance(page_or_locator, Locator):
                await page_or_locator.wait_for(timeout=self.timeout, state="hidden")


class FunctionWait(WaitStrategy):
    """函数条件等待策略"""

    def __init__(
        self,
        function: Callable[[], bool],
        timeout: int = 30000,
        check_interval: float = 0.1,
    ):
        self.function = function
        self.timeout = timeout
        self.check_interval = check_interval

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            import asyncio
            from playwright.async_api import TimeoutError

            start_time = page_or_locator.page.context if isinstance(page_or_locator, Locator) else None

            elapsed = 0
            while elapsed < self.timeout:
                if self.function():
                    return
                await asyncio.sleep(self.check_interval)
                elapsed += self.check_interval

            raise TimeoutError(f"Function wait timeout after {self.timeout}ms")


class Timeout(WaitStrategy):
    """固定超时等待策略"""

    def __init__(self, timeout: float):
        self.timeout = timeout

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            import asyncio
            await asyncio.sleep(self.timeout)


class CustomWait(WaitStrategy):
    """自定义等待策略"""

    def __init__(self, script: str, timeout: int = 30000):
        self.script = script
        self.timeout = timeout

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            if isinstance(page_or_locator, Page):
                await page_or_locator.wait_for_function(self.script, timeout=self.timeout)


class WaitForLoadState(WaitStrategy):
    """加载状态等待策略"""

    def __init__(self, state: str = "load", timeout: int = 30000):
        self.state = state
        self.timeout = timeout

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            if isinstance(page_or_locator, Page):
                await page_or_locator.wait_for_load_state(self.state, timeout=self.timeout)


class WaitForURL(WaitStrategy):
    """URL模式等待策略"""

    def __init__(self, url_pattern: str, timeout: int = 30000):
        self.url_pattern = url_pattern
        self.timeout = timeout

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            if isinstance(page_or_locator, Page):
                await page_or_locator.wait_for_url(self.url_pattern, timeout=self.timeout)


class WaitAll(WaitStrategy):
    """组合等待策略 - 等待所有策略都满足"""

    def __init__(self, strategies: list):
        self.strategies = strategies

    async def wait(self, page_or_locator):
        for strategy in self.strategies:
            await strategy.wait(page_or_locator)


class WaitAny(WaitStrategy):
    """组合等待策略 - 任意一个策略满足即可"""

    def __init__(self, strategies: list):
        self.strategies = strategies

    async def wait(self, page_or_locator):
        if PLAYWRIGHT_AVAILABLE:
            import asyncio

            async def try_wait(strategy):
                try:
                    await strategy.wait(page_or_locator)
                    return True
                except Exception:
                    return False

            results = await asyncio.gather(
                *[try_wait(s) for s in self.strategies],
                return_exceptions=True
            )

            if not any(results):
                raise Exception("All wait strategies failed")
