# -*- coding: utf-8 -*-
"""
Playwright 浏览器管理器
Playwright Browser Manager
"""
import asyncio
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = Any
    Browser = Any
    BrowserContext = Any
    Playwright = Any

from ..config import BrowserConfig
from ..anti_crawler.fingerprint import FingerprintManager, Fingerprint
from src.utils.logging import Logger, setup_logger


@dataclass
class PageResult:
    """页面爬取结果"""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    raw_html: Optional[str] = None
    screenshot: Optional[bytes] = None
    status_code: Optional[int] = None
    error_message: Optional[str] = None


class PlaywrightManager:
    """Playwright 浏览器管理器"""

    def __init__(
        self,
        config: BrowserConfig,
        logger: Optional[Logger] = None,
        fingerprint_manager: Optional[FingerprintManager] = None,
    ):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError(
                "Playwright is not installed. Install with: pip install playwright && playwright install"
            )

        self.config = config
        self.logger = logger or setup_logger(experiment_name="playwright")
        self.fingerprint_manager = fingerprint_manager

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()

    async def start(self):
        """启动浏览器"""
        self._playwright = await async_playwright().start()

        browser_type = self.config.browser_type.lower()
        if browser_type == "chromium":
            self._browser = await self._playwright.chromium.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )
        elif browser_type == "firefox":
            self._browser = await self._playwright.firefox.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )
        elif browser_type == "webkit":
            self._browser = await self._playwright.webkit.launch(
                headless=self.config.headless,
                slow_mo=self.config.slow_mo,
            )
        else:
            raise ValueError(f"Unknown browser type: {browser_type}")

        # 创建上下文
        context_options = self._get_context_options()
        self._context = await self._browser.new_context(**context_options)

        self.logger.info(f"Browser started: {browser_type} (headless={self.config.headless})")

    async def close(self):
        """关闭浏览器"""
        if self._context:
            await self._context.close()
            self._context = None

        if self._browser:
            await self._browser.close()
            self._browser = None

        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self.logger.info("Browser closed")

    def _get_context_options(self) -> Dict[str, Any]:
        """获取上下文选项"""
        options = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            "ignore_https_errors": True,
        }

        if self.config.user_agent:
            options["user_agent"] = self.config.user_agent

        # 注入指纹
        if self.fingerprint_manager and self.fingerprint_manager.fingerprint:
            fp = self.fingerprint_manager.fingerprint
            options["viewport"] = {
                "width": fp.screen_width,
                "height": fp.screen_height,
            }

        return options

    async def new_page(self) -> Page:
        """创建新页面"""
        if self._context is None:
            raise RuntimeError("Browser not started")
        return await self._context.new_page()

    async def navigate(
        self,
        url: str,
        wait_until: str = None,
        timeout: int = None,
    ) -> PageResult:
        """
        导航到URL并等待页面加载

        Args:
            url: 目标URL
            wait_until: 等待条件
            timeout: 超时时间（毫秒）

        Returns:
            PageResult 对象
        """
        page = await self.new_page()
        result = PageResult(url=url)

        if timeout is None:
            timeout = self.config.timeout

        if wait_until is None:
            wait_until = self.config.wait_until

        try:
            # 导航
            response = await page.goto(url, wait_until=wait_until, timeout=timeout)
            result.status_code = response.status if response else None

            # 获取内容
            result.raw_html = await page.content()
            result.title = await page.title()

            # 提取正文
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(result.raw_html, 'lxml')

            # 移除脚本和样式
            for tag in soup(['script', 'style']):
                tag.decompose()

            result.content = soup.get_text(separator='\n', strip=True)

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"Navigation failed: {url} - {e}")

        finally:
            await page.close()

        return result

    async def click_and_wait(
        self,
        selector: str,
        wait_selector: Optional[str] = None,
        timeout: int = 30000,
    ) -> bool:
        """
        点击元素并等待

        Args:
            selector: 点击的选择器
            wait_selector: 等待出现的选择器
            timeout: 超时时间

        Returns:
            是否成功
        """
        page = await self.new_page()
        try:
            await page.click(selector, timeout=timeout)
            if wait_selector:
                await page.wait_for_selector(wait_selector, timeout=timeout)
            return True
        except Exception as e:
            self.logger.warning(f"Click failed: {selector} - {e}")
            return False
        finally:
            await page.close()

    async def scroll_and_extract(
        self,
        url: str,
        max_scrolls: int = 10,
        scroll_delay: float = 1.0,
    ) -> PageResult:
        """
        滚动页面并提取内容（用于加载更多内容的页面）

        Args:
            url: 目标URL
            max_scrolls: 最大滚动次数
            scroll_delay: 滚动间隔（秒）

        Returns:
            PageResult 对象
        """
        page = await self.new_page()
        result = PageResult(url=url)

        try:
            await page.goto(url, wait_until=self.config.wait_until, timeout=self.config.timeout)

            # 滚动加载
            for _ in range(max_scrolls):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(scroll_delay)

                # 检查是否还有更多内容
                before_height = await page.evaluate("document.body.scrollHeight")
                await asyncio.sleep(scroll_delay)
                after_height = await page.evaluate("document.body.scrollHeight")

                if before_height == after_height:
                    break

            result.raw_html = await page.content()
            result.title = await page.title()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(result.raw_html, 'lxml')
            for tag in soup(['script', 'style']):
                tag.decompose()
            result.content = soup.get_text(separator='\n', strip=True)

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"Scroll and extract failed: {url} - {e}")

        finally:
            await page.close()

        return result

    async def fill_form_and_submit(
        self,
        url: str,
        form_data: Dict[str, str],
        submit_selector: str,
    ) -> PageResult:
        """
        填写表单并提交

        Args:
            url: 表单页面URL
            form_data: 表单数据 {selector: value}
            submit_selector: 提交按钮选择器

        Returns:
            PageResult 对象
        """
        page = await self.new_page()
        result = PageResult(url=url)

        try:
            await page.goto(url, wait_until=self.config.wait_until, timeout=self.config.timeout)

            # 填写表单
            for selector, value in form_data.items():
                await page.fill(selector, value)

            # 点击提交
            await page.click(submit_selector)

            # 等待导航完成
            await page.wait_for_load_state(self.config.wait_until, timeout=self.config.timeout)

            result.raw_html = await page.content()
            result.title = await page.title()

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(result.raw_html, 'lxml')
            for tag in soup(['script', 'style']):
                tag.decompose()
            result.content = soup.get_text(separator='\n', strip=True)

        except Exception as e:
            result.error_message = str(e)
            self.logger.error(f"Form submit failed: {url} - {e}")

        finally:
            await page.close()

        return result

    async def take_screenshot(self, url: str, path: str) -> bool:
        """
        截取页面快照

        Args:
            url: 目标URL
            path: 保存路径

        Returns:
            是否成功
        """
        page = await self.new_page()
        try:
            await page.goto(url, wait_until=self.config.wait_until, timeout=self.config.timeout)
            await page.screenshot(path=path, full_page=True)
            return True
        except Exception as e:
            self.logger.error(f"Screenshot failed: {url} - {e}")
            return False
        finally:
            await page.close()
