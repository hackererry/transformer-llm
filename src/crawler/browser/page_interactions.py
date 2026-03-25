# -*- coding: utf-8 -*-
"""
页面交互模块
Page Interaction Module
"""
import asyncio
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass

try:
    from playwright.async_api import Page, Locator, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    Page = Any
    Locator = Any
    PlaywrightTimeout = Exception


@dataclass
class InteractionResult:
    """交互结果"""
    success: bool
    message: str
    data: Optional[Any] = None


class PageInteractions:
    """页面交互操作"""

    def __init__(self, page: Page):
        if not PLAYWRIGHT_AVAILABLE:
            raise ImportError("Playwright is not installed")
        self.page = page

    async def click(
        self,
        selector: str,
        timeout: int = 30000,
        delay: int = 0,
    ) -> InteractionResult:
        """点击元素"""
        try:
            if delay > 0:
                await self.page.locator(selector).click(delay=delay, timeout=timeout)
            else:
                await self.page.locator(selector).click(timeout=timeout)
            return InteractionResult(success=True, message=f"Clicked: {selector}")
        except PlaywrightTimeout:
            return InteractionResult(success=False, message=f"Timeout clicking: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Click failed: {selector} - {e}")

    async def fill(
        self,
        selector: str,
        value: str,
        timeout: int = 30000,
    ) -> InteractionResult:
        """填写输入框"""
        try:
            await self.page.locator(selector).fill(value, timeout=timeout)
            return InteractionResult(success=True, message=f"Filled: {selector}")
        except PlaywrightTimeout:
            return InteractionResult(success=False, message=f"Timeout filling: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Fill failed: {selector} - {e}")

    async def hover(self, selector: str, timeout: int = 30000) -> InteractionResult:
        """悬停到元素"""
        try:
            await self.page.locator(selector).hover(timeout=timeout)
            return InteractionResult(success=True, message=f"Hovered: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Hover failed: {selector} - {e}")

    async def select_option(
        self,
        selector: str,
        value: str,
        timeout: int = 30000,
    ) -> InteractionResult:
        """选择下拉选项"""
        try:
            await self.page.locator(selector).select_option(value, timeout=timeout)
            return InteractionResult(success=True, message=f"Selected: {selector} = {value}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Select failed: {selector} - {e}")

    async def check(self, selector: str, timeout: int = 30000) -> InteractionResult:
        """勾选复选框"""
        try:
            await self.page.locator(selector).check(timeout=timeout)
            return InteractionResult(success=True, message=f"Checked: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Check failed: {selector} - {e}")

    async def uncheck(self, selector: str, timeout: int = 30000) -> InteractionResult:
        """取消勾选复选框"""
        try:
            await self.page.locator(selector).uncheck(timeout=timeout)
            return InteractionResult(success=True, message=f"Unchecked: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Uncheck failed: {selector} - {e}")

    async def type_text(
        self,
        selector: str,
        text: str,
        delay: int = 100,
        timeout: int = 30000,
    ) -> InteractionResult:
        """逐字输入文本"""
        try:
            await self.page.locator(selector).type(text, delay=delay, timeout=timeout)
            return InteractionResult(success=True, message=f"Typed to: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Type failed: {selector} - {e}")

    async def press_key(
        self,
        selector: str,
        key: str,
        timeout: int = 30000,
    ) -> InteractionResult:
        """按键"""
        try:
            await self.page.locator(selector).press(key, timeout=timeout)
            return InteractionResult(success=True, message=f"Pressed {key} on: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Press failed: {selector} - {e}")

    async def scroll_to_element(
        self,
        selector: str,
        timeout: int = 30000,
    ) -> InteractionResult:
        """滚动到元素"""
        try:
            element = self.page.locator(selector)
            await element.scroll_into_view_if_needed(timeout=timeout)
            return InteractionResult(success=True, message=f"Scrolled to: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Scroll failed: {selector} - {e}")

    async def wait_for_selector(
        self,
        selector: str,
        timeout: int = 30000,
        state: str = "visible",
    ) -> InteractionResult:
        """等待选择器"""
        try:
            await self.page.locator(selector).wait_for(timeout=timeout, state=state)
            return InteractionResult(success=True, message=f"Selector visible: {selector}")
        except PlaywrightTimeout:
            return InteractionResult(success=False, message=f"Selector timeout: {selector}")
        except Exception as e:
            return InteractionResult(success=False, message=f"Wait failed: {selector} - {e}")

    async def wait_for_navigation(
        self,
        timeout: int = 30000,
        wait_until: str = "load",
    ) -> InteractionResult:
        """等待导航"""
        try:
            await self.page.wait_for_load_state(wait_until, timeout=timeout)
            return InteractionResult(success=True, message="Navigation complete")
        except PlaywrightTimeout:
            return InteractionResult(success=False, message="Navigation timeout")
        except Exception as e:
            return InteractionResult(success=False, message=f"Navigation failed: {e}")

    async def execute_script(self, script: str) -> Any:
        """执行JavaScript"""
        return await self.page.evaluate(script)

    async def get_attribute(
        self,
        selector: str,
        attribute: str,
    ) -> Optional[str]:
        """获取元素属性"""
        try:
            return await self.page.locator(selector).get_attribute(attribute)
        except Exception:
            return None

    async def get_text(self, selector: str) -> Optional[str]:
        """获取元素文本"""
        try:
            return await self.page.locator(selector).text_content()
        except Exception:
            return None

    async def get_inner_html(self, selector: str) -> Optional[str]:
        """获取元素内部HTML"""
        try:
            return await self.page.locator(selector).inner_html()
        except Exception:
            return None

    async def is_visible(self, selector: str) -> bool:
        """检查元素是否可见"""
        try:
            return await self.page.locator(selector).is_visible()
        except Exception:
            return False

    async def is_enabled(self, selector: str) -> bool:
        """检查元素是否可用"""
        try:
            return await self.page.locator(selector).is_enabled()
        except Exception:
            return False

    async def is_checked(self, selector: str) -> bool:
        """检查复选框是否被勾选"""
        try:
            return await self.page.locator(selector).is_checked()
        except Exception:
            return False

    async def count(self, selector: str) -> int:
        """计算匹配元素数量"""
        try:
            return await self.page.locator(selector).count()
        except Exception:
            return 0

    async def wait_for_all(
        self,
        selectors: List[str],
        timeout: int = 30000,
    ) -> Dict[str, bool]:
        """等待多个选择器"""
        results = {}
        for selector in selectors:
            try:
                await self.page.locator(selector).wait_for(timeout=timeout)
                results[selector] = True
            except Exception:
                results[selector] = False
        return results


class InfiniteScroller:
    """无限滚动处理器"""

    def __init__(self, page: Page):
        self.page = page
        self.max_scrolls: int = 50
        self.scroll_delay: float = 1.0
        self.stop_condition: Optional[Callable[[], bool]] = None

    async def scroll(
        self,
        max_scrolls: int = 50,
        scroll_delay: float = 1.0,
        stop_condition: Optional[Callable[[], bool]] = None,
    ) -> int:
        """
        执行无限滚动

        Args:
            max_scrolls: 最大滚动次数
            scroll_delay: 滚动间隔（秒）
            stop_condition: 停止条件函数

        Returns:
            实际滚动次数
        """
        self.max_scrolls = max_scrolls
        self.scroll_delay = scroll_delay
        self.stop_condition = stop_condition

        scroll_count = 0
        last_height = 0

        for _ in range(self.max_scrolls):
            # 检查停止条件
            if self.stop_condition and self.stop_condition():
                break

            # 获取当前高度
            current_height = await self.page.evaluate("document.body.scrollHeight")

            # 滚动到底部
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(self.scroll_delay)

            # 检查是否到达底部
            new_height = await self.page.evaluate("document.body.scrollHeight")
            if new_height == current_height:
                # 再等一下确认没有更多内容
                await asyncio.sleep(self.scroll_delay)
                new_height = await self.page.evaluate("document.body.scrollHeight")
                if new_height == current_height:
                    break

            last_height = new_height
            scroll_count += 1

        return scroll_count

    async def get_loaded_items(self, item_selector: str) -> List[str]:
        """获取已加载的项目"""
        items = []
        count = await self.page.locator(item_selector).count()
        for i in range(count):
            try:
                text = await self.page.locator(item_selector).nth(i).text_content()
                if text:
                    items.append(text.strip())
            except Exception:
                pass
        return items
