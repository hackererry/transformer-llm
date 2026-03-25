# -*- coding: utf-8 -*-
"""
浏览器指纹防护模块
Browser Fingerprint Protection
"""
import random
import hashlib
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Fingerprint:
    """浏览器指纹"""
    screen_width: int
    screen_height: int
    screen_color_depth: int
    timezone: str
    language: str
    platform: str
    hardware_concurrency: int
    device_memory: int

    # Canvas 指纹
    canvas_hash: Optional[str] = None

    # WebGL 指纹
    webgl_vendor: Optional[str] = None
    webgl_renderer: Optional[str] = None

    # 插件
    plugins: List[str] = None

    def __post_init__(self):
        if self.plugins is None:
            self.plugins = []

    @property
    def hash(self) -> str:
        """计算指纹哈希"""
        data = f"{self.screen_width},{self.screen_height},{self.screen_color_depth},"
        data += f"{self.timezone},{self.language},{self.platform},"
        data += f"{self.hardware_concurrency},{self.device_memory}"
        if self.canvas_hash:
            data += f",{self.canvas_hash}"
        if self.webgl_vendor:
            data += f",{self.webgl_vendor}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]


class FingerprintGenerator:
    """指纹生成器"""

    # 常用屏幕分辨率
    SCREEN_RESOLUTIONS = [
        (1920, 1080),
        (2560, 1440),
        (1366, 768),
        (1536, 864),
        (1440, 900),
        (1280, 720),
        (1600, 900),
        (3840, 2160),
    ]

    # 时区
    TIMEZONES = [
        "Asia/Shanghai",
        "Asia/Tokyo",
        "Asia/Singapore",
        "America/New_York",
        "America/Los_Angeles",
        "Europe/London",
        "Europe/Paris",
    ]

    # 语言
    LANGUAGES = [
        "zh-CN",
        "zh-TW",
        "en-US",
        "en-GB",
        "ja-JP",
        "ko-KR",
    ]

    # 平台
    PLATFORMS = [
        "Win32",
        "MacIntel",
        "Linux x86_64",
    ]

    def __init__(self, seed: Optional[int] = None):
        """初始化指纹生成器"""
        if seed is not None:
            random.seed(seed)
        else:
            random.seed(datetime.now().microsecond)

        self.screen_width, self.screen_height = random.choice(self.SCREEN_RESOLUTIONS)
        self.screen_color_depth = random.choice([24, 32])
        self.timezone = random.choice(self.TIMEZONES)
        self.language = random.choice(self.LANGUAGES)
        self.platform = random.choice(self.PLATFORMS)
        self.hardware_concurrency = random.choice([2, 4, 8, 16])
        self.device_memory = random.choice([2, 4, 8, 16])

    def generate(self) -> Fingerprint:
        """生成指纹"""
        return Fingerprint(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            screen_color_depth=self.screen_color_depth,
            timezone=self.timezone,
            language=self.language,
            platform=self.platform,
            hardware_concurrency=self.hardware_concurrency,
            device_memory=self.device_memory,
        )

    def generate_randomized(self) -> Fingerprint:
        """生成随机化指纹（每次不同）"""
        screen_width, screen_height = random.choice(self.SCREEN_RESOLUTIONS)

        return Fingerprint(
            screen_width=screen_width,
            screen_height=screen_height,
            screen_color_depth=random.choice([24, 32]),
            timezone=random.choice(self.TIMEZONES),
            language=random.choice(self.LANGUAGES),
            platform=random.choice(self.PLATFORMS),
            hardware_concurrency=random.choice([2, 4, 8, 16]),
            device_memory=random.choice([2, 4, 8, 16]),
        )


class FingerprintManager:
    """指纹管理器"""

    def __init__(self):
        self.fingerprint: Optional[Fingerprint] = None
        self.original_values: Dict[str, Any] = {}

    def set_fingerprint(self, fingerprint: Fingerprint):
        """设置指纹"""
        self.fingerprint = fingerprint

    def apply_to_browser(self, page: Any):
        """
        应用指纹到浏览器页面

        Args:
            page: Playwright Page 对象
        """
        if self.fingerprint is None:
            return

        # 设置视口
        page.set_viewport_size({
            "width": self.fingerprint.screen_width,
            "height": self.fingerprint.screen_height,
        })

    def get_headers(self) -> Dict[str, str]:
        """获取伪造的请求头"""
        if self.fingerprint is None:
            return {}

        return {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": self.fingerprint.language,
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }

    def get_viewport(self) -> Dict[str, int]:
        """获取视口大小"""
        if self.fingerprint is None:
            return {"width": 1920, "height": 1080}

        return {
            "width": self.fingerprint.screen_width,
            "height": self.fingerprint.screen_height,
        }

    def get_navigator_props(self) -> Dict[str, Any]:
        """获取 navigator 对象属性"""
        if self.fingerprint is None:
            return {}

        return {
            "hardwareConcurrency": self.fingerprint.hardware_concurrency,
            "deviceMemory": self.fingerprint.device_memory,
            "platform": self.fingerprint.platform,
            "language": self.fingerprint.language,
            "languages": [self.fingerprint.language, "en"],
        }

    @staticmethod
    def generate_canvas_hash() -> str:
        """生成随机的 Canvas 指纹哈希"""
        return hashlib.sha256(str(random.random()).encode()).hexdigest()[:16]

    @staticmethod
    def generate_webgl_info() -> tuple:
        """生成随机的 WebGL 信息"""
        vendors = [
            "Intel Inc.",
            "NVIDIA Corporation",
            "AMD",
            "Apple Inc.",
        ]
        renderers = [
            "Intel Iris OpenGL Engine",
            "NVIDIA GeForce GTX 1060",
            "AMD Radeon Pro 5500M",
            "Apple M1",
        ]
        return random.choice(vendors), random.choice(renderers)


class FingerprintInjector:
    """指纹注入器，用于修改浏览器指纹"""

    # JavaScript 代码模板
    CANVAS_OVERRIDE_SCRIPT = """
    (function() {
        const originalGetContext = HTMLCanvasElement.prototype.getContext;
        HTMLCanvasElement.prototype.getContext = function(type, attributes) {
            const context = originalGetContext.call(this, type, attributes);
            if (type === '2d') {
                const originalFillText = context.fillText;
                context.fillText = function() {
                    // 添加微小的随机噪声
                    // 实际实现需要更复杂的逻辑
                    return originalFillText.apply(this, arguments);
                };
            }
            return context;
        };
    })();
    """

    WEBGL_OVERRIDE_SCRIPT = """
    (function() {
        const getParameter = WebGLRenderingContext.prototype.getParameter;
        WebGLRenderingContext.prototype.getParameter = function(param) {
            if (param === 37445) {
                return '%VENDOR%';
            }
            if (param === 37446) {
                return '%RENDERER%';
            }
            return getParameter.call(this, param);
        };
    })();
    """

    def __init__(self, fingerprint: Fingerprint):
        self.fingerprint = fingerprint

    def get_canvas_override_script(self) -> str:
        """获取 Canvas 覆盖脚本"""
        return self.CANVAS_OVERRIDE_SCRIPT

    def get_webgl_override_script(self) -> str:
        """获取 WebGL 覆盖脚本"""
        vendor, renderer = FingerprintManager.generate_webgl_info()
        script = self.WEBGL_OVERRIDE_SCRIPT.replace('%VENDOR%', vendor)
        script = script.replace('%RENDERER%', renderer)
        return script

    def inject(self, page: Any):
        """
        注入指纹修改脚本到页面

        Args:
            page: Playwright Page 对象
        """
        page.add_init_script(self.get_webgl_override_script())
