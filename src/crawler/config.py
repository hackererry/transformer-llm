# -*- coding: utf-8 -*-
"""
爬虫配置管理模块
Configuration Management for Crawler
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import json


@dataclass
class CrawlerConfig:
    """爬虫基础配置"""

    # 请求配置
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    request_timeout: int = 30
    retry_times: int = 3
    retry_delay: float = 2.0
    max_redirects: int = 10

    # 爬取配置
    delay: float = 1.0
    delay_jitter: float = 0.3
    max_concurrent: int = 5
    max_depth: int = 3
    max_pages: int = 1000

    # 输出配置
    output_dir: str = "./output/crawled"
    clean_text: bool = True
    save_raw: bool = False

    # robots.txt 配置
    respect_robots: bool = True

    # 断点续爬
    resume_file: Optional[str] = None

    # 浏览器模式
    use_browser: bool = False
    headless: bool = True

    # 代理配置
    proxy_url: Optional[str] = None

    # 数据库配置
    db_url: Optional[str] = None

    # Redis配置
    redis_url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawlerConfig":
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class SiteConfig:
    """站点配置"""
    name: str
    start_url: str
    max_pages: int = 100
    link_pattern: str = ""
    content_type: str = "auto"
    delay: float = 1.0
    depth: int = 3
    respect_robots: bool = True


@dataclass
class ProxyConfig:
    """代理配置"""
    proxy_type: str = "http"
    proxy_host: str = ""
    proxy_port: int = 8080
    proxy_user: Optional[str] = None
    proxy_password: Optional[str] = None

    @property
    def proxy_url(self) -> Optional[str]:
        """生成代理URL"""
        if not self.proxy_host:
            return None
        if self.proxy_user and self.proxy_password:
            return f"{self.proxy_type}://{self.proxy_user}:{self.proxy_password}@{self.proxy_host}:{self.proxy_port}"
        return f"{self.proxy_type}://{self.proxy_host}:{self.proxy_port}"


@dataclass
class BrowserConfig:
    """浏览器配置"""
    browser_type: str = "chromium"
    headless: bool = True
    viewport_width: int = 1920
    viewport_height: int = 1080
    user_agent: Optional[str] = None
    slow_mo: float = 0
    timeout: int = 30000

    # 等待策略
    wait_until: str = "networkidle"
    wait_for_selector: Optional[str] = None
    wait_for_timeout: int = 5000


@dataclass
class CrawlerSettings:
    """爬虫全局设置"""

    # 爬虫配置
    crawler: CrawlerConfig = field(default_factory=CrawlerConfig)

    # 代理配置
    proxy: ProxyConfig = field(default_factory=ProxyConfig)

    # 浏览器配置
    browser: BrowserConfig = field(default_factory=BrowserConfig)

    # 日志配置
    log_dir: str = "./logs"
    log_level: str = "INFO"

    # 指标配置
    metrics_enabled: bool = True
    metrics_interval: int = 10

    # 并发配置
    max_concurrent_requests: int = 10
    max_concurrent_per_domain: int = 2

    @classmethod
    def from_file(cls, path: str) -> "CrawlerSettings":
        """从JSON文件加载"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CrawlerSettings":
        """从字典创建"""
        settings = cls()
        if 'crawler' in data:
            settings.crawler = CrawlerConfig(**data['crawler'])
        if 'proxy' in data:
            settings.proxy = ProxyConfig(**data['proxy'])
        if 'browser' in data:
            settings.browser = BrowserConfig(**data['browser'])
        if 'log_dir' in data:
            settings.log_dir = data['log_dir']
        if 'log_level' in data:
            settings.log_level = data['log_level']
        if 'metrics_enabled' in data:
            settings.metrics_enabled = data['metrics_enabled']
        if 'metrics_interval' in data:
            settings.metrics_interval = data['metrics_interval']
        if 'max_concurrent_requests' in data:
            settings.max_concurrent_requests = data['max_concurrent_requests']
        if 'max_concurrent_per_domain' in data:
            settings.max_concurrent_per_domain = data['max_concurrent_per_domain']
        return settings

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'crawler': self.crawler.to_dict(),
            'proxy': {
                'proxy_type': self.proxy.proxy_type,
                'proxy_host': self.proxy.proxy_host,
                'proxy_port': self.proxy.proxy_port,
                'proxy_user': self.proxy.proxy_user,
                'proxy_password': self.proxy.proxy_password,
            },
            'browser': {
                'browser_type': self.browser.browser_type,
                'headless': self.browser.headless,
                'viewport_width': self.browser.viewport_width,
                'viewport_height': self.browser.viewport_height,
                'user_agent': self.browser.user_agent,
                'slow_mo': self.browser.slow_mo,
                'timeout': self.browser.timeout,
                'wait_until': self.browser.wait_until,
                'wait_for_selector': self.browser.wait_for_selector,
                'wait_for_timeout': self.browser.wait_for_timeout,
            },
            'log_dir': self.log_dir,
            'log_level': self.log_level,
            'metrics_enabled': self.metrics_enabled,
            'metrics_interval': self.metrics_interval,
            'max_concurrent_requests': self.max_concurrent_requests,
            'max_concurrent_per_domain': self.max_concurrent_per_domain,
        }

    def save(self, path: str):
        """保存到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
