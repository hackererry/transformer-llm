# -*- coding: utf-8 -*-
"""
网络爬虫模块测试
"""
import pytest
import json
import os
import tempfile
from unittest.mock import Mock, patch, AsyncMock

from src.crawler.config import CrawlerConfig, SiteConfig
from src.crawler.engine import RobotChecker, TextExtractor, AsyncCrawler, CrawlResult


class TestCrawlerConfig:
    """测试爬虫配置"""

    def test_default_config(self):
        """测试默认配置"""
        config = CrawlerConfig()
        assert config.request_timeout == 30
        assert config.retry_times == 3
        assert config.delay == 1.0
        assert config.max_concurrent == 5
        assert config.respect_robots is True

    def test_custom_config(self):
        """测试自定义配置"""
        config = CrawlerConfig(
            delay=2.0,
            max_concurrent=10,
            clean_text=False
        )
        assert config.delay == 2.0
        assert config.max_concurrent == 10
        assert config.clean_text is False


class TestSiteConfig:
    """测试站点配置"""

    def test_site_config_creation(self):
        """测试站点配置创建"""
        config = SiteConfig(
            name="test_site",
            start_url="https://example.com",
            max_pages=50,
            depth=2
        )
        assert config.name == "test_site"
        assert config.start_url == "https://example.com"
        assert config.max_pages == 50
        assert config.depth == 2


class TestRobotChecker:
    """测试 robots.txt 检查器"""

    def test_can_fetch_allowed(self):
        """测试允许爬取的URL"""
        checker = RobotChecker("TestBot")
        # 如果 robots.txt 解析失败，默认允许
        result = checker.can_fetch("https://httpbin.org/anything")
        assert result is True

    def test_robots_txt_parsing(self):
        """测试 robots.txt 解析"""
        checker = RobotChecker("TestBot")
        # 测试解析器初始化
        parser = checker._get_parser("https://example.com/path")
        # 结果可能是 None（解析失败）或 RobotFileParser 对象
        assert parser is None or hasattr(parser, 'can_fetch')


class TestTextExtractor:
    """测试文本提取器"""

    @pytest.fixture
    def extractor(self):
        return TextExtractor()

    def test_extract_title_from_title_tag(self, extractor):
        """测试从 title 标签提取标题"""
        html = """
        <html>
        <head><title>Test Article Title</title></head>
        <body><p>Content</p></body>
        </html>
        """
        text, title = extractor.extract(html, "https://example.com")
        assert title == "Test Article Title"

    def test_extract_title_with_suffix(self, extractor):
        """测试标题后缀移除"""
        html = """
        <html>
        <head><title>Article Title - Example Site</title></head>
        <body><p>Content</p></body>
        </html>
        """
        text, title = extractor.extract(html, "https://example.com")
        assert "Article Title" in title

    def test_extract_news_content(self, extractor):
        """测试新闻内容提取"""
        html = """
        <html>
        <body>
            <nav>Navigation</nav>
            <article>
                <h1>News Title</h1>
                <p>This is the first paragraph of news content.</p>
                <p>This is the second paragraph.</p>
            </article>
            <footer>Footer</footer>
        </body>
        </html>
        """
        text, title = extractor.extract(html, "https://example.com", content_type='news')
        assert "first paragraph" in text
        assert "second paragraph" in text

    def test_extract_novel_content(self, extractor):
        """测试小说内容提取"""
        html = """
        <html>
        <body>
            <div class="chapter-content">
                <p>第一章 开始</p>
                <p>这是小说的正文内容。</p>
                <p>故事还在继续。</p>
            </div>
        </body>
        </html>
        """
        text, title = extractor.extract(html, "https://example.com", content_type='novel')
        assert "正文内容" in text

    def test_remove_noise_elements(self, extractor):
        """测试噪音元素移除"""
        html = """
        <html>
        <body>
            <script>alert('noise');</script>
            <nav>Navigation Menu</nav>
            <div class="content">
                <p>Main content here.</p>
            </div>
            <footer>Footer Content</footer>
        </body>
        </html>
        """
        text, _ = extractor.extract(html, "https://example.com")
        assert "alert" not in text
        assert "Main content" in text


class TestAsyncCrawler:
    """测试异步网络爬虫"""

    @pytest.fixture
    def config(self):
        return CrawlerConfig(delay=0.1, retry_times=1, max_concurrent=1)

    @pytest.fixture
    def site_config(self):
        return SiteConfig(
            name="test_site",
            start_url="https://example.com",
            max_pages=10,
            depth=1
        )

    @pytest.fixture
    def crawler(self, config, site_config):
        return AsyncCrawler(config, site_config)

    def test_normalize_url(self, crawler):
        """测试URL规范化"""
        # 测试相对路径
        url = crawler.normalize_url("/path", "https://example.com")
        assert url == "https://example.com/path"

        # 测试移除fragment
        url = crawler.normalize_url("https://example.com/path#anchor")
        assert url == "https://example.com/path"

    def test_is_valid_url(self, crawler):
        """测试URL有效性检查"""
        # 有效URL
        assert crawler.is_valid_url("https://example.com/page")
        assert crawler.is_valid_url("https://example.com/article/123")

        # 无效URL
        assert not crawler.is_valid_url("https://example.com/image.jpg")
        assert not crawler.is_valid_url("https://example.com/style.css")
        assert not crawler.is_valid_url("ftp://example.com/file")

    def test_is_valid_url_with_base(self, crawler):
        """测试同域检查"""
        base = "https://example.com"
        assert crawler.is_valid_url("https://example.com/page", base)
        assert not crawler.is_valid_url("https://other.com/page", base)

    def test_is_valid_url_with_pattern(self, crawler):
        """测试链接模式匹配"""
        pattern = r"/article/.*"
        assert crawler.is_valid_url("https://example.com/article/123", pattern=pattern)
        assert not crawler.is_valid_url("https://example.com/page/123", pattern=pattern)

    def test_extract_links(self, crawler):
        """测试链接提取"""
        html = """
        <html>
        <body>
            <a href="/page1">Link 1</a>
            <a href="https://example.com/page2">Link 2</a>
            <a href="https://other.com/page3">Link 3</a>
            <a href="/image.jpg">Image</a>
        </body>
        </html>
        """
        links = crawler.extract_links(html, "https://example.com")

        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
        # 外部链接被过滤
        assert "https://other.com/page3" not in links
        # 图片链接被过滤
        assert "https://example.com/image.jpg" not in links

    def test_compute_content_hash(self, crawler):
        """测试内容哈希计算"""
        content = "Test content"
        hash1 = crawler.compute_content_hash(content)
        hash2 = crawler.compute_content_hash(content)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_clean_text(self, crawler):
        """测试文本清洗"""
        dirty_text = "  Hello   \n\n\n  World  \n\n  https://example.com  "
        clean = crawler._clean_text(dirty_text)
        assert "https://example.com" not in clean
        assert "Hello" in clean
        assert "World" in clean


class TestCrawlResult:
    """测试爬取结果数据类"""

    def test_crawl_result_creation(self):
        """测试 CrawlResult 创建"""
        result = CrawlResult(
            url="https://example.com",
            title="Test Title",
            content="Test content",
            status_code=200,
            depth=0
        )
        assert result.url == "https://example.com"
        assert result.title == "Test Title"
        assert result.content == "Test content"
        assert result.status_code == 200
        assert result.depth == 0
        assert result.error_message is None

    def test_crawl_result_with_error(self):
        """测试带错误的 CrawlResult"""
        result = CrawlResult(
            url="https://example.com",
            error_message="Timeout",
            status_code=None
        )
        assert result.error_message == "Timeout"
        assert result.status_code is None


class TestCrawlerIntegration:
    """爬虫集成测试"""

    @pytest.fixture
    def config(self):
        return CrawlerConfig(
            delay=0.1,
            retry_times=1,
            clean_text=True,
            max_concurrent=1
        )

    @pytest.fixture
    def site_config(self):
        return SiteConfig(
            name="test_site",
            start_url="https://example.com",
            max_pages=10,
            depth=1
        )

    def test_crawler_initialization(self, config, site_config):
        """测试爬虫初始化"""
        crawler = AsyncCrawler(config, site_config)
        assert crawler is not None
        assert crawler.config == config
        assert crawler.site_config == site_config
        assert crawler.robot_checker is not None
        assert crawler.text_extractor is not None

    def test_crawler_stats_initialization(self, config, site_config):
        """测试爬虫统计初始化"""
        crawler = AsyncCrawler(config, site_config)
        assert crawler.stats['pages_crawled'] == 0
        assert crawler.stats['pages_failed'] == 0
        assert crawler.stats['bytes_downloaded'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
