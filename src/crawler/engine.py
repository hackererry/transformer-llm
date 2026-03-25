# -*- coding: utf-8 -*-
"""
爬虫引擎
Crawler Engine - Core async crawler using httpx
"""
import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from .config import CrawlerConfig, SiteConfig
from .storage.database import Database, create_sqlite_db
from .storage.file_storage import FileStorage
from src.utils.logging import Logger, setup_logger
from src.utils.metrics import MetricsTracker


@dataclass
class CrawlResult:
    """爬取结果"""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    raw_html: Optional[str] = None
    content_hash: Optional[str] = None
    content_type: Optional[str] = None
    domain: Optional[str] = None
    depth: int = 0
    status_code: Optional[int] = None
    error_message: Optional[str] = None
    bytes_downloaded: int = 0
    crawl_time: float = 0.0


class RobotChecker:
    """robots.txt 检查器"""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._parsers: Dict[str, RobotFileParser] = {}
        self._cache: Dict[str, bool] = {}

    def _get_parser(self, url: str) -> Optional[RobotFileParser]:
        """获取或创建 robots.txt 解析器"""
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"

        if base_url not in self._parsers:
            parser = RobotFileParser()
            robots_url = f"{base_url}/robots.txt"
            try:
                parser.set_url(robots_url)
                parser.read()
                self._parsers[base_url] = parser
            except Exception:
                self._parsers[base_url] = None

        return self._parsers.get(base_url)

    def can_fetch(self, url: str) -> bool:
        """检查是否允许爬取"""
        if url in self._cache:
            return self._cache[url]

        parser = self._get_parser(url)
        if parser is None:
            self._cache[url] = True
            return True

        result = parser.can_fetch(self.user_agent, url)
        self._cache[url] = result
        return result


class TextExtractor:
    """网页文本提取器"""

    REMOVE_TAGS = ['script', 'style', 'nav', 'footer', 'header', 'aside',
                   'iframe', 'noscript', 'form', 'button', 'input']

    CONTENT_TAGS = ['article', 'main', 'div', 'section']

    NOISE_PATTERNS = [
        r'nav', r'menu', r'sidebar', r'footer', r'header', r'comment',
        r'advertisement', r'ad-', r'ads-', r'banner', r'social', r'share',
        r'related', r'recommend', r'breadcrumb', r'pagination'
    ]

    def __init__(self):
        self.noise_regex = re.compile('|'.join(self.NOISE_PATTERNS), re.IGNORECASE)

    def extract(self, html: str, url: str, content_type: str = 'auto') -> Tuple[str, str]:
        """从HTML中提取文本"""
        soup = BeautifulSoup(html, 'lxml')
        title = self._extract_title(soup)

        if content_type == 'news':
            text = self._extract_news(soup)
        elif content_type == 'novel':
            text = self._extract_novel(soup)
        elif content_type == 'wiki':
            text = self._extract_wiki(soup)
        else:
            text = self._extract_auto(soup)

        return text, title

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """提取页面标题"""
        og_title = soup.find('meta', property='og:title')
        if og_title and og_title.get('content'):
            return og_title['content'].strip()

        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            for sep in [' - ', ' | ', ' _ ', '——']:
                if sep in title:
                    title = title.split(sep)[0].strip()
            return title

        return ""

    def _extract_news(self, soup: BeautifulSoup) -> str:
        """提取新闻正文"""
        self._remove_noise(soup)
        article = soup.find('article')
        if not article:
            for tag in ['div', 'section']:
                for elem in soup.find_all(tag, class_=True):
                    classes = ' '.join(elem.get('class', []))
                    if any(kw in classes.lower() for kw in ['content', 'article', 'post', 'body', 'text']):
                        article = elem
                        break
                if article:
                    break

        if article:
            return self._get_text(article)
        return self._extract_auto(soup)

    def _extract_novel(self, soup: BeautifulSoup) -> str:
        """提取小说章节"""
        self._remove_noise(soup)
        content_div = None
        content_patterns = ['chapter-content', 'content', 'text-content',
                           'chapter', 'article-content', 'novel-content']

        for pattern in content_patterns:
            elem = soup.find(class_=re.compile(pattern, re.IGNORECASE))
            if elem:
                content_div = elem
                break

        if not content_div:
            content_div = self._find_content_by_density(soup)

        if content_div:
            return self._get_text(content_div)
        return self._extract_auto(soup)

    def _extract_wiki(self, soup: BeautifulSoup) -> str:
        """提取百科内容"""
        self._remove_noise(soup)
        wiki_content = soup.find('div', class_=re.compile(r'(wiki|content|body)', re.IGNORECASE))

        if wiki_content:
            for tag in ['table', 'sup', 'span']:
                for elem in wiki_content.find_all(tag):
                    elem.decompose()
            return self._get_text(wiki_content)

        return self._extract_auto(soup)

    def _extract_auto(self, soup: BeautifulSoup) -> str:
        """自动提取正文"""
        self._remove_noise(soup)
        content = self._find_content_by_density(soup)
        if content:
            return self._get_text(content)
        return self._get_text(soup.body if soup.body else soup)

    def _remove_noise(self, soup: BeautifulSoup):
        """移除噪音元素"""
        for tag in self.REMOVE_TAGS:
            for elem in soup.find_all(tag):
                elem.decompose()

        for elem in soup.find_all(class_=self.noise_regex):
            elem.decompose()

        for elem in soup.find_all(id=self.noise_regex):
            elem.decompose()

    def _find_content_by_density(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        """通过文本密度找到正文区域"""
        candidates = []

        for tag in self.CONTENT_TAGS:
            for elem in soup.find_all(tag):
                text = elem.get_text()
                text_len = len(text.strip())
                links = elem.find_all('a')
                link_text_len = sum(len(link.get_text()) for link in links)

                if text_len > 0:
                    text_density = (text_len - link_text_len) / text_len
                else:
                    text_density = 0

                tag_count = len(elem.find_all())
                if tag_count > 0:
                    tag_density = text_len / tag_count
                else:
                    tag_density = text_len

                score = text_len * text_density * 0.5 + tag_density
                candidates.append((elem, score, text_len))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            for elem, score, text_len in candidates:
                if text_len >= 100:
                    return elem

        return None

    def _get_text(self, elem) -> str:
        """从元素中提取纯文本"""
        text = elem.get_text(separator='\n')
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if line:
                lines.append(line)
        return '\n'.join(lines)


class AsyncCrawler:
    """异步网络爬虫"""

    def __init__(
        self,
        config: CrawlerConfig,
        site_config: SiteConfig,
        logger: Optional[Logger] = None,
        metrics: Optional[MetricsTracker] = None,
        database: Optional[Database] = None,
        file_storage: Optional[FileStorage] = None,
    ):
        self.config = config
        self.site_config = site_config
        self.logger = logger or setup_logger(experiment_name=f"crawler_{site_config.name}")
        self.metrics = metrics or MetricsTracker()
        self.database = database
        self.file_storage = file_storage

        self.robot_checker = RobotChecker(config.user_agent)
        self.text_extractor = TextExtractor()

        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.pending_urls: Set[Tuple[str, int]] = set()

        self._client: Optional[httpx.AsyncClient] = None
        self._semaphore: Optional[asyncio.Semaphore] = None

        self.stats = {
            'pages_crawled': 0,
            'pages_failed': 0,
            'bytes_downloaded': 0,
            'start_time': 0,
        }

    async def __aenter__(self):
        """异步上下文管理器入口"""
        await self._init_client()
        self.stats['start_time'] = time.time()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器退出"""
        await self.close()

    async def _init_client(self):
        """初始化HTTP客户端"""
        limits = httpx.Limits(
            max_connections=self.config.max_concurrent,
            max_keepalive_connections=self.config.max_concurrent,
        )
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.request_timeout),
            limits=limits,
            headers={'User-Agent': self.config.user_agent},
            follow_redirects=True,
            max_redirects=self.config.max_redirects,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def normalize_url(self, url: str, base_url: str = None) -> str:
        """规范化URL"""
        if base_url:
            url = urljoin(base_url, url)

        parsed = urlparse(url)
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''
        ))
        return normalized

    def is_valid_url(self, url: str, base_url: str = None, pattern: str = None) -> bool:
        """检查URL是否有效"""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ('http', 'https'):
                return False

            if base_url:
                base_parsed = urlparse(base_url)
                if parsed.netloc != base_parsed.netloc:
                    return False

            if pattern:
                if not re.search(pattern, url):
                    return False

            skip_patterns = [
                r'\.(jpg|jpeg|png|gif|bmp|svg|ico|css|js|pdf|zip|rar)$',
                r'(login|register|search|tag|category|#)',
            ]
            for skip in skip_patterns:
                if re.search(skip, url, re.IGNORECASE):
                    return False

            return True
        except Exception:
            return False

    def compute_content_hash(self, content: str) -> str:
        """计算内容哈希"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def crawl_page(self, url: str, depth: int = 0) -> CrawlResult:
        """爬取单个页面"""
        start_time = time.time()
        result = CrawlResult(url=url, depth=depth)

        if self.config.respect_robots and not self.robot_checker.can_fetch(url):
            result.error_message = "Blocked by robots.txt"
            self.logger.warning(f"Blocked by robots.txt: {url}")
            return result

        delay = self.config.delay
        if self.config.delay_jitter > 0:
            import random
            delay += random.uniform(-self.config.delay_jitter, self.config.delay_jitter)
        await asyncio.sleep(max(0.1, delay))

        async with self._semaphore:
            for attempt in range(self.config.retry_times):
                try:
                    response = await self._client.get(url)
                    result.status_code = response.status_code

                    if response.status_code != 200:
                        continue

                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' not in content_type:
                        result.error_message = "Not HTML content"
                        return result

                    result.raw_html = response.text
                    result.bytes_downloaded = len(response.content)

                    result.content, result.title = self.text_extractor.extract(
                        result.raw_html, url, self.site_config.content_type
                    )

                    if self.config.clean_text and result.content:
                        result.content = self._clean_text(result.content)

                    result.content_hash = self.compute_content_hash(result.content or '')

                    parsed = urlparse(url)
                    result.domain = parsed.netloc
                    result.content_type = self.site_config.content_type

                    result.crawl_time = time.time() - start_time

                    self.logger.info(f"Crawled: {result.title or url} ({result.status_code})")

                    return result

                except httpx.TimeoutException:
                    result.error_message = "Timeout"
                except httpx.HTTPError as e:
                    result.error_message = str(e)
                except Exception as e:
                    result.error_message = str(e)

                if attempt < self.config.retry_times - 1:
                    await asyncio.sleep(self.config.retry_delay)

            self.failed_urls.add(url)
            self.stats['pages_failed'] += 1
            result.crawl_time = time.time() - start_time
            return result

    def _clean_text(self, text: str) -> str:
        """清洗文本"""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def extract_links(self, html: str, base_url: str, pattern: str = None) -> List[str]:
        """从HTML中提取链接"""
        soup = BeautifulSoup(html, 'lxml')
        links = []

        for a in soup.find_all('a', href=True):
            href = a['href']
            url = self.normalize_url(href, base_url)

            if self.is_valid_url(url, base_url, pattern):
                links.append(url)

        return list(set(links))

    async def crawl_site(self) -> List[CrawlResult]:
        """爬取整个站点"""
        results = []
        self.pending_urls = {(self.site_config.start_url, 0)}

        self.logger.info(f"Starting crawl: {self.site_config.name}")
        self.logger.info(f"Start URL: {self.site_config.start_url}")

        while self.pending_urls and len(results) < self.site_config.max_pages:
            batch_size = min(self.config.max_concurrent, len(self.pending_urls))
            batch = list(self.pending_urls)[:batch_size]
            self.pending_urls = self.pending_urls - set(batch)

            tasks = []
            for url, depth in batch:
                if url in self.visited_urls or depth > self.site_config.depth:
                    continue
                self.visited_urls.add(url)
                tasks.append(self.crawl_page(url, depth))

            if not tasks:
                continue

            batch_results = await asyncio.gather(*tasks)

            for result in batch_results:
                if result.content and not result.error_message:
                    results.append(result)
                    self.stats['pages_crawled'] += 1
                    self.stats['bytes_downloaded'] += result.bytes_downloaded

                    # 1. 写入文件（立即存储）
                    file_path = ""
                    if self.file_storage:
                        file_path = self.file_storage.write_content(result.content)
                        self.logger.debug(f"Wrote content to {file_path}")

                    # 2. 更新数据库状态
                    if self.database:
                        self.database.save_page({
                            'url': result.url,
                            'content_hash': result.content_hash,
                            'file_path': file_path,
                            'domain': result.domain,
                        })

                    if result.raw_html and result.depth < self.site_config.depth:
                        links = self.extract_links(
                            result.raw_html,
                            result.url,
                            self.site_config.link_pattern
                        )
                        for link in links:
                            if link not in self.visited_urls:
                                self.pending_urls.add((link, result.depth + 1))

                    if self.metrics:
                        self.metrics.update({
                            'pages_crawled': 1,
                            'bytes_downloaded': result.bytes_downloaded,
                            'crawl_time': result.crawl_time,
                        }, count=1)

        elapsed = time.time() - self.stats['start_time']
        self.logger.info(
            f"Crawl completed: {len(results)} pages, "
            f"{len(self.failed_urls)} failed, "
            f"{self.stats['bytes_downloaded'] / 1024 / 1024:.2f} MB, "
            f"{elapsed:.1f}s"
        )

        return results

    def save_results(self, results: List[CrawlResult], output_dir: str):
        """保存爬取结果"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        all_text = []
        for result in results:
            all_text.append(f"# {result.title or result.url}\n")
            all_text.append(f"# URL: {result.url}\n")
            all_text.append(result.content or "")
            all_text.append("\n" + "=" * 50 + "\n")

        output_file = os.path.join(output_dir, f"{self.site_config.name}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(all_text))

        self.logger.info(f"Saved {len(results)} pages to {output_file}")

        import json
        meta_file = os.path.join(output_dir, f"{self.site_config.name}_meta.json")
        meta = {
            'site_name': self.site_config.name,
            'total_pages': len(results),
            'failed_pages': len(self.failed_urls),
            'urls': [r.url for r in results],
            'failed_urls': list(self.failed_urls),
            'stats': self.stats,
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


async def crawl_url(
    url: str,
    config: Optional[CrawlerConfig] = None,
    content_type: str = 'auto',
    clean_text: bool = True,
) -> Optional[CrawlResult]:
    """爬取单个URL"""
    if config is None:
        config = CrawlerConfig()

    site_config = SiteConfig(
        name=urlparse(url).netloc.replace('.', '_'),
        start_url=url,
        content_type=content_type,
    )

    crawler = AsyncCrawler(config, site_config)
    async with crawler:
        result = await crawler.crawl_page(url)
        return result


async def crawl_sites(
    config_path: str,
    output_dir: str = "./output/crawled",
    max_concurrent: int = 2,
) -> int:
    """并发爬取所有站点"""
    import yaml

    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)

    global_config = config_data.get('global', {})
    sites = config_data.get('sites', [])

    if not sites:
        print("错误: 配置文件中没有站点")
        return 0

    print(f"=" * 50)
    print(f"开始批量爬取: {len(sites)} 个站点")
    print(f"请求间隔: {global_config.get('delay', 1.0)}s")
    print(f"最大并发: {max_concurrent}")
    print(f"每个站点最大页数: {global_config.get('max_pages', 100)}")
    print(f"=" * 50)

    # 创建爬虫配置
    crawler_config = CrawlerConfig(
        delay=global_config.get('delay', 1.0),
        max_depth=global_config.get('depth', 3),
        max_pages=global_config.get('max_pages', 30),
        max_concurrent=max_concurrent,  # 每个站点并发数
        respect_robots=False,  # 测试环境设为False
    )

    # 创建站点配置
    site_configs = []
    for site in sites:
        site_config = SiteConfig(
            name=site['name'],
            start_url=site['start_url'],
            max_pages=site.get('max_pages', global_config.get('max_pages', 30)),
            depth=site.get('depth', global_config.get('depth', 3)),
            content_type=site.get('content_type', 'news'),
            delay=site.get('delay', global_config.get('delay', 2.0)),
        )
        site_configs.append(site_config)

    # 创建输出目录
    import os
    os.makedirs(output_dir, exist_ok=True)

    # 创建统一数据库
    db_path = os.path.join(output_dir, "crawler.db")
    db = create_sqlite_db(db_path)

    # 创建统一文件存储
    data_dir = os.path.join(output_dir, "data")
    file_storage = FileStorage(data_dir)

    # 并发控制
    semaphore = asyncio.Semaphore(max_concurrent)

    async def crawl_site(site_config: SiteConfig) -> List[CrawlResult]:
        """爬取单个站点"""
        logger = setup_logger(
            log_dir=os.path.join(output_dir, "logs"),
            experiment_name=f"crawler_{site_config.name}"
        )

        metrics = MetricsTracker()

        crawler = AsyncCrawler(
            config=crawler_config,
            site_config=site_config,
            logger=logger,
            metrics=metrics,
            database=db,
            file_storage=file_storage,
        )

        try:
            async with crawler:
                results = await crawler.crawl_site()

            logger.info(f"✓ {site_config.name} 完成: {len(results)} 页")

            avg_metrics = metrics.average()
            if avg_metrics:
                logger.info(f"  指标: {avg_metrics}")

            return results

        except Exception as e:
            logger.error(f"✗ {site_config.name} 失败: {e}")
            return []

    async def bounded_crawl(site_config: SiteConfig):
        async with semaphore:
            return await crawl_site(site_config)

    # 并发执行
    tasks = [bounded_crawl(sc) for sc in site_configs]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计
    total_pages = 0
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"站点 {site_configs[i].name} 出错: {result}")
        else:
            total_pages += len(result)

    print(f"\n" + "=" * 50)
    print(f"批量爬取完成!")
    print(f"总页数: {total_pages}")
    print(f"输出目录: {output_dir}")
    print(f"=" * 50)

    db.close()
    return total_pages
