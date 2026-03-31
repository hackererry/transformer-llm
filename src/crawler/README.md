# 爬虫层 (src/crawler/)

异步网络爬虫系统，支持多站点批量爬取、反反爬虫、浏览器自动化、数据存储。

## 目录结构

```
src/crawler/
├── cli.py              # 命令行接口
├── engine.py           # 爬虫引擎（httpx + BeautifulSoup）
├── config.py           # CrawlerConfig / SiteConfig — 配置管理
├── browser/            # 浏览器自动化
│   ├── playwright_manager.py  # Playwright 管理器
│   ├── page_interactions.py    # 页面交互
│   └── wait_strategies.py     # 等待策略
├── anti_crawler/       # 反反爬虫
│   ├── proxy_pool.py           # 代理池
│   ├── user_agent_pool.py      # User-Agent 池
│   └── fingerprint.py          # 浏览器指纹
└── storage/            # 数据存储
    ├── database.py             # SQLite 存储
    ├── file_storage.py         # 文件存储
    ├── redis_cache.py          # Redis 缓存
    └── preprocessed_output.py  # 预处理输出
```

## 核心组件

### cli.py — 命令行接口

```bash
# 批量爬取
python -m src.crawler.cli run

# 指定配置和并行数
python -m src.crawler.cli run --config configs/crawler/crawler_config.yaml --parallel 3

# 指定输出目录
python -m src.crawler.cli run --output-dir ./crawled --parallel 2

# 查看爬虫状态
python -m src.crawler.cli status
```

### engine.py — 爬虫引擎

`CrawlEngine` 核心功能：

| 特性 | 说明 |
|------|------|
| 异步爬取 | `asyncio` + `httpx` 并发 |
| robots.txt | 自动检查并遵守 robots.txt |
| 重试机制 | 自动重试失败请求 |
| 速率限制 | 可配置请求间隔 |
| 内容提取 | BeautifulSoup 提取正文 |
| 去重 | 基于 URL/内容哈希去重 |
| 深度控制 | 可配置爬取深度 |

**CrawlResult 数据结构：**

```python
@dataclass
class CrawlResult:
    url: str              # URL
    title: str            # 标题
    content: str          # 正文内容
    raw_html: str         # 原始 HTML
    content_hash: str     # 内容哈希
    domain: str           # 域名
    depth: int            # 爬取深度
    status_code: int      # HTTP 状态码
    bytes_downloaded: int # 下载字节数
    crawl_time: float     # 爬取耗时
```

### config.py — 爬虫配置

**CrawlerConfig 全局配置：**

| 参数 | 说明 |
|------|------|
| `delay` | 请求间隔（秒） |
| `max_pages` | 每个站点最多爬取页数 |
| `depth` | 爬取深度 |
| `max_concurrent` | 每个站点最大并发数 |
| `timeout` | 请求超时（秒） |
| `retry_times` | 重试次数 |
| `respect_robots_txt` | 是否遵守 robots.txt |

**SiteConfig 站点配置：**

| 参数 | 说明 |
|------|------|
| `name` | 站点名称 |
| `start_url` | 起始 URL |
| `max_pages` | 最多爬取页数 |
| `content_type` | 内容类型（news/novel/wiki/auto） |
| `delay` | 请求间隔（覆盖全局） |
| `selectors` | CSS 选择器（自定义提取规则） |

**配置文件格式（YAML）：**

```yaml
global:
  delay: 2.0
  max_pages: 30
  depth: 3
  max_concurrent: 2
  timeout: 30
  retry_times: 3
  respect_robots_txt: true

sites:
  - name: example_news
    start_url: https://news.example.com/
    max_pages: 50
    content_type: news
    delay: 2.0
  - name: example_wiki
    start_url: https://wiki.example.com/
    max_pages: 100
    content_type: wiki
    delay: 3.0
```

### content_type 内容类型

| 类型 | 说明 | 提取策略 |
|------|------|---------|
| `news` | 新闻资讯 | 自动提取正文、标题、时间 |
| `novel` | 小说/文学 | 保留段落格式 |
| `wiki` | 百科/知识 | 提取主要内容、目录 |
| `auto` | 自动检测 | 根据页面结构自动选择 |

### storage/ — 数据存储

| 存储类型 | 说明 |
|---------|------|
| `database.py` | SQLite — 页面内容、爬取记录、统计信息 |
| `file_storage.py` | 文件系统 — 按域名/日期组织 |
| `redis_cache.py` | Redis — URL 去重缓存 |
| `preprocessed_output.py` | 预处理输出 — 直接对接数据处理模块 |

**数据库表：**

| 表名 | 说明 |
|------|------|
| `crawler_pages` | 爬取的页面（URL、标题、内容、哈希、时间） |
| `crawl_stats` | 爬取统计（每日/每站点统计） |

## 使用示例

```python
from src.crawler.engine import CrawlEngine
from src.crawler.config import CrawlerConfig, SiteConfig
from src.crawler.storage.database import create_sqlite_db

# 创建数据库
create_sqlite_db("crawler.db")

# 配置
config = CrawlerConfig(
    delay=2.0,
    max_pages=30,
    max_concurrent=2,
)

sites = [
    SiteConfig(
        name="example",
        start_url="https://example.com/",
        max_pages=50,
        content_type="news",
    ),
]

# 爬取
engine = CrawlEngine(config)
results = await engine.crawl_sites(sites)

for result in results:
    print(f"Title: {result.title}")
    print(f"Content length: {len(result.content)}")
```

## 反反爬虫

### anti_crawler/

| 模块 | 说明 |
|------|------|
| `proxy_pool.py` | 代理池，自动切换代理 |
| `user_agent_pool.py` | User-Agent 随机切换 |
| `fingerprint.py` | 浏览器指纹模拟 |

### browser/ — Playwright 浏览器自动化

| 模块 | 说明 |
|------|------|
| `playwright_manager.py` | Playwright 浏览器管理器 |
| `page_interactions.py` | 页面交互（滚动、点击、等待） |
| `wait_strategies.py` | 等待策略（网络空闲、内容加载） |

**适用场景：**
- JavaScript 渲染页面
- 懒加载内容
- 需要登录的页面
- 反爬严格的站点

## 最佳实践

1. **遵守 robots.txt** — `respect_robots_txt: true`
2. **合理请求间隔** — `delay: 2.0` 或更高
3. **设置超时** — 避免长时间等待
4. **启用重试** — `retry_times: 3`
5. **使用代理** — 大规模爬取时轮换代理
6. **监控统计** — 定期查看爬虫状态
