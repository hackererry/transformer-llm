# -*- coding: utf-8 -*-
"""
爬虫数据库存储模块
Crawler Database Storage Module

使用统一的数据库工具（src/utils/database.py）

数据库文件：db/transform.db

Usage:
    from src.crawler.storage.database import get_crawler_repo, get_crawl_stats_repo

    # 获取爬虫页面仓库
    page_repo = get_crawler_repo()

    # 保存页面
    page_repo.save_page({
        'url': 'https://example.com',
        'content_hash': 'abc123',
        'file_path': '/path/to/file.txt',
        'domain': 'example.com',
    })

    # 获取爬虫统计仓库
    stats_repo = get_crawl_stats_repo()

    # 更新统计
    stats_repo.increment_stats(pages_crawled=1)
"""
import os
from typing import Optional

from src.utils.database import DatabaseManager, get_default_db_path
from src.utils.repository import (
    CrawlerPageRepository,
    CrawlStatsRepository,
)


def get_db_path() -> str:
    """获取爬虫数据库路径（统一到 transform.db）"""
    return get_default_db_path()


def init_db() -> None:
    """初始化数据库"""
    DatabaseManager.init_database(get_db_path())


def get_crawler_repo(db_path: Optional[str] = None) -> CrawlerPageRepository:
    """获取爬虫页面仓库实例

    Args:
        db_path: 数据库路径，默认使用统一数据库

    Returns:
        CrawlerPageRepository 实例
    """
    if db_path is None:
        db_path = get_db_path()
    return CrawlerPageRepository(db_path)


def get_crawl_stats_repo(db_path: Optional[str] = None) -> CrawlStatsRepository:
    """获取爬虫统计仓库实例

    Args:
        db_path: 数据库路径，默认使用统一数据库

    Returns:
        CrawlStatsRepository 实例
    """
    if db_path is None:
        db_path = get_db_path()
    return CrawlStatsRepository(db_path)


# ========== 兼容旧接口的便捷函数 ==========
# 这些函数保留以保持向后兼容性，但内部使用新的 Repository

def create_sqlite_db(db_path: Optional[str] = None) -> "LegacyDatabase":
    """创建兼容旧接口的数据库对象

    Args:
        db_path: 数据库路径（将被忽略，使用统一数据库）

    Returns:
        LegacyDatabase 实例
    """
    init_db()
    return LegacyDatabase()


class LegacyDatabase:
    """兼容旧接口的数据库类

    此类保留以兼容现有代码，新代码应直接使用 Repository
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_db_path()
        self._page_repo = CrawlerPageRepository(self.db_path)
        self._stats_repo = CrawlStatsRepository(self.db_path)

    def create_tables(self):
        """创建所有表"""
        DatabaseManager.init_database(self.db_path)

    def drop_tables(self):
        """删除所有表（谨慎使用）"""
        from src.utils.database_schema import drop_all_tables
        conn = DatabaseManager.get_connection(self.db_path)
        try:
            drop_all_tables(conn)
        finally:
            conn.close()

    def close(self):
        """关闭连接（无需操作，使用的是轻量级 sqlite3）"""
        pass

    # ========== 页面操作（委托给 Repository）==========

    def save_page(self, page_data: dict) -> Optional[int]:
        """保存爬取页面"""
        return self._page_repo.save_page(page_data)

    def get_page_by_url(self, url: str) -> Optional[dict]:
        """根据URL获取页面"""
        return self._page_repo.find_by_url(url)

    def get_page_by_hash(self, content_hash: str) -> Optional[dict]:
        """根据内容哈希获取页面"""
        rows = self._page_repo.find_by_content_hash(content_hash)
        return rows[0] if rows else None

    def is_url_crawled(self, url: str) -> bool:
        """检查URL是否已爬取"""
        return self._page_repo.is_crawled(url)

    def get_pages_by_domain(self, domain: str, limit: int = 100, offset: int = 0) -> list:
        """获取指定域名的页面"""
        return self._page_repo.find_by_domain(domain, limit)

    def get_crawled_count(self) -> int:
        """获取已爬取页面数量"""
        return self._page_repo.get_crawled_count()

    # ========== 统计操作（委托给 Repository）==========

    def update_stats(
        self,
        pages_crawled: int = 0,
        pages_failed: int = 0,
        pages_queued: int = 0,
        bytes_downloaded: int = 0,
        total_time: float = 0.0,
    ):
        """更新今日统计"""
        self._stats_repo.increment_stats(
            pages_crawled=pages_crawled,
            pages_failed=pages_failed,
            pages_queued=pages_queued,
            bytes_downloaded=bytes_downloaded,
            total_time=total_time,
        )

    def get_stats(self, days: int = 7) -> list:
        """获取最近N天的统计"""
        return self._stats_repo.get_recent_stats(days)
