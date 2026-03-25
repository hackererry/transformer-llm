# -*- coding: utf-8 -*-
"""
数据库存储模块
Database Storage Module using SQLAlchemy
"""
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Float,
    Index,
    UniqueConstraint,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool

Base = declarative_base()


class CrawledPage(Base):
    """爬取记录表（简化版）"""
    __tablename__ = "crawled_pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    url = Column(String(2048), nullable=False, unique=True)  # URL去重
    content_hash = Column(String(64), index=True)  # 内容去重
    file_path = Column(String(512))  # 所属文件
    domain = Column(String(256), index=True)  # 域名统计
    is_crawled = Column(Boolean, default=False)
    crawled_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('url', name='uix_url'),
        Index('idx_hash', 'content_hash'),
    )


class CrawlStats(Base):
    """爬取统计表"""
    __tablename__ = "crawl_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, default=datetime.utcnow, unique=True, index=True)

    # 计数
    pages_crawled = Column(Integer, default=0)
    pages_failed = Column(Integer, default=0)
    pages_queued = Column(Integer, default=0)

    # 流量
    bytes_downloaded = Column(Integer, default=0)

    # 时间
    total_time = Column(Float, default=0.0)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Database:
    """数据库管理类"""

    def __init__(
        self,
        db_url: str = "sqlite:///./crawler.db",
        pool_size: int = 5,
        max_overflow: int = 10,
    ):
        """
        初始化数据库

        Args:
            db_url: 数据库连接URL
            pool_size: 连接池大小
            max_overflow: 最大溢出连接数
        """
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            echo=False,
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self):
        """创建所有表"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """删除所有表"""
        Base.metadata.drop_all(bind=self.engine)

    def get_session(self) -> Session:
        """获取会话"""
        return self.SessionLocal()

    def close(self):
        """关闭连接"""
        self.engine.dispose()

    # ========== 页面操作 ==========

    def save_page(self, page_data: Dict[str, Any]) -> Optional[int]:
        """保存爬取页面（简化版）"""
        with self.get_session() as session:
            try:
                page = CrawledPage(
                    url=page_data['url'],
                    content_hash=page_data.get('content_hash'),
                    file_path=page_data.get('file_path'),
                    domain=page_data.get('domain'),
                    is_crawled=page_data.get('is_crawled', True),
                    crawled_at=datetime.utcnow(),
                )
                session.add(page)
                session.commit()
                return page.id
            except Exception:
                session.rollback()
                return None

    def get_page_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """根据URL获取页面"""
        with self.get_session() as session:
            page = session.query(CrawledPage).filter(CrawledPage.url == url).first()
            if page:
                return self._page_to_dict(page)
            return None

    def get_page_by_hash(self, content_hash: str) -> Optional[Dict[str, Any]]:
        """根据内容哈希获取页面"""
        with self.get_session() as session:
            page = session.query(CrawledPage).filter(
                CrawledPage.content_hash == content_hash
            ).first()
            if page:
                return self._page_to_dict(page)
            return None

    def is_url_crawled(self, url: str) -> bool:
        """检查URL是否已爬取"""
        with self.get_session() as session:
            return session.query(CrawledPage).filter(
                CrawledPage.url == url,
                CrawledPage.is_crawled == True
            ).count() > 0

    def get_pages_by_domain(
        self,
        domain: str,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """获取指定域名的页面"""
        with self.get_session() as session:
            pages = session.query(CrawledPage).filter(
                CrawledPage.domain == domain,
                CrawledPage.is_crawled == True
            ).limit(limit).offset(offset).all()
            return [self._page_to_dict(p) for p in pages]

    def get_crawled_count(self) -> int:
        """获取已爬取页面数量"""
        with self.get_session() as session:
            return session.query(CrawledPage).filter(
                CrawledPage.is_crawled == True
            ).count()

    # ========== 统计操作 ==========

    def update_stats(
        self,
        pages_crawled: int = 0,
        pages_failed: int = 0,
        pages_queued: int = 0,
        bytes_downloaded: int = 0,
        total_time: float = 0.0,
    ):
        """更新今日统计"""
        with self.get_session() as session:
            try:
                today = datetime.utcnow().date()
                stats = session.query(CrawlStats).filter(
                    CrawlStats.date >= today
                ).first()

                if stats:
                    stats.pages_crawled += pages_crawled
                    stats.pages_failed += pages_failed
                    stats.pages_queued += pages_queued
                    stats.bytes_downloaded += bytes_downloaded
                    stats.total_time += total_time
                else:
                    stats = CrawlStats(
                        pages_crawled=pages_crawled,
                        pages_failed=pages_failed,
                        pages_queued=pages_queued,
                        bytes_downloaded=bytes_downloaded,
                        total_time=total_time,
                    )
                    session.add(stats)
                session.commit()
            except Exception:
                session.rollback()

    def get_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取最近N天的统计"""
        with self.get_session() as session:
            from datetime import timedelta
            start_date = datetime.utcnow() - timedelta(days=days)
            stats = session.query(CrawlStats).filter(
                CrawlStats.date >= start_date
            ).order_by(CrawlStats.date.desc()).all()
            return [self._stats_to_dict(s) for s in stats]

    # ========== 辅助方法 ==========

    def _page_to_dict(self, page: CrawledPage) -> Dict[str, Any]:
        """页面对象转字典（简化版）"""
        return {
            'id': page.id,
            'url': page.url,
            'content_hash': page.content_hash,
            'file_path': page.file_path,
            'domain': page.domain,
            'is_crawled': page.is_crawled,
            'crawled_at': page.crawled_at.isoformat() if page.crawled_at else None,
        }

    def _stats_to_dict(self, stats: CrawlStats) -> Dict[str, Any]:
        """统计对象转字典"""
        return {
            'id': stats.id,
            'date': stats.date.isoformat() if stats.date else None,
            'pages_crawled': stats.pages_crawled,
            'pages_failed': stats.pages_failed,
            'pages_queued': stats.pages_queued,
            'bytes_downloaded': stats.bytes_downloaded,
            'total_time': stats.total_time,
        }


# ========== 便捷函数 ==========

def create_sqlite_db(db_path: str = "./crawler.db") -> Database:
    """创建SQLite数据库"""
    db_url = f"sqlite:///{db_path}"
    db = Database(db_url)
    db.create_tables()
    return db


def create_postgres_db(
    host: str = "localhost",
    port: int = 5432,
    database: str = "crawler",
    user: str = "postgres",
    password: str = "",
) -> Database:
    """创建PostgreSQL数据库"""
    db_url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    db = Database(db_url)
    db.create_tables()
    return db
