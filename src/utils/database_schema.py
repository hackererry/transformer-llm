# -*- coding: utf-8 -*-
"""
数据库表定义
Database Schema Definitions

包含所有数据库表的创建语句
"""
import sqlite3
from typing import List

# ========== 爬虫相关表 ==========

SCHEMA_CRAWLER_PAGES = """
CREATE TABLE IF NOT EXISTS crawler_pages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    content_hash TEXT,
    file_path TEXT,
    domain TEXT,
    is_crawled INTEGER DEFAULT 0,
    crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_CRAWLER_PAGES_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_crawler_content_hash ON crawler_pages(content_hash);
CREATE INDEX IF NOT EXISTS idx_crawler_domain ON crawler_pages(domain);
CREATE INDEX IF NOT EXISTS idx_crawler_is_crawled ON crawler_pages(is_crawled);
"""

SCHEMA_CRAWL_STATS = """
CREATE TABLE IF NOT EXISTS crawl_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT UNIQUE NOT NULL,
    pages_crawled INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    pages_queued INTEGER DEFAULT 0,
    bytes_downloaded INTEGER DEFAULT 0,
    total_time REAL DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

SCHEMA_CRAWL_STATS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_crawl_stats_date ON crawl_stats(date);
"""

# ========== 清洗相关表 ==========

SCHEMA_CLEANED_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS cleaned_documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_file_path TEXT,
    original_md5 TEXT,
    cleaned_md5 TEXT,
    cleaned_content TEXT,
    original_size INTEGER DEFAULT 0,
    cleaned_size INTEGER DEFAULT 0,
    line_count INTEGER DEFAULT 0,
    quality_score REAL DEFAULT 1.0,
    pii_count INTEGER DEFAULT 0,
    quality_filtered INTEGER DEFAULT 0,
    dedup_filtered INTEGER DEFAULT 0,
    run_id TEXT,
    source_url TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(original_file_path)
);
"""

SCHEMA_CLEANED_DOCUMENTS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_cleaned_original_md5 ON cleaned_documents(original_md5);
CREATE INDEX IF NOT EXISTS idx_cleaned_cleaned_md5 ON cleaned_documents(cleaned_md5);
CREATE INDEX IF NOT EXISTS idx_cleaned_run_id ON cleaned_documents(run_id);
CREATE INDEX IF NOT EXISTS idx_cleaned_created_at ON cleaned_documents(created_at);
"""

SCHEMA_CLEANING_RUNS = """
CREATE TABLE IF NOT EXISTS cleaning_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    input_dir TEXT,
    output_dir TEXT,
    total_files INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    total_input_lines INTEGER DEFAULT 0,
    total_output_lines INTEGER DEFAULT 0,
    total_pii_detected INTEGER DEFAULT 0,
    total_quality_filtered INTEGER DEFAULT 0,
    total_dedup_filtered INTEGER DEFAULT 0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP
);
"""

SCHEMA_CLEANING_RUNS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_cleaning_runs_started_at ON cleaning_runs(started_at);
CREATE INDEX IF NOT EXISTS idx_cleaning_runs_finished_at ON cleaning_runs(finished_at);
"""


def init_all_tables(conn: sqlite3.Connection) -> None:
    """初始化所有表

    Args:
        conn: sqlite3.Connection 对象
    """
    # 爬虫相关表
    conn.executescript(SCHEMA_CRAWLER_PAGES)
    conn.executescript(SCHEMA_CRAWLER_PAGES_INDEXES)

    conn.executescript(SCHEMA_CRAWL_STATS)
    conn.executescript(SCHEMA_CRAWL_STATS_INDEXES)

    # 清洗相关表
    conn.executescript(SCHEMA_CLEANED_DOCUMENTS)
    conn.executescript(SCHEMA_CLEANED_DOCUMENTS_INDEXES)

    conn.executescript(SCHEMA_CLEANING_RUNS)
    conn.executescript(SCHEMA_CLEANING_RUNS_INDEXES)


def drop_all_tables(conn: sqlite3.Connection) -> None:
    """删除所有表（谨慎使用）

    Args:
        conn: sqlite3.Connection 对象
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    for (table_name,) in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.commit()


def get_all_schemas() -> List[str]:
    """获取所有表的创建语句

    Returns:
        SQL 语句列表
    """
    return [
        SCHEMA_CRAWLER_PAGES.strip(),
        SCHEMA_CRAWL_STATS.strip(),
        SCHEMA_CLEANED_DOCUMENTS.strip(),
        SCHEMA_CLEANING_RUNS.strip(),
    ]
