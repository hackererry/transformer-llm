# -*- coding: utf-8 -*-
"""
数据库工具模块测试
测试 DatabaseManager 和 Repository 的功能
"""
import pytest
import sys
import os
import tempfile
import sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.database import DatabaseManager, get_default_db_path, get_connection
from src.utils.database_schema import init_all_tables, get_all_schemas
from src.utils.repository import (
    CrawlerPageRepository,
    CrawlStatsRepository,
    CleanedDocumentRepository,
    CleaningRunRepository,
)


@pytest.fixture
def temp_db():
    """创建临时数据库"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    conn = DatabaseManager.get_connection(db_path)
    init_all_tables(conn)
    conn.close()

    yield db_path

    try:
        os.unlink(db_path)
    except Exception:
        pass


class TestDatabaseManager:
    """DatabaseManager 测试类"""

    def test_get_connection(self, temp_db):
        """测试获取数据库连接"""
        conn = DatabaseManager.get_connection(temp_db)
        assert conn is not None
        assert isinstance(conn, sqlite3.Connection)

        # 验证 row_factory
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        row = cursor.fetchone()
        assert row["test"] == 1
        conn.close()

    def test_init_database(self, temp_db):
        """测试初始化数据库"""
        # 数据库已通过 fixture 初始化，这里验证表存在
        assert DatabaseManager.table_exists("crawler_pages", temp_db)
        assert DatabaseManager.table_exists("crawl_stats", temp_db)
        assert DatabaseManager.table_exists("cleaned_documents", temp_db)
        assert DatabaseManager.table_exists("cleaning_runs", temp_db)

    def test_execute_query(self, temp_db):
        """测试执行查询"""
        # 插入测试数据
        DatabaseManager.execute_update(
            "INSERT INTO crawler_pages (url) VALUES (?)",
            ("https://example.com",),
            temp_db
        )

        # 查询
        rows = DatabaseManager.execute_query(
            "SELECT * FROM crawler_pages WHERE url = ?",
            ("https://example.com",),
            temp_db
        )
        assert len(rows) == 1
        assert rows[0]["url"] == "https://example.com"

    def test_execute_update(self, temp_db):
        """测试执行更新"""
        # 插入
        rows = DatabaseManager.execute_update(
            "INSERT INTO crawler_pages (url) VALUES (?)",
            ("https://example.com",),
            temp_db
        )
        assert rows == 1

        # 更新
        rows = DatabaseManager.execute_update(
            "UPDATE crawler_pages SET domain = ? WHERE url = ?",
            ("example.com", "https://example.com"),
            temp_db
        )
        assert rows == 1

    def test_table_exists(self, temp_db):
        """测试检查表是否存在"""
        assert DatabaseManager.table_exists("crawler_pages", temp_db)
        assert not DatabaseManager.table_exists("nonexistent_table", temp_db)

    def test_get_table_info(self, temp_db):
        """测试获取表结构信息"""
        info = DatabaseManager.get_table_info("crawler_pages", temp_db)
        assert len(info) > 0
        # 验证包含 id 列
        assert any(col["name"] == "id" for col in info)


class TestCrawlerPageRepository:
    """CrawlerPageRepository 测试类"""

    def test_save_page(self, temp_db):
        """测试保存页面"""
        repo = CrawlerPageRepository(temp_db)
        page_id = repo.save_page({
            "url": "https://example.com/page1",
            "content_hash": "abc123",
            "file_path": "/data/page1.txt",
            "domain": "example.com",
        })
        assert page_id is not None
        assert page_id > 0

    def test_find_by_url(self, temp_db):
        """测试根据 URL 查找"""
        repo = CrawlerPageRepository(temp_db)
        repo.save_page({
            "url": "https://example.com/page1",
            "content_hash": "abc123",
            "domain": "example.com",
        })

        page = repo.find_by_url("https://example.com/page1")
        assert page is not None
        assert page["url"] == "https://example.com/page1"
        assert page["content_hash"] == "abc123"

    def test_find_by_content_hash(self, temp_db):
        """测试根据内容哈希查找"""
        repo = CrawlerPageRepository(temp_db)
        repo.save_page({
            "url": "https://example.com/page1",
            "content_hash": "abc123",
            "domain": "example.com",
        })

        pages = repo.find_by_content_hash("abc123")
        assert len(pages) == 1
        assert pages[0]["content_hash"] == "abc123"

    def test_find_by_domain(self, temp_db):
        """测试根据域名查找"""
        repo = CrawlerPageRepository(temp_db)
        repo.save_page({
            "url": "https://example.com/page1",
            "content_hash": "abc123",
            "domain": "example.com",
        })
        repo.save_page({
            "url": "https://example.com/page2",
            "content_hash": "def456",
            "domain": "example.com",
        })

        pages = repo.find_by_domain("example.com")
        assert len(pages) == 2

    def test_is_crawled(self, temp_db):
        """测试检查 URL 是否已爬取"""
        repo = CrawlerPageRepository(temp_db)
        assert not repo.is_crawled("https://example.com/page1")

        repo.save_page({
            "url": "https://example.com/page1",
            "content_hash": "abc123",
            "domain": "example.com",
        })

        assert repo.is_crawled("https://example.com/page1")


class TestCrawlStatsRepository:
    """CrawlStatsRepository 测试类"""

    def test_increment_stats(self, temp_db):
        """测试增量更新统计"""
        repo = CrawlStatsRepository(temp_db)
        repo.increment_stats(pages_crawled=5, bytes_downloaded=1024)

        stats = repo.get_recent_stats(1)
        assert len(stats) >= 1

    def test_get_or_create_today(self, temp_db):
        """测试获取或创建今日统计"""
        repo = CrawlStatsRepository(temp_db)
        stats = repo.get_or_create_today()
        assert stats is not None
        assert "date" in stats


class TestCleanedDocumentRepository:
    """CleanedDocumentRepository 测试类"""

    def test_save_document(self, temp_db):
        """测试保存文档"""
        repo = CleanedDocumentRepository(temp_db)
        doc_id = repo.save_document({
            "original_file_path": "/data/test.txt",
            "original_md5": "abc123",
            "cleaned_md5": "def456",
            "cleaned_content": "清洗后的内容",
            "original_size": 1000,
            "cleaned_size": 800,
            "line_count": 10,
            "quality_score": 0.85,
            "pii_count": 2,
        })
        assert doc_id is not None

    def test_find_by_original_md5(self, temp_db):
        """测试根据原始 MD5 查找"""
        repo = CleanedDocumentRepository(temp_db)
        repo.save_document({
            "original_file_path": "/data/test.txt",
            "original_md5": "abc123",
            "cleaned_md5": "def456",
            "cleaned_content": "清洗后的内容",
        })

        doc = repo.find_by_original_md5("abc123")
        assert doc is not None
        assert doc["original_md5"] == "abc123"

    def test_is_processed(self, temp_db):
        """测试检查文档是否已处理"""
        repo = CleanedDocumentRepository(temp_db)
        assert not repo.is_processed("abc123")

        repo.save_document({
            "original_file_path": "/data/test.txt",
            "original_md5": "abc123",
            "cleaned_md5": "def456",
            "cleaned_content": "清洗后的内容",
        })

        assert repo.is_processed("abc123")

    def test_save_documents_batch(self, temp_db):
        """测试批量保存"""
        repo = CleanedDocumentRepository(temp_db)
        docs = []
        for i in range(5):
            docs.append({
                "original_file_path": f"/data/test_{i}.txt",
                "original_md5": f"md5_{i}",
                "cleaned_md5": f"cleaned_md5_{i}",
                "cleaned_content": f"内容 {i}",
            })

        count = repo.save_documents_batch(docs)
        assert count == 5

    def test_delete_by_run_id(self, temp_db):
        """测试删除指定运行的所有文档"""
        repo = CleanedDocumentRepository(temp_db)
        run_repo = CleaningRunRepository(temp_db)

        run_id = run_repo.start_run("/input", "/output")

        repo.save_document({
            "original_file_path": "/data/test1.txt",
            "original_md5": "md5_1",
            "cleaned_md5": "cleaned_1",
            "cleaned_content": "内容1",
        }, run_id)

        repo.save_document({
            "original_file_path": "/data/test2.txt",
            "original_md5": "md5_2",
            "cleaned_md5": "cleaned_2",
            "cleaned_content": "内容2",
        }, run_id)

        count = repo.delete_by_run_id(run_id)
        assert count == 2

        docs = repo.find_by_run_id(run_id)
        assert len(docs) == 0

    def test_export_to_directory(self, temp_db):
        """测试导出到目录"""
        repo = CleanedDocumentRepository(temp_db)

        repo.save_document({
            "original_file_path": "/data/test1.txt",
            "original_md5": "md5_1",
            "cleaned_md5": "cleaned_1",
            "cleaned_content": "内容1",
        })

        with tempfile.TemporaryDirectory() as export_dir:
            count = repo.export_to_directory(None, export_dir)
            assert count == 1
            files = os.listdir(export_dir)
            assert len(files) == 1

    def test_get_stats(self, temp_db):
        """测试获取统计信息"""
        repo = CleanedDocumentRepository(temp_db)

        for i in range(3):
            repo.save_document({
                "original_file_path": f"/data/test_{i}.txt",
                "original_md5": f"md5_{i}",
                "cleaned_md5": f"cleaned_{i}",
                "cleaned_content": f"内容 {i}",
                "original_size": 1000,
                "cleaned_size": 800,
                "line_count": 10,
                "pii_count": 5,
            })

        stats = repo.get_stats()
        assert stats["total_documents"] == 3
        assert stats["total_original_size"] == 3000
        assert stats["total_cleaned_size"] == 2400
        assert stats["total_lines"] == 30
        assert stats["total_pii"] == 15


class TestCleaningRunRepository:
    """CleaningRunRepository 测试类"""

    def test_start_run(self, temp_db):
        """测试开始运行"""
        repo = CleaningRunRepository(temp_db)
        run_id = repo.start_run("/input", "/output")

        assert run_id is not None
        assert len(run_id) > 0

        run = repo.find_by_run_id(run_id)
        assert run is not None
        assert run["input_dir"] == "/input"
        assert run["output_dir"] == "/output"

    def test_update_stats(self, temp_db):
        """测试更新统计"""
        repo = CleaningRunRepository(temp_db)
        run_id = repo.start_run("/input", "/output")

        repo.update_stats(
            run_id,
            total_files=10,
            success_count=8,
            error_count=2,
            total_input_lines=1000,
            total_output_lines=800,
        )

        run = repo.find_by_run_id(run_id)
        assert run["total_files"] == 10
        assert run["success_count"] == 8
        assert run["error_count"] == 2

    def test_finish_run(self, temp_db):
        """测试结束运行"""
        repo = CleaningRunRepository(temp_db)
        run_id = repo.start_run("/input", "/output")

        repo.finish_run(run_id)

        run = repo.find_by_run_id(run_id)
        assert run["finished_at"] is not None

    def test_get_active_runs(self, temp_db):
        """测试获取活跃运行"""
        repo = CleaningRunRepository(temp_db)
        run_id = repo.start_run("/input1", "/output1")
        repo.start_run("/input2", "/output2")

        active = repo.get_active_runs()
        assert len(active) >= 2

    def test_delete_run(self, temp_db):
        """测试删除运行"""
        repo = CleaningRunRepository(temp_db)
        doc_repo = CleanedDocumentRepository(temp_db)

        run_id = repo.start_run("/input", "/output")

        doc_repo.save_document({
            "original_file_path": "/data/test.txt",
            "original_md5": "md5",
            "cleaned_md5": "cleaned",
            "cleaned_content": "内容",
        }, run_id)

        result = repo.delete_run(run_id)
        assert result is True

        run = repo.find_by_run_id(run_id)
        assert run is None
