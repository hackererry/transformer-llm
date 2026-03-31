# -*- coding: utf-8 -*-
"""
数据清洗数据库模块测试
测试 CleanedDocument 和 CleaningRun 的存储、查询功能

使用统一的数据库工具（src/util/database.py）
"""
import pytest
import sys
import os
import tempfile
import hashlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.cleaning_db import (
    LegacyCleaningDatabase as CleaningDatabase,
    create_cleaning_db,
)


# 共享的样本文档数据 fixture
@pytest.fixture
def sample_doc_data():
    """样本文档数据"""
    return {
        "original_file_path": "/path/to/test.txt",
        "original_md5": "abc123def456",
        "cleaned_md5": "def456abc789",
        "cleaned_content": "这是清洗后的文本内容。\n第二行内容。",
        "original_size": 1000,
        "cleaned_size": 500,
        "line_count": 2,
        "quality_score": 0.85,
        "pii_count": 3,
        "quality_filtered": 5,
        "dedup_filtered": 2,
    }


@pytest.fixture
def temp_db():
    """创建临时数据库"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = CleaningDatabase(db_path)
    db.create_tables()

    yield db

    db.close()
    try:
        os.unlink(db_path)
    except Exception:
        pass


class TestCleaningDatabase:
    """CleaningDatabase 测试类"""

    def test_create_tables(self, temp_db):
        """测试创建表"""
        # 表已通过 fixture 创建，这里验证它们存在
        from src.utils.database import DatabaseManager
        assert DatabaseManager.table_exists("cleaned_documents", temp_db.db_path)
        assert DatabaseManager.table_exists("cleaning_runs", temp_db.db_path)

    def test_save_cleaned_document(self, temp_db, sample_doc_data):
        """测试保存文档"""
        doc_id = temp_db.save_cleaned_document(sample_doc_data)
        assert doc_id is not None
        assert doc_id > 0

    def test_save_and_get_document(self, temp_db, sample_doc_data):
        """测试保存并获取文档"""
        doc_id = temp_db.save_cleaned_document(sample_doc_data)
        doc = temp_db.get_document_by_id(doc_id)

        assert doc is not None
        assert doc["original_file_path"] == sample_doc_data["original_file_path"]
        assert doc["original_md5"] == sample_doc_data["original_md5"]
        assert doc["cleaned_md5"] == sample_doc_data["cleaned_md5"]
        assert doc["cleaned_content"] == sample_doc_data["cleaned_content"]
        assert doc["line_count"] == sample_doc_data["line_count"]
        assert doc["quality_score"] == sample_doc_data["quality_score"]

    def test_get_document_by_original_md5(self, temp_db, sample_doc_data):
        """测试通过原始MD5获取文档"""
        temp_db.save_cleaned_document(sample_doc_data)
        doc = temp_db.get_document_by_original_md5(sample_doc_data["original_md5"])

        assert doc is not None
        assert doc["original_md5"] == sample_doc_data["original_md5"]

    def test_get_document_by_cleaned_md5(self, temp_db, sample_doc_data):
        """测试通过清洗后MD5获取文档"""
        temp_db.save_cleaned_document(sample_doc_data)
        doc = temp_db.get_document_by_cleaned_md5(sample_doc_data["cleaned_md5"])

        assert doc is not None
        assert doc["cleaned_md5"] == sample_doc_data["cleaned_md5"]

    def test_get_document_by_path(self, temp_db, sample_doc_data):
        """测试通过文件路径获取文档"""
        temp_db.save_cleaned_document(sample_doc_data)
        doc = temp_db.get_document_by_path(sample_doc_data["original_file_path"])

        assert doc is not None
        assert doc["original_file_path"] == sample_doc_data["original_file_path"]

    def test_is_document_processed(self, temp_db, sample_doc_data):
        """测试检查文档是否已处理"""
        assert not temp_db.is_document_processed(sample_doc_data["original_md5"])

        temp_db.save_cleaned_document(sample_doc_data)

        assert temp_db.is_document_processed(sample_doc_data["original_md5"])

    def test_is_path_processed(self, temp_db, sample_doc_data):
        """测试检查路径是否已处理"""
        assert not temp_db.is_path_processed(sample_doc_data["original_file_path"])

        temp_db.save_cleaned_document(sample_doc_data)

        assert temp_db.is_path_processed(sample_doc_data["original_file_path"])

    def test_update_existing_document(self, temp_db, sample_doc_data):
        """测试更新已存在的文档"""
        doc_id1 = temp_db.save_cleaned_document(sample_doc_data)

        # 修改数据
        sample_doc_data["cleaned_content"] = "更新后的内容"
        sample_doc_data["line_count"] = 10

        doc_id2 = temp_db.save_cleaned_document(sample_doc_data)

        # 应该返回相同的ID
        assert doc_id1 == doc_id2

        # 验证更新
        doc = temp_db.get_document_by_id(doc_id1)
        assert doc["cleaned_content"] == "更新后的内容"
        assert doc["line_count"] == 10

    def test_batch_save(self, temp_db, sample_doc_data):
        """测试批量保存"""
        docs = []
        for i in range(5):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            docs.append(doc)

        count = temp_db.save_documents_batch(docs)
        assert count == 5

        assert temp_db.get_document_count() == 5

    def test_delete_document(self, temp_db, sample_doc_data):
        """测试删除文档"""
        doc_id = temp_db.save_cleaned_document(sample_doc_data)
        assert temp_db.get_document_count() == 1

        temp_db.delete_document(doc_id)

        assert temp_db.get_document_count() == 0
        assert temp_db.get_document_by_id(doc_id) is None

    def test_get_all_documents(self, temp_db, sample_doc_data):
        """测试获取所有文档"""
        # 保存多个文档
        for i in range(3):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            temp_db.save_cleaned_document(doc)

        docs = temp_db.get_all_documents()
        assert len(docs) == 3


class TestCleaningRun:
    """CleaningRun 测试类"""

    def test_start_run(self, temp_db):
        """测试开始运行"""
        run_id = temp_db.start_run("/input", "/output")

        assert run_id is not None
        assert len(run_id) > 0

        run = temp_db.get_run(run_id)
        assert run is not None
        assert run["input_dir"] == "/input"
        assert run["output_dir"] == "/output"
        assert run["total_files"] == 0
        assert run["success_count"] == 0

    def test_update_run_stats(self, temp_db):
        """测试更新运行统计"""
        run_id = temp_db.start_run("/input", "/output")

        temp_db.update_run_stats(
            run_id,
            total_files=10,
            success_count=8,
            error_count=2,
            total_input_lines=1000,
            total_output_lines=800,
            total_pii_detected=50,
            total_quality_filtered=100,
            total_dedup_filtered=50,
        )

        run = temp_db.get_run(run_id)
        assert run["total_files"] == 10
        assert run["success_count"] == 8
        assert run["error_count"] == 2
        assert run["total_input_lines"] == 1000
        assert run["total_output_lines"] == 800

    def test_finish_run(self, temp_db, sample_doc_data):
        """测试结束运行"""
        run_id = temp_db.start_run("/input", "/output")

        # 保存一些文档
        for i in range(3):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            doc["pii_count"] = 2
            doc["quality_filtered"] = 1
            doc["dedup_filtered"] = 1
            doc["line_count"] = 5
            temp_db.save_cleaned_document(doc, run_id)

        temp_db.finish_run(run_id)

        run = temp_db.get_run(run_id)
        assert run["finished_at"] is not None
        assert run["success_count"] == 3
        assert run["total_output_lines"] == 15
        assert run["total_pii_detected"] == 6
        assert run["total_quality_filtered"] == 3
        assert run["total_dedup_filtered"] == 3

    def test_get_documents_by_run(self, temp_db, sample_doc_data):
        """测试获取指定运行的文档"""
        run_id = temp_db.start_run("/input", "/output")

        # 保存文档到指定运行
        for i in range(3):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            temp_db.save_cleaned_document(doc, run_id)

        docs = temp_db.get_documents_by_run(run_id)
        assert len(docs) == 3

        # 验证所有文档都属于该运行
        for doc in docs:
            assert doc["run_id"] == run_id

    def test_delete_run(self, temp_db, sample_doc_data):
        """测试删除运行及其文档"""
        run_id = temp_db.start_run("/input", "/output")

        # 保存文档
        for i in range(3):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            temp_db.save_cleaned_document(doc, run_id)

        assert temp_db.get_document_count() == 3

        temp_db.delete_run(run_id)

        assert temp_db.get_run(run_id) is None
        assert temp_db.get_document_count() == 0

    def test_get_all_runs(self, temp_db):
        """测试获取所有运行"""
        for i in range(3):
            temp_db.start_run(f"/input_{i}", f"/output_{i}")

        runs = temp_db.get_all_runs()
        assert len(runs) == 3


class TestExport:
    """导出功能测试"""

    @pytest.fixture
    def export_dir(self):
        """创建临时导出目录"""
        with tempfile.TemporaryDirectory() as d:
            yield d

    def test_export_all_documents(self, temp_db, sample_doc_data, export_dir):
        """测试导出所有文档"""
        # 保存文档
        for i in range(3):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            temp_db.save_cleaned_document(doc)

        count = temp_db.export_to_directory(None, export_dir)
        assert count == 3

        # 验证文件存在
        files = os.listdir(export_dir)
        assert len(files) == 3

    def test_export_by_run(self, temp_db, sample_doc_data, export_dir):
        """测试导出指定运行的文档"""
        run_id = temp_db.start_run("/input", "/output")

        # 保存文档到指定运行
        for i in range(2):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            temp_db.save_cleaned_document(doc, run_id)

        # 保存其他运行
        other_run_id = temp_db.start_run("/input2", "/output2")
        doc = sample_doc_data.copy()
        doc["original_file_path"] = "/path/to/other.txt"
        doc["original_md5"] = "other_md5"
        doc["cleaned_md5"] = "other_cleaned_md5"
        doc["cleaned_content"] = "其他内容"
        temp_db.save_cleaned_document(doc, other_run_id)

        count = temp_db.export_to_directory(run_id, export_dir)
        assert count == 2

        # 验证只导出了指定运行的文档
        files = os.listdir(export_dir)
        assert len(files) == 2

    def test_export_handles_filename_conflicts(self, temp_db, sample_doc_data, export_dir):
        """测试导出时处理文件名冲突"""
        # 保存两个相同文件名的文档（使用不同的路径）
        doc1 = sample_doc_data.copy()
        doc1["original_file_path"] = "/path/to/same_name.txt"
        doc1["original_md5"] = "md5_1"
        doc1["cleaned_md5"] = "cleaned_md5_1"
        doc1["cleaned_content"] = "内容1"
        temp_db.save_cleaned_document(doc1)

        # 第二个文档使用不同的路径
        doc2 = sample_doc_data.copy()
        doc2["original_file_path"] = "/other/same_name.txt"
        doc2["original_md5"] = "md5_2"
        doc2["cleaned_md5"] = "cleaned_md5_2"
        doc2["cleaned_content"] = "内容2"
        temp_db.save_cleaned_document(doc2)

        count = temp_db.export_to_directory(None, export_dir)
        assert count == 2

        files = os.listdir(export_dir)
        assert len(files) == 2


class TestStats:
    """统计功能测试"""

    def test_get_stats(self, temp_db, sample_doc_data):
        """测试获取统计信息"""
        # 保存文档
        for i in range(3):
            doc = sample_doc_data.copy()
            doc["original_file_path"] = f"/path/to/test_{i}.txt"
            doc["original_md5"] = f"md5_{i}"
            doc["cleaned_md5"] = f"cleaned_md5_{i}"
            doc["cleaned_content"] = f"内容 {i}"
            doc["original_size"] = 1000
            doc["cleaned_size"] = 800
            doc["line_count"] = 10
            doc["pii_count"] = 5
            temp_db.save_cleaned_document(doc)

        # 创建一个完成的运行
        temp_db.start_run("/input", "/output")

        stats = temp_db.get_stats()

        assert stats["total_documents"] == 3
        assert stats["total_original_size"] == 3000
        assert stats["total_cleaned_size"] == 2400
        assert stats["total_lines"] == 30
        assert stats["total_pii"] == 15


class TestCreateCleaningDb:
    """create_cleaning_db 函数测试"""

    def test_create_with_path(self):
        """测试使用指定路径创建"""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = create_cleaning_db(db_path)
            assert db is not None

            # 验证可以创建表
            db.create_tables()
            doc_id = db.save_cleaned_document({
                "original_file_path": "/test.txt",
                "original_md5": "test",
                "cleaned_md5": "test",
                "cleaned_content": "test",
            })
            assert doc_id is not None

            db.close()
        finally:
            try:
                os.unlink(db_path)
            except Exception:
                pass

    def test_create_without_path(self):
        """测试不指定路径时创建默认数据库"""
        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            # 切换到临时目录
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                db = create_cleaning_db()

                # 检查数据库连接是否成功
                doc_id = db.save_cleaned_document({
                    "original_file_path": "/test.txt",
                    "original_md5": "test",
                    "cleaned_md5": "test",
                    "cleaned_content": "test",
                })
                assert doc_id is not None

                db.close()
            finally:
                os.chdir(old_cwd)
