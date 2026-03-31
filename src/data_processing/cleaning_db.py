# -*- coding: utf-8 -*-
"""
数据清洗数据库存储模块
Cleaning Database Storage Module

使用统一的数据库工具（src/utils/database.py）

数据库文件：db/transform.db

Usage:
    from src.data_processing.cleaning_db import get_cleaned_doc_repo, get_cleaning_run_repo

    # 获取清洗文档仓库
    doc_repo = get_cleaned_doc_repo()

    # 保存文档
    doc_repo.save_document({
        "original_file_path": "/path/to/original.txt",
        "original_md5": "abc123...",
        "cleaned_md5": "def456...",
        "cleaned_content": "清洗后的文本...",
    }, run_id="xxx")

    # 获取清洗运行仓库
    run_repo = get_cleaning_run_repo()

    # 开始运行
    run_id = run_repo.start_run("/input/dir", "/output/dir")
"""
from typing import Optional

from src.utils.database import DatabaseManager, get_default_db_path
from src.utils.repository import (
    CleanedDocumentRepository,
    CleaningRunRepository,
)

# 兼容旧接口的别名（将在类定义后赋值）
CleaningDatabase = None
CleanedDocument = None  # 不再使用 ORM 模型
CleaningRun = None  # 不再使用 ORM 模型


def get_db_path() -> str:
    """获取清洗数据库路径（统一到 transform.db）"""
    return get_default_db_path()


def init_db() -> None:
    """初始化数据库"""
    DatabaseManager.init_database(get_db_path())


def get_cleaned_doc_repo(db_path: Optional[str] = None) -> CleanedDocumentRepository:
    """获取清洗文档仓库实例

    Args:
        db_path: 数据库路径，默认使用统一数据库

    Returns:
        CleanedDocumentRepository 实例
    """
    if db_path is None:
        db_path = get_db_path()
    return CleanedDocumentRepository(db_path)


def get_cleaning_run_repo(db_path: Optional[str] = None) -> CleaningRunRepository:
    """获取清洗运行仓库实例

    Args:
        db_path: 数据库路径，默认使用统一数据库

    Returns:
        CleaningRunRepository 实例
    """
    if db_path is None:
        db_path = get_db_path()
    return CleaningRunRepository(db_path)


# ========== 兼容旧接口的便捷类 ==========
# 这些类和函数保留以兼容现有代码，新代码应直接使用 Repository

def create_cleaning_db(db_path: Optional[str] = None) -> "LegacyCleaningDatabase":
    """创建兼容旧接口的数据库对象

    Args:
        db_path: 数据库路径（将被忽略，使用统一数据库）

    Returns:
        LegacyCleaningDatabase 实例
    """
    init_db()
    return LegacyCleaningDatabase()


class LegacyCleaningDatabase:
    """兼容旧接口的清洗数据库类

    此类保留以兼容现有代码，新代码应直接使用 Repository
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_db_path()
        self._doc_repo = CleanedDocumentRepository(self.db_path)
        self._run_repo = CleaningRunRepository(self.db_path)

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

    # ========== 文档操作（委托给 Repository）==========

    def save_cleaned_document(
        self,
        doc_data: dict,
        run_id: Optional[str] = None,
    ) -> Optional[int]:
        """保存清洗后的文档"""
        return self._doc_repo.save_document(doc_data, run_id)

    def save_documents_batch(
        self,
        docs: list,
        run_id: Optional[str] = None,
    ) -> int:
        """批量保存文档"""
        return self._doc_repo.save_documents_batch(docs, run_id)

    def get_document_by_id(self, doc_id: int) -> Optional[dict]:
        """根据ID获取文档"""
        return self._doc_repo.read(doc_id)

    def get_document_by_original_md5(self, md5: str) -> Optional[dict]:
        """根据原始文件MD5获取文档"""
        return self._doc_repo.find_by_original_md5(md5)

    def get_document_by_cleaned_md5(self, md5: str) -> Optional[dict]:
        """根据清洗后内容MD5获取文档"""
        return self._doc_repo.find_by_cleaned_md5(md5)

    def get_document_by_path(self, file_path: str) -> Optional[dict]:
        """根据文件路径获取文档"""
        return self._doc_repo.find_by_original_path(file_path)

    def is_document_processed(self, original_md5: str) -> bool:
        """检查文档是否已处理"""
        return self._doc_repo.is_processed(original_md5)

    def is_path_processed(self, file_path: str) -> bool:
        """检查文件路径是否已处理"""
        return self._doc_repo.find_by_original_path(file_path) is not None

    def get_documents_by_run(self, run_id: str) -> list:
        """获取指定运行的所有文档"""
        return self._doc_repo.find_by_run_id(run_id)

    def get_all_documents(self, limit: int = 1000, offset: int = 0) -> list:
        """获取所有文档（分页）"""
        return self._doc_repo.list(limit, offset)

    def get_document_count(self) -> int:
        """获取文档总数"""
        return self._doc_repo.count()

    def delete_document(self, doc_id: int) -> bool:
        """删除文档"""
        return self._doc_repo.delete(doc_id)

    def delete_documents_by_run(self, run_id: str) -> int:
        """删除指定运行的所有文档"""
        return self._doc_repo.delete_by_run_id(run_id)

    # ========== 运行记录操作（委托给 Repository）==========

    def start_run(self, input_dir: str, output_dir: Optional[str] = None) -> str:
        """开始一个新的清洗运行"""
        return self._run_repo.start_run(input_dir, output_dir)

    def update_run_stats(self, run_id: str, **kwargs) -> None:
        """更新运行统计"""
        self._run_repo.update_stats(run_id, **kwargs)

    def finish_run(self, run_id: str) -> None:
        """结束清洗运行"""
        self._run_repo.finish_run(run_id)

    def get_run(self, run_id: str) -> Optional[dict]:
        """获取运行记录"""
        return self._run_repo.find_by_run_id(run_id)

    def get_all_runs(self) -> list:
        """获取所有运行记录"""
        return self._run_repo.list()

    def delete_run(self, run_id: str) -> bool:
        """删除运行记录"""
        return self._run_repo.delete_run(run_id)

    # ========== 导出功能 ==========

    def export_to_directory(
        self,
        run_id: Optional[str] = None,
        output_dir: str = ".",
    ) -> int:
        """将数据库中的文档导出到目录"""
        return self._doc_repo.export_to_directory(run_id, output_dir)

    # ========== 统计功能 ==========

    def get_stats(self) -> dict:
        """获取统计信息"""
        return self._doc_repo.get_stats()


# ========== 兼容旧接口的别名（放在文件末尾以解决循环依赖）==========
CleaningDatabase = LegacyCleaningDatabase
