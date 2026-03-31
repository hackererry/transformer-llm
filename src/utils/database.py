# -*- coding: utf-8 -*-
"""
统一数据库管理器
Unified Database Manager

数据库文件统一存放在 db/transform.db

Usage:
    from src.utils.database import DatabaseManager, get_default_db_path

    # 获取默认数据库路径
    db_path = get_default_db_path()

    # 获取数据库连接
    conn = DatabaseManager.get_connection()

    # 初始化数据库
    DatabaseManager.init_database()

    # 备份数据库
    backup_path = DatabaseManager.backup_database()
"""
import os
import sqlite3
import shutil
from datetime import datetime
from typing import Optional, List, Tuple, Any
from contextlib import contextmanager

# 默认数据库路径
DEFAULT_DB_NAME = "transform.db"


def get_project_root() -> str:
    """获取项目根目录"""
    # src/utils/database.py -> src/utils -> src -> 项目根目录
    current_file = os.path.abspath(__file__)
    src_dir = os.path.dirname(os.path.dirname(current_file))
    project_root = os.path.dirname(src_dir)
    return project_root


def get_default_db_path() -> str:
    """获取默认数据库路径

    Returns:
        db/transform.db 的绝对路径
    """
    db_dir = os.path.join(get_project_root(), "db")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, DEFAULT_DB_NAME)


class DatabaseManager:
    """统一数据库管理器

    提供数据库连接、初始化、备份等基础功能
    """

    @staticmethod
    def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
        """获取数据库连接

        Args:
            db_path: 数据库路径，默认使用 db/transform.db

        Returns:
            sqlite3.Connection 对象
        """
        if db_path is None:
            db_path = get_default_db_path()

        # 确保目录存在
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        # 启用外键约束
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    @contextmanager
    def get_cursor(db_path: Optional[str] = None):
        """上下文管理器，获取游标

        Args:
            db_path: 数据库路径，默认使用 db/transform.db

        Usage:
            with DatabaseManager.get_cursor() as cursor:
                cursor.execute("SELECT * FROM table")
                rows = cursor.fetchall()
        """
        conn = DatabaseManager.get_connection(db_path)
        try:
            cursor = conn.cursor()
            yield cursor
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def init_database(db_path: Optional[str] = None) -> None:
        """初始化数据库（创建所有表）

        Args:
            db_path: 数据库路径，默认使用 db/transform.db
        """
        from src.utils.database_schema import init_all_tables

        conn = DatabaseManager.get_connection(db_path)
        try:
            init_all_tables(conn)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def backup_database(
        db_path: Optional[str] = None,
        backup_path: Optional[str] = None,
    ) -> str:
        """备份数据库

        Args:
            db_path: 数据库路径，默认使用 db/transform.db
            backup_path: 备份文件路径，默认使用 db/transform_backup_YYYYMMDD_HHMMSS.db

        Returns:
            备份文件路径
        """
        if db_path is None:
            db_path = get_default_db_path()

        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_dir = os.path.dirname(db_path)
            backup_path = os.path.join(db_dir, f"transform_backup_{timestamp}.db")

        shutil.copy2(db_path, backup_path)
        return backup_path

    @staticmethod
    def execute_query(
        query: str,
        params: Tuple = (),
        db_path: Optional[str] = None,
    ) -> List[sqlite3.Row]:
        """执行查询

        Args:
            query: SQL 查询语句
            params: 查询参数
            db_path: 数据库路径

        Returns:
            查询结果列表
        """
        with DatabaseManager.get_cursor(db_path) as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    @staticmethod
    def execute_update(
        query: str,
        params: Tuple = (),
        db_path: Optional[str] = None,
    ) -> int:
        """执行更新（INSERT/UPDATE/DELETE）

        Args:
            query: SQL 语句
            params: 参数
            db_path: 数据库路径

        Returns:
            影响的行数
        """
        with DatabaseManager.get_cursor(db_path) as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    @staticmethod
    def table_exists(table_name: str, db_path: Optional[str] = None) -> bool:
        """检查表是否存在

        Args:
            table_name: 表名
            db_path: 数据库路径

        Returns:
            表是否存在
        """
        query = """
            SELECT name FROM sqlite_master
            WHERE type='table' AND name=?
        """
        rows = DatabaseManager.execute_query(query, (table_name,), db_path)
        return len(rows) > 0

    @staticmethod
    def get_table_info(table_name: str, db_path: Optional[str] = None) -> List[dict]:
        """获取表结构信息

        Args:
            table_name: 表名
            db_path: 数据库路径

        Returns:
            表字段信息列表
        """
        query = f"PRAGMA table_info({table_name})"
        rows = DatabaseManager.execute_query(query, (), db_path)
        return [dict(row) for row in rows]

    @staticmethod
    def row_to_dict(row: sqlite3.Row) -> dict:
        """将 sqlite3.Row 转换为字典

        Args:
            row: sqlite3.Row 对象

        Returns:
            字典
        """
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def rows_to_list(rows: List[sqlite3.Row]) -> List[dict]:
        """将 sqlite3.Row 列表转换为字典列表

        Args:
            rows: sqlite3.Row 列表

        Returns:
            字典列表
        """
        return [DatabaseManager.row_to_dict(row) for row in rows]


# ========== 便捷函数 ==========

def get_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """获取数据库连接（便捷函数）"""
    return DatabaseManager.get_connection(db_path)


def init_database(db_path: Optional[str] = None) -> None:
    """初始化数据库（便捷函数）"""
    DatabaseManager.init_database(db_path)


def backup_database(
    db_path: Optional[str] = None,
    backup_path: Optional[str] = None,
) -> str:
    """备份数据库（便捷函数）"""
    return DatabaseManager.backup_database(db_path, backup_path)
