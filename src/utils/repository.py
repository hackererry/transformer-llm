# -*- coding: utf-8 -*-
"""
数据仓库模块
Data Repository Module

提供各种数据实体的 CRUD 操作
"""
import os
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

from src.utils.database import DatabaseManager, get_default_db_path


# ========== 基类 ==========

class BaseRepository(ABC):
    """数据仓库基类

    提供基本的 CRUD 操作模板
    """

    TABLE_NAME: str = ""

    def __init__(self, db_path: Optional[str] = None):
        """初始化仓库

        Args:
            db_path: 数据库路径，默认使用 db/transform.db
        """
        self.db_path = db_path or get_default_db_path()

    def create(self, data: Dict[str, Any]) -> Optional[int]:
        """创建记录

        Args:
            data: 字典形式的数据

        Returns:
            新记录 ID，失败返回 None
        """
        columns = list(data.keys())
        placeholders = ["?" for _ in columns]
        values = [data[col] for col in columns]

        query = f"""
            INSERT INTO {self.TABLE_NAME} ({', '.join(columns)})
            VALUES ({', '.join(placeholders)})
        """
        try:
            return DatabaseManager.execute_update(query, tuple(values), self.db_path)
        except Exception as e:
            # 处理唯一约束冲突
            if "UNIQUE constraint failed" in str(e):
                return None
            raise e

    def read(self, id: int) -> Optional[Dict[str, Any]]:
        """读取记录

        Args:
            id: 记录 ID

        Returns:
            字典形式的记录，不存在返回 None
        """
        query = f"SELECT * FROM {self.TABLE_NAME} WHERE id = ?"
        rows = DatabaseManager.execute_query(query, (id,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def update(self, id: int, data: Dict[str, Any]) -> bool:
        """更新记录

        Args:
            id: 记录 ID
            data: 要更新的数据

        Returns:
            是否更新成功
        """
        set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
        values = list(data.values()) + [id]
        query = f"UPDATE {self.TABLE_NAME} SET {set_clause} WHERE id = ?"
        rows = DatabaseManager.execute_update(query, tuple(values), self.db_path)
        return rows > 0

    def delete(self, id: int) -> bool:
        """删除记录

        Args:
            id: 记录 ID

        Returns:
            是否删除成功
        """
        query = f"DELETE FROM {self.TABLE_NAME} WHERE id = ?"
        rows = DatabaseManager.execute_update(query, (id,), self.db_path)
        return rows > 0

    def list(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出记录

        Args:
            limit: 返回数量限制
            offset: 偏移量

        Returns:
            记录列表
        """
        query = f"SELECT * FROM {self.TABLE_NAME} LIMIT ? OFFSET ?"
        rows = DatabaseManager.execute_query(query, (limit, offset), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def count(self) -> int:
        """统计记录数量

        Returns:
            记录总数
        """
        query = f"SELECT COUNT(*) as cnt FROM {self.TABLE_NAME}"
        rows = DatabaseManager.execute_query(query, (), self.db_path)
        return rows[0]["cnt"] if rows else 0


# ========== 爬虫相关 Repository ==========

class CrawlerPageRepository(BaseRepository):
    """爬虫页面仓库

    操作 crawler_pages 表
    """

    TABLE_NAME = "crawler_pages"

    def find_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """根据 URL 查找页面

        Args:
            url: 页面 URL

        Returns:
            页面记录字典
        """
        query = "SELECT * FROM crawler_pages WHERE url = ?"
        rows = DatabaseManager.execute_query(query, (url,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def find_by_content_hash(self, content_hash: str) -> List[Dict[str, Any]]:
        """根据内容哈希查找页面

        Args:
            content_hash: 内容 SHA-256 哈希

        Returns:
            页面记录列表
        """
        query = "SELECT * FROM crawler_pages WHERE content_hash = ?"
        rows = DatabaseManager.execute_query(query, (content_hash,), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def find_by_domain(self, domain: str, limit: int = 100) -> List[Dict[str, Any]]:
        """根据域名查找页面

        Args:
            domain: 域名
            limit: 返回数量限制

        Returns:
            页面记录列表
        """
        query = """
            SELECT * FROM crawler_pages
            WHERE domain = ? AND is_crawled = 1
            LIMIT ?
        """
        rows = DatabaseManager.execute_query(query, (domain, limit), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def is_crawled(self, url: str) -> bool:
        """检查 URL 是否已爬取

        Args:
            url: 页面 URL

        Returns:
            是否已爬取
        """
        query = "SELECT COUNT(*) as cnt FROM crawler_pages WHERE url = ? AND is_crawled = 1"
        rows = DatabaseManager.execute_query(query, (url,), self.db_path)
        return rows[0]["cnt"] > 0 if rows else False

    def save_page(self, data: Dict[str, Any]) -> Optional[int]:
        """保存页面（如果已存在则更新）

        Args:
            data: 页面数据，包含 url, content_hash, file_path, domain

        Returns:
            页面 ID
        """
        url = data.get("url")
        existing = self.find_by_url(url)
        if existing:
            # 更新现有记录
            self.update(existing["id"], {
                k: v for k, v in data.items()
                if k != "url" and k != "id"
            })
            return existing["id"]
        else:
            # 创建新记录
            data["is_crawled"] = data.get("is_crawled", 1)
            return self.create(data)

    def get_crawled_count(self) -> int:
        """获取已爬取页面数量

        Returns:
            已爬取页面数量
        """
        query = "SELECT COUNT(*) as cnt FROM crawler_pages WHERE is_crawled = 1"
        rows = DatabaseManager.execute_query(query, (), self.db_path)
        return rows[0]["cnt"] if rows else 0


class CrawlStatsRepository(BaseRepository):
    """爬虫统计仓库

    操作 crawl_stats 表
    """

    TABLE_NAME = "crawl_stats"

    def get_by_date(self, date: str) -> Optional[Dict[str, Any]]:
        """根据日期获取统计

        Args:
            date: 日期字符串 (YYYY-MM-DD)

        Returns:
            统计记录
        """
        query = "SELECT * FROM crawl_stats WHERE date = ?"
        rows = DatabaseManager.execute_query(query, (date,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def get_or_create_today(self) -> Dict[str, Any]:
        """获取或创建今天的统计记录

        Returns:
            今天的统计记录
        """
        today = datetime.now().strftime("%Y-%m-%d")
        existing = self.get_by_date(today)
        if existing:
            return existing

        # 创建新记录
        self.create({"date": today})
        return self.get_by_date(today)

    def increment_stats(
        self,
        pages_crawled: int = 0,
        pages_failed: int = 0,
        pages_queued: int = 0,
        bytes_downloaded: int = 0,
        total_time: float = 0.0,
    ) -> None:
        """增量更新今日统计

        Args:
            pages_crawled: 成功爬取数
            pages_failed: 失败数
            pages_queued: 排队数
            bytes_downloaded: 下载字节数
            total_time: 耗时
        """
        today = self.get_or_create_today()
        today_id = today["id"]

        # 读取当前值并更新
        query = """
            UPDATE crawl_stats
            SET pages_crawled = pages_crawled + ?,
                pages_failed = pages_failed + ?,
                pages_queued = pages_queued + ?,
                bytes_downloaded = bytes_downloaded + ?,
                total_time = total_time + ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """
        DatabaseManager.execute_update(
            query,
            (pages_crawled, pages_failed, pages_queued, bytes_downloaded, total_time, today_id),
            self.db_path,
        )

    def get_recent_stats(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取最近 N 天的统计

        Args:
            days: 天数

        Returns:
            统计记录列表
        """
        query = """
            SELECT * FROM crawl_stats
            ORDER BY date DESC
            LIMIT ?
        """
        rows = DatabaseManager.execute_query(query, (days,), self.db_path)
        return DatabaseManager.rows_to_list(rows)


# ========== 清洗相关 Repository ==========

class CleanedDocumentRepository(BaseRepository):
    """清洗文档仓库

    操作 cleaned_documents 表
    """

    TABLE_NAME = "cleaned_documents"

    def find_by_original_md5(self, md5: str) -> Optional[Dict[str, Any]]:
        """根据原始文件 MD5 查找文档

        Args:
            md5: 原始文件 MD5

        Returns:
            文档记录
        """
        query = "SELECT * FROM cleaned_documents WHERE original_md5 = ?"
        rows = DatabaseManager.execute_query(query, (md5,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def find_by_cleaned_md5(self, md5: str) -> Optional[Dict[str, Any]]:
        """根据清洗后内容 MD5 查找文档

        Args:
            md5: 清洗后内容 MD5

        Returns:
            文档记录
        """
        query = "SELECT * FROM cleaned_documents WHERE cleaned_md5 = ?"
        rows = DatabaseManager.execute_query(query, (md5,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def find_by_original_path(self, file_path: str) -> Optional[Dict[str, Any]]:
        """根据原始文件路径查找文档

        Args:
            file_path: 原始文件路径

        Returns:
            文档记录
        """
        query = "SELECT * FROM cleaned_documents WHERE original_file_path = ?"
        rows = DatabaseManager.execute_query(query, (file_path,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def find_by_run_id(self, run_id: str) -> List[Dict[str, Any]]:
        """根据运行 ID 查找所有文档

        Args:
            run_id: 运行 ID

        Returns:
            文档记录列表
        """
        query = "SELECT * FROM cleaned_documents WHERE run_id = ? ORDER BY id"
        rows = DatabaseManager.execute_query(query, (run_id,), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def is_processed(self, original_md5: str) -> bool:
        """检查文档是否已处理

        Args:
            original_md5: 原始文件 MD5

        Returns:
            是否已处理
        """
        doc = self.find_by_original_md5(original_md5)
        return doc is not None

    def save_document(self, data: Dict[str, Any], run_id: Optional[str] = None) -> Optional[int]:
        """保存清洗后的文档（如果已存在则更新）

        Args:
            data: 文档数据
            run_id: 运行 ID

        Returns:
            文档 ID
        """
        file_path = data.get("original_file_path")
        existing = self.find_by_original_path(file_path) if file_path else None

        if existing:
            # 更新现有记录
            update_data = {k: v for k, v in data.items() if k not in ("id", "original_file_path")}
            if run_id:
                update_data["run_id"] = run_id
            self.update(existing["id"], update_data)
            return existing["id"]
        else:
            # 创建新记录
            if run_id:
                data["run_id"] = run_id
            return self.create(data)

    def save_documents_batch(
        self,
        docs: List[Dict[str, Any]],
        run_id: Optional[str] = None,
    ) -> int:
        """批量保存文档

        Args:
            docs: 文档数据列表
            run_id: 运行 ID

        Returns:
            成功保存的数量
        """
        count = 0
        for doc_data in docs:
            if self.save_document(doc_data, run_id):
                count += 1
        return count

    def delete_by_run_id(self, run_id: str) -> int:
        """删除指定运行的所有文档

        Args:
            run_id: 运行 ID

        Returns:
            删除的记录数
        """
        query = "DELETE FROM cleaned_documents WHERE run_id = ?"
        return DatabaseManager.execute_update(query, (run_id,), self.db_path)

    def export_to_directory(
        self,
        run_id: Optional[str] = None,
        output_dir: str = ".",
    ) -> int:
        """将文档导出到目录

        Args:
            run_id: 运行 ID（None 则导出所有）
            output_dir: 输出目录

        Returns:
            导出文件数量
        """
        os.makedirs(output_dir, exist_ok=True)

        if run_id:
            query = "SELECT * FROM cleaned_documents WHERE run_id = ?"
            rows = DatabaseManager.execute_query(query, (run_id,), self.db_path)
        else:
            query = "SELECT * FROM cleaned_documents"
            rows = DatabaseManager.execute_query(query, (), self.db_path)

        count = 0
        for row in rows:
            doc = DatabaseManager.row_to_dict(row)
            orig_path = doc.get("original_file_path")
            if orig_path:
                basename = os.path.basename(orig_path)
            else:
                basename = f"doc_{doc['id']}.txt"

            out_path = os.path.join(output_dir, basename)
            # 处理文件名冲突
            counter = 1
            base_name = os.path.splitext(basename)[0]
            ext = os.path.splitext(basename)[1] or ".txt"
            while os.path.exists(out_path):
                basename = f"{base_name}_{counter}{ext}"
                out_path = os.path.join(output_dir, basename)
                counter += 1

            with open(out_path, "w", encoding="utf-8") as f:
                f.write(doc.get("cleaned_content", ""))

            count += 1

        return count

    def search_by_content(self, keyword: str, limit: int = 100) -> List[Dict[str, Any]]:
        """在内容中搜索

        Args:
            keyword: 搜索关键词
            limit: 返回数量限制

        Returns:
            匹配的文档列表
        """
        query = """
            SELECT * FROM cleaned_documents
            WHERE cleaned_content LIKE ?
            LIMIT ?
        """
        rows = DatabaseManager.execute_query(query, (f"%{keyword}%", limit), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def get_all_processed_md5s(self) -> set:
        """获取所有非空 original_md5 的集合

        Returns:
            所有已处理文档的 original_md5 集合
        """
        query = "SELECT original_md5 FROM cleaned_documents WHERE original_md5 IS NOT NULL AND original_md5 != ''"
        rows = DatabaseManager.execute_query(query, (), self.db_path)
        return {row["original_md5"] for row in rows}

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息

        Returns:
            统计字典
        """
        query = """
            SELECT
                COUNT(*) as total_documents,
                COALESCE(SUM(original_size), 0) as total_original_size,
                COALESCE(SUM(cleaned_size), 0) as total_cleaned_size,
                COALESCE(SUM(line_count), 0) as total_lines,
                COALESCE(SUM(pii_count), 0) as total_pii
            FROM cleaned_documents
        """
        rows = DatabaseManager.execute_query(query, (), self.db_path)
        return DatabaseManager.row_to_dict(rows[0]) if rows else {}


class CleaningRunRepository(BaseRepository):
    """清洗运行仓库

    操作 cleaning_runs 表
    """

    TABLE_NAME = "cleaning_runs"

    def find_by_run_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        """根据运行 ID 查找

        Args:
            run_id: 运行 ID

        Returns:
            运行记录
        """
        query = "SELECT * FROM cleaning_runs WHERE run_id = ?"
        rows = DatabaseManager.execute_query(query, (run_id,), self.db_path)
        if rows:
            return DatabaseManager.row_to_dict(rows[0])
        return None

    def start_run(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """开始一个新的清洗运行

        Args:
            input_dir: 输入目录
            output_dir: 输出目录

        Returns:
            运行 ID
        """
        run_id = str(uuid.uuid4())
        self.create({
            "run_id": run_id,
            "input_dir": input_dir,
            "output_dir": output_dir,
            "started_at": datetime.now().isoformat(),
        })
        return run_id

    def update_stats(
        self,
        run_id: str,
        total_files: Optional[int] = None,
        success_count: Optional[int] = None,
        error_count: Optional[int] = None,
        total_input_lines: Optional[int] = None,
        total_output_lines: Optional[int] = None,
        total_pii_detected: Optional[int] = None,
        total_quality_filtered: Optional[int] = None,
        total_dedup_filtered: Optional[int] = None,
    ) -> None:
        """更新运行统计

        Args:
            run_id: 运行 ID
            total_files: 总文件数
            success_count: 成功数
            error_count: 错误数
            total_input_lines: 输入总行数
            total_output_lines: 输出总行数
            total_pii_detected: 检测到的 PII 总数
            total_quality_filtered: 质量过滤总数
            total_dedup_filtered: 去重过滤总数
        """
        run = self.find_by_run_id(run_id)
        if not run:
            return

        updates = {}
        if total_files is not None:
            updates["total_files"] = total_files
        if success_count is not None:
            updates["success_count"] = success_count
        if error_count is not None:
            updates["error_count"] = error_count
        if total_input_lines is not None:
            updates["total_input_lines"] = total_input_lines
        if total_output_lines is not None:
            updates["total_output_lines"] = total_output_lines
        if total_pii_detected is not None:
            updates["total_pii_detected"] = total_pii_detected
        if total_quality_filtered is not None:
            updates["total_quality_filtered"] = total_quality_filtered
        if total_dedup_filtered is not None:
            updates["total_dedup_filtered"] = total_dedup_filtered

        if updates:
            self.update(run["id"], updates)

    def finish_run(self, run_id: str) -> None:
        """结束清洗运行

        Args:
            run_id: 运行 ID
        """
        run = self.find_by_run_id(run_id)
        if not run:
            return

        # 汇总关联文档的统计
        doc_repo = CleanedDocumentRepository(self.db_path)
        docs = doc_repo.find_by_run_id(run_id)

        update_data = {
            "finished_at": datetime.now().isoformat(),
        }

        if docs:
            update_data["success_count"] = len(docs)
            update_data["total_output_lines"] = sum(d.get("line_count", 0) or 0 for d in docs)
            update_data["total_pii_detected"] = sum(d.get("pii_count", 0) or 0 for d in docs)
            update_data["total_quality_filtered"] = sum(d.get("quality_filtered", 0) or 0 for d in docs)
            update_data["total_dedup_filtered"] = sum(d.get("dedup_filtered", 0) or 0 for d in docs)

        self.update(run["id"], update_data)

    def get_active_runs(self) -> List[Dict[str, Any]]:
        """获取正在运行的记录

        Returns:
            运行记录列表
        """
        query = "SELECT * FROM cleaning_runs WHERE finished_at IS NULL ORDER BY started_at DESC"
        rows = DatabaseManager.execute_query(query, (), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def get_completed_runs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取已完成的运行记录

        Args:
            limit: 返回数量限制

        Returns:
            运行记录列表
        """
        query = """
            SELECT * FROM cleaning_runs
            WHERE finished_at IS NOT NULL
            ORDER BY started_at DESC
            LIMIT ?
        """
        rows = DatabaseManager.execute_query(query, (limit,), self.db_path)
        return DatabaseManager.rows_to_list(rows)

    def delete_run(self, run_id: str, delete_documents: bool = True) -> bool:
        """删除运行记录

        Args:
            run_id: 运行 ID
            delete_documents: 是否同时删除关联的文档

        Returns:
            是否删除成功
        """
        if delete_documents:
            doc_repo = CleanedDocumentRepository(self.db_path)
            doc_repo.delete_by_run_id(run_id)

        run = self.find_by_run_id(run_id)
        if run:
            return self.delete(run["id"])
        return False


# ========== 便捷函数 ==========

def get_crawler_page_repo(db_path: Optional[str] = None) -> CrawlerPageRepository:
    """获取爬虫页面仓库实例"""
    return CrawlerPageRepository(db_path)


def get_crawl_stats_repo(db_path: Optional[str] = None) -> CrawlStatsRepository:
    """获取爬虫统计仓库实例"""
    return CrawlStatsRepository(db_path)


def get_cleaned_document_repo(db_path: Optional[str] = None) -> CleanedDocumentRepository:
    """获取清洗文档仓库实例"""
    return CleanedDocumentRepository(db_path)


def get_cleaning_run_repo(db_path: Optional[str] = None) -> CleaningRunRepository:
    """获取清洗运行仓库实例"""
    return CleaningRunRepository(db_path)
