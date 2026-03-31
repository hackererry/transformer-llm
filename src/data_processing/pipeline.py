#!/usr/bin/env python
"""
数据清洗流水线
提供组合式流式清洗能力，串联 TextCleaner / PII 脱敏 / 去重 / 质量过滤
"""
import hashlib
import os
import time
from typing import (
    Iterator, List, Optional, Set, Tuple, Callable, Any, Dict
)

from .clean_text import TextCleaner
from .pii_remover import (
    remove_pii, remove_pii_with_count,
    PII_PATTERNS, PII_PLACEHOLDERS, COMPOUND_PATTERNS
)
from .deduplicate import (
    deduplicate_lines_from_text,
    exact_deduplicate,
    near_deduplicate,
)
from .quality_filter import compute_quality_score
from .cleaning_db import CleaningDatabase  # 兼容旧接口


# =====================
# 核心流水线类
# =====================

class CleaningPipeline:
    """
    可组合的清洗流水线

    使用示例:
        pipeline = CleaningPipeline()
        pipeline.add(TextCleaner.fix_encoding_issues)
        pipeline.add(TextCleaner.remove_extra_whitespace)
        pipeline.apply(line)  # 对单行应用

        # 流式处理文件
        for cleaned_line, line_no, out_no in pipeline.process_stream(input_path, output_path):
            ...
    """

    def __init__(self):
        self._steps: List[Callable[[str], str]] = []
        self._quality_threshold: Optional[float] = None
        self._dedup_enabled: bool = False
        self._dedup_seen: Optional[Set[str]] = None

    def add(self, func: Callable[[str], str]) -> "CleaningPipeline":
        """添加清洗步骤（函数签名为 str -> str）"""
        self._steps.append(func)
        return self

    def add_quality_filter(self, threshold: float) -> "CleaningPipeline":
        """添加质量过滤步骤"""
        self._quality_threshold = threshold
        return self

    def add_dedup(self) -> "CleaningPipeline":
        """启用行级去重（MD5 哈希）"""
        self._dedup_enabled = True
        self._dedup_seen = set()
        return self

    def apply(self, text: str) -> str:
        """对单行文本应用流水线"""
        for step in self._steps:
            text = step(text)
        return text

    def process_stream(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        encoding: str = "utf-8",
        show_progress: bool = True,
    ) -> Iterator[Tuple[str, int, int]]:
        """
        流式处理文件：逐行读取 -> 逐行清洗 -> 逐行写出

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径（None 则返回生成器）
            encoding: 文件编码
            show_progress: 是否打印进度

        Yields:
            (cleaned_line, input_line_no, output_line_no)
        """
        output_file = None
        if output_path:
            output_file = open(output_path, "w", encoding=encoding, buffering=8192)

        try:
            output_counter = 0
            input_counter = 0
            start_time = time.time()

            with open(input_path, "r", encoding=encoding, errors="ignore") as f:
                for line_no, raw_line in enumerate(f, 1):
                    input_counter += 1
                    line = raw_line.rstrip("\n\r")

                    # 空行跳过
                    if not line.strip():
                        continue

                    # 1) 应用清洗步骤
                    cleaned = self.apply(line)

                    # 2) 质量过滤
                    if self._quality_threshold is not None:
                        score = compute_quality_score(cleaned)
                        if score < self._quality_threshold:
                            continue

                    # 3) 行级去重
                    if self._dedup_enabled and self._dedup_seen is not None:
                        line_hash = hashlib.md5(
                            cleaned.encode(encoding)
                        ).hexdigest()
                        if line_hash in self._dedup_seen:
                            continue
                        self._dedup_seen.add(line_hash)

                    # 4) 写出
                    output_counter += 1
                    if output_file:
                        output_file.write(cleaned + "\n")
                    yield cleaned, line_no, output_counter

                    # 进度显示
                    if show_progress and input_counter % 10000 == 0:
                        elapsed = time.time() - start_time
                        rate = input_counter / elapsed if elapsed > 0 else 0
                        print(f"    Processed {input_counter:,} lines, "
                              f"output {output_counter:,} lines ({rate:.0f} lines/s)")

        finally:
            if output_file:
                output_file.close()

    def reset(self) -> None:
        """重置流水线状态（用于复用）"""
        if self._dedup_seen is not None:
            self._dedup_seen.clear()


# =====================
# 预构建流水线工厂
# =====================

def build_light_pipeline() -> CleaningPipeline:
    """
    轻量流水线：编码修复 + 空白规范化
    适用场景：预处理速度优先
    """
    pipeline = CleaningPipeline()
    pipeline.add(TextCleaner.fix_encoding_issues)
    pipeline.add(TextCleaner.remove_extra_whitespace)
    return pipeline


def build_standard_pipeline(
    remove_pii: bool = False,
    remove_urls: bool = True,
    remove_page_numbers: bool = True,
    quality_threshold: Optional[float] = None,
    enable_dedup: bool = True,
) -> CleaningPipeline:
    """
    标准流水线：完整清洗步骤

    Args:
        remove_pii: 是否移除 PII（包含邮箱、手机号、身份证等）
        remove_urls: 是否移除 URL
        remove_page_numbers: 是否移除页码
        quality_threshold: 质量分数阈值（None 则跳过质量过滤）
        enable_dedup: 是否启用行级去重（默认启用，per-file 作用域）
    """
    pipeline = CleaningPipeline()

    # Step 1: 编码修复（最先执行）
    pipeline.add(TextCleaner.fix_encoding_issues)

    # Step 2: 特殊字符移除
    pipeline.add(TextCleaner.remove_special_chars)

    # Step 3: URL 移除（文本类杂质）
    # 注意：邮箱由 pii_remover 统一处理（remove_pii=True 时），
    # 不在这里单独处理，避免重复替换导致行为不一致
    if remove_urls:
        pipeline.add(TextCleaner.remove_urls)

    # Step 4: 书籍元数据和格式标记
    pipeline.add(TextCleaner.remove_book_metadata)
    pipeline.add(TextCleaner.remove_format_markers)

    # Step 5: 页码移除
    if remove_page_numbers:
        pipeline.add(TextCleaner.remove_page_numbers)

    # Step 6: 空白规范化（倒数第二步）
    pipeline.add(TextCleaner.remove_extra_whitespace)

    # Step 7: 空行移除
    pipeline.add(TextCleaner.remove_empty_lines)

    # Step 8: PII 脱敏
    if remove_pii:
        pipeline.add(remove_pii)

    # Step 9: 质量过滤（需要阈值）
    if quality_threshold is not None:
        pipeline.add_quality_filter(quality_threshold)

    # Step 10: 去重（放在最后，因为已经清洗完成）
    if enable_dedup:
        pipeline.add_dedup()

    return pipeline


# =====================
# 一站式流式清洗函数
# =====================

def stream_clean_pipeline(
    input_path: str,
    output_path: str,
    remove_pii: bool = True,
    remove_urls: bool = True,
    remove_page_numbers: bool = True,
    quality_threshold: Optional[float] = None,
    enable_dedup: bool = True,
    dedup_threshold: int = 3,
    show_progress: bool = True,
    encoding: str = "utf-8",
    db: Optional[CleaningDatabase] = None,
    run_id: Optional[str] = None,
) -> dict:
    """
    一站式流式清洗函数（供 CLI 直接调用）

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        remove_pii: 是否移除 PII（包含邮箱、手机号、身份证等）
        remove_urls: 是否移除 URL
        remove_page_numbers: 是否移除页码
        quality_threshold: 质量分数阈值（None 跳过）
        enable_dedup: 是否启用行级去重（默认启用）
        dedup_threshold: 去重阈值，出现次数 >= threshold 时才去重（默认 3）
        show_progress: 是否打印进度
        encoding: 文件编码
        db: 数据库实例（可选，用于存储清洗结果）
        run_id: 运行ID（可选）

    Returns:
        处理统计字典
    """
    import hashlib

    stats = {
        "input_lines": 0,
        "output_lines": 0,
        "quality_filtered": 0,
        "dedup_filtered": 0,
        "pii_detected": 0,
    }

    start_time = time.time()

    # 计算原始文件信息
    original_size = 0
    if os.path.exists(input_path):
        original_size = os.path.getsize(input_path)
    original_md5 = ""

    # 第一遍：读取并清洗所有行
    cleaned_lines: List[Tuple[str, str]] = []  # (line_hash, cleaned_line)
    line_counts: Dict[str, int] = {}

    with open(input_path, "r", encoding=encoding, errors="ignore") as fin:
        for line in fin:
            stats["input_lines"] += 1
            line = line.rstrip("\n\r")

            if not line.strip():
                continue

            # 编码修复
            line = TextCleaner.fix_encoding_issues(line)

            # 特殊字符
            line = TextCleaner.remove_special_chars(line)

            # URL
            if remove_urls:
                line = TextCleaner.remove_urls(line)

            # 书籍元数据和格式
            line = TextCleaner.remove_book_metadata(line)
            line = TextCleaner.remove_format_markers(line)

            # 页码
            if remove_page_numbers:
                line = TextCleaner.remove_page_numbers(line)

            # 空白规范化
            line = TextCleaner.remove_extra_whitespace(line)
            line = TextCleaner.remove_empty_lines(line)

            if not line.strip():
                continue

            # PII 移除
            if remove_pii:
                cleaned, counts = remove_pii_with_count(line)
                pii_count = sum(counts.values())
                if pii_count > 0:
                    stats["pii_detected"] += pii_count
                line = cleaned

            # 质量过滤
            if quality_threshold is not None:
                score = compute_quality_score(line)
                if score < quality_threshold:
                    stats["quality_filtered"] += 1
                    continue

            # 记录行哈希和计数
            line_hash = hashlib.md5(line.encode(encoding)).hexdigest()
            cleaned_lines.append((line_hash, line))
            line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

            if show_progress and stats["input_lines"] % 10000 == 0:
                print(f"    Pass 1: Processed {stats['input_lines']:,} lines")

    # 确定需要去重的行哈希（出现次数 >= 阈值）
    if enable_dedup and dedup_threshold > 1:
        dedup_hashes: Set[str] = {
            h for h, count in line_counts.items() if count >= dedup_threshold
        }
    elif enable_dedup:
        dedup_hashes = set(line_counts.keys())
    else:
        dedup_hashes = set()

    # 第二遍：输出去重后的行
    seen_hashes: Set[str] = set()
    output_lines = []
    for line_hash, line in cleaned_lines:
        if line_hash in dedup_hashes:
            if line_hash in seen_hashes:
                stats["dedup_filtered"] += 1
                continue
            seen_hashes.add(line_hash)

        stats["output_lines"] += 1
        output_lines.append(line)

    # 写出到文件（如果指定了 output_path）
    if output_path:
        cleaned_content = "\n".join(output_lines)
        with open(output_path, "w", encoding=encoding, buffering=8192) as fout:
            fout.write(cleaned_content + "\n")

        # 计算清洗后的 MD5
        cleaned_md5 = hashlib.md5(("\n".join(output_lines)).encode(encoding)).hexdigest()
    else:
        cleaned_md5 = hashlib.md5(("\n".join(output_lines)).encode(encoding)).hexdigest()

    # 写入数据库（如果指定了 db）
    if db is not None:
        cleaned_content = "\n".join(output_lines)
        cleaned_size = len(cleaned_content.encode(encoding))

        # 计算质量分数（基于输出内容）
        avg_quality = 1.0
        if output_lines and quality_threshold is not None:
            total_score = sum(compute_quality_score(line) for line in output_lines)
            avg_quality = total_score / len(output_lines) if output_lines else 1.0

        doc_data = {
            "original_file_path": os.path.abspath(input_path),
            "original_md5": original_md5 or hashlib.md5(open(input_path, "rb").read()).hexdigest(),
            "cleaned_md5": cleaned_md5,
            "cleaned_content": cleaned_content,
            "original_size": original_size,
            "cleaned_size": cleaned_size,
            "line_count": stats["output_lines"],
            "quality_score": avg_quality,
            "pii_count": stats["pii_detected"],
            "quality_filtered": stats["quality_filtered"],
            "dedup_filtered": stats["dedup_filtered"],
        }
        db.save_cleaned_document(doc_data, run_id)

    elapsed = time.time() - start_time
    if show_progress:
        rate = stats["input_lines"] / elapsed if elapsed > 0 else 0
        print(f"    Done: {stats['input_lines']:,} input, {stats['output_lines']:,} output "
              f"({rate:.0f} lines/s)")

    return stats


# =====================
# 文档级去重函数
# =====================

def deduplicate_documents(
    input_dir: str,
    output_dir: str,
    dedup_level: str = "exact",
    jaccard_threshold: float = 0.8,
    ngram_n: int = 5,
    recursive: bool = False,
    show_progress: bool = True,
) -> dict:
    """
    文档级去重（跨文件）

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（去重后的文件）
        dedup_level: 去重级别，"exact" (SHA-256) 或 "near" (N-gram+Jaccard)
        jaccard_threshold: Jaccard 相似度阈值（仅 near 模式有效）
        ngram_n: N-gram 大小（仅 near 模式有效）
        recursive: 是否递归扫描子目录
        show_progress: 是否显示进度

    Returns:
        统计字典
    """
    import os

    stats = {
        "total_files": 0,
        "unique_files": 0,
        "duplicates": 0,
    }

    # 扫描文件
    input_files = []
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if fname.lower().endswith(".txt"):
                fpath = os.path.join(root, fname)
                input_files.append(fpath)
        if not recursive:
            break

    input_files.sort()
    stats["total_files"] = len(input_files)

    if show_progress:
        print(f"  Found {len(input_files)} files")

    # 加载所有文档
    if show_progress:
        print(f"  Loading documents...")
    documents = []
    valid_files = []

    for fpath in input_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                lines = [l.strip() for l in f if l.strip()]
                if lines:
                    # 将文件的所有行合并为一个文档
                    documents.append("\n".join(lines))
                    valid_files.append(fpath)
        except Exception as e:
            if show_progress:
                print(f"    Warning: Failed to read {fpath}: {e}")

    if show_progress:
        print(f"  Loaded {len(documents)} documents")

    # 去重
    if dedup_level == "exact":
        if show_progress:
            print(f"  Deduplicating (exact SHA-256)...")
        original_count = len(documents)
        documents = exact_deduplicate(documents)
        stats["duplicates"] = original_count - len(documents)
    else:  # near
        if show_progress:
            print(f"  Deduplicating (near N-gram+Jaccard, threshold={jaccard_threshold})...")
        original_count = len(documents)
        documents = near_deduplicate(documents, threshold=jaccard_threshold, n=ngram_n)
        stats["duplicates"] = original_count - len(documents)

    stats["unique_files"] = len(documents)

    # 写出（需要关联去重后的文档和原文件）
    # 找出被保留的文档对应的原文件
    import hashlib
    kept_files = []
    for fpath, doc in zip(valid_files, documents):
        # 重新计算哈希来匹配
        doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
        if not hasattr(deduplicate_documents, '_kept_hashes'):
            deduplicate_documents._kept_hashes = set()

    # 重新处理：保留去重后的文档
    deduplicate_documents._kept_hashes = set()
    for doc in documents:
        doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
        deduplicate_documents._kept_hashes.add(doc_hash)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 找出保留的文件
    kept_files = []
    for fpath in valid_files:
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
            if lines:
                doc = "\n".join(lines)
                doc_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
                if doc_hash in deduplicate_documents._kept_hashes:
                    kept_files.append((fpath, doc))

    # 写出
    if show_progress:
        print(f"  Writing {len(kept_files)} unique documents...")
    for idx, (orig_path, doc) in enumerate(kept_files):
        # 保持原文件名
        fname = os.path.basename(orig_path)
        out_path = os.path.join(output_dir, fname)

        # 处理文件名冲突
        counter = 1
        base_name = fname
        while os.path.exists(out_path):
            name, ext = os.path.splitext(base_name)
            fname = f"{name}_{counter}{ext}"
            out_path = os.path.join(output_dir, fname)
            counter += 1

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(doc + "\n")

        if show_progress and (idx + 1) % 100 == 0:
            print(f"    Written {idx + 1}/{len(kept_files)} files")

    # 清理全局状态
    if hasattr(deduplicate_documents, '_kept_hashes'):
        del deduplicate_documents._kept_hashes

    if show_progress:
        print(f"  Done: {stats['unique_files']} unique, {stats['duplicates']} duplicates removed")

    return stats
