#!/usr/bin/env python
"""
数据清洗脚本

对原始文本进行完整清洗流水线处理：
  - 编码修复、空行/空白规范化
  - 特殊字符、URL、页码移除
  - 书籍元数据和格式标记移除
  - PII 个人信息脱敏
  - 质量分数过滤
  - 行级精确去重（per-file 作用域）
  - 文档级去重（跨文件，默认启用）
  - 清洗结果默认存储到数据库

Usage:
    # 单文件清洗（默认存储到数据库 + 文档级去重）
    python scripts/clean_data.py -i data/raw.txt -o data/clean.txt

    # 目录批量清洗（默认行为）
    python scripts/clean_data.py -I data/raw -O data/clean

    # 禁用文档级去重
    python scripts/clean_data.py -I data/raw -O data/clean --no-doc-dedup

    # 禁用数据库存储
    python scripts/clean_data.py -I data/raw -O data/clean --no-db

    # 并发清洗（4进程）
    python scripts/clean_data.py -I data/raw -O data/clean --no-doc-dedup --workers 4

    # 使用内存数据库
    python scripts/clean_data.py -I data/raw -O data/clean --db memory
"""
import os
import sys
import time
import argparse
from typing import List, Optional, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.pipeline import stream_clean_pipeline, deduplicate_documents
from src.data_processing.cleaning_db import create_cleaning_db, CleaningDatabase
import hashlib


# =====================
# 工具函数
# =====================

def scan_text_files(directory: str, recursive: bool = False) -> List[str]:
    """扫描目录下所有 .txt 文件"""
    files = []
    directory = os.path.abspath(directory)
    if recursive:
        for root, _, filenames in os.walk(directory):
            for fname in filenames:
                if fname.lower().endswith(".txt"):
                    files.append(os.path.join(root, fname))
    else:
        for fname in os.listdir(directory):
            full_path = os.path.join(directory, fname)
            if os.path.isfile(full_path) and fname.lower().endswith(".txt"):
                files.append(full_path)
    files.sort()
    return files


# =====================
# 处理函数
# =====================

def process_single_file(
    input_path: str,
    output_path: str,
    quality_threshold: float,
    dedup_threshold: int,
    verbose: bool,
    db: Optional[CleaningDatabase] = None,
    run_id: Optional[str] = None,
    db_only: bool = False,
) -> dict:
    """处理单个文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径（db_only=True 时可为 None）
        quality_threshold: 质量分数阈值
        dedup_threshold: 去重阈值
        verbose: 是否显示详细进度
        db: 数据库实例（可选）
        run_id: 运行ID（可选）
        db_only: 是否仅写入数据库，不输出文件
    """
    if not os.path.exists(input_path):
        print(f"  [ERROR] File not found: {input_path}")
        return {"error": "file not found"}

    file_size = os.path.getsize(input_path)
    print(f"  Input:  {input_path} ({file_size / 1024 / 1024:.1f} MB)")

    # 确定输出路径
    actual_output = None if db_only else output_path
    if actual_output:
        print(f"  Output: {output_path}")
    if db:
        print(f"  Database: enabled")

    start_time = time.perf_counter()

    try:
        stats = stream_clean_pipeline(
            input_path=input_path,
            output_path=actual_output,
            remove_pii=True,
            remove_urls=True,
            remove_page_numbers=True,
            quality_threshold=quality_threshold,
            enable_dedup=True,
            dedup_threshold=dedup_threshold,
            show_progress=verbose,
            db=db,
            run_id=run_id,
        )
    except Exception as e:
        print(f"  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    elapsed = time.time() - start_time

    # 处理跳过的情况
    if stats.get("skipped"):
        print(f"  Skipped (already processed in database)")
        stats["input_size_mb"] = file_size / 1024 / 1024
        stats["output_size_mb"] = 0
        stats["elapsed_seconds"] = elapsed
        stats["throughput_mb_per_sec"] = 0
        return stats

    # 补充统计
    stats["input_size_mb"] = file_size / 1024 / 1024
    if actual_output and os.path.exists(actual_output):
        stats["output_size_mb"] = os.path.getsize(actual_output) / 1024 / 1024
    else:
        stats["output_size_mb"] = 0
    stats["elapsed_seconds"] = elapsed
    stats["throughput_mb_per_sec"] = stats["input_size_mb"] / elapsed if elapsed > 0 else 0

    return stats


def _process_file_worker(
    input_path: str,
    output_path: Optional[str],
    quality_threshold: float,
    dedup_threshold: int,
) -> dict:
    """Worker 进程函数（模块顶层，可被 pickle）

    仅执行 CPU 密集型清洗计算，不接触数据库。
    返回 stats（含 doc_data）或错误信息。
    """
    try:
        stats = stream_clean_pipeline(
            input_path=input_path,
            output_path=output_path,
            remove_pii=True,
            remove_urls=True,
            remove_page_numbers=True,
            quality_threshold=quality_threshold,
            enable_dedup=True,
            dedup_threshold=dedup_threshold,
            show_progress=False,
            db=None,
            skip_db_write=True,
        )
        return {"stats": stats, "file": os.path.basename(input_path)}
    except Exception as e:
        return {"error": str(e), "file": os.path.basename(input_path)}


def process_directory(
    input_dir: str,
    output_dir: str,
    quality_threshold: float,
    dedup_threshold: int,
    recursive: bool,
    verbose: bool,
    db: Optional[CleaningDatabase] = None,
    db_only: bool = False,
    workers: int = 1,
) -> List[dict]:
    """批量处理目录

    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        quality_threshold: 质量分数阈值
        dedup_threshold: 去重阈值
        recursive: 是否递归处理子目录
        verbose: 是否显示详细进度
        db: 数据库实例（可选）
        db_only: 是否仅写入数据库
        workers: 并发进程数（默认1=串行）
    """
    input_files = scan_text_files(input_dir, recursive=recursive)

    if not input_files:
        print(f"[WARNING] No .txt files found in {input_dir}")
        return []

    if not db_only:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Found {len(input_files)} files to process")
    if db:
        print(f"Database: enabled")
    if db_only:
        print("Mode: database only (no file output)")
    if workers > 1:
        print(f"Workers: {workers}")
    print("=" * 60)

    # 开始数据库运行记录
    run_id = None
    if db:
        run_id = db.start_run(input_dir, output_dir if not db_only else None)
        print(f"Run ID: {run_id}")

    all_stats = []
    total_input_lines = 0
    total_output_lines = 0
    total_pii_detected = 0
    total_quality_filtered = 0
    total_dedup_filtered = 0
    success_count = 0
    fail_count = 0
    skip_count = 0

    # ---- 并发路径 (workers > 1) ----
    if workers > 1:
        # 预过滤已处理文件（MD5 检查）
        files_to_process = input_files
        if db:
            processed_md5s = db.get_all_processed_md5s()
            if processed_md5s:
                filtered = []
                for fpath in input_files:
                    with open(fpath, "rb") as f:
                        md5 = hashlib.md5(f.read()).hexdigest()
                    if md5 not in processed_md5s:
                        filtered.append(fpath)
                    else:
                        skip_count += 1
                        if verbose:
                            print(f"  Skipped (DB MD5): {os.path.basename(fpath)}")
                skipped_by_db = len(input_files) - len(filtered)
                if skipped_by_db > 0:
                    print(f"  Skipped {skipped_by_db} already-processed files (DB MD5 check)")
                files_to_process = filtered

        # 预创建所有输出目录
        file_tasks = []
        for input_path in files_to_process:
            if not db_only:
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_subdir = os.path.dirname(output_path)
                if output_subdir:
                    os.makedirs(output_subdir, exist_ok=True)
            else:
                output_path = None
            file_tasks.append((input_path, output_path))

        total_tasks = len(file_tasks)
        doc_data_list = []
        completed = 0

        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for input_path, output_path in file_tasks:
                future = executor.submit(
                    _process_file_worker,
                    input_path, output_path,
                    quality_threshold, dedup_threshold,
                )
                futures[future] = input_path

            for future in as_completed(futures):
                result = future.result()
                completed += 1
                fname = result.get("file", "?")

                if "error" in result:
                    print(f"  [{completed}/{total_tasks}] {fname} - ERROR: {result['error']}")
                    all_stats.append({"file": fname, "stats": {"error": result["error"]}})
                    fail_count += 1
                else:
                    stats = result["stats"]
                    if verbose:
                        print(f"  [{completed}/{total_tasks}] {fname}: "
                              f"{stats.get('input_lines', 0):,} -> {stats.get('output_lines', 0):,} lines")

                    all_stats.append({"file": fname, "stats": stats})
                    success_count += 1
                    total_input_lines += stats.get("input_lines", 0)
                    total_output_lines += stats.get("output_lines", 0)
                    total_pii_detected += stats.get("pii_detected", 0)
                    total_quality_filtered += stats.get("quality_filtered", 0)
                    total_dedup_filtered += stats.get("dedup_filtered", 0)

                    if "doc_data" in stats:
                        doc_data_list.append(stats["doc_data"])

        # 批量写入数据库
        if db and doc_data_list:
            batch_count = db.save_documents_batch(doc_data_list, run_id)
            if verbose:
                print(f"  Batch saved {batch_count} documents to DB")

    # ---- 串行路径 (workers <= 1，原有逻辑) ----
    else:
        for idx, input_path in enumerate(input_files, 1):
            if not db_only:
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                output_subdir = os.path.dirname(output_path)

                if output_subdir:
                    os.makedirs(output_subdir, exist_ok=True)
            else:
                output_path = None

            print(f"\n[{idx}/{len(input_files)}] {os.path.basename(input_path)}")

            stats = process_single_file(
                input_path, output_path, quality_threshold, dedup_threshold,
                verbose, db, run_id, db_only
            )
            all_stats.append({
                "file": os.path.basename(input_path),
                "stats": stats,
            })

            if "error" not in stats:
                if stats.get("skipped"):
                    skip_count += 1
                else:
                    success_count += 1
                total_input_lines += stats.get("input_lines", 0)
                total_output_lines += stats.get("output_lines", 0)
                total_pii_detected += stats.get("pii_detected", 0)
                total_quality_filtered += stats.get("quality_filtered", 0)
                total_dedup_filtered += stats.get("dedup_filtered", 0)
            else:
                fail_count += 1

    # 结束数据库运行记录
    if db and run_id:
        db.update_run_stats(
            run_id,
            total_files=len(input_files),
            success_count=success_count,
            error_count=fail_count,
            total_input_lines=total_input_lines,
            total_output_lines=total_output_lines,
            total_pii_detected=total_pii_detected,
            total_quality_filtered=total_quality_filtered,
            total_dedup_filtered=total_dedup_filtered,
        )
        db.finish_run(run_id)

    # 全局总结
    total_elapsed = sum(s.get("stats", {}).get("elapsed_seconds", 0) for s in all_stats)

    print("\n" + "=" * 60)
    print("Clean Data Summary")
    print("=" * 60)
    print(f"  Total files:        {len(input_files)}")
    print(f"  Success:            {success_count}")
    print(f"  Skipped:            {skip_count}")
    print(f"  Failed:             {fail_count}")
    print(f"  Total input:        {total_input_lines:,} lines")
    print(f"  Total output:       {total_output_lines:,} lines")
    print(f"  PII detected:       {total_pii_detected:,} occurrences")
    print(f"  Quality filtered:   {total_quality_filtered:,} lines")
    print(f"  Dedup filtered:     {total_dedup_filtered:,} lines")
    if total_input_lines > 0:
        retention_rate = total_output_lines / total_input_lines * 100
        print(f"  Retention rate:     {retention_rate:.1f}%")
    print(f"  Total time:         {total_elapsed:.1f}s")
    if db:
        doc_count = db.get_document_count()
        print(f"  Documents in DB:    {doc_count}")

    return all_stats


def process_directory_with_doc_dedup(
    input_dir: str,
    output_dir: str,
    quality_threshold: float,
    dedup_threshold: int,
    dedup_level: str,
    jaccard_threshold: float,
    ngram_n: int,
    recursive: bool,
    verbose: bool,
    db: Optional[CleaningDatabase] = None,
    db_only: bool = False,
) -> dict:
    """
    批量处理目录（带文档级去重），无中间文件

    1. 计算所有文档的哈希
    2. 确定需要保留的文档
    3. 直接清洗并输出到目标目录
    """
    input_files = scan_text_files(input_dir, recursive=recursive)

    if not input_files:
        print(f"[WARNING] No .txt files found in {input_dir}")
        return {"total_files": 0, "duplicates": 0}

    print(f"Found {len(input_files)} files")
    if db:
        print(f"Database: enabled")
    if db_only:
        print("Mode: database only (no file output)")

    # 开始数据库运行记录
    run_id = None
    if db:
        run_id = db.start_run(input_dir, output_dir if not db_only else None)
        print(f"Run ID: {run_id}")

    # 第一步：加载所有文档内容并计算哈希
    if verbose:
        print("  Computing document hashes...")
    documents = []
    file_info = []  # (file_path, rel_path, content)

    # 获取数据库中已处理的 MD5 集合，用于早期过滤
    processed_md5s = set()
    if db:
        processed_md5s = db.get_all_processed_md5s()
        if processed_md5s and verbose:
            print(f"  DB already has {len(processed_md5s)} processed documents")

    db_skip_count = 0
    for fpath in input_files:
        try:
            with open(fpath, "rb") as f:
                raw_content = f.read()
            original_md5 = hashlib.md5(raw_content).hexdigest()

            # 数据库级 MD5 过滤：跳过已处理文件
            if processed_md5s and original_md5 in processed_md5s:
                db_skip_count += 1
                continue

            content = raw_content.decode("utf-8", errors="ignore")
            if content.strip():
                rel_path = os.path.relpath(fpath, input_dir)
                documents.append(content)
                file_info.append((fpath, rel_path, content))
        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to read {fpath}: {e}")

    total_files = len(file_info)
    if db_skip_count > 0:
        print(f"  Skipped {db_skip_count} already-processed files (DB MD5 check)")
    print(f"Loaded {total_files} documents to process")
    print(f"Loaded {total_files} documents")

    # 第二步：文档级去重
    print(f"  Deduplicating documents (level={dedup_level})...")
    original_count = len(documents)

    if dedup_level == "exact":
        # 精确去重：SHA-256
        seen_hashes: Dict[str, str] = {}  # hash -> file_path
        keep_files = []  # 需要保留的文件信息

        for fpath, rel_path, content in file_info:
            doc_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            if doc_hash not in seen_hashes:
                seen_hashes[doc_hash] = fpath
                keep_files.append((fpath, rel_path))
    else:
        # 近似去重：N-gram + Jaccard
        keep_files = []
        kept_contents = []

        for fpath, rel_path, content in file_info:
            is_duplicate = False
            for kept_content in kept_contents:
                from src.data_processing.deduplicate import ngram_signature, jaccard_similarity
                sig1 = ngram_signature(content, n=ngram_n)
                sig2 = ngram_signature(kept_content, n=ngram_n)
                if jaccard_similarity(sig1, sig2) > jaccard_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept_contents.append(content)
                keep_files.append((fpath, rel_path))

    duplicates = total_files - len(keep_files)
    print(f"  Document dedup: {duplicates} duplicates removed, {len(keep_files)} unique files")

    if duplicates == 0:
        print("  (No duplicates found)")

    # 第三步：清洗并输出保留的文档
    if not db_only:
        os.makedirs(output_dir, exist_ok=True)

    print(f"  Cleaning {len(keep_files)} unique files...")
    print("=" * 60)

    all_stats = []
    total_input_lines = 0
    total_output_lines = 0
    total_pii_detected = 0
    total_quality_filtered = 0
    total_line_dedup_filtered = 0
    success_count = 0
    fail_count = 0
    skip_count = 0

    for idx, (input_path, rel_path) in enumerate(keep_files, 1):
        if not db_only:
            output_path = os.path.join(output_dir, rel_path)
            output_subdir = os.path.dirname(output_path)

            if output_subdir:
                os.makedirs(output_subdir, exist_ok=True)
        else:
            output_path = None

        if verbose:
            print(f"\n[{idx}/{len(keep_files)}] {os.path.basename(input_path)}")

        stats = process_single_file(
            input_path, output_path, quality_threshold, dedup_threshold,
            verbose, db, run_id, db_only
        )
        all_stats.append({
            "file": os.path.basename(input_path),
            "stats": stats,
        })

        if "error" not in stats:
            if stats.get("skipped"):
                skip_count += 1
            else:
                success_count += 1
            total_input_lines += stats.get("input_lines", 0)
            total_output_lines += stats.get("output_lines", 0)
            total_pii_detected += stats.get("pii_detected", 0)
            total_quality_filtered += stats.get("quality_filtered", 0)
            total_line_dedup_filtered += stats.get("dedup_filtered", 0)
        else:
            fail_count += 1

    # 结束数据库运行记录
    if db and run_id:
        db.update_run_stats(
            run_id,
            total_files=total_files,
            success_count=success_count,
            error_count=fail_count,
            total_input_lines=total_input_lines,
            total_output_lines=total_output_lines,
            total_pii_detected=total_pii_detected,
            total_quality_filtered=total_quality_filtered,
            total_dedup_filtered=total_line_dedup_filtered,
        )
        db.finish_run(run_id)

    # 全局总结
    total_elapsed = sum(s.get("stats", {}).get("elapsed_seconds", 0) for s in all_stats)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Raw files:           {total_files}")
    print(f"  After doc dedup:    {len(keep_files)}")
    print(f"  Duplicates removed:  {duplicates}")
    print(f"  DB skipped:         {db_skip_count}")
    print(f"  Success:            {success_count}")
    print(f"  Skipped (pipeline): {skip_count}")
    print(f"  Failed:             {fail_count}")
    print(f"  Total input:        {total_input_lines:,} lines")
    print(f"  Total output:       {total_output_lines:,} lines")
    print(f"  PII detected:       {total_pii_detected:,} occurrences")
    print(f"  Quality filtered:   {total_quality_filtered:,} lines")
    print(f"  Line dedup filtered: {total_line_dedup_filtered:,} lines")
    if total_input_lines > 0:
        retention_rate = total_output_lines / total_input_lines * 100
        print(f"  Retention rate:    {retention_rate:.1f}%")
    print(f"  Total time:         {total_elapsed:.1f}s")
    if db:
        doc_count = db.get_document_count()
        print(f"  Documents in DB:    {doc_count}")

    return {
        "total_files": total_files,
        "unique_files": len(keep_files),
        "duplicates": duplicates,
    }


# =====================
# 参数解析
# =====================

def parse_args():
    parser = argparse.ArgumentParser(
        description="数据清洗脚本 - 完整清洗流水线（PII + 质量过滤 + 去重）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单文件清洗（默认存储到数据库 + 文档级去重）
  python scripts/clean_data.py -i data/raw.txt -o data/clean.txt

  # 目录批量清洗（默认行为）
  python scripts/clean_data.py -I data/raw -O data/clean

  # 禁用文档级去重
  python scripts/clean_data.py -I data/raw -O data/clean --no-doc-dedup

  # 禁用数据库存储
  python scripts/clean_data.py -I data/raw -O data/clean --no-db

  # 使用内存数据库
  python scripts/clean_data.py -I data/raw -O data/clean --db memory

  # 目录清洗 + 近似去重
  python scripts/clean_data.py -I data/raw -O data/clean --dedup-level near

  # 并发清洗（4进程，仅 --no-doc-dedup 模式有效）
  python scripts/clean_data.py -I data/raw -O data/clean --no-doc-dedup --workers 4
        """,
    )

    # --- 输入输出 ---
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-i", "--input", type=str, required=False,
        help="输入文件路径",
    )
    io_group.add_argument(
        "-o", "--output", type=str, required=False,
        help="输出文件路径",
    )
    io_group.add_argument(
        "-I", "--input_dir", type=str, required=False,
        help="输入目录路径",
    )
    io_group.add_argument(
        "-O", "--output_dir", type=str, required=False,
        help="输出目录路径",
    )
    io_group.add_argument(
        "--recursive", action="store_true",
        help="递归扫描子目录",
    )

    # --- 清洗参数 ---
    clean_group = parser.add_argument_group("Cleaning Options")
    clean_group.add_argument(
        "--quality-threshold", type=float, default=0.5,
        help="质量分数阈值 (0.0~1.0)，低于此值被过滤（默认 0.5）",
    )
    clean_group.add_argument(
        "--dedup-threshold", type=int, default=3,
        help="行级去重阈值，出现次数 >= threshold 时才去重（默认 3）",
    )

    # --- 文档级去重 ---
    doc_dedup_group = parser.add_argument_group("Document-level Deduplication")
    doc_dedup_group.add_argument(
        "--no-doc-dedup", action="store_true",
        help="禁用文档级去重（默认启用）",
    )
    doc_dedup_group.add_argument(
        "--dedup-level", choices=["exact", "near"], default="exact",
        help="文档级去重算法: exact(SHA-256) 或 near(N-gram+Jaccard，默认 exact)",
    )
    doc_dedup_group.add_argument(
        "--jaccard-threshold", type=float, default=0.8,
        help="近似去重 Jaccard 相似度阈值（默认 0.8，仅 near 模式有效）",
    )
    doc_dedup_group.add_argument(
        "--ngram-n", type=int, default=5,
        help="N-gram 大小（默认 5，仅 near 模式有效）",
    )

    # --- 通用选项 ---
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "-v", "--verbose", action="store_true",
        help="显示详细处理进度",
    )
    misc_group.add_argument(
        "--dry-run", action="store_true",
        help="模拟运行（不实际写出文件）",
    )
    misc_group.add_argument(
        "--encoding", type=str, default="utf-8",
        help="输入文件编码（默认 utf-8）",
    )
    misc_group.add_argument(
        "--workers", type=int, default=1,
        help="并发进程数（默认1=串行，0=自动=CPU核数，仅 --no-doc-dedup 模式有效）",
    )

    # --- 数据库选项 ---
    db_group = parser.add_argument_group("Database Options")
    db_group.add_argument(
        "--db", type=str, default=None,
        help="数据库路径（默认 db/transform.db，如设为 'memory' 则使用内存数据库）",
    )
    db_group.add_argument(
        "--no-db", action="store_true",
        help="禁用数据库存储（默认启用）",
    )
    db_group.add_argument(
        "--db-only", action="store_true",
        help="仅写入数据库，不输出 TXT 文件",
    )
    db_group.add_argument(
        "--export-dir", type=str, default=None,
        help="从数据库导出到指定目录",
    )
    db_group.add_argument(
        "--run-id", type=str, default=None,
        help="指定运行ID（用于查看运行记录）",
    )

    args = parser.parse_args()

    # --- 参数校验 ---
    # 导出模式只需要 --db 和 --export-dir
    if args.export_dir is not None:
        if args.input is not None or args.input_dir is not None:
            parser.error("导出模式不能同时指定输入文件/目录")
        if args.output is not None or args.output_dir is not None:
            parser.error("导出模式不能同时指定输出文件/目录")
        return args

    # 查看运行记录模式
    if args.run_id is not None and args.export_dir is None:
        if args.input is not None or args.input_dir is not None:
            parser.error("查看运行记录模式不能同时指定输入文件/目录")
        if args.output is not None or args.output_dir is not None:
            parser.error("查看运行记录模式不能同时指定输出文件/目录")
        return args

    has_input = args.input is not None or args.input_dir is not None
    if not has_input:
        parser.error("必须指定 --input 或 --input_dir")

    # 互斥校验
    if args.input is not None and args.input_dir is not None:
        parser.error("不能同时指定 --input 和 --input_dir")

    if args.output is not None and args.output_dir is not None:
        parser.error("不能同时指定 --output 和 --output_dir")

    return args


# =====================
# 主入口
# =====================

def main():
    args = parse_args()

    print("=" * 60)
    print("Data Cleaning Pipeline (Full Mode)")
    print("=" * 60)
    print(f"Quality threshold: {args.quality_threshold}")
    print(f"Dedup threshold: {args.dedup_threshold}")
    print(f"Doc dedup: {not args.no_doc_dedup}")
    if not args.no_doc_dedup:
        print(f"Doc dedup level: {args.dedup_level}")
    print(f"Dry run: {args.dry_run}")
    print(f"Database: {not args.no_db}")

    # 解析 workers 参数
    workers = args.workers
    if workers == 0:
        workers = os.cpu_count() or 4
    print(f"Workers: {workers}")

    # --- 导出模式 ---
    if args.export_dir is not None:
        print(f"Database: {args.db or 'db/transform.db'}")
        print(f"Export to: {args.export_dir}")
        print("=" * 60)

        db = create_cleaning_db(args.db) if args.db else create_cleaning_db()

        if args.run_id:
            # 导出指定运行
            print(f"Exporting run: {args.run_id}")
            count = db.export_to_directory(args.run_id, args.export_dir)
            print(f"\nExported {count} documents from run {args.run_id}")
        else:
            # 导出所有文档
            print("Exporting all documents...")
            count = db.export_to_directory(None, args.export_dir)
            print(f"\nExported {count} documents")

        print("=" * 60)
        return

    # --- 查看运行记录模式 ---
    if args.run_id is not None and args.export_dir is None:
        print(f"Database: {args.db or 'db/transform.db'}")
        print("=" * 60)

        db = create_cleaning_db(args.db) if args.db else create_cleaning_db()
        run = db.get_run(args.run_id)

        if run:
            print(f"Run ID: {run['run_id']}")
            print(f"Input dir: {run['input_dir']}")
            print(f"Output dir: {run['output_dir']}")
            print(f"Started: {run['started_at']}")
            print(f"Finished: {run['finished_at']}")
            print(f"Total files: {run['total_files']}")
            print(f"Success: {run['success_count']}")
            print(f"Error: {run['error_count']}")
            print(f"Total input lines: {run['total_input_lines']:,}")
            print(f"Total output lines: {run['total_output_lines']:,}")
            print(f"Total PII detected: {run['total_pii_detected']:,}")
            print(f"Total quality filtered: {run['total_quality_filtered']:,}")
            print(f"Total dedup filtered: {run['total_dedup_filtered']:,}")
        else:
            print(f"Run not found: {args.run_id}")

        print("=" * 60)
        return

    # --- 初始化数据库（默认启用） ---
    db = None
    if not args.no_db:
        db = create_cleaning_db(args.db) if args.db else create_cleaning_db()
        print(f"Database: {args.db or 'db/transform.db'}")

    if args.db_only:
        print("Mode: database only (no file output)")

    # --- 单文件模式 ---
    if args.input is not None:
        output_path = None if args.db_only else (args.output or (os.path.splitext(args.input)[0] + "_cleaned.txt"))

        if args.dry_run:
            print(f"[Dry-run] Would clean {args.input} -> {output_path or 'database only'}")
        else:
            stats = process_single_file(
                args.input, output_path, args.quality_threshold,
                args.dedup_threshold, args.verbose, db, None, args.db_only
            )
            if "error" not in stats:
                print("\n" + "=" * 60)
                print("File Clean Summary")
                print("=" * 60)
                print(f"  Input lines:      {stats.get('input_lines', 0):,}")
                print(f"  Output lines:     {stats.get('output_lines', 0):,}")
                print(f"  PII detected:     {stats.get('pii_detected', 0):,}")
                print(f"  Quality filtered: {stats.get('quality_filtered', 0):,}")
                print(f"  Dedup filtered:   {stats.get('dedup_filtered', 0):,}")
                print(f"  Elapsed:          {stats.get('elapsed_seconds', 0):.1f}s")
                print(f"  Throughput:       {stats.get('throughput_mb_per_sec', 0):.2f} MB/s")
                if not args.db_only:
                    print(f"  Output:           {output_path}")
                if db:
                    print(f"  Database:         saved")

    # --- 目录批量模式 ---
    elif args.input_dir is not None:
        output_dir = None if args.db_only else (args.output_dir or (args.input_dir + "_cleaned"))

        if args.dry_run:
            files = scan_text_files(args.input_dir, recursive=args.recursive)
            if not args.no_doc_dedup:
                print(f"[Dry-run] Would dedup {len(files)} docs (level={args.dedup_level}) and clean -> {output_dir or 'database only'}")
            else:
                print(f"[Dry-run] Would clean {len(files)} files -> {output_dir or 'database only'}")
        else:
            if not args.no_doc_dedup:
                # 文档级去重 + 清洗（无中间文件）
                process_directory_with_doc_dedup(
                    input_dir=args.input_dir,
                    output_dir=output_dir or ".",
                    quality_threshold=args.quality_threshold,
                    dedup_threshold=args.dedup_threshold,
                    dedup_level=args.dedup_level,
                    jaccard_threshold=args.jaccard_threshold,
                    ngram_n=args.ngram_n,
                    recursive=args.recursive,
                    verbose=args.verbose,
                    db=db,
                    db_only=args.db_only,
                )
            else:
                # 仅清洗（无文档级去重）
                process_directory(
                    args.input_dir, output_dir or ".", args.quality_threshold,
                    args.dedup_threshold, args.recursive, args.verbose,
                    db=db, db_only=args.db_only, workers=workers
                )

    print("=" * 60)


if __name__ == "__main__":
    main()
