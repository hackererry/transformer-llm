#!/usr/bin/env python
"""
文档格式转换工具

支持 PDF, CSV, JSON, EPUB, Parquet 格式转换为纯文本 TXT

Usage:
    # 单文件转换
    python scripts/convert_document.py -i input.pdf -o output.txt

    # 批量转换目录
    python scripts/convert_document.py -I input_dir -O output_dir

    # 合并多个TXT文件
    python scripts/convert_document.py --merge file1.txt file2.txt -o merged.txt
"""
import os
import sys
import argparse
from typing import List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.document_converter import (
    convert_to_txt,
    batch_convert,
    merge_txt_files,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="文档格式转换工具 - 支持 PDF, CSV, JSON, EPUB, Parquet 转换为 TXT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单文件转换
  python scripts/convert_document.py -i input.pdf -o output.txt

  # 批量转换目录
  python scripts/convert_document.py -I input_dir -O output_dir

  # 指定并发线程数
  python scripts/convert_document.py -I input_dir -O output_dir --workers 8

  # 合并多个TXT文件
  python scripts/convert_document.py --merge file1.txt file2.txt -o merged.txt

  # 合并目录下所有TXT文件
  python scripts/convert_document.py --merge_dir input_dir -o merged.txt
        """,
    )

    # --- 转换模式 ---
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "-i", "--input", type=str,
        help="输入文件路径 (支持 .pdf, .csv, .json, .epub, .parquet)",
    )
    io_group.add_argument(
        "-o", "--output", type=str,
        help="输出文件路径",
    )
    io_group.add_argument(
        "-I", "--input_dir", type=str,
        help="输入目录路径（批量转换）",
    )
    io_group.add_argument(
        "-O", "--output_dir", type=str,
        help="输出目录路径（批量转换）",
    )

    # --- 合并模式 ---
    merge_group = parser.add_argument_group("Merge Mode")
    merge_group.add_argument(
        "--merge", nargs='+', type=str,
        help="合并多个TXT文件",
    )
    merge_group.add_argument(
        "--merge_dir", type=str,
        help="合并目录下所有TXT文件",
    )

    # --- 批量转换选项 ---
    batch_group = parser.add_argument_group("Batch Convert Options")
    batch_group.add_argument(
        "--workers", type=int, default=None,
        help="并发线程数（默认为 CPU 核数）",
    )

    args = parser.parse_args()

    # --- 参数校验 ---
    # 合并模式
    if args.merge is not None or args.merge_dir is not None:
        if args.input is not None or args.input_dir is not None:
            parser.error("合并模式不能与转换模式同时使用")
        if args.merge_dir is not None and args.merge is not None:
            parser.error("不能同时指定 --merge 和 --merge_dir")
        return args

    # 转换模式
    has_input = args.input is not None or args.input_dir is not None
    if not has_input:
        parser.error("必须指定 --input/--input_dir 或 --merge/--merge_dir")

    if args.input is not None and args.input_dir is not None:
        parser.error("不能同时指定 --input 和 --input_dir")

    return args


def main():
    args = parse_args()

    print("=" * 60)
    print("Document Converter - PDF/CSV/JSON/EPUB/Parquet to TXT")
    print("=" * 60)

    # --- 合并模式 ---
    if args.merge is not None:
        txt_files = [os.path.abspath(f) for f in args.merge]
        # 过滤存在的文件
        existing_files = [f for f in txt_files if os.path.exists(f)]
        if len(existing_files) != len(txt_files):
            missing = set(txt_files) - set(existing_files)
            print(f"[WARNING] 跳过不存在的文件: {missing}")

        if not existing_files:
            print("[ERROR] 没有有效的文件可以合并")
            return

        output_path = args.output or "merged.txt"
        merge_txt_files(existing_files, output_path)
        print(f"合并完成: {output_path}")

    elif args.merge_dir is not None:
        input_dir = os.path.abspath(args.merge_dir)
        if not os.path.isdir(input_dir):
            print(f"[ERROR] 目录不存在: {input_dir}")
            return

        # 扫描目录下所有TXT文件
        txt_files = []
        for fname in os.listdir(input_dir):
            fpath = os.path.join(input_dir, fname)
            if os.path.isfile(fpath) and fname.lower().endswith('.txt'):
                txt_files.append(fpath)

        if not txt_files:
            print(f"[ERROR] 在 {input_dir} 中未找到 .txt 文件")
            return

        txt_files.sort()
        output_path = args.output or os.path.join(input_dir, "merged.txt")
        merge_txt_files(txt_files, output_path)
        print(f"合并完成: {output_path}")

    # --- 单文件转换 ---
    elif args.input is not None:
        input_path = os.path.abspath(args.input)
        output_path = os.path.abspath(args.output) if args.output else None

        print(f"Input:  {input_path}")
        if output_path:
            print(f"Output: {output_path}")
        print("=" * 60)

        try:
            result = convert_to_txt(input_path, output_path)
            print("=" * 60)
            print(f"转换成功: {result}")
        except Exception as e:
            print("=" * 60)
            print(f"[ERROR] 转换失败: {e}")
            import traceback
            traceback.print_exc()

    # --- 批量转换目录 ---
    elif args.input_dir is not None:
        input_dir = os.path.abspath(args.input_dir)
        output_dir = os.path.abspath(args.output_dir) if args.output_dir else None

        if not os.path.isdir(input_dir):
            print(f"[ERROR] 目录不存在: {input_dir}")
            return

        print(f"Input dir:  {input_dir}")
        if output_dir:
            print(f"Output dir: {output_dir}")
        if args.workers:
            print(f"Workers:    {args.workers}")
        print("=" * 60)

        success_files = batch_convert(input_dir, output_dir, max_workers=args.workers)

        print("=" * 60)
        if success_files:
            print(f"转换成功: {len(success_files)} 个文件")

    print("=" * 60)


if __name__ == "__main__":
    main()
