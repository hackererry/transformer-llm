#!/usr/bin/env python
"""
数据预处理脚本 v2.0
支持多文件、增量处理、流式编码、采样训练tokenizer

Usage:
    # 方式1: 目录输入（自动扫描所有txt文件）
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data \
        --max_seq_length 512 \
        --vocab_size 32000

    # 方式2: 文件列表输入
    python scripts/preprocess_data.py \
        --train_files dataset/data/train1.txt dataset/data/train2.txt \
        --output_dir ./preprocessed_data \
        --max_seq_length 512

    # 方式3: 带验证集
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --validation_file dataset/data/val.txt \
        --output_dir ./preprocessed_data

    # 增量处理: 检测到文件变化后自动重新处理
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data \
        --force_reprocess  # 强制重新处理
"""
import os
import sys
import argparse
import hashlib
import json
import time
import random
from typing import List, Dict, Any, Optional, Iterator, Tuple
from contextlib import contextmanager
from dataclasses import dataclass

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import HuggingFaceBPETokenizer, save_preprocessed_data
import torch


# ============ 配置类 ============

@dataclass
class PreprocessConfig:
    """预处理配置"""
    train_files: List[str]
    validation_file: Optional[str]
    output_dir: str
    max_seq_length: int = 512
    vocab_size: int = 32000
    min_frequency: int = 2
    shard_size: int = 10000
    tokenizer_sample_bytes: int = 100 * 1024 * 1024  # 100MB for tokenizer training
    force_reprocess: bool = False


# ============ 性能监控 ============

class PreprocessPerformanceMonitor:
    """数据预处理性能监控器"""

    def __init__(self, enabled: bool = True, print_interval: int = 2000):
        self.enabled = enabled
        self.print_interval = print_interval
        self.reset()

    def reset(self):
        self.timings: Dict[str, List[float]] = {
            "file_hash": [],
            "load_text": [],
            "tokenizer_train": [],
            "tokenizer_save": [],
            "tokenization": [],
            "shard_save": [],
            "total": [],
        }
        self.stage_start: Optional[float] = None
        self._call_counts: Dict[str, int] = {}

    @contextmanager
    def measure(self, stage: str):
        if not self.enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timings[stage].append(elapsed)
            count = self._call_counts.get(stage, 0) + 1
            self._call_counts[stage] = count
            if count % self.print_interval == 0 or count == 1:
                print(f"  [Timing] {stage}: {elapsed:.3f}s (batch #{count})")

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, times in self.timings.items():
            if stage == "total":
                continue
            if len(times) > 0:
                summary[stage] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
            else:
                summary[stage] = {"count": 0, "total": 0, "mean": 0, "min": 0, "max": 0}
        return summary

    def print_summary(self, dataset_name: str = ""):
        summary = self.get_summary()
        total_time = self.timings.get("total", [0])[0] if self.timings.get("total") else 0

        print("\n" + "=" * 70)
        print(f"              Preprocessing Performance Summary {dataset_name}")
        print("=" * 70)
        print(f"{'Stage':<25} {'Count':>8} {'Total(s)':>12} {'Mean(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}")
        print("-" * 70)

        for stage, stats in summary.items():
            if stats["count"] > 0:
                stage_name = stage.replace("_", " ").title()
                print(f"{stage_name:<25} {stats['count']:>8} {stats['total']:>12.3f} {stats['mean']*1000:>12.2f} {stats['min']*1000:>12.2f} {stats['max']*1000:>12.2f}")

        print("-" * 70)

        if total_time > 0:
            print(f"\nTime Breakdown:")
            for stage, stats in summary.items():
                if stats["total"] > 0:
                    pct = stats["total"] / total_time * 100
                    stage_name = stage.replace("_", " ").title()
                    print(f"  {stage_name:<23}: {pct:>6.2f}%  ({stats['total']:.3f}s)")
            print(f"\n  {'Total':<23}: 100.00%  ({total_time:.3f}s)")

        print("=" * 70 + "\n")


# ============ 核心函数 ============

def parse_args():
    parser = argparse.ArgumentParser(description="数据预处理脚本 v2.0")

    # 数据输入（支持两种方式）
    input_group = parser.add_argument_group("数据输入")
    input_group.add_argument("--train_dir", type=str, default=None,
                             help="训练数据目录（自动扫描所有 .txt 文件）")
    input_group.add_argument("--train_files", type=str, nargs="+", default=None,
                             help="训练数据文件列表")
    input_group.add_argument("--validation_file", type=str, default=None,
                             help="验证数据文件路径")

    # 输出配置
    output_group = parser.add_argument_group("输出配置")
    output_group.add_argument("--output_dir", type=str, default="./preprocessed_data",
                             help="输出目录")
    output_group.add_argument("--force_reprocess", action="store_true",
                             help="强制重新处理，即使文件未变化")

    # 预处理参数
    preprocess_group = parser.add_argument_group("预处理参数")
    preprocess_group.add_argument("--max_seq_length", type=int, default=512,
                                 help="最大序列长度")
    preprocess_group.add_argument("--vocab_size", type=int, default=32000,
                                 help="词表大小")
    preprocess_group.add_argument("--min_frequency", type=int, default=2,
                                 help="BPE最小频率")
    preprocess_group.add_argument("--shard_size", type=int, default=10000,
                                 help="每个分片的样本数量")
    preprocess_group.add_argument("--tokenizer_sample_bytes", type=int, default=100 * 1024 * 1024,
                                 help="用于训练tokenizer的采样字节数（默认100MB）")

    args = parser.parse_args()

    # 验证输入
    if args.train_dir is None and args.train_files is None:
        parser.error("必须指定 --train_dir 或 --train_files")

    if args.train_dir is not None and args.train_files is not None:
        parser.error("不能同时指定 --train_dir 和 --train_files")

    return args


def compute_file_hash(file_path: str) -> str:
    """计算文件的MD5哈希值（流式读取，不占用过多内存）"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def compute_files_hash(file_paths: List[str]) -> Dict[str, str]:
    """计算多个文件的哈希值"""
    hashes = {}
    for path in sorted(file_paths):
        hashes[path] = compute_file_hash(path)
    return hashes


def scan_text_files(directory: str) -> List[str]:
    """扫描目录下所有 .txt 文件"""
    text_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            full_path = os.path.join(directory, filename)
            text_files.append(full_path)
    text_files.sort()
    return text_files


def sample_text_for_tokenizer(
    file_paths: List[str],
    target_bytes: int = 100 * 1024 * 1024,
) -> List[str]:
    """
    从多个文件中采样文本用于训练tokenizer
    采用分层采样，确保各文件都有代表性样本

    Args:
        file_paths: 文件路径列表
        target_bytes: 目标采样字节数

    Returns:
        采样后的文本列表
    """
    print(f"\n[Tokenizer Sampling] 目标采样: {target_bytes / 1024 / 1024:.1f}MB")

    # 获取各文件大小
    file_sizes = []
    for path in file_paths:
        size = os.path.getsize(path)
        file_sizes.append((path, size))

    total_size = sum(s for _, s in file_sizes)
    print(f"[Tokenizer Sampling] 总文件大小: {total_size / 1024 / 1024:.1f}MB")

    sampled_texts = []
    bytes_collected = 0

    # 按比例分层采样
    for path, size in file_sizes:
        if bytes_collected >= target_bytes:
            break

        # 计算该文件应采样的比例
        ratio = min(target_bytes / total_size, 1.0)
        sample_size = int(size * ratio * len(file_paths))  # 稍微多采样一点以防不足

        # 如果文件较小，直接全部读取
        if size <= sample_size * 2:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts = [line.strip() for line in f if line.strip()]
            sampled_texts.extend(texts)
            bytes_collected += size
        else:
            # 随机采样
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) > 0:
                # 按行数比例采样
                num_sample = max(1, int(len(lines) * ratio * len(file_paths)))
                num_sample = min(num_sample, len(lines))

                # 随机选择样本行
                random.seed(42)  # 保证可复现
                sampled_indices = random.sample(range(len(lines)), num_sample)
                for idx in sampled_indices:
                    sampled_texts.append(lines[idx])
                    bytes_collected += len(lines[idx].encode("utf-8"))

                    if bytes_collected >= target_bytes:
                        break

    print(f"[Tokenizer Sampling] 实际采样: {len(sampled_texts)} 行, ~{bytes_collected / 1024 / 1024:.1f}MB")

    # 打乱顺序
    random.shuffle(sampled_texts)

    return sampled_texts


def stream_text_files(file_paths: List[str]) -> Iterator[str]:
    """
    流式读取多个文本文件
    每次只读取一行，避免内存溢出

    Args:
        file_paths: 文件路径列表

    Yields:
        每行文本
    """
    for path in file_paths:
        print(f"  Reading: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def load_all_texts(file_paths: List[str]) -> List[str]:
    """加载所有文本文件到内存（适用于中等规模数据）"""
    texts = []
    for path in file_paths:
        print(f"  Loading: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def stream_encode_and_save(
    file_paths: List[str],
    tokenizer,
    max_seq_length: int,
    output_dir: str,
    dataset_name: str,
    shard_size: int = 10000,
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    流式编码并保存数据

    流程：
    1. 流式读取文本
    2. 编码并累积 token
    3. 当累积超过 max_seq_length 时创建样本
    4. 当样本数达到 shard_size 时保存分片

    Args:
        file_paths: 文件路径列表
        tokenizer: 分词器
        max_seq_length: 最大序列长度
        output_dir: 输出目录
        dataset_name: 数据集名称（train/val）
        shard_size: 每个分片的样本数
        perf_monitor: 性能监控器

    Returns:
        (shard_files, metadata) 元组
    """
    print(f"\n{'='*60}")
    print(f"Processing {dataset_name} dataset: {len(file_paths)} files")
    print(f"{'='*60}")

    examples = []
    current_chunk = []
    total_tokens = 0
    shard_index = 0
    shard_files = []

    # 用于元数据
    original_files = [os.path.basename(p) for p in file_paths]

    def flush_chunk():
        nonlocal current_chunk
        if not current_chunk:
            return
        # Padding
        if len(current_chunk) < max_seq_length:
            current_chunk = current_chunk + [tokenizer.pad_token_id] * (max_seq_length - len(current_chunk))
        return current_chunk[:max_seq_length]

    def create_sample(tokens):
        return {
            "input_ids": tokens,
            "labels": tokens.copy(),
        }

    def save_shard():
        nonlocal examples, shard_index
        if not examples:
            return None

        shard_filename = f"{dataset_name}_{shard_index:03d}.pt"
        shard_path = os.path.join(output_dir, shard_filename)

        shard_metadata = {
            "max_seq_length": max_seq_length,
            "vocab_size": tokenizer.vocab_size,
            "num_examples": len(examples),
            "shard_index": shard_index,
            "total_tokens": total_tokens,
            "original_files": original_files,
        }

        save_preprocessed_data(
            examples=examples,
            output_path=shard_path,
            metadata=shard_metadata,
        )

        shard_files.append(shard_filename)
        print(f"  Saved shard {shard_index}: {len(examples)} samples")
        shard_index += 1
        examples = []

    # 流式处理
    print("Tokenizing and creating samples...")
    for text in stream_text_files(file_paths):
        if perf_monitor:
            with perf_monitor.measure("tokenization"):
                tokens = tokenizer.encode(text, add_special_tokens=False)
        else:
            tokens = tokenizer.encode(text, add_special_tokens=False)

        total_tokens += len(tokens)
        current_chunk.extend(tokens)

        # 当chunk足够大时，创建样本
        while len(current_chunk) >= max_seq_length:
            sample = current_chunk[:max_seq_length]
            examples.append(create_sample(sample))
            # 滑动窗口，有一半重叠
            current_chunk = current_chunk[max_seq_length // 2:]

            # 当样本数达到分片大小时保存
            if len(examples) >= shard_size:
                if perf_monitor:
                    with perf_monitor.measure("shard_save"):
                        save_shard()
                else:
                    save_shard()

    # 处理剩余数据
    if len(current_chunk) > 0:
        sample = flush_chunk()
        if sample:
            examples.append(create_sample(sample))

    # 保存最后的分片
    if examples:
        if perf_monitor:
            with perf_monitor.measure("shard_save"):
                save_shard()
        else:
            save_shard()

    # 总元数据
    metadata = {
        "max_seq_length": max_seq_length,
        "vocab_size": tokenizer.vocab_size,
        "num_examples": sum(len(torch.load(os.path.join(output_dir, f), weights_only=False).get("examples", [])) for f in shard_files) if shard_files else len(examples),
        "num_shards": len(shard_files),
        "total_tokens": total_tokens,
        "original_files": original_files,
    }

    print(f"\n{dataset_name} processed: {metadata['num_examples']} samples, {metadata['num_shards']} shards")

    return shard_files, metadata


def check_incremental(
    output_dir: str,
    file_hashes: Dict[str, str],
) -> bool:
    """
    检查是否可以增量处理

    Returns:
        True 如果所有文件都未变化，可以跳过处理
    """
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")

    if not os.path.exists(dataset_info_path):
        return False

    try:
        with open(dataset_info_path, "r", encoding="utf-8") as f:
            old_info = json.load(f)

        # 检查文件哈希
        old_hashes = old_info.get("file_hashes", {})
        for path, hash_val in file_hashes.items():
            if old_hashes.get(path) != hash_val:
                print(f"  File changed: {path}")
                return False

        # 检查配置是否变化
        # （如果有需要比较的参数，在这里添加）

        print("  All files unchanged, can skip processing")
        return True

    except (json.JSONDecodeError, KeyError):
        return False


def preprocess_dataset_incremental(
    config: PreprocessConfig,
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> bool:
    """
    增量预处理主流程

    Returns:
        True 如果执行了处理，False 如果跳过了处理
    """
    # 计算所有文件的哈希
    all_files = config.train_files.copy()
    if config.validation_file:
        all_files.append(config.validation_file)

    with perf_monitor.measure("file_hash") if perf_monitor else NullContextManager():
        file_hashes = compute_files_hash(all_files)

    print("\nFile hashes:")
    for path, hash_val in file_hashes.items():
        print(f"  {os.path.basename(path)}: {hash_val}")

    # 检查是否可以增量处理
    if not config.force_reprocess and check_incremental(config.output_dir, file_hashes):
        print("\n[Incremental] Skipping preprocessing - no files changed")
        return False

    # ========== 步骤1: 采样并训练tokenizer ==========
    print(f"\n{'='*60}")
    print("Step 1: Training Tokenizer")
    print(f"{'='*60}")

    # 如果tokenizer已存在且未变化，可以跳过
    tokenizer_path = os.path.join(config.output_dir, "tokenizer")
    tokenizer_exists = os.path.exists(tokenizer_path)

    # 检查tokenizer是否需要重新训练
    skip_tokenizer = False
    if tokenizer_exists and not config.force_reprocess:
        try:
            # 尝试加载tokenizer验证
            test_tokenizer = HuggingFaceBPETokenizer.load(tokenizer_path)
            if test_tokenizer.vocab_size == config.vocab_size:
                print(f"  Tokenizer exists and matches vocab_size, skipping training")
                skip_tokenizer = True
                tokenizer = test_tokenizer
        except Exception:
            pass

    if not skip_tokenizer:
        with perf_monitor.measure("tokenizer_train") if perf_monitor else NullContextManager():
            print(f"\n  Sampling texts for tokenizer training ({config.tokenizer_sample_bytes / 1024 / 1024:.0f}MB)...")
            sample_texts = sample_text_for_tokenizer(
                config.train_files,
                target_bytes=config.tokenizer_sample_bytes,
            )

            print(f"  Training tokenizer (vocab_size={config.vocab_size})...")
            tokenizer = HuggingFaceBPETokenizer()
            tokenizer.train(sample_texts, vocab_size=config.vocab_size, min_frequency=config.min_frequency)
            print(f"  Tokenizer trained. Vocab size: {tokenizer.vocab_size}")

        # 保存tokenizer
        with perf_monitor.measure("tokenizer_save") if perf_monitor else NullContextManager():
            print(f"\n  Saving tokenizer to {tokenizer_path}")
            tokenizer.save(tokenizer_path)
    else:
        print(f"  Skipping tokenizer training (already exists)")

    # ========== 步骤2: 流式编码训练数据 ==========
    print(f"\n{'='*60}")
    print("Step 2: Encoding Training Data")
    print(f"{'='*60}")

    train_shard_files, train_metadata = stream_encode_and_save(
        file_paths=config.train_files,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        output_dir=config.output_dir,
        dataset_name="train",
        shard_size=config.shard_size,
        perf_monitor=perf_monitor,
    )

    # ========== 步骤3: 流式编码验证数据 ==========
    val_shard_files = None
    val_metadata = None
    if config.validation_file:
        print(f"\n{'='*60}")
        print("Step 3: Encoding Validation Data")
        print(f"{'='*60}")

        val_shard_files, val_metadata = stream_encode_and_save(
            file_paths=[config.validation_file],
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            output_dir=config.output_dir,
            dataset_name="val",
            shard_size=config.shard_size,
            perf_monitor=perf_monitor,
        )

    # ========== 步骤4: 保存数据集信息 ==========
    print(f"\n{'='*60}")
    print("Step 4: Saving Dataset Info")
    print(f"{'='*60}")

    dataset_info = {
        "version": "2.0",
        "file_hashes": file_hashes,
        "train_files": config.train_files,
        "validation_file": config.validation_file,
        "max_seq_length": config.max_seq_length,
        "vocab_size": config.vocab_size,
        "min_frequency": config.min_frequency,
        "shard_size": config.shard_size,
        "train_metadata": train_metadata,
        "validation_metadata": val_metadata,
        "files": {
            "train": train_shard_files,
            "validation": val_shard_files,
            "tokenizer": "tokenizer",
        },
    }

    dataset_info_path = os.path.join(config.output_dir, "dataset_info.json")
    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    print(f"  Dataset info saved to {dataset_info_path}")

    return True


# ============ Null Context Manager ============

class NullContextManager:
    """啥也不做的上下文管理器"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __call__(self, func):
        return func


# ============ Main ============

def main():
    args = parse_args()

    # 构建配置
    if args.train_dir:
        train_files = scan_text_files(args.train_dir)
        if not train_files:
            print(f"Error: No .txt files found in {args.train_dir}")
            return
        print(f"Found {len(train_files)} training files in {args.train_dir}")
    else:
        train_files = args.train_files

    config = PreprocessConfig(
        train_files=train_files,
        validation_file=args.validation_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        shard_size=args.shard_size,
        tokenizer_sample_bytes=args.tokenizer_sample_bytes,
        force_reprocess=args.force_reprocess,
    )

    # 创建性能监控器
    perf_monitor = PreprocessPerformanceMonitor(enabled=True)
    total_start_time = time.perf_counter()

    print("=" * 60)
    print("Data Preprocessing Script v2.0")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Tokenizer sample: {config.tokenizer_sample_bytes / 1024 / 1024:.0f}MB")
    print(f"Force reprocess: {config.force_reprocess}")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 执行预处理
    processed = preprocess_dataset_incremental(config, perf_monitor)

    # 记录总时间
    total_time = time.perf_counter() - total_start_time
    perf_monitor.timings["total"].append(total_time)

    print("\n" + "=" * 60)
    if processed:
        print("Preprocessing completed!")
    else:
        print("Preprocessing skipped (no changes detected)")
    print("=" * 60)

    # 打印性能统计
    perf_monitor.print_summary()

    # 打印输出结构
    print(f"\nOutput structure:")
    print(f"  {config.output_dir}/")
    print(f"  ├── train_000.pt")
    print(f"  ├── train_001.pt")
    print(f"  ├── ... (shards)")
    if config.validation_file:
        print(f"  ├── val_000.pt")
    print(f"  ├── tokenizer/")
    print(f"  │   └── tokenizer.json")
    print(f"  └── dataset_info.json")

    if processed:
        print(f"\nTo train with preprocessed data:")
        print(f"  python scripts/pretrain.py --preprocessed_data {config.output_dir} ...")


if __name__ == "__main__":
    main()
