#!/usr/bin/env python
"""
数据预处理脚本 v3.0
支持多文件、增量处理、流式编码、采样训练tokenizer

v3.0 新特性:
- 增量添加新数据：只处理新文件，追加分片
- 修改文件检测：删除旧分片，重新处理该文件
- 删除文件处理：保留分片，标记元数据为 orphaned
- Tokenizer 一致性：支持冻结/扩展/重训三种模式
- 版本迁移：自动迁移 v2.0 格式的 dataset_info.json

Usage:
    # 方式1: 目录输入（自动扫描所有txt文件）
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data \
        --max_seq_length 512 \
        --vocab_size 32000

    # 方式2: 增量添加新数据（默认模式）
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data

    # 方式3: 查看变化但不处理（dry-run）
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data \
        --dry-run

    # 方式4: 完全重新处理
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data \
        --full

    # 方式5: Tokenizer 扩展模式
    python scripts/preprocess_data.py \
        --train_dir dataset/data \
        --output_dir ./preprocessed_data \
        --tokenizer-mode extend
"""
import os
import sys
import argparse
import hashlib
import json
import time
import random
import shutil
from typing import List, Dict, Any, Optional, Iterator, Tuple, Set
from contextlib import contextmanager
from dataclasses import dataclass, field

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import HuggingFaceBPETokenizer, save_preprocessed_data
import torch


# ============ 数据结构类 ============

@dataclass
class FileChange:
    """文件变化记录"""
    file_path: str
    change_type: str  # "new", "modified", "deleted", "unchanged"
    old_hash: Optional[str] = None
    new_hash: Optional[str] = None


@dataclass
class TokenizerPolicy:
    """Tokenizer 策略配置"""
    mode: str = "frozen"  # "frozen" | "extend" | "retrain"
    extend_threshold: int = 100  # 触发扩展的新词汇阈值


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
    incremental: bool = True  # 启用增量处理
    dry_run: bool = False  # 仅检测变化，不处理
    tokenizer_mode: str = "frozen"  # frozen | extend | retrain


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
            "shard_create": [],  # 分片生成耗时（包含tokenization和样本创建）
            "shard_save": [],    # 分片保存耗时
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


# ============ Null Context Manager ============

class NullContextManager:
    """啥也不做的上下文管理器"""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def __call__(self, func):
        return func


# ============ 核心函数 ============

def parse_args():
    parser = argparse.ArgumentParser(description="数据预处理脚本 v3.0")

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

    # 增量处理参数
    incremental_group = parser.add_argument_group("增量处理参数")
    incremental_group.add_argument("--incremental", action="store_true", default=True,
                                   help="启用增量处理模式（默认）")
    incremental_group.add_argument("--full", action="store_true",
                                   help="完全重新处理所有数据")
    incremental_group.add_argument("--dry-run", action="store_true",
                                   help="仅检测变化，不实际处理")
    incremental_group.add_argument("--tokenizer-mode", choices=["frozen", "extend", "retrain"],
                                   default="frozen",
                                   help="Tokenizer 处理模式: frozen(冻结), extend(扩展), retrain(重训)")

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

    # 兼容旧参数
    parser.add_argument("--force_reprocess", action="store_true",
                       help="强制重新处理（等同于 --full）")

    args = parser.parse_args()

    # 验证输入
    if args.train_dir is None and args.train_files is None:
        parser.error("必须指定 --train_dir 或 --train_files")

    if args.train_dir is not None and args.train_files is not None:
        parser.error("不能同时指定 --train_dir 和 --train_files")

    # --full 覆盖 --incremental
    if args.full:
        args.incremental = False

    # --force_reprocess 等同于 --full
    if args.force_reprocess:
        args.incremental = False

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


# ============ dataset_info.json v3.0 结构 ============

def create_dataset_info_v3(
    config: PreprocessConfig,
    tokenizer_vocab_size: int,
) -> Dict[str, Any]:
    """创建 v3.0 格式的 dataset_info.json"""
    return {
        "version": "3.0",
        "config": {
            "max_seq_length": config.max_seq_length,
            "vocab_size": config.vocab_size,
            "min_frequency": config.min_frequency,
            "shard_size": config.shard_size,
        },
        "files": {},  # 文件级别的元数据
        "shards": {},  # 分片级别的元数据
        "summary": {
            "total_files": 0,
            "total_shards": 0,
            "total_examples": 0,
            "next_shard_index": 0,
        },
    }


def load_dataset_info(output_dir: str) -> Optional[Dict[str, Any]]:
    """加载 dataset_info.json"""
    info_path = os.path.join(output_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        return None

    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_dataset_info(output_dir: str, info: Dict[str, Any]) -> None:
    """保存 dataset_info.json"""
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"  Dataset info saved to {info_path}")


# ============ 版本迁移 ============

def migrate_dataset_info_v2_to_v3(old_info: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    将 v2.0 格式的 dataset_info.json 迁移到 v3.0

    v2.0 结构:
    {
        "version": "2.0",
        "file_hashes": {"file1.txt": "hash1"},
        "train_files": [...],
        "train_metadata": {...},
        ...
    }

    v3.0 结构:
    {
        "version": "3.0",
        "config": {...},
        "files": {
            "file1.txt": {
                "hash": "hash1",
                "status": "legacy",
                "shards": ["train_000.pt"],
                "num_examples": 5000
            }
        },
        "shards": {
            "train_000.pt": {
                "index": 0,
                "source_files": ["file1.txt"],
                "num_examples": 5000
            }
        },
        "summary": {...}
    }
    """
    print("  Migrating dataset_info.json from v2.0 to v3.0...")

    new_info = {
        "version": "3.0",
        "config": {
            "max_seq_length": old_info.get("max_seq_length", 512),
            "vocab_size": old_info.get("vocab_size", 32000),
            "min_frequency": old_info.get("min_frequency", 2),
            "shard_size": old_info.get("shard_size", 10000),
        },
        "files": {},
        "shards": {},
        "summary": {
            "total_files": 0,
            "total_shards": 0,
            "total_examples": 0,
            "next_shard_index": 0,
        },
    }

    # 迁移文件信息
    file_hashes = old_info.get("file_hashes", {})
    train_files = old_info.get("train_files", [])
    train_metadata = old_info.get("train_metadata", {})
    train_shard_files = old_info.get("files", {}).get("train", [])

    # 构建 files 信息
    for file_path, file_hash in file_hashes.items():
        file_name = os.path.basename(file_path)
        new_info["files"][file_name] = {
            "hash": file_hash,
            "status": "legacy",  # 标记为遗留数据
            "shards": train_shard_files.copy(),
            "num_examples": train_metadata.get("num_examples", 0) // max(len(file_hashes), 1),
            "full_path": file_path,
        }

    # 构建 shards 信息
    total_examples = train_metadata.get("num_examples", 0)
    for idx, shard_file in enumerate(train_shard_files):
        # 从现有分片读取实际样本数
        shard_path = os.path.join(output_dir, shard_file)
        try:
            data = torch.load(shard_path, map_location="cpu", weights_only=False)
            num_examples = len(data.get("examples", []))
        except:
            num_examples = train_metadata.get("num_examples", 0) // max(len(train_shard_files), 1)

        new_info["shards"][shard_file] = {
            "index": idx,
            "source_files": list(new_info["files"].keys()),  # 所有文件
            "num_examples": num_examples,
        }
        new_info["summary"]["total_examples"] += num_examples

    new_info["summary"]["total_files"] = len(new_info["files"])
    new_info["summary"]["total_shards"] = len(train_shard_files)
    new_info["summary"]["next_shard_index"] = len(train_shard_files)

    print(f"  Migration complete: {new_info['summary']['total_files']} files, "
          f"{new_info['summary']['total_shards']} shards")

    return new_info


# ============ 文件变化检测 ============

def detect_file_changes(
    current_files: List[str],
    dataset_info: Dict[str, Any],
) -> List[FileChange]:
    """
    检测文件变化

    Args:
        current_files: 当前文件列表（完整路径）
        dataset_info: v3.0 格式的 dataset_info.json

    Returns:
        FileChange 列表
    """
    changes = []

    # 获取已记录的文件信息
    recorded_files = dataset_info.get("files", {})

    # 标准化路径用于比较
    current_basenames = {os.path.basename(f): f for f in current_files}
    recorded_basenames = set(recorded_files.keys())

    # 检测新增和修改的文件
    for file_path in current_files:
        file_name = os.path.basename(file_path)
        new_hash = compute_file_hash(file_path)

        if file_name not in recorded_basenames:
            changes.append(FileChange(
                file_path=file_path,
                change_type="new",
                old_hash=None,
                new_hash=new_hash,
            ))
        else:
            old_hash = recorded_files[file_name].get("hash")
            if old_hash != new_hash:
                changes.append(FileChange(
                    file_path=file_path,
                    change_type="modified",
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))
            else:
                changes.append(FileChange(
                    file_path=file_path,
                    change_type="unchanged",
                    old_hash=old_hash,
                    new_hash=new_hash,
                ))

    # 检测删除的文件
    for file_name in recorded_basenames:
        if file_name not in current_basenames:
            old_info = recorded_files[file_name]
            changes.append(FileChange(
                file_path=old_info.get("full_path", file_name),
                change_type="deleted",
                old_hash=old_info.get("hash"),
                new_hash=None,
            ))

    return changes


def print_file_changes(changes: List[FileChange]) -> None:
    """打印文件变化摘要"""
    new_files = [c for c in changes if c.change_type == "new"]
    modified_files = [c for c in changes if c.change_type == "modified"]
    deleted_files = [c for c in changes if c.change_type == "deleted"]
    unchanged_files = [c for c in changes if c.change_type == "unchanged"]

    print("\n" + "=" * 60)
    print("File Change Detection Summary")
    print("=" * 60)

    if new_files:
        print(f"\nNew files ({len(new_files)}):")
        for c in new_files:
            print(f"  + {os.path.basename(c.file_path)}")

    if modified_files:
        print(f"\nModified files ({len(modified_files)}):")
        for c in modified_files:
            print(f"  * {os.path.basename(c.file_path)}")

    if deleted_files:
        print(f"\nDeleted files ({len(deleted_files)}):")
        for c in deleted_files:
            print(f"  - {os.path.basename(c.file_path)}")

    if unchanged_files:
        print(f"\nUnchanged files: {len(unchanged_files)}")

    print("=" * 60)


# ============ Tokenizer 策略 ============

def ensure_tokenizer_consistency(
    config: PreprocessConfig,
    output_dir: str,
    changes: List[FileChange],
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> Tuple[Any, bool]:
    """
    确保 Tokenizer 一致性

    Args:
        config: 预处理配置
        output_dir: 输出目录
        changes: 文件变化列表
        perf_monitor: 性能监控器

    Returns:
        (tokenizer, needs_full_reprocess) 元组
        - tokenizer: 加载或训练好的 tokenizer
        - needs_full_reprocess: 是否需要全量重处理
    """
    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer_exists = os.path.exists(tokenizer_path)

    # frozen 模式：使用现有 tokenizer
    if config.tokenizer_mode == "frozen":
        if tokenizer_exists:
            print(f"  Loading existing tokenizer (frozen mode)")
            tokenizer = HuggingFaceBPETokenizer.load(tokenizer_path)
            return tokenizer, False
        else:
            # 没有 tokenizer，需要训练
            print(f"  No existing tokenizer, training new one")
            tokenizer = _train_tokenizer(config, perf_monitor)
            tokenizer.save(tokenizer_path)
            return tokenizer, True

    # retrain 模式：重新训练 tokenizer
    elif config.tokenizer_mode == "retrain":
        print(f"  Retraining tokenizer (retrain mode)")
        tokenizer = _train_tokenizer(config, perf_monitor)
        tokenizer.save(tokenizer_path)
        return tokenizer, True  # 需要全量重处理

    # extend 模式：检测新词汇，必要时扩展
    elif config.tokenizer_mode == "extend":
        if not tokenizer_exists:
            print(f"  No existing tokenizer, training new one")
            tokenizer = _train_tokenizer(config, perf_monitor)
            tokenizer.save(tokenizer_path)
            return tokenizer, True

        # 加载现有 tokenizer
        tokenizer = HuggingFaceBPETokenizer.load(tokenizer_path)

        # 检测新文件中的 OOV（词表外词汇）
        new_files = [c.file_path for c in changes if c.change_type == "new"]
        if new_files:
            oov_count = _detect_oov(tokenizer, new_files, config.tokenizer_sample_bytes)
            print(f"  Detected {oov_count} potential new tokens in new files")

            if oov_count > config.extend_threshold:
                print(f"  OOV count ({oov_count}) exceeds threshold ({config.extend_threshold}), "
                      f"retraining tokenizer")
                tokenizer = _train_tokenizer(config, perf_monitor)
                tokenizer.save(tokenizer_path)
                return tokenizer, True  # 需要全量重处理

        return tokenizer, False

    return None, False


def _train_tokenizer(
    config: PreprocessConfig,
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> Any:
    """训练 tokenizer"""
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

    return tokenizer


def _detect_oov(tokenizer: Any, file_paths: List[str], sample_bytes: int) -> int:
    """检测词表外词汇数量"""
    # 采样文本
    sample_texts = sample_text_for_tokenizer(file_paths, target_bytes=min(sample_bytes, 10 * 1024 * 1024))

    # 统计 OOV
    vocab = set(tokenizer.vocab.keys())
    oov_tokens = set()

    for text in sample_texts[:1000]:  # 限制处理量
        # 简单的字符级分词来检测 OOV
        for char in text:
            if char not in vocab and char.strip():
                oov_tokens.add(char)

    return len(oov_tokens)


# ============ 增量处理核心函数 ============

def incremental_add_new_files(
    new_files: List[str],
    tokenizer: Any,
    config: PreprocessConfig,
    dataset_info: Dict[str, Any],
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> Dict[str, Any]:
    """
    处理新增文件，追加分片

    Args:
        new_files: 新增文件列表
        tokenizer: 分词器
        config: 预处理配置
        dataset_info: 当前 dataset_info
        perf_monitor: 性能监控器

    Returns:
        更新后的 dataset_info
    """
    if not new_files:
        return dataset_info

    print(f"\n{'='*60}")
    print(f"Incremental: Processing {len(new_files)} new files")
    print(f"{'='*60}")

    # 获取下一个分片索引
    next_shard_index = dataset_info["summary"]["next_shard_index"]

    # 编码新文件
    shard_files, metadata, file_metadata = stream_encode_and_save_incremental(
        file_paths=new_files,
        tokenizer=tokenizer,
        max_seq_length=config.max_seq_length,
        output_dir=config.output_dir,
        dataset_name="train",
        shard_size=config.shard_size,
        start_shard_index=next_shard_index,
        perf_monitor=perf_monitor,
    )

    # 更新 dataset_info
    for file_path in new_files:
        file_name = os.path.basename(file_path)
        file_hash = compute_file_hash(file_path)
        dataset_info["files"][file_name] = {
            "hash": file_hash,
            "status": "processed",
            "shards": shard_files.copy(),
            "num_examples": file_metadata.get(file_name, {}).get("num_examples", 0),
            "full_path": file_path,
        }

    # 更新分片信息
    for shard_file in shard_files:
        dataset_info["shards"][shard_file] = {
            "index": next_shard_index,
            "source_files": [os.path.basename(f) for f in new_files],
            "num_examples": metadata.get("shard_examples", {}).get(shard_file, 0),
        }
        next_shard_index += 1

    # 更新摘要
    dataset_info["summary"]["total_files"] = len(dataset_info["files"])
    dataset_info["summary"]["total_shards"] = len(dataset_info["shards"])
    dataset_info["summary"]["total_examples"] += metadata.get("num_examples", 0)
    dataset_info["summary"]["next_shard_index"] = next_shard_index

    return dataset_info


def handle_modified_files(
    modified_files: List[FileChange],
    tokenizer: Any,
    config: PreprocessConfig,
    dataset_info: Dict[str, Any],
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> Dict[str, Any]:
    """
    处理修改的文件：删除旧分片，重新处理

    Args:
        modified_files: 修改的文件列表
        tokenizer: 分词器
        config: 预处理配置
        dataset_info: 当前 dataset_info
        perf_monitor: 性能监控器

    Returns:
        更新后的 dataset_info
    """
    if not modified_files:
        return dataset_info

    print(f"\n{'='*60}")
    print(f"Incremental: Processing {len(modified_files)} modified files")
    print(f"{'='*60}")

    for change in modified_files:
        file_name = os.path.basename(change.file_path)
        print(f"\n  Processing modified file: {file_name}")

        # 获取旧分片
        old_file_info = dataset_info["files"].get(file_name, {})
        old_shards = old_file_info.get("shards", [])

        # 删除旧分片
        for shard_file in old_shards:
            shard_path = os.path.join(config.output_dir, shard_file)
            if os.path.exists(shard_path):
                os.remove(shard_path)
                print(f"    Removed old shard: {shard_file}")

            # 从 shards 记录中删除
            if shard_file in dataset_info["shards"]:
                del dataset_info["shards"][shard_file]

        # 重新处理文件
        next_shard_index = dataset_info["summary"]["next_shard_index"]
        shard_files, metadata, file_metadata = stream_encode_and_save_incremental(
            file_paths=[change.file_path],
            tokenizer=tokenizer,
            max_seq_length=config.max_seq_length,
            output_dir=config.output_dir,
            dataset_name="train",
            shard_size=config.shard_size,
            start_shard_index=next_shard_index,
            perf_monitor=perf_monitor,
        )

        # 更新文件信息
        dataset_info["files"][file_name] = {
            "hash": change.new_hash,
            "status": "processed",
            "shards": shard_files,
            "num_examples": file_metadata.get(file_name, {}).get("num_examples", 0),
            "full_path": change.file_path,
        }

        # 更新分片信息
        for shard_file in shard_files:
            dataset_info["shards"][shard_file] = {
                "index": next_shard_index,
                "source_files": [file_name],
                "num_examples": metadata.get("shard_examples", {}).get(shard_file, 0),
            }
            next_shard_index += 1

        dataset_info["summary"]["next_shard_index"] = next_shard_index

    # 重新计算摘要
    dataset_info["summary"]["total_shards"] = len(dataset_info["shards"])
    dataset_info["summary"]["total_examples"] = sum(
        s.get("num_examples", 0) for s in dataset_info["shards"].values()
    )

    return dataset_info


def handle_deleted_files(
    deleted_files: List[FileChange],
    dataset_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    处理删除的文件：保留分片，标记为 orphaned

    Args:
        deleted_files: 删除的文件列表
        dataset_info: 当前 dataset_info

    Returns:
        更新后的 dataset_info
    """
    if not deleted_files:
        return dataset_info

    print(f"\n{'='*60}")
    print(f"Incremental: Marking {len(deleted_files)} deleted files as orphaned")
    print(f"{'='*60}")

    for change in deleted_files:
        file_name = os.path.basename(change.file_path)
        print(f"  Marking as orphaned: {file_name}")

        if file_name in dataset_info["files"]:
            # 更新状态为 orphaned
            dataset_info["files"][file_name]["status"] = "orphaned"
            dataset_info["files"][file_name]["orphaned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # 更新摘要（total_files 只计算 processed 状态的文件）
    dataset_info["summary"]["total_files"] = sum(
        1 for f in dataset_info["files"].values()
        if f.get("status") == "processed"
    )

    return dataset_info


# ============ 流式编码和保存 ============

def sample_text_for_tokenizer(
    file_paths: List[str],
    target_bytes: int = 100 * 1024 * 1024,
) -> List[str]:
    """
    从多个文件中采样文本用于训练tokenizer
    采用分层采样，确保各文件都有代表性样本
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

        ratio = min(target_bytes / total_size, 1.0)
        sample_size = int(size * ratio * len(file_paths))

        if size <= sample_size * 2:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                texts = [line.strip() for line in f if line.strip()]
            sampled_texts.extend(texts)
            bytes_collected += size
        else:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]

            if len(lines) > 0:
                num_sample = max(1, int(len(lines) * ratio * len(file_paths)))
                num_sample = min(num_sample, len(lines))

                random.seed(42)
                sampled_indices = random.sample(range(len(lines)), num_sample)
                for idx in sampled_indices:
                    sampled_texts.append(lines[idx])
                    bytes_collected += len(lines[idx].encode("utf-8"))

                    if bytes_collected >= target_bytes:
                        break

    print(f"[Tokenizer Sampling] 实际采样: {len(sampled_texts)} 行, ~{bytes_collected / 1024 / 1024:.1f}MB")

    random.shuffle(sampled_texts)
    return sampled_texts


def stream_text_files(file_paths: List[str]) -> Iterator[str]:
    """流式读取多个文本文件"""
    for path in file_paths:
        print(f"  Reading: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def stream_raw_files(file_paths: List[str]) -> Iterator[str]:
    """
    流式读取原始文本文件（不清洗，仅 yield 非空行）

    重要：本脚本假设输入已经是清洗后的文本。
    如需清洗，请先运行 scripts/clean_data.py。

    Args:
        file_paths: 文件路径列表

    Yields:
        每行文本（非空）
    """
    for path in file_paths:
        print(f"  Reading: {path}")
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def stream_encode_and_save_incremental(
    file_paths: List[str],
    tokenizer,
    max_seq_length: int,
    output_dir: str,
    dataset_name: str,
    shard_size: int = 10000,
    start_shard_index: int = 0,
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> Tuple[List[str], Dict[str, Any], Dict[str, Any]]:
    """
    流式编码并保存数据（增量版本）

    Returns:
        (shard_files, metadata, file_metadata) 元组
        - shard_files: 保存的分片文件列表
        - metadata: 总体元数据
        - file_metadata: 每个文件的元数据
    """
    print(f"\nProcessing {len(file_paths)} files...")

    examples = []
    current_chunk = []
    total_tokens = 0
    shard_index = start_shard_index
    shard_files = []

    # 每个文件的统计
    file_metadata = {}
    current_file_examples = 0
    current_file_name = None

    # 分片样本数统计
    shard_examples = {}

    def flush_chunk():
        nonlocal current_chunk
        if not current_chunk:
            return
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
            "source_files": [os.path.basename(f) for f in file_paths],
        }

        save_preprocessed_data(
            examples=examples,
            output_path=shard_path,
            metadata=shard_metadata,
        )

        shard_files.append(shard_filename)
        shard_examples[shard_filename] = len(examples)
        print(f"    Saved shard {shard_index}: {len(examples)} samples")
        shard_index += 1
        examples = []

    # 流式处理（带数据质量增强）
    print("Tokenizing and creating samples...")
    shard_start_time = time.perf_counter() if perf_monitor else None

    # 使用原始文本流（假设已清洗）
    text_iterator = stream_raw_files(file_paths)

    for text in text_iterator:
        tokens = tokenizer.encode(text, add_special_tokens=False)

        total_tokens += len(tokens)
        current_chunk.extend(tokens)

        while len(current_chunk) >= max_seq_length:
            sample = current_chunk[:max_seq_length]
            examples.append(create_sample(sample))
            current_file_examples += 1
            current_chunk = current_chunk[max_seq_length // 2:]

            if len(examples) >= shard_size:
                # 记录分片生成耗时（tokenization + 样本创建）
                if perf_monitor and shard_start_time:
                    elapsed = time.perf_counter() - shard_start_time
                    perf_monitor.timings["shard_create"].append(elapsed)
                    perf_monitor._call_counts["shard_create"] = perf_monitor._call_counts.get("shard_create", 0) + 1

                # 保存分片
                if perf_monitor:
                    with perf_monitor.measure("shard_save"):
                        save_shard()
                else:
                    save_shard()

                # 重置计时器
                shard_start_time = time.perf_counter() if perf_monitor else None

    # 处理剩余数据
    if len(current_chunk) > 0:
        sample = flush_chunk()
        if sample:
            examples.append(create_sample(sample))
            current_file_examples += 1

    # 保存最后的分片
    if examples:
        # 记录最后一个分片生成耗时
        if perf_monitor and shard_start_time:
            elapsed = time.perf_counter() - shard_start_time
            perf_monitor.timings["shard_create"].append(elapsed)
            perf_monitor._call_counts["shard_create"] = perf_monitor._call_counts.get("shard_create", 0) + 1

        if perf_monitor:
            with perf_monitor.measure("shard_save"):
                save_shard()
        else:
            save_shard()

    # 计算每个文件的样本数（简化：按文件大小比例分配）
    total_examples = sum(shard_examples.values())
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        total_size = sum(os.path.getsize(f) for f in file_paths)
        file_examples = int(total_examples * file_size / total_size) if total_size > 0 else 0
        file_metadata[file_name] = {"num_examples": file_examples}

    metadata = {
        "num_examples": total_examples,
        "num_shards": len(shard_files),
        "total_tokens": total_tokens,
        "shard_examples": shard_examples,
    }

    print(f"  Processed: {metadata['num_examples']} samples, {metadata['num_shards']} shards")

    return shard_files, metadata, file_metadata


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
    流式编码并保存数据（兼容旧版本）
    """
    shard_files, metadata, _ = stream_encode_and_save_incremental(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        output_dir=output_dir,
        dataset_name=dataset_name,
        shard_size=shard_size,
        start_shard_index=0,
        perf_monitor=perf_monitor,
    )
    return shard_files, metadata


# ============ 全量处理 ============

def full_preprocess(
    config: PreprocessConfig,
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> bool:
    """
    全量预处理（兼容旧版本行为）
    """
    print(f"\n{'='*60}")
    print("Full Preprocessing Mode")
    print(f"{'='*60}")

    # 清理旧数据
    if os.path.exists(config.output_dir):
        print(f"  Cleaning output directory: {config.output_dir}")
        for f in os.listdir(config.output_dir):
            if f.endswith(".pt") or f == "dataset_info.json":
                os.remove(os.path.join(config.output_dir, f))

    os.makedirs(config.output_dir, exist_ok=True)

    # 训练 tokenizer
    print(f"\n{'='*60}")
    print("Step 1: Training Tokenizer")
    print(f"{'='*60}")

    tokenizer_path = os.path.join(config.output_dir, "tokenizer")
    tokenizer = _train_tokenizer(config, perf_monitor)
    tokenizer.save(tokenizer_path)

    # 编码训练数据
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

    # 编码验证数据
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

    # 创建 v3.0 格式的 dataset_info
    dataset_info = create_dataset_info_v3(config, tokenizer.vocab_size)

    # 添加文件信息
    for file_path in config.train_files:
        file_name = os.path.basename(file_path)
        dataset_info["files"][file_name] = {
            "hash": compute_file_hash(file_path),
            "status": "processed",
            "shards": train_shard_files.copy(),
            "num_examples": train_metadata.get("num_examples", 0) // len(config.train_files),
            "full_path": file_path,
        }

    # 添加分片信息
    for idx, shard_file in enumerate(train_shard_files):
        dataset_info["shards"][shard_file] = {
            "index": idx,
            "source_files": [os.path.basename(f) for f in config.train_files],
            "num_examples": train_metadata.get("shard_examples", {}).get(shard_file, 0),
        }

    # 更新摘要
    dataset_info["summary"]["total_files"] = len(config.train_files)
    dataset_info["summary"]["total_shards"] = len(train_shard_files)
    dataset_info["summary"]["total_examples"] = train_metadata.get("num_examples", 0)
    dataset_info["summary"]["next_shard_index"] = len(train_shard_files)

    save_dataset_info(config.output_dir, dataset_info)

    return True


# ============ 增量处理主流程 ============

def incremental_preprocess(
    config: PreprocessConfig,
    perf_monitor: Optional[PreprocessPerformanceMonitor] = None,
) -> bool:
    """
    增量预处理主流程
    """
    print(f"\n{'='*60}")
    print("Incremental Preprocessing Mode")
    print(f"{'='*60}")

    os.makedirs(config.output_dir, exist_ok=True)

    # 加载或创建 dataset_info
    dataset_info = load_dataset_info(config.output_dir)

    if dataset_info is None:
        print("  No existing dataset_info.json, creating new one")
        dataset_info = create_dataset_info_v3(config, config.vocab_size)
        # 首次运行，执行全量处理
        return full_preprocess(config, perf_monitor)

    # 版本迁移
    version = dataset_info.get("version", "1.0")
    if version == "2.0":
        dataset_info = migrate_dataset_info_v2_to_v3(dataset_info, config.output_dir)
        save_dataset_info(config.output_dir, dataset_info)
    elif version != "3.0":
        print(f"  Unsupported version {version}, performing full reprocess")
        return full_preprocess(config, perf_monitor)

    # 检测文件变化
    print("\nDetecting file changes...")
    changes = detect_file_changes(config.train_files, dataset_info)
    print_file_changes(changes)

    # 统计变化
    new_files = [c for c in changes if c.change_type == "new"]
    modified_files = [c for c in changes if c.change_type == "modified"]
    deleted_files = [c for c in changes if c.change_type == "deleted"]
    unchanged_files = [c for c in changes if c.change_type == "unchanged"]

    # dry-run 模式：只显示变化，不处理
    if config.dry_run:
        print("\n[Dry-run] No changes made. Run without --dry-run to apply changes.")
        return False

    # 如果没有变化，跳过处理
    if not new_files and not modified_files and not deleted_files:
        print("\n[Incremental] No changes detected, skipping preprocessing")
        return False

    # Tokenizer 一致性检查
    print(f"\n{'='*60}")
    print("Tokenizer Consistency Check")
    print(f"{'='*60}")
    print(f"  Mode: {config.tokenizer_mode}")

    tokenizer, needs_full_reprocess = ensure_tokenizer_consistency(
        config, config.output_dir, changes, perf_monitor
    )

    if needs_full_reprocess:
        print("  Tokenizer changed, performing full reprocess")
        return full_preprocess(config, perf_monitor)

    # 增量处理
    # 1. 处理删除的文件
    dataset_info = handle_deleted_files(deleted_files, dataset_info)

    # 2. 处理修改的文件
    dataset_info = handle_modified_files(
        [c for c in changes if c.change_type == "modified"],
        tokenizer, config, dataset_info, perf_monitor
    )

    # 3. 处理新增的文件
    dataset_info = incremental_add_new_files(
        [c.file_path for c in changes if c.change_type == "new"],
        tokenizer, config, dataset_info, perf_monitor
    )

    # 保存更新后的 dataset_info
    save_dataset_info(config.output_dir, dataset_info)

    return True


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
        incremental=args.incremental,
        dry_run=args.dry_run,
        tokenizer_mode=args.tokenizer_mode,
    )

    # 创建性能监控器
    perf_monitor = PreprocessPerformanceMonitor(enabled=True)
    total_start_time = time.perf_counter()

    print("=" * 60)
    print("Data Preprocessing Script v3.0")
    print("=" * 60)
    print(f"Output directory: {config.output_dir}")
    print(f"Max sequence length: {config.max_seq_length}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Tokenizer sample: {config.tokenizer_sample_bytes / 1024 / 1024:.0f}MB")
    print(f"Mode: {'Full' if not config.incremental else 'Incremental'}")
    print(f"Tokenizer mode: {config.tokenizer_mode}")
    print("Note: Input is assumed to be pre-cleaned.")
    print("      Run 'python scripts/clean_data.py' first if needed.")
    if config.dry_run:
        print(f"Dry-run: True (no changes will be made)")

    # 执行预处理
    if config.incremental:
        processed = incremental_preprocess(config, perf_monitor)
    else:
        processed = full_preprocess(config, perf_monitor)

    # 记录总时间
    total_time = time.perf_counter() - total_start_time
    perf_monitor.timings["total"].append(total_time)

    print("\n" + "=" * 60)
    if processed:
        print("Preprocessing completed!")
    else:
        print("Preprocessing skipped (no changes detected or dry-run)")
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
