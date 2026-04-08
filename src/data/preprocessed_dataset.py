"""
预处理好缓存的数据集
直接从 .pt 缓存加载，无需 tokenizer
支持分片加载以避免内存溢出
"""
import os
import gc
import json
import random
import threading
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Dict, Any, Optional, List, Iterator
import math


class PreprocessedDataset(Dataset):
    """
    预处理好缓存的数据集

    缓存格式：
    {
        "version": "1.0",
        "metadata": {
            "max_seq_length": 512,
            "vocab_size": 32000,
            "num_examples": 10000,
            "original_file": "train.txt",
            "data_hash": "abc123def456",
        },
        "examples": [
            {"input_ids": [...], "labels": [...]},
            ...
        ]
    }
    """

    VERSION = "1.0"

    def __init__(
        self,
        data_path: str,
        validate_metadata: bool = True,
        metadata_check: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            data_path: 预处理缓存文件路径 (.pt)
            validate_metadata: 是否验证元数据兼容性
            metadata_check: 需要验证的元数据字典 (如 {"max_seq_length": 512, "vocab_size": 32000})
        """
        self.data_path = data_path
        self.validate_metadata = validate_metadata
        self.metadata_check = metadata_check

        # 加载缓存
        self.data = self._load_cache()
        self.metadata = self.data["metadata"]
        self.examples = self.data["examples"]

        # 验证元数据
        if validate_metadata and metadata_check:
            self._validate_metadata()

    def _load_cache(self) -> Dict[str, Any]:
        """加载缓存文件"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Preprocessed data file not found: {self.data_path}")

        print(f"Loading preprocessed data from {self.data_path}")
        data = torch.load(self.data_path, map_location="cpu", weights_only=False)

        # 检查版本
        version = data.get("version", "unknown")
        if version != self.VERSION:
            print(f"Warning: Cache version {version} differs from expected {self.VERSION}")

        return data

    def _validate_metadata(self) -> None:
        """验证元数据兼容性"""
        if self.metadata_check is None:
            return

        for key, expected_value in self.metadata_check.items():
            if key not in self.metadata:
                raise ValueError(f"Metadata key '{key}' not found in cache. Available keys: {list(self.metadata.keys())}")

            actual_value = self.metadata[key]
            if actual_value != expected_value:
                raise ValueError(
                    f"Metadata mismatch for '{key}': "
                    f"expected {expected_value}, got {actual_value}. "
                    f"Please re-preprocess data with correct parameters."
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        result = {"input_ids": item["input_ids"], "labels": item["labels"]}
        if "attention_mask" in item:
            result["attention_mask"] = item["attention_mask"]
        return result

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据集元数据"""
        return self.metadata.copy()


def save_preprocessed_data(
    examples: list,
    output_path: str,
    metadata: Dict[str, Any],
    version: str = "1.0",
    source_files: Optional[List[str]] = None,
) -> None:
    """
    保存预处理好的数据为缓存格式

    Args:
        examples: 数据样本列表，每个样本包含 input_ids 和 labels
        output_path: 输出文件路径
        metadata: 元数据字典
        version: 缓存版本
        source_files: 源文件列表（v3.0 增量处理支持）
    """
    # 如果 metadata 中没有 source_files 且提供了参数，则添加
    if source_files and "source_files" not in metadata:
        metadata = metadata.copy()
        metadata["source_files"] = source_files

    data = {
        "version": version,
        "metadata": metadata,
        "examples": examples,
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(data, output_path)
    print(f"Saved preprocessed data to {output_path}")
    print(f"  - Examples: {len(examples)}")
    print(f"  - Metadata: {metadata}")


class ShardedPreprocessedDataset(IterableDataset):
    """
    分片预处理好缓存的数据集
    支持动态加载分片，避免内存溢出

    使用方式：
        dataset = ShardedPreprocessedDataset(
            data_dir="./preprocessed_data",
            prefix="train",
            shuffle=True,
            preload=False,  # 默认惰性加载
        )

        # DataLoader 会自动按需加载分片
        dataloader = DataLoader(dataset, batch_size=8)
    """

    def __init__(
        self,
        data_dir: str,
        prefix: str = "train",
        num_shards: int = None,
        metadata_check: Optional[Dict[str, Any]] = None,
        shuffle: bool = False,
        seed: int = 42,
        preload: bool = False,
    ):
        """
        Args:
            data_dir: 预处理数据目录
            prefix: 数据文件前缀 (如 "train", "val")
            num_shards: 分片总数（如果为None，从目录扫描）
            metadata_check: 需要验证的元数据
            shuffle: 是否启用两级 shuffle（分片间 + 分片内）
            seed: 随机种子
            preload: 是否在初始化时预加载所有分片到内存
        """
        self.data_dir = data_dir
        self.prefix = prefix
        self.metadata_check = metadata_check
        self.shuffle = shuffle
        self.seed = seed

        # 发现所有分片
        self.shard_files = self._discover_shards()

        if num_shards is not None:
            self.num_shards = num_shards
        else:
            self.num_shards = len(self.shard_files)

        if self.num_shards == 0:
            raise ValueError(f"No shards found in {data_dir} with prefix {prefix}")

        self.preload = preload
        self._all_examples: Optional[List[Dict]] = None
        self._total_examples: Optional[int] = None
        self._epoch: int = 0

        if self.preload:
            self._preload_all_shards()
            print(f"  Mode: PRELOAD (all {self.num_shards} shards loaded)")
        else:
            self._load_manifest()
            print(f"  Mode: LAZY (shard-by-shard, prefetch enabled)")
            print(f"  Total shards: {self.num_shards}")
            if self._total_examples is not None:
                print(f"  Total examples: {self._total_examples} (from manifest)")

    def _load_manifest(self):
        """从 dataset_info.json 加载元数据（不加载实际数据）"""
        info_path = os.path.join(self.data_dir, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            # 提取每个分片的样本数
            shards_info = info.get("shards", {})
            self._shard_example_counts = {}
            for shard_file in self.shard_files:
                shard_info = shards_info.get(shard_file, {})
                self._shard_example_counts[shard_file] = shard_info.get("num_examples", 0)
            # 优先使用当前 prefix 匹配到的分片总数，避免 val 误用全局总数
            shard_total = sum(self._shard_example_counts.values())
            if shard_total > 0:
                self._total_examples = shard_total
            # 否则保持 None，由 __len__ 回退到加载分片计数
        else:
            # 无 manifest，回退到加载分片计数
            self._shard_example_counts = None

    def _discover_shards(self) -> List[str]:
        """发现目录下所有匹配的分片文件"""
        shard_files = []
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if filename.startswith(f"{self.prefix}_") and filename.endswith(".pt"):
                shard_files.append(filename)

        shard_files.sort()
        return shard_files

    def _preload_all_shards(self) -> None:
        """预加载所有分片到内存"""
        print(f"Preloading {self.num_shards} shards into memory...")
        all_examples = []
        for i, shard_file in enumerate(self.shard_files):
            shard_path = os.path.join(self.data_dir, shard_file)
            print(f"  Loading shard {i + 1}/{self.num_shards}: {shard_file}")
            data = torch.load(shard_path, map_location="cpu", weights_only=False)

            # 验证元数据
            if self.metadata_check:
                metadata = data.get("metadata", {})
                for key, expected_value in self.metadata_check.items():
                    if key in metadata and metadata[key] != expected_value:
                        raise ValueError(
                            f"Metadata mismatch in shard {i}: "
                            f"{key}={metadata[key]}, expected {expected_value}"
                        )

            examples = data.get("examples", [])
            all_examples.extend(examples)

        self._all_examples = all_examples
        self._total_examples = len(all_examples)
        print(f"  Total examples loaded: {self._total_examples}")

    def _load_shard(self, shard_index: int) -> List[Dict]:
        """加载指定分片"""
        if shard_index >= len(self.shard_files):
            raise IndexError(f"Shard index {shard_index} out of range (total {len(self.shard_files)})")

        shard_path = os.path.join(self.data_dir, self.shard_files[shard_index])
        data = torch.load(shard_path, map_location="cpu", weights_only=False)

        # 验证元数据
        if self.metadata_check:
            metadata = data.get("metadata", {})
            for key, expected_value in self.metadata_check.items():
                if key in metadata and metadata[key] != expected_value:
                    raise ValueError(
                        f"Metadata mismatch in shard {shard_index}: "
                        f"{key}={metadata[key]}, expected {expected_value}"
                    )

        return data.get("examples", [])

    def _get_shuffled_indices(self) -> List[int]:
        """获取打乱后的样本索引顺序"""
        g = random.Random(self.seed + self._epoch)
        indices = list(range(self._total_examples))
        g.shuffle(indices)
        return indices

    def set_epoch(self, epoch: int):
        """设置当前 epoch（用于每 epoch 不同的 shuffle 顺序）"""
        self._epoch = epoch

    def __iter__(self) -> Iterator:
        """迭代器，每次返回一个样本"""
        if self._all_examples is not None:
            # 预加载模式：从内存读取
            if self.shuffle:
                indices = self._get_shuffled_indices()
                for idx in indices:
                    item = self._all_examples[idx]
                    result = {"input_ids": item["input_ids"], "labels": item["labels"]}
                    if "attention_mask" in item:
                        result["attention_mask"] = item["attention_mask"]
                    yield result
            else:
                for item in self._all_examples:
                    result = {"input_ids": item["input_ids"], "labels": item["labels"]}
                    if "attention_mask" in item:
                        result["attention_mask"] = item["attention_mask"]
                    yield result
        else:
            # 惰性加载模式：分片级 + 分片内两级 shuffle，后台预取
            rng = random.Random(self.seed + self._epoch)
            shard_indices = list(range(self.num_shards))
            if self.shuffle:
                rng.shuffle(shard_indices)  # 第一级：分片间 shuffle

            # 预取缓冲区
            prefetched_examples = None
            prefetch_thread = None
            prefetch_error = None

            def _prefetch(shard_idx):
                nonlocal prefetched_examples, prefetch_error
                try:
                    prefetched_examples = self._load_shard(shard_idx)
                except Exception as e:
                    prefetch_error = e

            for i, shard_idx in enumerate(shard_indices):
                # 获取当前分片数据
                if i == 0:
                    # 第一个分片：同步加载
                    examples = self._load_shard(shard_idx)
                else:
                    # 等待预取完成
                    if prefetch_thread is not None:
                        prefetch_thread.join()
                    if prefetch_error is not None:
                        raise prefetch_error
                    examples = prefetched_examples
                    prefetched_examples = None

                # 启动下一个分片的预取（后台线程）
                prefetch_thread = None
                if i + 1 < len(shard_indices):
                    prefetch_thread = threading.Thread(
                        target=_prefetch,
                        args=(shard_indices[i + 1],),
                        daemon=True,
                    )
                    prefetch_thread.start()

                # 第二级：分片内 shuffle
                if self.shuffle:
                    rng.shuffle(examples)

                # 产出当前分片的样本（直接传 list，避免 tensor→list→tensor 往返）
                for item in examples:
                    result = {"input_ids": item["input_ids"], "labels": item["labels"]}
                    if "attention_mask" in item:
                        result["attention_mask"] = item["attention_mask"]
                    yield result
                del examples

            # 等待最后一个预取线程结束（清理）
            if prefetch_thread is not None:
                prefetch_thread.join()

    def __len__(self) -> int:
        """返回总样本数"""
        if self._total_examples is not None:
            return self._total_examples
        if self._all_examples is not None:
            return len(self._all_examples)
        # 如果没有 manifest 且没有预加载，回退到加载分片统计
        if not hasattr(self, "_cached_len"):
            total = 0
            for shard_file in self.shard_files:
                shard_path = os.path.join(self.data_dir, shard_file)
                data = torch.load(shard_path, map_location="cpu", weights_only=False)
                total += len(data.get("examples", []))
            self._cached_len = total
        return self._cached_len


def create_sharded_dataset(
    data_dir: str,
    prefix: str = "train",
    num_shards: int = None,
    metadata_check: Optional[Dict[str, Any]] = None,
    preload: bool = False,
    **kwargs,
) -> ShardedPreprocessedDataset:
    """
    创建分片数据集的工厂函数
    """
    return ShardedPreprocessedDataset(
        data_dir=data_dir,
        prefix=prefix,
        num_shards=num_shards,
        metadata_check=metadata_check,
        preload=preload,
        **kwargs,
    )
