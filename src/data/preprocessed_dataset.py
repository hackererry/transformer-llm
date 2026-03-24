"""
预处理好缓存的数据集
直接从 .pt 缓存加载，无需 tokenizer
支持分片加载以避免内存溢出
"""
import os
import json
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
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }

    def get_metadata(self) -> Dict[str, Any]:
        """获取数据集元数据"""
        return self.metadata.copy()


def save_preprocessed_data(
    examples: list,
    output_path: str,
    metadata: Dict[str, Any],
    version: str = "1.0",
) -> None:
    """
    保存预处理好的数据为缓存格式

    Args:
        examples: 数据样本列表，每个样本包含 input_ids 和 labels
        output_path: 输出文件路径
        metadata: 元数据字典
        version: 缓存版本
    """
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
            num_shards=10,
            metadata_check={"max_seq_length": 512},
        )

        # DataLoader 会自动按需加载分片
        dataloader = DataLoader(dataset, batch_size=8, num_workers=4)
    """

    def __init__(
        self,
        data_dir: str,
        prefix: str = "train",
        num_shards: int = None,
        metadata_check: Optional[Dict[str, Any]] = None,
        shuffle: bool = False,
        seed: int = 42,
        preload: bool = True,
    ):
        """
        Args:
            data_dir: 预处理数据目录
            prefix: 数据文件前缀 (如 "train", "val")
            num_shards: 分片总数（如果为None，从目录扫描）
            metadata_check: 需要验证的元数据
            shuffle: 是否打乱分片顺序
            seed: 随机种子
            preload: 是否在初始化时预加载所有分片到内存（提升训练速度）
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

        # 预加载所有分片到内存（性能优化）
        self.preload = preload
        self._all_examples: Optional[List[Dict]] = None
        self._total_examples: Optional[int] = None

        if self.preload:
            self._preload_all_shards()

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
        print(f"Loading shard {shard_index + 1}/{len(self.shard_files)}: {self.shard_files[shard_index]}")

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
        import random
        g = random.Random(self.seed)
        indices = list(range(self._total_examples))
        g.shuffle(indices)
        return indices

    def __iter__(self) -> Iterator:
        """迭代器，每次返回一个样本"""
        if self._all_examples is not None:
            # 预加载模式：从内存读取
            if self.shuffle:
                indices = self._get_shuffled_indices()
                for idx in indices:
                    item = self._all_examples[idx]
                    yield {
                        "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                        "labels": torch.tensor(item["labels"], dtype=torch.long),
                    }
            else:
                for item in self._all_examples:
                    yield {
                        "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                        "labels": torch.tensor(item["labels"], dtype=torch.long),
                    }
        else:
            # 非预加载模式：动态加载（内存友好但速度慢）
            for shard_idx in range(self.num_shards):
                examples = self._load_shard(shard_idx)
                for item in examples:
                    yield {
                        "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
                        "labels": torch.tensor(item["labels"], dtype=torch.long),
                    }

    def __len__(self) -> int:
        """返回总样本数"""
        if self._total_examples is not None:
            return self._total_examples
        if self._all_examples is not None:
            return len(self._all_examples)
        # 如果没有预加载，需要加载所有分片统计
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
    preload: bool = True,
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
