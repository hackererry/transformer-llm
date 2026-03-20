"""
并行化模块
支持多进程数据加载和并行计算
"""
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, Sampler
from typing import Optional, List, Dict, Any, Callable, Iterator
import os
import math
from collections import defaultdict


class ParallelDataLoader:
    """
    并行数据加载器
    使用多进程加速数据加载
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        num_workers: int = 4,
        shuffle: bool = True,
        collate_fn: Optional[Callable] = None,
        drop_last: bool = False,
        pin_memory: bool = False,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        """
        Args:
            dataset: 数据集
            batch_size: 批次大小
            num_workers: 工作进程数
            shuffle: 是否打乱
            collate_fn: 数据整理函数
            drop_last: 是否丢弃最后不完整批次
            pin_memory: 是否固定内存
            prefetch_factor: 预取因子
            persistent_workers: 是否保持工作进程存活
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

        # 创建数据加载器
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers if num_workers > 0 else False,
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class ChunkedBatchSampler(Sampler):
    """
    分块批次采样器
    将数据分成块，每个块内的样本长度相近，提高padding效率
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        drop_last: bool = False,
        length_fn: Optional[Callable] = None,
        sort_key: Optional[Callable] = None,
    ):
        """
        Args:
            data_source: 数据源
            batch_size: 批次大小
            drop_last: 是否丢弃最后不完整批次
            length_fn: 获取样本长度的函数
            sort_key: 排序键函数
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.length_fn = length_fn or (lambda x: len(x["input_ids"]))
        self.sort_key = sort_key

    def __iter__(self) -> Iterator[List[int]]:
        # 获取所有样本的长度
        lengths = [(i, self.length_fn(self.data_source[i])) for i in range(len(self.data_source))]

        # 按长度排序
        lengths.sort(key=lambda x: x[1])

        # 分批
        batches = []
        for i in range(0, len(lengths), self.batch_size):
            batch = [lengths[j][0] for j in range(i, min(i + self.batch_size, len(lengths)))]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)

        # 打乱批次顺序
        import random
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        return (len(self.data_source) + self.batch_size - 1) // self.batch_size


class BucketBatchSampler(Sampler):
    """
    桶批次采样器
    将样本按长度分桶，每个批次从同一个桶中采样
    """

    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        num_buckets: int = 10,
        drop_last: bool = False,
        length_fn: Optional[Callable] = None,
    ):
        """
        Args:
            data_source: 数据源
            batch_size: 批次大小
            num_buckets: 桶数量
            drop_last: 是否丢弃最后不完整批次
            length_fn: 获取样本长度的函数
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.length_fn = length_fn or (lambda x: len(x["input_ids"]))

        # 构建桶
        self._build_buckets()

    def _build_buckets(self):
        """构建桶"""
        # 获取所有样本长度
        lengths = [self.length_fn(self.data_source[i]) for i in range(len(self.data_source))]

        # 确定桶边界
        min_len, max_len = min(lengths), max(lengths)
        bucket_size = (max_len - min_len) / self.num_buckets

        # 分配样本到桶
        self.buckets = defaultdict(list)
        for i, length in enumerate(lengths):
            bucket_idx = min(int((length - min_len) / bucket_size), self.num_buckets - 1)
            self.buckets[bucket_idx].append(i)

    def __iter__(self) -> Iterator[List[int]]:
        import random

        # 打乱每个桶内的样本
        for bucket_idx in self.buckets:
            random.shuffle(self.buckets[bucket_idx])

        batches = []
        for bucket_idx in range(self.num_buckets):
            bucket = self.buckets[bucket_idx]
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        # 打乱批次顺序
        random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self) -> int:
        total = 0
        for bucket in self.buckets.values():
            if self.drop_last:
                total += len(bucket) // self.batch_size
            else:
                total += (len(bucket) + self.batch_size - 1) // self.batch_size
        return total


class ParallelProcessor:
    """
    并行处理器
    使用多进程并行处理数据
    """

    def __init__(
        self,
        num_workers: int = None,
        process_fn: Optional[Callable] = None,
    ):
        """
        Args:
            num_workers: 工作进程数，默认为CPU核心数
            process_fn: 处理函数
        """
        self.num_workers = num_workers or os.cpu_count()
        self.process_fn = process_fn

    def process(self, items: List[Any]) -> List[Any]:
        """
        并行处理项目列表

        Args:
            items: 要处理的项目列表

        Returns:
            处理后的项目列表
        """
        if self.process_fn is None:
            raise ValueError("process_fn must be provided")

        if self.num_workers <= 1:
            return [self.process_fn(item) for item in items]

        with mp.Pool(self.num_workers) as pool:
            results = pool.map(self.process_fn, items)

        return results

    def process_in_chunks(
        self,
        items: List[Any],
        chunk_size: int = 100,
    ) -> List[Any]:
        """
        分块并行处理

        Args:
            items: 要处理的项目列表
            chunk_size: 每块大小

        Returns:
            处理后的项目列表
        """
        if self.process_fn is None:
            raise ValueError("process_fn must be provided")

        if self.num_workers <= 1:
            return [self.process_fn(item) for item in items]

        # 分块
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]

        def process_chunk(chunk):
            return [self.process_fn(item) for item in chunk]

        with mp.Pool(self.num_workers) as pool:
            results = pool.map(process_chunk, chunks)

        # 展平结果
        return [item for chunk in results for item in chunk]


def get_optimal_num_workers(
    dataset_size: int,
    batch_size: int,
    available_memory_mb: float = None,
) -> int:
    """
    计算最优的工作进程数

    Args:
        dataset_size: 数据集大小
        batch_size: 批次大小
        available_memory_mb: 可用内存(MB)

    Returns:
        推荐的工作进程数
    """
    cpu_count = os.cpu_count() or 4

    # 基于CPU核心数的推荐
    recommended = min(cpu_count, 8)

    # 如果数据集很小，减少工作进程
    if dataset_size < 1000:
        recommended = min(recommended, 2)
    elif dataset_size < 10000:
        recommended = min(recommended, 4)

    # 考虑内存限制
    if available_memory_mb:
        # 假设每个工作进程需要约200MB内存
        max_by_memory = int(available_memory_mb / 200)
        recommended = min(recommended, max_by_memory)

    return max(1, recommended)


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
    drop_last: bool = False,
    use_bucket_sampling: bool = False,
    **kwargs,
) -> DataLoader:
    """
    创建优化的数据加载器

    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱
        num_workers: 工作进程数
        collate_fn: 数据整理函数
        drop_last: 是否丢弃最后不完整批次
        use_bucket_sampling: 是否使用桶采样
        **kwargs: 额外参数

    Returns:
        数据加载器
    """
    if num_workers is None:
        num_workers = get_optimal_num_workers(len(dataset), batch_size)

    if use_bucket_sampling:
        sampler = BucketBatchSampler(
            dataset,
            batch_size=batch_size,
            drop_last=drop_last,
        )
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=False,
            **kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=drop_last,
        pin_memory=False,
        **kwargs,
    )
