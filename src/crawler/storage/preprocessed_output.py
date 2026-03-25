# -*- coding: utf-8 -*-
"""
预训练数据输出模块
Preprocessed Output Module for Training Data
"""
import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class PreprocessedDocument:
    """预处理后的文档"""
    id: str
    text: str
    url: Optional[str] = None
    title: Optional[str] = None
    domain: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'text': self.text,
            'url': self.url,
            'title': self.title,
            'domain': self.domain,
            'content_hash': self.content_hash,
            'metadata': self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PreprocessedDocument":
        """从字典创建"""
        return cls(
            id=data['id'],
            text=data['text'],
            url=data.get('url'),
            title=data.get('title'),
            domain=data.get('domain'),
            content_hash=data.get('content_hash'),
            metadata=data.get('metadata'),
        )


class PreprocessedOutput:
    """预训练数据输出器"""

    def __init__(
        self,
        output_dir: str,
        shard_size: int = 10000,
        tokenizer = None,
    ):
        """
        初始化预训练数据输出器

        Args:
            output_dir: 输出目录
            shard_size: 每个分片的文档数量
            tokenizer: 可选的 tokenizer，用于验证和编码
        """
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.tokenizer = tokenizer

        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "tokenizer").mkdir(exist_ok=True)

        self._current_shard: List[PreprocessedDocument] = []
        self._shard_index = 0
        self._total_documents = 0

        self._metadata: Dict[str, Any] = {
            'total_documents': 0,
            'total_tokens': 0,
            'total_bytes': 0,
            'shards': [],
        }

    def add(self, document: PreprocessedDocument):
        """添加文档"""
        self._current_shard.append(document)
        self._total_documents += 1

        if len(self._current_shard) >= self.shard_size:
            self._flush_shard()

    def add_batch(self, documents: List[PreprocessedDocument]):
        """批量添加文档"""
        for doc in documents:
            self.add(doc)

    def _flush_shard(self):
        """刷新当前分片到磁盘"""
        if not self._current_shard:
            return

        shard_file = self.output_dir / f"train_{self._shard_index:05d}.pt"
        shard_json = self.output_dir / f"train_{self._shard_index:05d}.json"

        # 保存为 PyTorch 张量格式
        tensors = []
        for doc in self._current_shard:
            if self.tokenizer:
                encoding = self.tokenizer.encode(
                    doc.text,
                    truncation=True,
                    padding=False,
                )
                tensor = torch.tensor(encoding['input_ids'], dtype=torch.long)
            else:
                tensor = torch.tensor([ord(c) for c in doc.text], dtype=torch.long)

            tensors.append({
                'id': doc.id,
                'input_ids': tensor,
                'text': doc.text,
                'url': doc.url,
                'title': doc.title,
            })

        # 保存张量
        torch.save(tensors, shard_file)

        # 保存元数据
        shard_meta = {
            'index': self._shard_index,
            'document_count': len(self._current_shard),
            'documents': [doc.to_dict() for doc in self._current_shard],
        }
        with open(shard_json, 'w', encoding='utf-8') as f:
            json.dump(shard_meta, f, ensure_ascii=False, indent=2)

        # 更新全局元数据
        self._metadata['total_documents'] += len(self._current_shard)
        self._metadata['shards'].append({
            'index': self._shard_index,
            'file': str(shard_file),
            'document_count': len(self._current_shard),
        })

        # 清空当前分片
        self._current_shard = []
        self._shard_index += 1

    def close(self):
        """关闭并刷新所有数据"""
        self._flush_shard()
        self._save_metadata()

    def _save_metadata(self):
        """保存数据集元信息"""
        meta_file = self.output_dir / "dataset_info.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(self._metadata, f, ensure_ascii=False, indent=2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def total_documents(self) -> int:
        """获取已处理的文档总数"""
        return self._total_documents

    @property
    def shard_count(self) -> int:
        """获取分片数量"""
        return self._shard_index


class PreprocessedDataset:
    """预训练数据集读取器"""

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        tokenizer = None,
    ):
        """
        初始化数据集读取器

        Args:
            data_dir: 数据目录
            split: 数据集划分 ("train", "val")
            tokenizer: 可选的 tokenizer
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.tokenizer = tokenizer

        self._shard_files: List[Path] = []
        self._load_shards()

        self._metadata: Dict[str, Any] = {}
        self._load_metadata()

    def _load_shards(self):
        """加载分片文件列表"""
        pattern = f"{self.split}_*.pt"
        self._shard_files = sorted(self.data_dir.glob(pattern))

    def _load_metadata(self):
        """加载元数据"""
        meta_file = self.data_dir / "dataset_info.json"
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                self._metadata = json.load(f)

    def __len__(self) -> int:
        """获取数据集大小"""
        return self._metadata.get('total_documents', 0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """获取单个样本"""
        shard_index = index // 10000
        position = index % 10000

        if shard_index >= len(self._shard_files):
            raise IndexError(f"Index {index} out of range")

        shard_file = self._shard_files[shard_index]
        tensors = torch.load(shard_file)

        if position >= len(tensors):
            raise IndexError(f"Index {index} out of range in shard")

        item = tensors[position]

        result = {
            'id': item['id'],
            'text': item['text'],
            'url': item.get('url'),
            'title': item.get('title'),
        }

        if self.tokenizer:
            result['input_ids'] = item['input_ids']
        else:
            result['input_ids'] = item['input_ids']

        return result

    def iterate(self, batch_size: int = 8) -> Iterator[List[Dict[str, Any]]]:
        """迭代获取批次数据"""
        batch = []

        for i in range(len(self)):
            batch.append(self[i])

            if len(batch) >= batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


def convert_crawl_results_to_preprocessed(
    crawl_results: List[Dict[str, Any]],
    output_dir: str,
    shard_size: int = 10000,
    dedup: bool = True,
) -> Dict[str, Any]:
    """
    将爬取结果转换为预训练格式

    Args:
        crawl_results: 爬取结果列表
        output_dir: 输出目录
        shard_size: 分片大小
        dedup: 是否去重

    Returns:
        统计信息
    """
    seen_hashes = set()
    stats = {
        'total': 0,
        'duplicates': 0,
        'empty': 0,
        'added': 0,
    }

    with PreprocessedOutput(output_dir, shard_size=shard_size) as output:
        for result in crawl_results:
            stats['total'] += 1

            text = result.get('content', '')
            if not text:
                stats['empty'] += 1
                continue

            content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()

            if dedup and content_hash in seen_hashes:
                stats['duplicates'] += 1
                continue

            seen_hashes.add(content_hash)

            doc = PreprocessedDocument(
                id=content_hash[:16],
                text=text,
                url=result.get('url'),
                title=result.get('title'),
                domain=result.get('domain'),
                content_hash=content_hash,
                metadata={
                    'depth': result.get('depth', 0),
                    'crawled_at': result.get('crawled_at'),
                },
            )

            output.add(doc)
            stats['added'] += 1

    return stats


def load_preprocessed_dataset(
    data_dir: str,
    split: str = "train",
    tokenizer = None,
) -> PreprocessedDataset:
    """
    加载预训练数据集

    Args:
        data_dir: 数据目录
        split: 数据集划分
        tokenizer: 可选的 tokenizer

    Returns:
        PreprocessedDataset 实例
    """
    return PreprocessedDataset(data_dir, split, tokenizer)
