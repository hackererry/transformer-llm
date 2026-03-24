"""
数据处理模块测试
测试Tokenizer、数据集、数据整理器等
"""
import pytest
import torch
import sys
import os
import tempfile
import json
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    HuggingFaceBPETokenizer,
    get_tokenizer,
    PretrainDataset,
    PretrainIterableDataset,
    FinetuneDataset,
    TextFileDataset,
    MemoryMappedDataset,
    create_dataset,
    PreprocessedDataset,
    ShardedPreprocessedDataset,
    save_preprocessed_data,
    create_sharded_dataset,
    DataCollatorForLanguageModeling,
    DataCollatorForCausalLM,
    DataCollatorForSFT,
    DynamicBatchSampler,
    get_collator,
)
from src.data.tokenizer import BPETokenizer


class TestHuggingFaceBPETokenizer:
    """HuggingFace BPE Tokenizer测试"""

    def test_tokenizer_creation(self):
        """测试创建tokenizer"""
        tokenizer = HuggingFaceBPETokenizer()
        assert tokenizer is not None

    def test_tokenizer_training(self):
        """测试训练tokenizer"""
        tokenizer = HuggingFaceBPETokenizer()
        texts = [
            "hello world",
            "hello there",
            "world peace",
            "artificial intelligence",
            "machine learning",
        ]
        tokenizer.train(texts, vocab_size=100, min_frequency=1)
        assert tokenizer.vocab_size > 0

    def test_encode_decode(self):
        """测试编码和解码"""
        tokenizer = HuggingFaceBPETokenizer()
        texts = ["hello world", "test sentence"]
        tokenizer.train(texts, vocab_size=50, min_frequency=1)

        text = "hello world"
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(ids, list)

        decoded = tokenizer.decode(ids)
        assert isinstance(decoded, str)

    def test_special_tokens(self):
        """测试特殊token"""
        tokenizer = HuggingFaceBPETokenizer()
        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.eos_token == "</s>"

    def test_save_and_load(self):
        """测试保存和加载"""
        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["hello world test"], vocab_size=50, min_frequency=1)
        original_vocab_size = tokenizer.vocab_size

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(tmpdir)
            loaded = HuggingFaceBPETokenizer.load(tmpdir)

            assert loaded.vocab_size == original_vocab_size

    def test_encode_with_special_tokens(self):
        """测试添加特殊token的编码"""
        tokenizer = HuggingFaceBPETokenizer()
        texts = ["hello world"]
        tokenizer.train(texts, vocab_size=50, min_frequency=1)

        ids_with_special = tokenizer.encode("hello world", add_special_tokens=True)
        ids_without_special = tokenizer.encode("hello world", add_special_tokens=False)
        assert len(ids_with_special) >= len(ids_without_special)


class TestGetTokenizer:
    """Tokenizer工厂函数测试"""

    def test_get_empty_tokenizer(self):
        """测试获取空tokenizer"""
        tokenizer = get_tokenizer(tokenizer_type="bpe")
        assert tokenizer is not None

    def test_get_tokenizer_from_path(self):
        """测试从路径加载tokenizer"""
        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["test data"], vocab_size=50, min_frequency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(tmpdir)
            loaded = get_tokenizer(tmpdir, tokenizer_type="bpe")
            assert loaded.vocab_size == tokenizer.vocab_size


class TestDataCollatorForCausalLM:
    """因果LM数据整理器测试"""

    def test_collate(self):
        """测试整理功能"""
        collator = DataCollatorForCausalLM(pad_token_id=0, max_length=32)

        batch = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [1, 2, 3]},
        ]

        result = collator(batch)

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2
        assert result["input_ids"].shape[1] == 5  # 最长序列长度

    def test_padding(self):
        """测试padding"""
        collator = DataCollatorForCausalLM(pad_token_id=0)

        batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [1, 2, 3, 4, 5]},
        ]

        result = collator(batch)

        # 第一个样本应该被padding
        assert result["input_ids"][0, 3] == 0
        assert result["input_ids"][0, 4] == 0

    def test_labels_padding(self):
        """测试labels padding"""
        collator = DataCollatorForCausalLM(pad_token_id=0)

        batch = [
            {"input_ids": [1, 2, 3]},
        ]

        result = collator(batch)
        # labels应该用-100填充padding位置
        assert result["labels"][0, -1] == -100 or result["labels"][0, -1] in [1, 2, 3]

    def test_different_max_length(self):
        """测试不同的max_length"""
        collator = DataCollatorForCausalLM(pad_token_id=0, max_length=10)

        batch = [
            {"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
            {"input_ids": [1, 2, 3]},
        ]

        result = collator(batch)
        assert result["input_ids"].shape[1] == 10  # 截断到max_length

    def test_attention_mask(self):
        """测试attention_mask"""
        collator = DataCollatorForCausalLM(pad_token_id=0)

        batch = [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [1, 2, 3, 4, 5]},
        ]

        result = collator(batch)
        # 第一个样本的attention_mask后两位应该是0
        assert result["attention_mask"][0, 3] == 0
        assert result["attention_mask"][0, 4] == 0


class TestDataCollatorForSFT:
    """SFT数据整理器测试"""

    def test_collate(self):
        """测试整理功能"""
        collator = DataCollatorForSFT(pad_token_id=0)

        batch = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [-100, -100, 3, 4, 5],
            },
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, -100, 3],
            },
        ]

        result = collator(batch)

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result
        assert result["input_ids"].shape[0] == 2
        assert result["labels"].shape[0] == 2

    def test_sft_labels_with_ignore_index(self):
        """测试SFT labels使用ignore_index"""
        collator = DataCollatorForSFT(pad_token_id=0, ignore_index=-100)

        batch = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [1, 2, 3, 4, 5],  # 没有-100
            },
        ]

        result = collator(batch)
        # padding位置应该是ignore_index
        assert result["labels"][0, -1] == -100


class TestDataCollatorForLanguageModeling:
    """语言模型数据整理器测试"""

    def test_collate(self):
        """测试整理功能"""
        collator = DataCollatorForLanguageModeling(pad_token_id=0)

        batch = [
            {"input_ids": [1, 2, 3, 4, 5]},
            {"input_ids": [1, 2, 3]},
        ]

        result = collator(batch)

        assert "input_ids" in result
        assert "labels" in result
        assert "attention_mask" in result


class TestDynamicBatchSampler:
    """动态批采样器测试"""

    def test_sampler_creation(self):
        """测试采样器创建"""
        class SimpleDataset:
            def __len__(self):
                return 100
            def __getitem__(self, idx):
                return {"input_ids": [1] * 50}

        dataset = SimpleDataset()
        sampler = DynamicBatchSampler(dataset, max_tokens=200, max_batch_size=32)
        assert sampler.max_tokens == 200
        assert sampler.max_batch_size == 32

    def test_batching(self):
        """测试批处理"""
        class SimpleDataset:
            def __init__(self, lengths):
                self.lengths = lengths
            def __len__(self):
                return len(self.lengths)
            def __getitem__(self, idx):
                return {"input_ids": [1] * self.lengths[idx]}

        dataset = SimpleDataset([10, 20, 30, 40, 50])
        sampler = DynamicBatchSampler(dataset, max_tokens=100, max_batch_size=10)

        batches = list(sampler)
        # 验证批次数量
        assert len(batches) > 0


class TestGetCollator:
    """获取整理器工厂函数测试"""

    def test_get_causal_lm_collator(self):
        """测试获取因果LM整理器"""
        collator = get_collator("causal_lm", pad_token_id=0)
        assert isinstance(collator, DataCollatorForCausalLM)

    def test_get_sft_collator(self):
        """测试获取SFT整理器"""
        collator = get_collator("sft", pad_token_id=0)
        assert isinstance(collator, DataCollatorForSFT)

    def test_get_language_modeling_collator(self):
        """测试获取语言建模整理器"""
        collator = get_collator("language_modeling", pad_token_id=0)
        assert isinstance(collator, DataCollatorForLanguageModeling)

    def test_invalid_collator(self):
        """测试无效的整理器类型"""
        with pytest.raises(ValueError):
            get_collator("invalid_type")


class TestPreprocessedDataset:
    """预处理数据集测试"""

    def test_save_and_load_preprocessed_data(self):
        """测试保存和加载预处理数据"""
        examples = [
            {"input_ids": list(range(10)), "labels": list(range(10))},
            {"input_ids": list(range(20)), "labels": list(range(20))},
        ]
        metadata = {
            "max_seq_length": 20,
            "vocab_size": 32000,
            "num_examples": 2,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test.pt")
            save_preprocessed_data(examples, output_path, metadata)

            assert os.path.exists(output_path)

            dataset = PreprocessedDataset(output_path)
            assert len(dataset) == 2

            item = dataset[0]
            assert "input_ids" in item
            assert "labels" in item


class TestShardedPreprocessedDataset:
    """分片预处理数据集测试"""

    def test_create_sharded_dataset(self):
        """测试创建分片数据集"""
        examples = [
            {"input_ids": list(range(10)), "labels": list(range(10))},
            {"input_ids": list(range(20)), "labels": list(range(20))},
        ]
        metadata = {
            "max_seq_length": 20,
            "vocab_size": 32000,
            "num_examples": 2,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            # 保存多个分片
            for i in range(3):
                shard_examples = [
                    {"input_ids": list(range(10)), "labels": list(range(10))}
                    for _ in range(5)
                ]
                save_preprocessed_data(
                    shard_examples,
                    os.path.join(tmpdir, f"train_{i:03d}.pt"),
                    metadata,
                )

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                preload=False,
            )
            assert dataset.num_shards == 3

    def test_sharded_dataset_iteration(self):
        """测试分片数据集迭代"""
        examples = [
            {"input_ids": list(range(10)), "labels": list(range(10))},
        ]
        metadata = {
            "max_seq_length": 10,
            "vocab_size": 32000,
            "num_examples": 1,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessed_data(
                examples,
                os.path.join(tmpdir, "train_000.pt"),
                metadata,
            )

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                preload=False,
            )

            count = 0
            for item in dataset:
                count += 1
                assert "input_ids" in item
                assert "labels" in item

            assert count == 1


class TestCreateShardedDataset:
    """创建分片数据集工厂函数测试"""

    def test_create_function(self):
        """测试工厂函数"""
        examples = [{"input_ids": [1, 2], "labels": [1, 2]}]
        metadata = {"max_seq_length": 2, "vocab_size": 100, "num_examples": 1}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessed_data(
                examples,
                os.path.join(tmpdir, "train_000.pt"),
                metadata,
            )

            dataset = create_sharded_dataset(
                data_dir=tmpdir,
                prefix="train",
            )
            assert len(dataset) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
