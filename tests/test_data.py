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
    ShardedFinetuneDataset,
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
# BPETokenizer 已被 HuggingFaceBPETokenizer 替代


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

    def test_manifest_based_len(self):
        """测试 manifest-based __len__ 不需要加载分片"""
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
            # 保存分片
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

            # 创建 manifest
            manifest = {
                "version": "3.0",
                "summary": {},
                "shards": {
                    "train_000.pt": {"num_examples": 5},
                    "train_001.pt": {"num_examples": 5},
                    "train_002.pt": {"num_examples": 5},
                },
            }
            with open(os.path.join(tmpdir, "dataset_info.json"), "w") as f:
                json.dump(manifest, f)

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                preload=False,
            )

            # __len__ 应返回 manifest 中的数量，不加载分片
            assert len(dataset) == 15

    def test_two_level_shuffle(self):
        """测试两级 shuffle（分片间 + 分片内）"""
        examples = [{"input_ids": [i], "labels": [i]} for i in range(10)]
        metadata = {
            "max_seq_length": 1,
            "vocab_size": 32000,
            "num_examples": 10,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                save_preprocessed_data(
                    examples[i * 5:(i + 1) * 5],
                    os.path.join(tmpdir, f"train_{i:03d}.pt"),
                    metadata,
                )

            # shuffle=True
            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                shuffle=True,
                preload=False,
            )

            # 第一次迭代
            items1 = list(dataset)
            labels1 = [item["labels"][0] for item in items1]

            # 第二次迭代（同一 epoch，顺序相同）
            dataset2 = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                shuffle=True,
                preload=False,
            )
            items2 = list(dataset2)
            labels2 = [item["labels"][0] for item in items2]
            assert labels1 == labels2

            # 不同 epoch 应有不同顺序
            dataset3 = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                shuffle=True,
                preload=False,
                seed=42,
            )
            dataset3.set_epoch(0)
            items3 = list(dataset3)
            labels3 = [item["labels"][0] for item in items3]

            dataset4 = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                shuffle=True,
                preload=False,
                seed=42,
            )
            dataset4.set_epoch(1)
            items4 = list(dataset4)
            labels4 = [item["labels"][0] for item in items4]

            # 不同 epoch 顺序应不同（概率上几乎必然）
            assert labels3 != labels4

    def test_set_epoch_changes_order(self):
        """测试 set_epoch 改变顺序"""
        examples = [{"input_ids": [i], "labels": [i]} for i in range(6)]
        metadata = {"max_seq_length": 1, "vocab_size": 32000, "num_examples": 6}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessed_data(
                examples,
                os.path.join(tmpdir, "train_000.pt"),
                metadata,
            )

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                shuffle=True,
                preload=False,
                seed=123,
            )

            dataset.set_epoch(0)
            order0 = [item["labels"][0] for item in dataset]

            dataset.set_epoch(1)
            order1 = [item["labels"][0] for item in dataset]

            # 不同 epoch 应有不同顺序
            assert order0 != order1

    def test_lazy_mode_with_manifest(self):
        """测试惰性模式 + manifest"""
        examples = [{"input_ids": [i], "labels": [i]} for i in range(3)]
        metadata = {"max_seq_length": 1, "vocab_size": 32000, "num_examples": 3}

        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(2):
                save_preprocessed_data(
                    examples,
                    os.path.join(tmpdir, f"train_{i:03d}.pt"),
                    metadata,
                )

            manifest = {
                "version": "3.0",
                "summary": {},
                "shards": {
                    "train_000.pt": {"num_examples": 3},
                    "train_001.pt": {"num_examples": 3},
                },
            }
            with open(os.path.join(tmpdir, "dataset_info.json"), "w") as f:
                json.dump(manifest, f)

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                preload=False,
                shuffle=False,
            )

            assert len(dataset) == 6
            count = sum(1 for _ in dataset)
            assert count == 6


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


class TestShardedPreprocessedDatasetWithAttentionMask:
    """分片预处理数据集 attention_mask 支持测试"""

    def test_iter_with_attention_mask(self):
        """测试迭代时返回 attention_mask"""
        examples = [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, -100, 3],
                "attention_mask": [1, 1, 1],
            },
        ]
        metadata = {"max_seq_length": 3, "vocab_size": 100, "num_examples": 1}

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

            items = list(dataset)
            assert len(items) == 1
            assert "attention_mask" in items[0]
            assert items[0]["attention_mask"] == [1, 1, 1]

    def test_iter_without_attention_mask(self):
        """测试没有 attention_mask 时不返回该字段"""
        examples = [
            {"input_ids": [1, 2, 3], "labels": [1, 2, 3]},
        ]
        metadata = {"max_seq_length": 3, "vocab_size": 100, "num_examples": 1}

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

            items = list(dataset)
            assert len(items) == 1
            assert "attention_mask" not in items[0]
            assert "input_ids" in items[0]
            assert "labels" in items[0]

    def test_preload_with_attention_mask(self):
        """测试预加载模式返回 attention_mask"""
        examples = [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, 2, 3],
                "attention_mask": [1, 1, 1],
            },
            {
                "input_ids": [4, 5],
                "labels": [-100, 5],
                "attention_mask": [1, 1],
            },
        ]
        metadata = {"max_seq_length": 3, "vocab_size": 100, "num_examples": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessed_data(
                examples,
                os.path.join(tmpdir, "train_000.pt"),
                metadata,
            )

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="train",
                preload=True,
            )

            items = list(dataset)
            assert len(items) == 2
            for item in items:
                assert "attention_mask" in item

    def test_sft_preprocessed_data_with_collator(self):
        """测试 SFT 预处理数据与 DataCollatorForSFT 配合"""
        examples = [
            {
                "input_ids": [1, 2, 3, 4, 5],
                "labels": [-100, -100, 3, 4, 5],
                "attention_mask": [1, 1, 1, 1, 1],
            },
            {
                "input_ids": [6, 7, 8],
                "labels": [-100, -100, 8],
                "attention_mask": [1, 1, 1],
            },
        ]
        metadata = {"max_seq_length": 5, "vocab_size": 100, "num_examples": 2}

        with tempfile.TemporaryDirectory() as tmpdir:
            save_preprocessed_data(
                examples,
                os.path.join(tmpdir, "sft_train_000.pt"),
                metadata,
            )

            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="sft_train",
                preload=False,
            )

            items = list(dataset)
            assert len(items) == 2

            # 使用 SFT collator
            collator = DataCollatorForSFT(pad_token_id=0)
            batch = collator(items)

            assert batch["input_ids"].shape == (2, 5)
            assert batch["labels"].shape == (2, 5)
            assert batch["attention_mask"].shape == (2, 5)
            # 第二个样本 padding 位置的 labels 应为 -100
            assert batch["labels"][1, 3].item() == -100
            assert batch["labels"][1, 4].item() == -100
            # 第二个样本 padding 位置的 attention_mask 应为 0
            assert batch["attention_mask"][1, 3].item() == 0
            assert batch["attention_mask"][1, 4].item() == 0


class TestSFTPreprocessScript:
    """SFT 数据预处理脚本测试"""

    def test_format_prompt_alpaca(self):
        """测试 alpaca 模板格式化"""
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.preprocess_sft_data import format_prompt

        item = {"instruction": "What is 2+2?", "input": "", "output": "4"}
        prompt, full_text = format_prompt(item, "alpaca")
        assert "### Instruction:" in prompt
        assert "### Response:" in prompt
        assert full_text == prompt + "4"

    def test_format_prompt_alpaca_with_input(self):
        """测试 alpaca 模板带 input 字段"""
        from scripts.preprocess_sft_data import format_prompt

        item = {"instruction": "Translate", "input": "hello", "output": "你好"}
        prompt, full_text = format_prompt(item, "alpaca")
        assert "### Input:" in prompt
        assert "hello" in prompt

    def test_format_prompt_simple(self):
        """测试 simple 模板"""
        from scripts.preprocess_sft_data import format_prompt

        item = {"instruction": "Hi", "input": "", "output": "Hello"}
        prompt, full_text = format_prompt(item, "simple")
        assert "Instruction:" in prompt
        assert "Output:" in prompt

    def test_load_sft_data_jsonl(self):
        """测试加载 JSONL 数据"""
        from scripts.preprocess_sft_data import load_sft_data

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "test.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"instruction": "a", "output": "b"}) + "\n")
                f.write(json.dumps({"instruction": "c", "output": "d"}) + "\n")

            data = load_sft_data(jsonl_path)
            assert len(data) == 2

    def test_load_sft_data_json(self):
        """测试加载 JSON 数据"""
        from scripts.preprocess_sft_data import load_sft_data

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "test.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump([{"instruction": "a", "output": "b"}], f)

            data = load_sft_data(json_path)
            assert len(data) == 1

    def test_process_sft_data(self):
        """测试 SFT 数据处理"""
        from scripts.preprocess_sft_data import process_sft_data

        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["hello world test instruction output"], vocab_size=50, min_frequency=1)

        data = [
            {"instruction": "What is 2+2?", "input": "", "output": "4"},
            {"instruction": "Hello", "input": "world", "output": "test"},
        ]

        examples = process_sft_data(data, tokenizer, max_seq_length=512, template="alpaca")

        assert len(examples) == 2
        for ex in examples:
            assert "input_ids" in ex
            assert "labels" in ex
            assert "attention_mask" in ex
            assert len(ex["input_ids"]) == len(ex["labels"])
            assert len(ex["input_ids"]) == len(ex["attention_mask"])
            # prompt 部分应有 -100
            assert -100 in ex["labels"]

    def test_end_to_end_preprocess(self):
        """测试端到端预处理流程"""
        from scripts.preprocess_sft_data import (
            load_sft_data, process_sft_data, save_shards
        )

        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["hello world test instruction output"], vocab_size=50, min_frequency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试数据
            jsonl_path = os.path.join(tmpdir, "train.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i in range(5):
                    f.write(json.dumps({
                        "instruction": f"Test instruction {i}",
                        "input": f"Input {i}",
                        "output": f"Output {i}",
                    }) + "\n")

            # 处理
            data = load_sft_data(jsonl_path)
            examples = process_sft_data(data, tokenizer, max_seq_length=512, template="alpaca")

            metadata = {
                "max_seq_length": 512,
                "vocab_size": tokenizer.vocab_size,
                "template": "alpaca",
            }

            shard_files = save_shards(examples, tmpdir, "sft_train", shard_size=3, metadata=metadata)

            assert len(shard_files) == 2  # 5 examples / 3 per shard = 2 shards

            # 验证可以通过 ShardedPreprocessedDataset 加载
            dataset = ShardedPreprocessedDataset(
                data_dir=tmpdir,
                prefix="sft_train",
                preload=False,
            )

            items = list(dataset)
            assert len(items) == 5
            for item in items:
                assert "input_ids" in item
                assert "labels" in item
                assert "attention_mask" in item


class TestShardedFinetuneDataset:
    """ShardedFinetuneDataset 测试"""

    def test_jsonl_lazy_loading(self):
        """测试 JSONL 文件的惰性加载"""
        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["hello world test"], vocab_size=50, min_frequency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "train.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for i in range(5):
                    f.write(json.dumps({
                        "instruction": f"Test instruction {i}",
                        "input": f"Input {i}",
                        "output": f"Output {i}",
                    }) + "\n")

            dataset = ShardedFinetuneDataset(
                data_path=jsonl_path,
                tokenizer=tokenizer,
                max_seq_length=512,
                template="alpaca",
            )

            try:
                assert len(dataset) == 5

                # 获取一个样本
                item = dataset[0]
                assert "input_ids" in item
                assert "labels" in item
                assert "attention_mask" in item
                assert item["input_ids"].dtype == torch.long
            finally:
                dataset._mmap.close()
                dataset._file.close()

    def test_mmap_indexing(self):
        """测试 mmap 索引正确性"""
        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["test data"], vocab_size=50, min_frequency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "train.jsonl")
            for i in range(3):
                line = json.dumps({
                    "instruction": f"Instr {i}",
                    "input": f"Inp {i}",
                    "output": f"Out {i}",
                })
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")

            dataset = ShardedFinetuneDataset(
                data_path=jsonl_path,
                tokenizer=tokenizer,
                max_seq_length=512,
                template="alpaca",
            )

            try:
                assert len(dataset) == 3

                # 所有样本都能正确读取
                for i in range(3):
                    item = dataset[i]
                    assert "input_ids" in item
                    assert item["input_ids"].shape[0] > 0
            finally:
                dataset._mmap.close()
                dataset._file.close()

    def test_labels_masking(self):
        """测试 prompt 部分标签为 -100"""
        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["hello world test data"], vocab_size=50, min_frequency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = os.path.join(tmpdir, "train.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "instruction": "What is 2+2?",
                    "input": "",
                    "output": "4",
                }) + "\n")

            dataset = ShardedFinetuneDataset(
                data_path=jsonl_path,
                tokenizer=tokenizer,
                max_seq_length=512,
                template="alpaca",
            )

            try:
                item = dataset[0]
                labels = item["labels"]

                # prompt 部分应为 -100
                assert -100 in labels.tolist()
            finally:
                dataset._mmap.close()
                dataset._file.close()

    def test_requires_jsonl(self):
        """测试只接受 .jsonl 格式"""
        tokenizer = HuggingFaceBPETokenizer()
        tokenizer.train(["test"], vocab_size=50, min_frequency=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "train.json")

            with pytest.raises(ValueError, match="requires .jsonl"):
                ShardedFinetuneDataset(
                    data_path=json_path,
                    tokenizer=tokenizer,
                    max_seq_length=512,
                )


# ============ 增量预处理测试 ============

class TestDetectFileChanges:
    """文件变化检测测试"""

    def test_detect_new_files(self):
        """测试检测新增文件"""
        # 创建测试文件
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建新文件
            new_file = os.path.join(tmpdir, "new.txt")
            with open(new_file, "w", encoding="utf-8") as f:
                f.write("new content")

            # 模拟 dataset_info
            dataset_info = {
                "version": "3.0",
                "files": {},  # 空的，表示没有已记录的文件
            }

            # 导入检测函数
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from scripts.preprocess_data import detect_file_changes, FileChange

            changes = detect_file_changes([new_file], dataset_info)

            assert len(changes) == 1
            assert changes[0].change_type == "new"
            assert changes[0].new_hash is not None

    def test_detect_modified_files(self):
        """测试检测修改的文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建文件
            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("original content")

            # 计算初始哈希
            import hashlib
            with open(file_path, "rb") as f:
                old_hash = hashlib.md5(f.read()).hexdigest()

            # 修改文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("modified content")

            # 模拟 dataset_info
            dataset_info = {
                "version": "3.0",
                "files": {
                    "test.txt": {
                        "hash": old_hash,
                        "status": "processed",
                    }
                },
            }

            from scripts.preprocess_data import detect_file_changes

            changes = detect_file_changes([file_path], dataset_info)

            modified = [c for c in changes if c.change_type == "modified"]
            assert len(modified) == 1

    def test_detect_deleted_files(self):
        """测试检测删除的文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # dataset_info 中有记录但文件不存在
            dataset_info = {
                "version": "3.0",
                "files": {
                    "deleted.txt": {
                        "hash": "abc123",
                        "status": "processed",
                    }
                },
            }

            from scripts.preprocess_data import detect_file_changes

            # 当前文件列表为空
            changes = detect_file_changes([], dataset_info)

            deleted = [c for c in changes if c.change_type == "deleted"]
            assert len(deleted) == 1
            assert deleted[0].old_hash == "abc123"

    def test_detect_unchanged_files(self):
        """测试检测未变化的文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建文件
            file_path = os.path.join(tmpdir, "test.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("content")

            # 计算哈希
            import hashlib
            with open(file_path, "rb") as f:
                file_hash = hashlib.md5(f.read()).hexdigest()

            # dataset_info 中有相同哈希
            dataset_info = {
                "version": "3.0",
                "files": {
                    "test.txt": {
                        "hash": file_hash,
                        "status": "processed",
                    }
                },
            }

            from scripts.preprocess_data import detect_file_changes

            changes = detect_file_changes([file_path], dataset_info)

            unchanged = [c for c in changes if c.change_type == "unchanged"]
            assert len(unchanged) == 1


class TestDatasetInfoMigration:
    """dataset_info 版本迁移测试"""

    def test_migrate_v2_to_v3(self):
        """测试从 v2.0 迁移到 v3.0"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建测试分片
            examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]
            metadata = {"max_seq_length": 10, "vocab_size": 100, "num_examples": 1}
            save_preprocessed_data(examples, os.path.join(tmpdir, "train_000.pt"), metadata)

            # v2.0 格式的 dataset_info
            v2_info = {
                "version": "2.0",
                "file_hashes": {"/path/to/file1.txt": "hash1"},
                "train_files": ["/path/to/file1.txt"],
                "max_seq_length": 10,
                "vocab_size": 100,
                "min_frequency": 2,
                "shard_size": 10000,
                "train_metadata": {"num_examples": 100},
                "files": {"train": ["train_000.pt"]},
            }

            from scripts.preprocess_data import migrate_dataset_info_v2_to_v3

            v3_info = migrate_dataset_info_v2_to_v3(v2_info, tmpdir)

            assert v3_info["version"] == "3.0"
            assert "config" in v3_info
            assert "files" in v3_info
            assert "shards" in v3_info
            assert "summary" in v3_info
            assert v3_info["config"]["max_seq_length"] == 10

    def test_migrate_preserves_shard_info(self):
        """测试迁移保留分片信息"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建多个测试分片
            for i in range(2):
                examples = [{"input_ids": [1, 2, 3], "labels": [1, 2, 3]}]
                metadata = {"max_seq_length": 10, "vocab_size": 100, "num_examples": 1}
                save_preprocessed_data(examples, os.path.join(tmpdir, f"train_{i:03d}.pt"), metadata)

            v2_info = {
                "version": "2.0",
                "file_hashes": {"/path/to/file1.txt": "hash1"},
                "train_files": ["/path/to/file1.txt"],
                "max_seq_length": 10,
                "vocab_size": 100,
                "min_frequency": 2,
                "shard_size": 10000,
                "train_metadata": {"num_examples": 2},
                "files": {"train": ["train_000.pt", "train_001.pt"]},
            }

            from scripts.preprocess_data import migrate_dataset_info_v2_to_v3

            v3_info = migrate_dataset_info_v2_to_v3(v2_info, tmpdir)

            assert len(v3_info["shards"]) == 2
            assert "train_000.pt" in v3_info["shards"]
            assert "train_001.pt" in v3_info["shards"]


class TestIncrementalAddNewFiles:
    """增量添加新文件测试"""

    def test_add_new_file_creates_shard(self):
        """测试添加新文件创建新分片"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建 tokenizer
            tokenizer = HuggingFaceBPETokenizer()
            tokenizer.train(["hello world test"], vocab_size=50, min_frequency=1)
            tokenizer_path = os.path.join(tmpdir, "tokenizer")
            tokenizer.save(tokenizer_path)

            # 创建新文件
            new_file = os.path.join(tmpdir, "new_data.txt")
            with open(new_file, "w", encoding="utf-8") as f:
                f.write("hello world\n" * 100)

            # 初始 dataset_info
            dataset_info = {
                "version": "3.0",
                "config": {
                    "max_seq_length": 10,
                    "vocab_size": 50,
                },
                "files": {},
                "shards": {},
                "summary": {
                    "total_files": 0,
                    "total_shards": 0,
                    "next_shard_index": 0,
                },
            }

            from scripts.preprocess_data import incremental_add_new_files, PreprocessConfig

            config = PreprocessConfig(
                train_files=[new_file],
                validation_files=[],
                output_dir=tmpdir,
                max_seq_length=10,
                vocab_size=50,
            )

            updated_info = incremental_add_new_files(
                [new_file], tokenizer, config, dataset_info
            )

            assert len(updated_info["files"]) == 1
            assert updated_info["summary"]["total_files"] == 1
            assert updated_info["summary"]["total_shards"] >= 1


class TestHandleDeletedFiles:
    """处理删除文件测试"""

    def test_deleted_file_marked_orphaned(self):
        """测试删除的文件被标记为 orphaned"""
        dataset_info = {
            "version": "3.0",
            "files": {
                "deleted.txt": {
                    "hash": "abc123",
                    "status": "processed",
                    "shards": ["train_000.pt"],
                }
            },
            "shards": {
                "train_000.pt": {"num_examples": 100}
            },
            "summary": {
                "total_files": 1,
                "total_shards": 1,
            },
        }

        from scripts.preprocess_data import handle_deleted_files, FileChange

        deleted_change = FileChange(
            file_path="/path/to/deleted.txt",
            change_type="deleted",
            old_hash="abc123",
            new_hash=None,
        )

        updated_info = handle_deleted_files([deleted_change], dataset_info)

        assert updated_info["files"]["deleted.txt"]["status"] == "orphaned"
        assert "orphaned_at" in updated_info["files"]["deleted.txt"]
        # 分片应该保留
        assert "train_000.pt" in updated_info["shards"]


class TestTokenizerPolicy:
    """Tokenizer 策略测试"""

    def test_frozen_mode_uses_existing(self):
        """测试 frozen 模式使用现有 tokenizer"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建并保存 tokenizer
            tokenizer = HuggingFaceBPETokenizer()
            tokenizer.train(["hello world"], vocab_size=50, min_frequency=1)
            tokenizer_path = os.path.join(tmpdir, "tokenizer")
            tokenizer.save(tokenizer_path)

            from scripts.preprocess_data import ensure_tokenizer_consistency, PreprocessConfig

            config = PreprocessConfig(
                train_files=[],
                validation_files=[],
                output_dir=tmpdir,
                tokenizer_mode="frozen",
            )

            loaded_tokenizer, needs_reprocess = ensure_tokenizer_consistency(
                config, tmpdir, []
            )

            assert loaded_tokenizer is not None
            assert loaded_tokenizer.vocab_size == tokenizer.vocab_size
            assert needs_reprocess == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
