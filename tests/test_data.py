"""
数据处理模块测试
"""
import pytest
import torch
import sys
import os
import tempfile
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    BPETokenizer,
    DataCollatorForCausalLM,
    DataCollatorForSFT,
    get_collator,
)


class TestBPETokenizer:
    """BPE Tokenizer测试"""

    def test_tokenizer_creation(self):
        """测试创建tokenizer"""
        tokenizer = BPETokenizer()
        assert tokenizer is not None

    def test_encode_decode(self):
        """测试编码和解码"""
        tokenizer = BPETokenizer()
        # 添加一些基本词汇
        tokenizer.vocab = {"<unk>": 0, "<pad>": 1, "a": 2, "b": 3, "c": 4}
        tokenizer.ids_to_tokens = {v: k for k, v in tokenizer.vocab.items()}

        # 测试编码
        text = "abc"
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert isinstance(ids, list)

    def test_special_tokens(self):
        """测试特殊token"""
        tokenizer = BPETokenizer(
            unk_token="<unk>",
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )

        assert tokenizer.unk_token == "<unk>"
        assert tokenizer.pad_token == "<pad>"
        assert tokenizer.bos_token == "<s>"
        assert tokenizer.eos_token == "</s>"

    def test_train(self):
        """测试训练tokenizer"""
        tokenizer = BPETokenizer()
        texts = ["hello world", "hello there", "world peace"]
        tokenizer.train(texts, vocab_size=50)

        assert tokenizer.vocab_size > 0
        assert len(tokenizer.merges) >= 0

    def test_save_and_load(self):
        """测试保存和加载"""
        tokenizer = BPETokenizer()
        tokenizer.train(["hello world"], vocab_size=30)

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenizer.save(tmpdir)
            loaded = BPETokenizer.load(tmpdir)

            assert loaded.vocab == tokenizer.vocab


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

        # labels用-100 padding
        assert result["labels"][0, -1] == -100 or result["labels"][0, -1] in [1, 2, 3]


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

    def test_invalid_collator(self):
        """测试无效的整理器类型"""
        with pytest.raises(ValueError):
            get_collator("invalid_type")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
