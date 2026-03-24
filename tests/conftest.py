"""
pytest配置文件
提供共享的fixture和配置
"""
import pytest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def tiny_config():
    """Tiny模型配置fixture"""
    from src.model import ModelConfig
    return ModelConfig.tiny()


@pytest.fixture
def small_config():
    """Small模型配置fixture"""
    from src.model import ModelConfig
    return ModelConfig.small()


@pytest.fixture
def medium_config():
    """Medium模型配置fixture"""
    from src.model import ModelConfig
    return ModelConfig.medium()


@pytest.fixture
def sample_tokenizer():
    """样本tokenizer fixture"""
    from src.data import HuggingFaceBPETokenizer
    tokenizer = HuggingFaceBPETokenizer()
    # 训练一个小型tokenizer用于测试
    texts = [
        "hello world",
        "this is a test",
        "transformers are powerful",
        "natural language processing",
        "machine learning is fun",
    ]
    tokenizer.train(texts, vocab_size=100, min_frequency=1)
    return tokenizer


@pytest.fixture
def sample_batch():
    """样本批次数据fixture"""
    return {
        "input_ids": torch.randint(0, 1000, (2, 32)),
        "attention_mask": torch.ones(2, 32),
        "labels": torch.randint(0, 1000, (2, 32)),
    }


@pytest.fixture
def temp_dir(tmp_path):
    """临时目录fixture"""
    return tmp_path


@pytest.fixture(autouse=True)
def reset_random_seed():
    """每个测试前重置随机种子"""
    torch.manual_seed(42)
    yield
