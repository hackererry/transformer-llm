"""
模型模块测试
"""
import pytest
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    ModelConfig,
    CausalLMModel,
    create_model,
    RMSNorm,
    SwiGLUFFN,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)


class TestModelConfig:
    """模型配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ModelConfig()
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8

    def test_tiny_config(self):
        """测试tiny配置"""
        config = ModelConfig.tiny()
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 6

    def test_small_config(self):
        """测试small配置"""
        config = ModelConfig.small()
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12

    def test_medium_config(self):
        """测试medium配置"""
        config = ModelConfig.medium()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24

    def test_head_dim_calculation(self):
        """测试head_dim计算"""
        config = ModelConfig(hidden_size=512, num_attention_heads=8)
        assert config.head_dim == 64

    def test_invalid_config(self):
        """测试无效配置"""
        with pytest.raises(AssertionError):
            ModelConfig(hidden_size=100, num_attention_heads=8)

    def test_to_dict(self):
        """测试转换为字典"""
        config = ModelConfig.tiny()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_size"] == 256

    def test_from_dict(self):
        """测试从字典创建"""
        d = {"hidden_size": 256, "num_hidden_layers": 6}
        config = ModelConfig.from_dict(d)
        assert config.hidden_size == 256


class TestRMSNorm:
    """RMSNorm测试"""

    def test_forward_shape(self):
        """测试输出形状"""
        norm = RMSNorm(512)
        x = torch.randn(2, 10, 512)
        output = norm(x)
        assert output.shape == x.shape

    def test_forward_values(self):
        """测试输出值"""
        norm = RMSNorm(512, eps=1e-6)
        x = torch.randn(2, 10, 512)
        output = norm(x)
        # 检查输出是否有限
        assert torch.isfinite(output).all()


class TestSwiGLUFFN:
    """SwiGLU FFN测试"""

    def test_forward_shape(self):
        """测试输出形状"""
        ffn = SwiGLUFFN(hidden_size=512, intermediate_size=1024)
        x = torch.randn(2, 10, 512)
        output = ffn(x)
        assert output.shape == (2, 10, 512)


class TestRotaryEmbedding:
    """旋转位置编码测试"""

    def test_cos_sin_shape(self):
        """测试cos和sin形状"""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 10, 512)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (10, 64)
        assert sin.shape == (10, 64)


class TestCausalLMModel:
    """因果语言模型测试"""

    def test_model_creation(self):
        """测试模型创建"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)
        assert model is not None

    def test_forward_shape(self):
        """测试前向传播形状"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids)
        assert "logits" in outputs
        assert outputs["logits"].shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_labels(self):
        """测试带标签的前向传播"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()

        outputs = model(input_ids, labels=labels)
        assert "loss" in outputs
        assert outputs["loss"].item() > 0

    def test_generation(self):
        """测试文本生成"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 10))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=False,
            )

        assert output.shape[1] == 30  # 10 + 20

    def test_kv_cache(self):
        """测试KV缓存"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)

        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids, use_cache=True)
        assert "past_key_values" in outputs
        assert len(outputs["past_key_values"]) == config.num_hidden_layers


class TestCreateModel:
    """模型工厂函数测试"""

    def test_create_tiny_model(self):
        """测试创建tiny模型"""
        model = create_model(config_name="tiny")
        assert isinstance(model, CausalLMModel)

    def test_create_with_config(self):
        """测试使用配置创建模型"""
        config = ModelConfig.tiny()
        model = create_model(config=config)
        assert isinstance(model, CausalLMModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
