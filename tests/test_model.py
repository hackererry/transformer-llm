"""
模型模块测试
测试所有模型组件：配置、Transformer、注意力、嵌入、LMHead等
"""
import pytest
import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    ModelConfig,
    CausalLMModel,
    TransformerModel,
    TransformerBlock,
    create_model,
    RMSNorm,
    SwiGLUFFN,
    FeedForward,
    LayerNorm,
    MLP,
    TransformerMLP,
    RotaryEmbedding,
    TokenEmbedding,
    TransformerEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
    Attention,
    FlashAttention,
    GroupedQueryAttention,
    CrossAttention,
    LMHead,
    TiedLMHead,
    AdaptiveLMHead,
    MLPHead,
    get_attention_class,
    is_flash_attention_available,
)


class TestModelConfig:
    """模型配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = ModelConfig()
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.vocab_size == 32000

    def test_tiny_config(self):
        """测试tiny配置"""
        config = ModelConfig.tiny()
        assert config.hidden_size == 256
        assert config.num_hidden_layers == 6
        assert config.num_attention_heads == 8
        assert config.intermediate_size == 512

    def test_small_config(self):
        """测试small配置"""
        config = ModelConfig.small()
        assert config.hidden_size == 512
        assert config.num_hidden_layers == 12
        assert config.num_attention_heads == 8
        assert config.intermediate_size == 1024

    def test_medium_config(self):
        """测试medium配置"""
        config = ModelConfig.medium()
        assert config.hidden_size == 1024
        assert config.num_hidden_layers == 24
        assert config.num_attention_heads == 16
        assert config.intermediate_size == 2048

    def test_head_dim_calculation(self):
        """测试head_dim计算"""
        config = ModelConfig(hidden_size=512, num_attention_heads=8)
        assert config.head_dim == 64

        config2 = ModelConfig(hidden_size=1024, num_attention_heads=16)
        assert config2.head_dim == 64

    def test_invalid_config_divisibility(self):
        """测试无效配置 - hidden_size必须能被num_attention_heads整除"""
        with pytest.raises(AssertionError):
            ModelConfig(hidden_size=100, num_attention_heads=8)

    def test_custom_config(self):
        """测试自定义配置"""
        config = ModelConfig(
            vocab_size=50000,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=4096,
            rope_theta=10000.0,
            rms_norm_eps=1e-5,
            hidden_dropout=0.1,
            attention_dropout=0.1,
        )
        assert config.vocab_size == 50000
        assert config.head_dim == 64

    def test_to_dict(self):
        """测试转换为字典"""
        config = ModelConfig.tiny()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["hidden_size"] == 256
        assert d["vocab_size"] == 32000
        assert "head_dim" in d

    def test_from_dict(self):
        """测试从字典创建"""
        d = {
            "vocab_size": 30000,
            "hidden_size": 256,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 512,
            "max_position_embeddings": 1024,
        }
        config = ModelConfig.from_dict(d)
        assert config.hidden_size == 256
        assert config.vocab_size == 30000


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
        assert torch.isfinite(output).all()

    def test_different_eps(self):
        """测试不同的eps值"""
        norm1 = RMSNorm(128, eps=1e-6)
        norm2 = RMSNorm(128, eps=1e-12)
        x = torch.randn(1, 5, 128)
        out1 = norm1(x)
        out2 = norm2(x)
        assert out1.shape == out2.shape

    def test_identity_with_unit_normalized(self):
        """测试单位规范化的输入"""
        norm = RMSNorm(64)
        x = torch.ones(1, 1, 64) * 0.1
        output = norm(x)
        assert output.shape == x.shape


class TestSwiGLUFFN:
    """SwiGLU FFN测试"""

    def test_forward_shape(self):
        """测试输出形状"""
        ffn = SwiGLUFFN(hidden_size=512, intermediate_size=1024)
        x = torch.randn(2, 10, 512)
        output = ffn(x)
        assert output.shape == (2, 10, 512)

    def test_forward_with_different_sizes(self):
        """测试不同尺寸的输入"""
        ffn = SwiGLUFFN(hidden_size=256, intermediate_size=512)
        x = torch.randn(4, 16, 256)
        output = ffn(x)
        assert output.shape == (4, 16, 256)

    def test_intermediate_size_calculation(self):
        """测试中间层大小"""
        config = ModelConfig(hidden_size=512, intermediate_size=2048)
        ffn = SwiGLUFFN(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )
        x = torch.randn(1, 1, config.hidden_size)
        output = ffn(x)
        assert output.shape == x.shape


class TestFeedForward:
    """标准FeedForward测试"""

    def test_forward_shape(self):
        """测试输出形状"""
        ff = FeedForward(hidden_size=512, intermediate_size=2048)
        x = torch.randn(2, 10, 512)
        output = ff(x)
        assert output.shape == (2, 10, 512)


class TestRotaryEmbedding:
    """旋转位置编码测试"""

    def test_cos_sin_shape(self):
        """测试cos和sin形状"""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 10, 64)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (10, 64)
        assert sin.shape == (10, 64)

    def test_different_seq_len(self):
        """测试不同序列长度"""
        rope = RotaryEmbedding(dim=32, max_position_embeddings=256)
        x = torch.randn(1, 20, 32)
        cos1, sin1 = rope(x, seq_len=20)

        x2 = torch.randn(1, 10, 32)
        cos2, sin2 = rope(x2, seq_len=10)

        assert cos1.shape == (20, 32)
        assert cos2.shape == (10, 32)

    def test_position_ids(self):
        """测试位置ID"""
        rope = RotaryEmbedding(dim=64, max_position_embeddings=512)
        x = torch.randn(2, 10, 64)
        cos, sin = rope(x, seq_len=10)
        assert cos.shape == (10, 64)


class TestApplyRotaryPosEmb:
    """RoPE应用函数测试"""

    def test_apply_rotary(self):
        """测试RoPE应用"""
        batch_size, seq_len, num_heads, head_dim = 2, 10, 8, 64
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)

        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape

    def test_rotate_half(self):
        """测试rotate_half函数"""
        x = torch.randn(2, 10, 64)
        out = rotate_half(x)
        assert out.shape == x.shape


class TestTransformerEmbedding:
    """Transformer嵌入测试"""

    def test_embedding_creation(self):
        """测试嵌入创建"""
        emb = TransformerEmbedding(
            vocab_size=1000,
            hidden_size=256,
            max_position_embeddings=512,
        )
        assert emb.token_embedding.embedding.num_embeddings == 1000

    def test_forward(self):
        """测试嵌入前向传播"""
        emb = TransformerEmbedding(
            vocab_size=1000,
            hidden_size=256,
            max_position_embeddings=512,
        )
        input_ids = torch.randint(0, 1000, (2, 32))
        position_ids = torch.arange(32).unsqueeze(0).expand(2, -1)

        hidden_states, (cos, sin) = emb(input_ids, position_ids)
        assert hidden_states.shape == (2, 32, 256)
        assert cos.shape[0] == 32


class TestAttention:
    """注意力机制测试"""

    def test_attention_creation(self):
        """测试注意力创建"""
        attn = Attention(
            hidden_size=256,
            num_attention_heads=8,
            head_dim=32,
        )
        assert attn.hidden_size == 256
        assert attn.num_heads == 8

    def test_attention_forward(self):
        """测试注意力前向传播"""
        attn = Attention(
            hidden_size=256,
            num_attention_heads=8,
            head_dim=32,
        )
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, 256)

        output, _ = attn(hidden_states)
        assert output.shape == hidden_states.shape

    def test_attention_with_mask(self):
        """测试带mask的注意力"""
        attn = Attention(
            hidden_size=256,
            num_attention_heads=8,
            head_dim=32,
        )
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, 256)
        attention_mask = torch.ones(batch_size, seq_len)

        output, _ = attn(hidden_states, attention_mask=attention_mask)
        assert output.shape == hidden_states.shape

    def test_attention_with_kv_cache(self):
        """测试带KV缓存的注意力"""
        attn = Attention(
            hidden_size=256,
            num_attention_heads=8,
            head_dim=32,
        )
        batch_size, seq_len = 1, 16
        hidden_states = torch.randn(batch_size, seq_len, 256)

        output, present_kv = attn(hidden_states, use_cache=True)
        assert present_kv is not None
        assert len(present_kv) == 2  # key and value


class TestCausalLMModel:
    """因果语言模型测试"""

    def test_model_creation(self):
        """测试模型创建"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)
        assert model is not None
        assert isinstance(model, nn.Module)

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

    def test_generation_greedy(self):
        """测试贪婪解码生成"""
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

    def test_generation_sampling(self):
        """测试采样生成"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 10))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=20,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
            )

        assert output.shape[1] == 30

    def test_generation_with_eos(self):
        """测试带EOS的生成"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 5))

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=50,
                do_sample=False,
                eos_token_id=0,
            )

        # 应该在max_new_tokens内结束
        assert output.shape[1] <= 55

    def test_kv_cache(self):
        """测试KV缓存"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)

        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids, use_cache=True)
        assert "past_key_values" in outputs
        assert len(outputs["past_key_values"]) == config.num_hidden_layers

    def test_prepare_inputs_for_generation(self):
        """测试生成输入准备"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)

        input_ids = torch.randint(0, config.vocab_size, (1, 20))
        past_kv = [None] * config.num_hidden_layers

        inputs = model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_kv
        )
        assert inputs["input_ids"].shape[1] == 1  # 只保留最后一个token

    def test_resize_token_embeddings(self):
        """测试调整词嵌入大小"""
        config = ModelConfig.tiny()
        model = CausalLMModel(config)

        old_size = model.config.vocab_size
        model.resize_token_embeddings(old_size + 100)
        assert model.config.vocab_size == old_size + 100

    def test_tied_embeddings(self):
        """测试共享词嵌入"""
        config = ModelConfig.tiny()
        config.tie_word_embeddings = True
        model = CausalLMModel(config)

        batch_size, seq_len = 1, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        outputs = model(input_ids, labels=input_ids)
        assert "loss" in outputs


class TestTransformerBlock:
    """Transformer块测试"""

    def test_block_forward(self):
        """测试块前向传播"""
        config = ModelConfig.tiny()
        block = TransformerBlock(config, layer_idx=0)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        output, _ = block(hidden_states)
        assert output.shape == hidden_states.shape

    def test_block_with_position_ids(self):
        """测试带position_ids的块前向传播"""
        config = ModelConfig.tiny()
        block = TransformerBlock(config, layer_idx=0)

        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        output, _ = block(hidden_states, position_ids=position_ids)
        assert output.shape == hidden_states.shape


class TestTransformerModel:
    """Transformer模型测试"""

    def test_transformer_creation(self):
        """测试Transformer创建"""
        config = ModelConfig.tiny()
        model = TransformerModel(config)
        assert model is not None

    def test_transformer_forward(self):
        """测试Transformer前向传播"""
        config = ModelConfig.tiny()
        model = TransformerModel(config)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        hidden_states, _ = model(input_ids)
        assert hidden_states.shape == (batch_size, seq_len, config.hidden_size)

    def test_get_set_input_embeddings(self):
        """测试获取和设置输入嵌入"""
        config = ModelConfig.tiny()
        model = TransformerModel(config)

        emb = model.get_input_embeddings()
        assert emb.num_embeddings == config.vocab_size


class TestCreateModel:
    """模型工厂函数测试"""

    def test_create_tiny_model(self):
        """测试创建tiny模型"""
        model = create_model(config_name="tiny")
        assert isinstance(model, CausalLMModel)

    def test_create_small_model(self):
        """测试创建small模型"""
        model = create_model(config_name="small")
        assert isinstance(model, CausalLMModel)

    def test_create_with_config(self):
        """测试使用配置创建模型"""
        config = ModelConfig.tiny()
        model = create_model(config=config)
        assert isinstance(model, CausalLMModel)


class TestLMHead:
    """LM Head测试"""

    def test_lm_head_creation(self):
        """测试LMHead创建"""
        head = LMHead(hidden_size=256, vocab_size=1000)
        assert head.vocab_size == 1000
        assert head.hidden_size == 256

    def test_lm_head_forward(self):
        """测试LMHead前向传播"""
        head = LMHead(hidden_size=256, vocab_size=1000)
        hidden_states = torch.randn(2, 10, 256)
        logits = head(hidden_states)
        assert logits.shape == (2, 10, 1000)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
