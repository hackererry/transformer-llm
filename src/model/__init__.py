"""
模型模块
导出所有模型相关组件
"""
from .config import ModelConfig, TrainingConfig, DataConfig
from .transformer import (
    TransformerModel,
    TransformerBlock,
    CausalLMModel,
    create_model,
)
from .attention import (
    Attention,
    FlashAttention,
    GroupedQueryAttention,
    CrossAttention,
    is_flash_attention_available,
    is_sdpa_available,
    get_attention_info,
    create_attention,
    get_attention_class,
)
from .embedding import (
    TokenEmbedding,
    RotaryEmbedding,
    TransformerEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)
from .layers import (
    RMSNorm,
    SwiGLUFFN,
    FeedForward,
    MLP,
    LayerNorm,
    TransformerMLP,
)
from .lm_head import (
    LMHead,
    TiedLMHead,
    AdaptiveLMHead,
    MLPHead,
    Pooler,
    SequenceSummary,
)

__all__ = [
    # 配置
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # 主模型
    "TransformerModel",
    "TransformerBlock",
    "CausalLMModel",
    "create_model",
    # 注意力
    "Attention",
    "FlashAttention",
    "GroupedQueryAttention",
    "CrossAttention",
    "is_flash_attention_available",
    "is_sdpa_available",
    "get_attention_info",
    "create_attention",
    "get_attention_class",
    # 嵌入
    "TokenEmbedding",
    "RotaryEmbedding",
    "TransformerEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
    # 层
    "RMSNorm",
    "SwiGLUFFN",
    "FeedForward",
    "MLP",
    "LayerNorm",
    "TransformerMLP",
    # LM头
    "LMHead",
    "TiedLMHead",
    "AdaptiveLMHead",
    "MLPHead",
    "Pooler",
    "SequenceSummary",
]
