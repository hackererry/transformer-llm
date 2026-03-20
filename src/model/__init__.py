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

# GPU优化的注意力（可选导入）
try:
    from .attention_gpu import (
        FlashAttention as GPUFlashAttention,
        ScaledDotProductAttention,
        get_attention_class,
        is_flash_attention_available,
    )
    _gpu_attention_available = True
except ImportError:
    _gpu_attention_available = False

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

# GPU优化（可选）
if _gpu_attention_available:
    __all__.extend([
        "GPUFlashAttention",
        "ScaledDotProductAttention",
        "get_attention_class",
        "is_flash_attention_available",
    ])
