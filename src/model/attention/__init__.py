"""
统一Attention模块
自动选择最优实现（CPU/GPU、Flash Attention、GQA等）
"""
import torch
from typing import Optional

from .base import (
    AttentionBase,
    apply_rotary_emb,
    apply_rotary_emb_qk,
    create_causal_mask,
    rotate_half,
    repeat_kv,
)

from .standard import (
    StandardAttention,
    ChunkedAttention,
    CrossAttention,
    Attention,  # 兼容性别名
)

from .flash import (
    FlashAttention,
    ScaledDotProductAttention,
    is_flash_attention_available,
    is_sdpa_available,
)

from .gqa import (
    GroupedQueryAttention,
    MultiQueryAttention,
    calculate_kv_cache_reduction,
)

from .streaming import (
    StreamingAttention,
    StreamingKVCache,
)

from .mla import (
    MultiHeadLatentAttention,
    MLAKVCache,
)


def create_attention(
    hidden_size: int,
    num_attention_heads: int,
    head_dim: int = None,
    num_key_value_heads: int = None,
    attention_dropout: float = 0.1,
    hidden_dropout: float = 0.1,
    max_position_embeddings: int = 2048,
    use_flash: bool = True,
    use_gqa: bool = False,
    use_chunked: bool = False,
    use_streaming_llm: bool = False,
    use_mla: bool = False,
    # MLA 专用参数
    kv_lora_rank: int = 512,
    q_lora_rank: int = 1536,
    rope_head_dim: int = 64,
    v_head_dim: int = 128,
    # StreamingLLM 参数
    sink_size: int = 4,
    streaming_window_size: int = 4096,
    **kwargs,
) -> AttentionBase:
    """
    工厂函数：根据配置自动选择最优Attention实现

    Args:
        hidden_size: 隐藏层大小
        num_attention_heads: 注意力头数
        head_dim: 每个头的维度（默认 hidden_size // num_attention_heads）
        num_key_value_heads: KV头数（GQA用，默认与num_attention_heads相同）
        attention_dropout: 注意力dropout率
        hidden_dropout: 隐藏层dropout率
        max_position_embeddings: 最大位置编码长度
        use_flash: 是否尝试使用Flash Attention
        use_gqa: 是否使用GQA（分组查询注意力）
        use_chunked: 是否使用分块注意力（节省内存）
        use_streaming_llm: 是否使用StreamingLLM（无限长度推理）
        use_mla: 是否使用MLA（Multi-Head Latent Attention）
        kv_lora_rank: MLA KV压缩维度
        q_lora_rank: MLA Q压缩维度
        rope_head_dim: MLA RoPE维度
        v_head_dim: MLA V维度
        sink_size: Attention Sink数量
        streaming_window_size: 滑动窗口大小
        **kwargs: 其他参数

    Returns:
        最优的Attention模块实例

    选择逻辑:
        1. 如果use_mla=True → MultiHeadLatentAttention
        2. 如果use_streaming_llm=True → StreamingAttention
        3. 如果use_gqa=True → GroupedQueryAttention
        4. 如果use_flash=True且GPU可用 → FlashAttention
        5. 如果use_chunked=True → ChunkedAttention
        6. 否则 → StandardAttention
    """
    head_dim = head_dim or hidden_size // num_attention_heads

    # 最高优先级：MLA
    if use_mla:
        return MultiHeadLatentAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            kv_lora_rank=kv_lora_rank,
            q_lora_rank=q_lora_rank,
            rope_head_dim=rope_head_dim,
            v_head_dim=v_head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
            use_flash_attn=use_flash,
            **kwargs,
        )

    # 优先使用 StreamingLLM
    if use_streaming_llm:
        return StreamingAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            sink_size=sink_size,
            window_size=streaming_window_size,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
            use_flash_attn=use_flash,
            **kwargs,
        )

    # 判断是否真正需要GQA
    is_gqa = use_gqa or (num_key_value_heads is not None and num_key_value_heads != num_attention_heads)

    if is_gqa:
        # 使用GQA
        return GroupedQueryAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads or num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
            use_flash_attn=use_flash,
            **kwargs,
        )

    elif use_flash and torch.cuda.is_available() and is_flash_attention_available():
        # 使用Flash Attention
        return FlashAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
            use_flash_attn=True,
            **kwargs,
        )

    elif use_chunked:
        # 使用分块注意力（节省内存）
        return ChunkedAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )

    else:
        # 使用标准注意力（兼容性最好）
        return StandardAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
            **kwargs,
        )


def get_attention_info() -> dict:
    """
    获取当前环境的Attention支持情况

    Returns:
        包含各种Attention可用性的字典
    """
    return {
        "cuda_available": torch.cuda.is_available(),
        "flash_attention": is_flash_attention_available(),
        "sdpa": is_sdpa_available(),
        "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "CPU",
    }


def get_attention_class(use_flash: bool = True, use_gqa: bool = False) -> type:
    """
    获取最优的Attention类（兼容性别名）

    Args:
        use_flash: 是否优先使用Flash Attention
        use_gqa: 是否使用GQA

    Returns:
        最优的Attention类
    """
    if use_gqa:
        return GroupedQueryAttention
    elif use_flash and torch.cuda.is_available() and is_flash_attention_available():
        return FlashAttention
    else:
        return StandardAttention


# 导出所有公共接口
__all__ = [
    # 基础类和函数
    "AttentionBase",
    "apply_rotary_emb",
    "apply_rotary_emb_qk",
    "create_causal_mask",
    "rotate_half",
    "repeat_kv",

    # Attention实现
    "StandardAttention",
    "ChunkedAttention",
    "CrossAttention",
    "FlashAttention",
    "ScaledDotProductAttention",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "StreamingAttention",
    "StreamingKVCache",
    "MultiHeadLatentAttention",
    "MLAKVCache",

    # 兼容性别名
    "Attention",

    # 工厂函数
    "create_attention",
    "get_attention_class",

    # 工具函数
    "is_flash_attention_available",
    "is_sdpa_available",
    "get_attention_info",
    "calculate_kv_cache_reduction",
]
