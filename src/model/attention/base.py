"""
Attention基础模块
包含公共函数和基类
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将张量分成两半并旋转
    用于RoPE计算

    Args:
        x: 输入张量 [..., head_dim]
    Returns:
        旋转后的张量 [..., head_dim]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    应用旋转位置编码

    Args:
        x: 输入张量 [batch, num_heads, seq_len, head_dim]
        cos: RoPE的cos缓存 [seq_len, head_dim] 或 [batch, seq_len, head_dim]
        sin: RoPE的sin缓存 [seq_len, head_dim] 或 [batch, seq_len, head_dim]
    Returns:
        应用RoPE后的张量
    """
    # 调整cos/sin形状
    seq_len = x.shape[2]
    if cos.dim() == 2:
        # [seq_len, head_dim] -> [1, 1, seq_len, head_dim]
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        # [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
        cos = cos[:, :seq_len].unsqueeze(1)
        sin = sin[:, :seq_len].unsqueeze(1)

    # 应用旋转
    return x * cos + rotate_half(x) * sin


def apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对Q和K应用旋转位置编码

    Args:
        q: Query张量 [batch, num_heads, seq_len, head_dim]
        k: Key张量 [batch, num_heads, seq_len, head_dim]
        cos: RoPE的cos缓存
        sin: RoPE的sin缓存
    Returns:
        应用RoPE后的Q和K
    """
    return apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    创建因果掩码（下三角矩阵）

    Args:
        seq_len: 序列长度
        device: 设备
        dtype: 数据类型
    Returns:
        因果掩码 [1, 1, seq_len, seq_len]，上三角为-inf
    """
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=dtype),
        diagonal=1,
    )
    causal_mask = causal_mask * -1e9  # 负无穷
    return causal_mask.unsqueeze(0).unsqueeze(0)


def repeat_kv(
    x: torch.Tensor,
    num_repeats: int,
) -> torch.Tensor:
    """
    重复KV头以匹配Q头数（用于GQA）

    Args:
        x: 输入张量 [batch, num_kv_heads, seq_len, head_dim]
        num_repeats: 重复次数 (num_heads / num_kv_heads)
    Returns:
        扩展后的张量 [batch, num_heads, seq_len, head_dim]
    """
    if num_repeats == 1:
        return x
    batch_size, num_heads, seq_len, head_dim = x.shape
    x = x[:, :, None, :, :].expand(
        batch_size, num_heads, num_repeats, seq_len, head_dim
    )
    return x.reshape(batch_size, num_heads * num_repeats, seq_len, head_dim)


class AttentionBase(nn.Module):
    """
    Attention基类
    定义公共接口和初始化逻辑
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings

        # 缩放因子
        self.scale = 1.0 / math.sqrt(head_dim)

        # Dropout层
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)

    def _init_weights(self, module: nn.Module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def get_head_dim(self) -> int:
        """获取每个头的维度"""
        return self.head_dim

    def get_num_heads(self) -> int:
        """获取注意力头数"""
        return self.num_attention_heads
