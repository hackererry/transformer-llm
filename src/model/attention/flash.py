"""
Flash Attention GPU优化实现
支持Flash Attention 2/3和PyTorch原生优化
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .base import AttentionBase, apply_rotary_emb, create_causal_mask


# 检测Flash Attention是否可用
FLASH_ATTENTION_AVAILABLE = False
FLASH_ATTENTION_V2_AVAILABLE = False
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
    FLASH_ATTENTION_V2_AVAILABLE = True
except ImportError:
    pass

# 检测PyTorch 2.0+ scaled_dot_product_attention
SDPA_AVAILABLE = hasattr(F, 'scaled_dot_product_attention')


def is_flash_attention_available() -> bool:
    """检查Flash Attention是否可用"""
    return FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available()


def is_sdpa_available() -> bool:
    """检查PyTorch SDPA是否可用"""
    return SDPA_AVAILABLE


class FlashAttention(AttentionBase):
    """
    Flash Attention 2 实现
    显著减少显存占用并加速计算

    特点：
    - 合并QKV投影（GPU优化）
    - 自动选择Flash Attention或回退实现
    - 支持KV缓存
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = None,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        head_dim = head_dim or hidden_size // num_attention_heads

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
        )

        # 合并的QKV投影（GPU优化：减少kernel调用）
        self.qkv_proj = nn.Linear(hidden_size, 3 * num_attention_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # 是否使用Flash Attention
        self.use_flash_attn = use_flash_attn and is_flash_attention_available()
        self.use_sdpa = not self.use_flash_attn and is_sdpa_available()

        # 初始化权重
        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: 注意力掩码
            cos, sin: RoPE位置编码
            past_key_value: KV缓存
            use_cache: 是否使用KV缓存
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算合并的QKV
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 应用RoPE
        if cos is not None and sin is not None:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # KV缓存
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # 选择注意力计算方式
        if self.use_flash_attn and hidden_states.device.type == "cuda":
            attn_output = self._flash_attention(q, k, v)
        elif self.use_sdpa:
            attn_output = self._sdpa_attention(q, k, v, attention_mask)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, present_key_value

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """使用Flash Attention计算"""
        # 转换为Flash Attention期望的格式
        # [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention要求FP16或BF16
        input_dtype = q.dtype
        if input_dtype == torch.float32:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)

        try:
            from flash_attn import flash_attn_func
            # causal=True 启用因果掩码
            output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=self.scale,
                causal=True,
            )
        except Exception:
            # 回退到标准注意力
            output = self._standard_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None
            )
            output = output.transpose(1, 2)

        # 转回原始格式
        output = output.transpose(1, 2)  # [batch, heads, seq, head_dim]

        if input_dtype == torch.float32:
            output = output.float()

        return output

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用PyTorch scaled_dot_product_attention"""
        # q, k, v: [batch, heads, seq, head_dim]

        # 使用is_causal参数自动处理因果掩码
        output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
            scale=self.scale,
        )

        return output

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准注意力计算（PyTorch原生）"""
        # q, k, v: [batch, heads, seq, head_dim]

        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 应用因果掩码
        seq_len = q.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        # 应用额外掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax和Dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 计算输出
        output = torch.matmul(attn_weights, v)

        return output


class ScaledDotProductAttention(AttentionBase):
    """
    优化的标准注意力实现
    使用PyTorch 2.0+的scaled_dot_product_attention
    适用于不支持Flash Attention的情况
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = None,
        attention_dropout: float = 0.1,
        use_sdpa: bool = True,
        **kwargs,
    ):
        head_dim = head_dim or hidden_size // num_attention_heads

        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
        )

        # 分离的QKV投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.use_sdpa = use_sdpa and is_sdpa_available()
        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # 计算QKV
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE
        if cos is not None and sin is not None:
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # KV缓存
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # 使用PyTorch的scaled_dot_product_attention
        if self.use_sdpa:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
                scale=self.scale,
            )
        else:
            # 回退实现
            output = self._manual_attention(q, k, v, attention_mask)

        # 重塑输出
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)

        return output, present_key_value

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """手动实现的注意力（兼容旧版PyTorch）"""
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 因果掩码
        seq_len = q.shape[2]
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        return torch.matmul(attn_weights, v)
