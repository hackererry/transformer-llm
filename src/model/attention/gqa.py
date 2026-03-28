"""
Grouped Query Attention (GQA) 实现
减少KV缓存，支持Flash Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .base import AttentionBase, apply_rotary_emb, repeat_kv


class GroupedQueryAttention(AttentionBase):
    """
    分组查询注意力 (Grouped Query Attention, GQA)

    特点：
    - 多个Query头共享一组Key/Value头
    - 减少KV缓存75%（如8个Query头用2个KV头）
    - 支持Flash Attention
    - 兼容CPU和GPU

    示例：
        num_heads=8, num_kv_heads=2
        - Q: 8个头
        - K: 2个头（每个被4个Q头共享）
        - V: 2个头（每个被4个Q头共享）
        - KV缓存减少: 8/2 = 4倍
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        num_key_value_heads: int = None,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
        )

        # KV头数，默认与Q头数相同（即MHA）
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.num_key_value_groups = num_attention_heads // self.num_key_value_heads

        # 验证头数配置
        assert num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

        # Q投影（完整头数）
        self.q_proj = nn.Linear(
            hidden_size,
            num_attention_heads * head_dim,
            bias=False
        )

        # 合并的KV投影（GPU优化：减少kernel调用）
        self.kv_proj = nn.Linear(
            hidden_size,
            2 * self.num_key_value_heads * head_dim,
            bias=False
        )

        # 输出投影
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim,
            hidden_size,
            bias=False
        )

        # Flash Attention支持
        self.use_flash_attn = use_flash_attn and self._check_flash_attention()

        # 初始化权重
        self.apply(self._init_weights)

    def _check_flash_attention(self) -> bool:
        """检查Flash Attention是否可用"""
        try:
            from flash_attn import flash_attn_func
            return torch.cuda.is_available()
        except ImportError:
            return False

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
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码
            position_ids: 位置ID
            cos, sin: RoPE位置编码
            past_key_value: KV缓存
            use_cache: 是否使用KV缓存
        Returns:
            output: [batch_size, seq_len, hidden_size]
            present_key_value: KV缓存（如果use_cache=True）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算Q
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        # 计算KV（合并投影）
        kv = self.kv_proj(hidden_states)
        kv = kv.view(
            batch_size, seq_len, 2, self.num_key_value_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)  # [2, batch, num_kv_heads, seq, head_dim]
        key_states = kv[0]
        value_states = kv[1]

        # 应用RoPE（在KV扩展前）
        if cos is not None and sin is not None:
            query_states = apply_rotary_emb(query_states, cos, sin)
            key_states = apply_rotary_emb(key_states, cos, sin)

        # 处理KV缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = None
        if use_cache:
            present_key_value = (key_states, value_states)

        # 扩展KV头以匹配Q头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 选择注意力计算方式
        if self.use_flash_attn and hidden_states.device.type == "cuda":
            attn_output = self._flash_attention(query_states, key_states, value_states)
        else:
            attn_output = self._standard_attention(
                query_states, key_states, value_states, attention_mask
            )

        # 重塑输出
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output, present_key_value

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """使用Flash Attention计算GQA"""
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

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准注意力计算"""
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

        # Softmax和Dropout（使用FP32提高数值稳定性）
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 计算输出
        output = torch.matmul(attn_weights, v)

        return output


class MultiQueryAttention(GroupedQueryAttention):
    """
    多查询注意力 (Multi-Query Attention, MQA)
    GQA的特例：num_kv_heads=1

    特点：
    - 所有Query头共享1组Key/Value
    - KV缓存减少87.5%（8头→1头KV）
    - 推理速度最快
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            num_key_value_heads=1,  # MQA: 只有1个KV头
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
            use_flash_attn=use_flash_attn,
            **kwargs,
        )


# 工具函数：计算GQA的KV缓存减少比例
def calculate_kv_cache_reduction(num_heads: int, num_kv_heads: int) -> float:
    """
    计算GQA相比MHA的KV缓存减少比例

    Args:
        num_heads: Query头数
        num_kv_heads: KV头数
    Returns:
        减少比例（0-1）
    """
    return 1.0 - (num_kv_heads / num_heads)
