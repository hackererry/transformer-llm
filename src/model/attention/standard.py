"""
标准Attention实现
CPU/GPU通用，兼容性好
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .base import AttentionBase, apply_rotary_emb_qk, create_causal_mask


class StandardAttention(AttentionBase):
    """
    标准多头注意力机制
    使用分离的QKV投影，兼容性好
    支持RoPE位置编码和KV缓存
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        **kwargs,  # 接受额外参数以保持接口兼容
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
        )

        # 分离的QKV投影（兼容性好）
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # 初始化权重
        self.apply(self._init_weights)

    # 兼容性别名
    @property
    def num_heads(self) -> int:
        """兼容性别名"""
        return self.num_attention_heads

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
            hidden_states: 输入张量 [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码 [batch_size, 1, seq_len, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
            cos: RoPE的cos缓存
            sin: RoPE的sin缓存
            past_key_value: 过去的KV缓存
            use_cache: 是否使用KV缓存
        Returns:
            attn_output: 注意力输出 [batch_size, seq_len, hidden_size]
            present_key_value: 当前的KV缓存(如果use_cache=True)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 重塑为多头形式 [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

        # 应用RoPE
        if cos is not None and sin is not None:
            query_states, key_states = apply_rotary_emb_qk(
                query_states, key_states, cos, sin
            )

        # 处理KV缓存
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        present_key_value = None
        if use_cache:
            present_key_value = (key_states, value_states)

        # 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) * self.scale

        # 应用注意力掩码
        if attention_mask is not None:
            # 处理不同形状的attention_mask
            if attention_mask.dim() == 2:
                # [batch, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                # [batch, 1, seq_len] -> [batch, 1, 1, seq_len]
                attention_mask = attention_mask.unsqueeze(1)

            # 将padding位置的注意力设为负无穷
            attention_mask = (1.0 - attention_mask.to(attn_weights.dtype)) * -1e9
            attn_weights = attn_weights + attention_mask

        # Softmax（使用FP32提高数值稳定性）
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        attn_weights = self.attn_dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, value_states)

        # 重塑回原始形状
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)

        # 输出投影
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, present_key_value


class ChunkedAttention(AttentionBase):
    """
    分块计算的Attention
    用于减少内存峰值，适用于长序列
    注意：这不是Flash Attention，只是分块计算
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        chunk_size: int = 64,
        **kwargs,
    ):
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
        )
        self.chunk_size = chunk_size

        # QKV投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        self.apply(self._init_weights)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """分块计算注意力以节省内存"""
        batch_size, seq_len, _ = hidden_states.shape

        # 计算QKV
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 重塑
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE
        if cos is not None and sin is not None:
            query, key = apply_rotary_emb_qk(query, key, cos, sin)

        # 分块计算注意力
        output = self._chunked_attention(query, key, value, attention_mask)

        # 重塑和输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return self.resid_dropout(output), None

    def _chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """分块计算注意力以减少内存使用"""
        batch_size, num_heads, seq_len, head_dim = query.shape

        # 如果序列较短，直接计算
        if seq_len <= self.chunk_size:
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            return torch.matmul(attn_weights, value)

        # 分块计算
        output = torch.zeros_like(query)
        chunk_size = self.chunk_size

        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            query_chunk = query[:, :, i:end_i, :]

            # 计算这个chunk对所有key的注意力
            attn_weights = torch.matmul(query_chunk, key.transpose(-2, -1)) * self.scale

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask[:, :, i:end_i, :]

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)

            output[:, :, i:end_i, :] = torch.matmul(attn_weights, value)

        return output


class CrossAttention(nn.Module):
    """
    交叉注意力
    用于编码器-解码器架构
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        # Q投影(来自解码器)
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        # KV投影(来自编码器)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            hidden_states: 解码器隐藏状态 [batch, seq_len, hidden]
            encoder_hidden_states: 编码器隐藏状态 [batch, encoder_seq_len, hidden]
            attention_mask: 解码器自注意力掩码
            encoder_attention_mask: 编码器注意力掩码
        """
        batch_size, seq_len, _ = hidden_states.shape
        encoder_seq_len = encoder_hidden_states.size(1)

        # 计算Q(来自解码器)
        query = self.q_proj(hidden_states)
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 计算KV(来自编码器)
        key = self.k_proj(encoder_hidden_states)
        value = self.v_proj(encoder_hidden_states)
        key = key.view(batch_size, encoder_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, encoder_seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 计算注意力
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if encoder_attention_mask is not None:
            attn_weights = attn_weights + encoder_attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, value)

        # 重塑
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return self.resid_dropout(output), None


# 兼容性别名
Attention = StandardAttention
