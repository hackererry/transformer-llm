"""
注意力机制模块
包含Multi-Head Attention和RoPE集成
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .embedding import apply_rotary_pos_emb
from .layers import RMSNorm


class Attention(nn.Module):
    """
    多头注意力机制
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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.attention_dropout = attention_dropout
        self.hidden_dropout = hidden_dropout
        self.max_position_embeddings = max_position_embeddings

        # QKV投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)

        # 缩放因子
        self.scale = 1.0 / math.sqrt(head_dim)

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

        # 重塑为多头形式
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
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids
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
            attn_weights = attn_weights + attention_mask

        # Softmax和Dropout
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


class FlashAttention(nn.Module):
    """
    Flash Attention的CPU实现
    注意: 真正的Flash Attention需要CUDA，这里提供内存优化的替代实现
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size

        # QKV投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)

        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        分块计算注意力以节省内存
        """
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
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # 分块计算注意力
        output = self._chunked_attention(query, key, value, attention_mask)

        # 重塑和输出投影
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return self.resid_dropout(output)

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


class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力(Grouped Query Attention, GQA)
    多个query头共享一个key/value头
    可减少KV缓存大小
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads

        # Q投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        # KV投影(使用较少的头)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * head_dim, bias=False)

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.resid_dropout = nn.Dropout(hidden_dropout)

        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # 计算QKV
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        # 重塑Q
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)

        # 重塑KV
        key = key.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 应用RoPE
        if cos is not None and sin is not None:
            query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)

        # 扩展KV以匹配Q的头数
        key = self._repeat_kv(key)
        value = self._repeat_kv(value)

        # 计算注意力
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        output = torch.matmul(attn_weights, value)

        # 重塑
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(output)
        return self.resid_dropout(output)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """重复KV头以匹配Q头数"""
        if self.num_key_value_groups == 1:
            return x
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(
            batch_size, num_heads, self.num_key_value_groups, seq_len, head_dim
        )
        return x.reshape(batch_size, num_heads * self.num_key_value_groups, seq_len, head_dim)


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
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim

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

        self.scale = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
        return self.resid_dropout(output)
