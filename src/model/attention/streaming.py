"""
StreamingLLM Attention 实现
支持无限长度推理的滑动窗口注意力 + Attention Sink
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

from .base import AttentionBase, apply_rotary_emb, repeat_kv


class StreamingKVCache:
    """
    StreamingLLM 的 KV 缓存管理器

    结构: [Sinks | Sliding Window]
    - Sinks: 前 sink_size 个 token，始终保留
    - Sliding Window: 最近 window_size 个 token，动态更新

    效果: 固定显存占用 O(sink_size + window_size)，支持无限长度推理
    """

    def __init__(
        self,
        sink_size: int = 4,
        window_size: int = 4096,
        num_layers: int = 12,
        num_heads: int = 8,
        head_dim: int = 64,
        num_kv_heads: int = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        self.sink_size = sink_size
        self.window_size = window_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = head_dim

        # 最大缓存长度 = sinks + window
        self.max_cache_len = sink_size + window_size

        # 缓存: [layers, 2 (K/V), batch, kv_heads, seq, head_dim]
        self.cache = None
        self.dtype = dtype or torch.float16
        self.device = device

        # 跟踪当前序列位置
        self.current_pos = 0

    def is_empty(self) -> bool:
        """检查缓存是否为空"""
        return self.cache is None

    def init_cache(self, batch_size: int):
        """初始化缓存张量"""
        self.cache = torch.zeros(
            self.num_layers, 2, batch_size,
            self.num_kv_heads, self.max_cache_len, self.head_dim,
            dtype=self.dtype, device=self.device
        )
        self.current_pos = 0

    def reset(self):
        """重置缓存"""
        self.cache = None
        self.current_pos = 0

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存并返回当前有效的 K/V

        Args:
            layer_idx: 层索引
            new_k: [batch, kv_heads, new_seq, head_dim]
            new_v: [batch, kv_heads, new_seq, head_dim]

        Returns:
            k, v: 有效的 K/V 缓存
        """
        batch_size = new_k.shape[0]
        new_seq_len = new_k.shape[2]

        if self.cache is None:
            self.init_cache(batch_size)

        start_pos = self.current_pos

        for i in range(new_seq_len):
            pos = start_pos + i

            if pos < self.sink_size:
                # 阶段1: 填充 attention sinks（锚点，始终保留）
                self.cache[layer_idx, 0, :, :, pos, :] = new_k[:, :, i, :]
                self.cache[layer_idx, 1, :, :, pos, :] = new_v[:, :, i, :]

            elif pos < self.max_cache_len:
                # 阶段2: 填充滑动窗口（未满）
                self.cache[layer_idx, 0, :, :, pos, :] = new_k[:, :, i, :]
                self.cache[layer_idx, 1, :, :, pos, :] = new_v[:, :, i, :]

            else:
                # 阶段3: 滑动窗口更新（环形缓冲）
                # 计算在窗口中的位置
                window_pos = self.sink_size + (pos - self.max_cache_len) % self.window_size
                self.cache[layer_idx, 0, :, :, window_pos, :] = new_k[:, :, i, :]
                self.cache[layer_idx, 1, :, :, window_pos, :] = new_v[:, :, i, :]

        self.current_pos = start_pos + new_seq_len

        return self.get_kv(layer_idx)

    def get_kv(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取当前层的有效 K/V"""
        if self.current_pos <= self.max_cache_len:
            # 未满: 返回实际长度的缓存
            k = self.cache[layer_idx, 0, :, :, :self.current_pos, :]
            v = self.cache[layer_idx, 1, :, :, :self.current_pos, :]
        else:
            # 已满: 返回完整缓存 [sinks + window]
            k = self.cache[layer_idx, 0]
            v = self.cache[layer_idx, 1]

        return k, v

    def get_seq_len(self) -> int:
        """获取当前有效序列长度"""
        return min(self.current_pos, self.max_cache_len)


class StreamingAttention(AttentionBase):
    """
    StreamingLLM 注意力实现

    特点:
    - 保留前 N 个 token 作为 attention sinks（锚点）
    - 滑动窗口处理后续 token
    - 支持无限长度推理
    - 可与 GQA 结合使用
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        sink_size: int = 4,
        window_size: int = 4096,
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

        self.sink_size = sink_size
        self.window_size = window_size

        # GQA 支持
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.num_key_value_groups = num_attention_heads // self.num_key_value_heads

        # Q 投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # 合并的 KV 投影
        self.kv_proj = nn.Linear(
            hidden_size,
            2 * self.num_key_value_heads * head_dim,
            bias=False
        )

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)

        # Flash Attention 支持
        self.use_flash_attn = use_flash_attn and self._check_flash_attention()

        self.apply(self._init_weights)

    def _check_flash_attention(self) -> bool:
        """检查 Flash Attention 是否可用"""
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
        past_key_value: Optional[StreamingKVCache] = None,
        use_cache: bool = False,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[StreamingKVCache]]:
        """
        前向传播

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: StreamingKVCache 实例
            layer_idx: 当前层索引
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(
            batch_size, seq_len, self.num_attention_heads, self.head_dim
        ).transpose(1, 2)

        # 计算 KV
        kv = self.kv_proj(hidden_states)
        kv = kv.view(
            batch_size, seq_len, 2, self.num_key_value_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        key_states = kv[0]
        value_states = kv[1]

        # 应用 RoPE
        if cos is not None and sin is not None:
            query_states = apply_rotary_emb(query_states, cos, sin)
            key_states = apply_rotary_emb(key_states, cos, sin)

        # 处理 Streaming KV 缓存
        if use_cache and past_key_value is not None:
            key_states, value_states = past_key_value.update(
                layer_idx, key_states, value_states
            )

        # 扩展 KV 头（GQA）
        if self.num_key_value_groups > 1:
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力
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

        return output, past_key_value

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """标准注意力计算，支持 StreamingLLM 的因果掩码"""
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # 创建 StreamingLLM 专用的因果掩码
        kv_len = k.shape[2]
        q_len = q.shape[2]

        causal_mask = self._create_streaming_mask(q_len, kv_len, q.device)
        attn_weights = attn_weights + causal_mask

        # 应用额外掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        return torch.matmul(attn_weights, v)

    def _create_streaming_mask(
        self,
        q_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        创建 StreamingLLM 的注意力掩码

        确保每个 token 只能 attend to:
        1. Attention sinks (前 sink_size 个)
        2. 滑动窗口内的 token
        """
        dtype = torch.float32

        # 基础因果掩码
        mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)

        # 计算 Q 的起始位置（在有 KV 缓存时）
        q_start = kv_len - q_len

        for i in range(q_len):
            q_pos = q_start + i

            # 因果掩码：不能关注未来的 token
            if q_pos + 1 < kv_len:
                mask[i, q_pos + 1:] = -1e9

            # 滑动窗口：不能关注窗口外的 token（除了 sinks）
            window_start = max(self.sink_size, q_pos - self.window_size + 1)
            if window_start > self.sink_size:
                mask[i, self.sink_size:window_start] = -1e9

        return mask.unsqueeze(0).unsqueeze(0)

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor
    ) -> torch.Tensor:
        """Flash Attention 实现"""
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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
            output = self._standard_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None
            )
            output = output.transpose(1, 2)

        output = output.transpose(1, 2)

        if input_dtype == torch.float32:
            output = output.float()

        return output
