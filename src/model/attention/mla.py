"""
Multi-Head Latent Attention (MLA)
DeepSeek-V2/V3 风格的注意力机制

核心思想:
1. KV 压缩: 将 KV 压缩到低维潜在空间，大幅减少 KV 缓存
2. 解耦 RoPE: 只对部分维度应用 RoPE，其余维度保持不变
3. 吸收投影: 推理时可将 up-projection 吸收到 Q 投影中

参考:
- DeepSeek-V2: https://arxiv.org/abs/2405.04434
- DeepSeek-V3: https://arxiv.org/abs/2412.19437
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .base import AttentionBase, apply_rotary_emb

# RotaryEmbedding 在上级目录
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class MLAKVCache:
    """
    MLA 专用的 KV 缓存

    缓存压缩后的 KV 而不是完整的 KV，大幅节省内存
    缓存内容:
    - compressed_kv: [batch, seq, kv_lora_rank] 压缩的 KV
    - k_rope: [batch, seq, num_heads, rope_head_dim] RoPE 部分的 K
    """

    def __init__(
        self,
        kv_lora_rank: int,
        rope_head_dim: int,
        num_heads: int,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        self.kv_lora_rank = kv_lora_rank
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.dtype = dtype or torch.float16
        self.device = device

        # 缓存张量
        self.compressed_kv = None  # [batch, seq, kv_lora_rank]
        self.k_rope = None        # [batch, seq, num_heads, rope_head_dim]

    def is_empty(self) -> bool:
        """检查缓存是否为空"""
        return self.compressed_kv is None

    def update(
        self,
        new_compressed_kv: torch.Tensor,
        new_k_rope: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新缓存

        Args:
            new_compressed_kv: [batch, new_seq, kv_lora_rank]
            new_k_rope: [batch, new_seq, num_heads, rope_head_dim]

        Returns:
            compressed_kv: 完整的压缩 KV 缓存
            k_rope: 完整的 RoPE K 缓存
        """
        if self.compressed_kv is None:
            self.compressed_kv = new_compressed_kv
            self.k_rope = new_k_rope
        else:
            self.compressed_kv = torch.cat([self.compressed_kv, new_compressed_kv], dim=1)
            self.k_rope = torch.cat([self.k_rope, new_k_rope], dim=1)

        return self.compressed_kv, self.k_rope

    def get_seq_len(self) -> int:
        """获取当前缓存序列长度"""
        if self.compressed_kv is None:
            return 0
        return self.compressed_kv.shape[1]

    def reset(self):
        """重置缓存"""
        self.compressed_kv = None
        self.k_rope = None


class MultiHeadLatentAttention(AttentionBase):
    """
    Multi-Head Latent Attention (MLA)

    特点:
    1. KV 压缩: KV 先压缩到低维，推理时只缓存压缩后的 KV
    2. 解耦 RoPE: 只对部分维度应用 RoPE，避免影响压缩效率
    3. 兼容 GQA: 可与分组查询注意力结合使用

    KV 缓存大小对比:
    - 标准 Attention: O(num_layers * batch * seq * num_heads * head_dim)
    - MLA: O(num_layers * batch * seq * kv_lora_rank)
    - 压缩比: kv_lora_rank / (num_heads * head_dim)，通常为 1/10 到 1/20
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        rope_head_dim: int = 64,
        v_head_dim: int = 128,
        attention_dropout: float = 0.1,
        hidden_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        use_flash_attn: bool = True,
        **kwargs,
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            num_attention_heads: 注意力头数
            head_dim: 每个注意力头的维度
            kv_lora_rank: KV 压缩维度（核心参数）
            q_lora_rank: Q 压缩维度（可选，None 表示不压缩 Q）
            rope_head_dim: 应用 RoPE 的维度（解耦 RoPE）
            v_head_dim: V 的 head 维度
            attention_dropout: 注意力 dropout
            hidden_dropout: 隐藏层 dropout
            max_position_embeddings: 最大位置编码长度
            rope_theta: RoPE 基数
            use_flash_attn: 是否尝试使用 Flash Attention
        """
        super().__init__(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            head_dim=head_dim,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            max_position_embeddings=max_position_embeddings,
        )

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.rope_head_dim = rope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = rope_theta

        # 计算 nope_dim（不应用 RoPE 的维度）
        # 注意：rope_head_dim 会在 _init_rope 中被调整以适应 head_dim
        self.qk_nope_head_dim = head_dim - rope_head_dim

        # 先初始化 RoPE（会调整 rope_head_dim）
        self._init_rope()

        # Q 压缩投影（DeepSeek-V3 风格）
        # Q 先压缩再恢复，可进一步节省参数
        if q_lora_rank is not None and q_lora_rank > 0:
            self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(
                q_lora_rank,
                num_attention_heads * head_dim,
                bias=False
            )
        else:
            # 直接投影
            self.q_proj = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)

        # KV 压缩投影（核心）
        # 输出: compressed_kv (kv_lora_rank) + k_rope (num_heads * rope_head_dim)
        self.kv_a_proj = nn.Linear(
            hidden_size,
            kv_lora_rank + num_attention_heads * rope_head_dim,
            bias=False,
        )

        # KV 上投影（恢复完整维度）
        # 从 compressed_kv 恢复到 k_nope + v
        self.kv_b_proj = nn.Linear(
            kv_lora_rank,
            num_attention_heads * (self.qk_nope_head_dim + v_head_dim),
            bias=False,
        )

        # 输出投影
        self.o_proj = nn.Linear(num_attention_heads * v_head_dim, hidden_size, bias=False)

        # Flash Attention 支持
        self.use_flash_attn = use_flash_attn and self._check_flash_attention()
        # PyTorch SDPA fallback（不需要额外安装）
        self.use_sdpa = not self.use_flash_attn and self._check_sdpa_available()

        # 初始化权重
        self._init_weights()

    def _init_rope(self):
        """初始化 RoPE（只应用于 rope_head_dim 维度）"""
        # 需要从上级目录导入
        import sys
        from pathlib import Path
        parent_dir = Path(__file__).parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        from embedding import RotaryEmbedding

        self.rotary_emb = RotaryEmbedding(
            dim=self.rope_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _check_flash_attention(self) -> bool:
        """检查 Flash Attention 是否可用"""
        try:
            from flash_attn import flash_attn_func
            return torch.cuda.is_available()
        except ImportError:
            return False

    def _check_sdpa_available(self) -> bool:
        """检查 PyTorch 内置 SDPA 是否可用（PyTorch 2.0+）"""
        return hasattr(F, 'scaled_dot_product_attention') and torch.cuda.is_available()

    def _init_weights(self):
        """初始化权重"""
        # Xavier 初始化
        for module in [self.q_a_proj if hasattr(self, 'q_a_proj') else self.q_proj,
                       self.kv_a_proj, self.kv_b_proj, self.o_proj]:
            if hasattr(self, 'q_a_proj') and module == (self.q_a_proj if hasattr(self, 'q_a_proj') else self.q_proj):
                nn.init.xavier_uniform_(module.weight)
            elif hasattr(self, 'q_proj') and module == self.q_proj:
                nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        past_key_value: Optional[MLAKVCache] = None,
        use_cache: bool = False,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Optional[MLAKVCache]]:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: 注意力掩码（可选）
            position_ids: 位置 ID（可选）
            cos: RoPE cos 值（可选）
            sin: RoPE sin 值（可选）
            past_key_value: MLA KV 缓存
            use_cache: 是否使用缓存
            layer_idx: 层索引

        Returns:
            output: [batch_size, seq_len, hidden_size]
            present_key_value: 更新后的 MLA KV 缓存（如果 use_cache=True）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Q 投影
        if hasattr(self, 'q_a_proj'):
            # 压缩再恢复
            q = self.q_b_proj(self.q_a_proj(hidden_states))
        else:
            q = self.q_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, num_heads, seq, head_dim]

        # 2. KV 压缩投影
        kv = self.kv_a_proj(hidden_states)

        # 分离 compressed_kv 和 k_rope
        compressed_kv = kv[..., :self.kv_lora_rank]  # [batch, seq, kv_lora_rank]
        k_rope = kv[..., self.kv_lora_rank:]          # [batch, seq, num_heads * rope_head_dim]

        # 3. KV 上投影
        kv_up = self.kv_b_proj(compressed_kv)
        kv_up = kv_up.view(
            batch_size, seq_len,
            self.num_attention_heads,
            self.qk_nope_head_dim + self.v_head_dim
        )

        # 分离 k_nope 和 v
        k_nope = kv_up[..., :self.qk_nope_head_dim]  # [batch, seq, num_heads, qk_nope_dim]
        v = kv_up[..., self.qk_nope_head_dim:]        # [batch, seq, num_heads, v_head_dim]

        # 4. 应用解耦 RoPE
        # 只对 Q 和 K 的 RoPE 部分应用位置编码
        k_rope = k_rope.view(batch_size, seq_len, self.num_attention_heads, self.rope_head_dim)

        # 获取 RoPE cos/sin
        # 注意：使用传入的 cos/sin（来自 TransformerEmbedding），维度是 head_dim
        # 或者使用 MLA 自己的 rotary_emb（维度是 rope_head_dim）
        if cos is None or sin is None:
            # 生成 RoPE 嵌入（使用 MLA 自己的 rotary_emb）
            cos, sin = self.rotary_emb(hidden_states, seq_len)
            rope_cos, rope_sin = cos, sin
        else:
            # 使用传入的 cos/sin（维度是 head_dim），需要为 MLA 单独生成
            rope_cos, rope_sin = self.rotary_emb(hidden_states, seq_len)

        # 重塑 Q 和 K 的 RoPE 部分
        q_rope = q[..., :self.rope_head_dim]  # [batch, num_heads, seq, rope_head_dim]
        k_rope = k_rope.transpose(1, 2)        # [batch, num_heads, seq, rope_head_dim]

        # 应用 RoPE（使用 MLA 自己的 rotary_emb 生成的 cos/sin）
        q_rope = apply_rotary_emb(q_rope, rope_cos, rope_sin)
        k_rope = apply_rotary_emb(k_rope, rope_cos, rope_sin)

        # 5. 组合完整的 K: concat(RoPE 部分, 非 RoPE 部分)
        # q: [batch, num_heads, seq, head_dim] 其中 head_dim = rope_head_dim + qk_nope_head_dim
        q_nope = q[..., self.rope_head_dim:]  # [batch, num_heads, seq, qk_nope_dim]
        q = torch.cat([q_rope, q_nope], dim=-1)  # [batch, num_heads, seq, head_dim]

        k_nope = k_nope.transpose(1, 2)  # [batch, num_heads, seq, qk_nope_dim]
        k = torch.cat([k_rope, k_nope], dim=-1)  # [batch, num_heads, seq, head_dim]

        # 6. 处理 KV 缓存
        present_key_value = None
        if use_cache and past_key_value is not None:
            # 更新缓存（存储压缩形式）
            compressed_kv_cached, k_rope_cached = past_key_value.update(
                compressed_kv,
                k_rope.transpose(1, 2)  # [batch, seq, num_heads, rope_head_dim]
            )

            # 从压缩缓存恢复完整 KV
            kv_up_cached = self.kv_b_proj(compressed_kv_cached)
            kv_up_cached = kv_up_cached.view(
                batch_size, -1,  # seq_len 变为缓存的长度
                self.num_attention_heads,
                self.qk_nope_head_dim + self.v_head_dim
            )

            k_nope_cached = kv_up_cached[..., :self.qk_nope_head_dim].transpose(1, 2)
            v = kv_up_cached[..., self.qk_nope_head_dim:].transpose(1, 2)

            k_rope_cached = k_rope_cached.transpose(1, 2)
            k = torch.cat([k_rope_cached, k_nope_cached], dim=-1)
        else:
            v = v.transpose(1, 2)  # [batch, num_heads, seq, v_head_dim]

        # 7. 创建新的缓存
        if use_cache:
            present_key_value = MLAKVCache(
                kv_lora_rank=self.kv_lora_rank,
                rope_head_dim=self.rope_head_dim,
                num_heads=self.num_attention_heads,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            present_key_value.update(compressed_kv, k_rope.transpose(1, 2))

        # 8. 注意力计算
        if self.use_flash_attn and hidden_states.device.type == "cuda":
            attn_output = self._flash_attention(q, k, v)
        elif self.use_sdpa and hidden_states.device.type == "cuda":
            attn_output = self._sdpa_attention(q, k, v)
        else:
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # 9. 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)
        output = self.resid_dropout(output)

        return output, present_key_value

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        使用 PyTorch 内置的 scaled_dot_product_attention（SDPA）
        支持 Flash Attention / Memory-Efficient Attention 自动选择

        Args:
            q: [batch, num_heads, q_seq, head_dim]
            k: [batch, num_heads, kv_seq, head_dim]
            v: [batch, num_heads, kv_seq, v_head_dim]

        Returns:
            output: [batch, num_heads, q_seq, v_head_dim]
        """
        scale = 1.0 / math.sqrt(self.rope_head_dim + self.qk_nope_head_dim)

        # SDPA 要求 KV 序列长度维度上的 head_dim 一致
        # 当 v_head_dim != head_dim 时，需要将 K 和 V 的 head_dim 对齐
        if self.v_head_dim != self.head_dim:
            head_dim = self.head_dim
            # Pad V to match head_dim, or pad K to match v_head_dim
            if self.v_head_dim < head_dim:
                v = F.pad(v, (0, head_dim - self.v_head_dim))
            else:
                k = F.pad(k, (0, self.v_head_dim - head_dim))
                head_dim = self.v_head_dim

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
                scale=scale,
            )

            # 移除 padding
            if self.v_head_dim < self.head_dim:
                output = output[..., :self.v_head_dim]
        else:
            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=True,
                scale=scale,
            )

        return output

    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        标准注意力计算

        Args:
            q: [batch, num_heads, q_seq, head_dim]
            k: [batch, num_heads, kv_seq, head_dim]
            v: [batch, num_heads, kv_seq, v_head_dim]
            attention_mask: 注意力掩码

        Returns:
            output: [batch, num_heads, q_seq, v_head_dim]
        """
        # 计算注意力分数
        # 注意: v_head_dim 可能与 head_dim 不同
        scale = 1.0 / math.sqrt(self.rope_head_dim + self.qk_nope_head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 创建因果掩码
        kv_seq_len = k.shape[2]
        q_seq_len = q.shape[2]

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(q_seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
            diagonal=kv_seq_len - q_seq_len + 1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))

        # 应用额外掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # 计算输出
        output = torch.matmul(attn_weights, v)

        return output

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Flash Attention 实现

        Args:
            q: [batch, num_heads, q_seq, head_dim]
            k: [batch, num_heads, kv_seq, head_dim]
            v: [batch, num_heads, kv_seq, v_head_dim]

        Returns:
            output: [batch, num_heads, q_seq, v_head_dim]
        """
        # Flash Attention 要求输入格式: [batch, seq, num_heads, head_dim]
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
                softmax_scale=1.0 / math.sqrt(self.head_dim),
                causal=True,
            )
        except Exception:
            # 回退到标准注意力
            output = self._standard_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None
            )
            output = output.transpose(1, 2)

        output = output.transpose(1, 2)  # [batch, num_heads, seq, v_head_dim]

        if input_dtype == torch.float32:
            output = output.float()

        return output
