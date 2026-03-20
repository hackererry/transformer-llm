"""
词嵌入和位置编码模块
包含Token Embedding和RoPE位置编码
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class TokenEmbedding(nn.Module):
    """
    Token嵌入层
    将token ID映射到嵌入向量
    """

    def __init__(self, vocab_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

        # 初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: token ID张量 [batch_size, seq_len]
        Returns:
            嵌入向量 [batch_size, seq_len, hidden_size]
        """
        embeddings = self.embedding(input_ids)
        # 缩放嵌入
        embeddings = embeddings * math.sqrt(self.hidden_size)
        return self.dropout(embeddings)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码(Rotary Position Embedding, RoPE)
    现代LLM(Llama, GPT-NeoX等)使用的位置编码方式
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # 预计算频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算cos和sin缓存
        self._set_cos_sin_cache(max_position_embeddings, device=None, dtype=None)

    def _set_cos_sin_cache(self, seq_len: int, device=None, dtype=None):
        """预计算cos和sin缓存"""
        if device is None:
            device = self.inv_freq.device
        if dtype is None:
            dtype = self.inv_freq.dtype

        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(device=device, dtype=dtype))
        # 重复以匹配所有维度
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入张量(用于获取dtype和device)
            seq_len: 序列长度
        Returns:
            cos, sin: 旋转位置编码
        """
        # 如果序列长度超过缓存，重新计算
        if seq_len > self.max_position_embeddings:
            self._set_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
            self.max_position_embeddings = seq_len

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将张量分成两半并旋转
    用于RoPE计算
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用旋转位置编码到query和key

    Args:
        q: query张量 [batch, num_heads, seq_len, head_dim]
        k: key张量 [batch, num_heads, seq_len, head_dim]
        cos: cos缓存 [seq_len, head_dim] 或 [batch, seq_len, head_dim]
        sin: sin缓存 [seq_len, head_dim] 或 [batch, seq_len, head_dim]
        position_ids: 位置ID [batch, seq_len]
    Returns:
        应用RoPE后的q和k
    """
    # 获取对应位置的cos和sin
    if position_ids is not None:
        # cos, sin: [seq_len, head_dim] -> [batch, seq_len, head_dim]
        cos = cos[position_ids]
        sin = sin[position_ids]

    # 调整形状以匹配q和k
    # cos: [batch, seq_len, head_dim] -> [batch, 1, seq_len, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # 应用旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


class LearnedPositionalEmbedding(nn.Module):
    """
    可学习的位置嵌入
    作为RoPE的备选方案
    """

    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        embeddings: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: token嵌入 [batch_size, seq_len, hidden_size]
            position_ids: 位置ID [batch_size, seq_len]
        Returns:
            添加位置编码后的嵌入
        """
        seq_len = embeddings.size(1)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=embeddings.device)
            position_ids = position_ids.unsqueeze(0).expand(embeddings.size(0), -1)

        position_embeddings = self.embedding(position_ids)
        return self.dropout(embeddings + position_embeddings)


class SinusoidalPositionalEmbedding(nn.Module):
    """
    正弦位置编码
    原始Transformer使用的位置编码
    """

    def __init__(
        self,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 预计算位置编码
        position = torch.arange(max_position_embeddings).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )
        pe = torch.zeros(1, max_position_embeddings, hidden_size)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入嵌入 [batch_size, seq_len, hidden_size]
        Returns:
            添加位置编码后的嵌入
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    完整的Transformer嵌入层
    包含Token嵌入和位置编码
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_position_embeddings: int = 2048,
        dropout: float = 0.1,
        position_embedding_type: str = "rope",
        rope_theta: float = 10000.0,
        head_dim: int = None,
    ):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, hidden_size, dropout)
        self.position_embedding_type = position_embedding_type
        self.head_dim = head_dim or hidden_size // 8  # 默认假设8个头

        if position_embedding_type == "rope":
            # RoPE在Attention层应用，这里只创建对象
            # 注意：RoPE的dim应该是head_dim，不是hidden_size
            self.position_embedding = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings,
                rope_theta,
            )
        elif position_embedding_type == "learned":
            self.position_embedding = LearnedPositionalEmbedding(
                hidden_size,
                max_position_embeddings,
                dropout,
            )
        elif position_embedding_type == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEmbedding(
                hidden_size,
                max_position_embeddings,
                dropout,
            )
        else:
            raise ValueError(f"Unknown position embedding type: {position_embedding_type}")

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: token ID张量 [batch_size, seq_len]
            position_ids: 位置ID [batch_size, seq_len]
        Returns:
            embeddings: 嵌入向量 [batch_size, seq_len, hidden_size]
            pos_embedding: 位置编码(对于RoPE返回cos, sin)
        """
        embeddings = self.token_embedding(input_ids)

        if self.position_embedding_type == "rope":
            # RoPE返回cos和sin，在attention层使用
            seq_len = input_ids.size(1)
            cos, sin = self.position_embedding(embeddings, seq_len)
            return embeddings, (cos, sin)
        else:
            # 其他位置编码直接加到嵌入上
            embeddings = self.position_embedding(embeddings, position_ids)
            return embeddings, None
