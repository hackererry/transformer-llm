"""
GPU优化的注意力机制
支持Flash Attention 2和混合精度训练
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math

# 检测Flash Attention是否可用
FLASH_ATTENTION_AVAILABLE = False
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    pass


def is_flash_attention_available() -> bool:
    """检查Flash Attention是否可用"""
    return FLASH_ATTENTION_AVAILABLE and torch.cuda.is_available()


class FlashAttention(nn.Module):
    """
    Flash Attention 2 实现
    显著减少显存占用并加速计算
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = None,
        attention_dropout: float = 0.1,
        max_position_embeddings: int = 2048,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings

        # QKV投影（合并以提高效率）
        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        # 是否使用Flash Attention
        self.use_flash_attn = use_flash_attn and is_flash_attention_available()

        # 缩放因子
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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

        # 计算QKV
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_attention_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 应用RoPE
        if cos is not None and sin is not None:
            q = self._apply_rotary_emb(q, cos, sin)
            k = self._apply_rotary_emb(k, cos, sin)

        # KV缓存
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_key_value = (k, v) if use_cache else None

        # 选择注意力计算方式
        if self.use_flash_attn and hidden_states.device.type == "cuda":
            # Flash Attention路径
            attn_output = self._flash_attention(q, k, v, attention_mask)
        else:
            # 标准注意力路径
            attn_output = self._standard_attention(q, k, v, attention_mask)

        # 输出投影
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output, present_key_value

    def _apply_rotary_emb(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """应用旋转位置编码"""
        # x: [batch, heads, seq, head_dim]
        seq_len = x.shape[2]

        # 调整cos/sin形状
        if cos.dim() == 2:
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, head_dim]
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        elif cos.dim() == 3:
            cos = cos[:, :seq_len].unsqueeze(1)  # [batch, 1, seq, head_dim]
            sin = sin[:, :seq_len].unsqueeze(1)

        # 旋转
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)

        return x * cos + rotated * sin

    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """使用Flash Attention计算"""
        # 转换为Flash Attention期望的格式
        # [batch, heads, seq, head_dim] -> [batch, seq, heads, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Flash Attention要求FP16或BF16
        dtype = q.dtype
        if dtype == torch.float32:
            q = q.half()
            k = k.half()
            v = v.half()

        # 调用Flash Attention
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
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attention_mask
            )
            output = output.transpose(1, 2)

        # 转回原始格式
        output = output.transpose(1, 2)  # [batch, heads, seq, head_dim]

        if dtype == torch.float32:
            output = output.float()

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


class ScaledDotProductAttention(nn.Module):
    """
    优化的标准注意力实现
    适用于不支持Flash Attention的情况
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        head_dim: int = None,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout

        # 分离的QKV投影
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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
            q, k = self._apply_rotary_emb(q, k, cos, sin)

        # 使用PyTorch的scaled_dot_product_attention（优化版本）
        # PyTorch 2.0+ 自动选择最优实现
        if hasattr(F, 'scaled_dot_product_attention'):
            # 创建因果掩码
            attn_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool),
                diagonal=1
            )

            output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
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

        return self.o_proj(output)

    def _apply_rotary_emb(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用旋转位置编码"""
        seq_len = q.shape[2]

        if cos.dim() == 2:
            cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
            sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # 旋转
        def rotate(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat([-x2, x1], dim=-1)

        q_embed = q * cos + rotate(q) * sin
        k_embed = k * cos + rotate(k) * sin

        return q_embed, k_embed

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


def get_attention_class(use_flash: bool = True):
    """
    工厂函数：获取最优的Attention类

    Args:
        use_flash: 是否尝试使用Flash Attention

    Returns:
        最优的Attention类
    """
    if use_flash and is_flash_attention_available():
        return FlashAttention
    else:
        return ScaledDotProductAttention
