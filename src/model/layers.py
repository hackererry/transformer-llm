"""
Transformer基础层模块
包含RMSNorm、SwiGLU FFN等组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    比LayerNorm更高效，不需要计算均值
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
        Returns:
            归一化后的张量
        """
        # 计算RMS
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class SwiGLUFFN(nn.Module):
    """
    SwiGLU前馈神经网络
    结合了Swish激活函数和GLU(Gated Linear Unit)结构
    公式: SwiGLU(x) = Swish(xW1) ⊗ (xW2)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        # 门控投影
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 上投影
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        # 下投影
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        # Dropout
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # SwiGLU: Swish(gate) * up
        gate = F.silu(self.gate_proj(x))  # SiLU = Swish
        up = self.up_proj(x)
        hidden = gate * up
        # 下投影
        output = self.down_proj(hidden)
        return self.dropout(output)


class FeedForward(nn.Module):
    """
    标准前馈神经网络
    使用GELU激活函数
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return self.dropout(x)


class MLP(nn.Module):
    """
    通用MLP模块
    支持多种激活函数
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation: str = "gelu",
        hidden_dropout: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.dropout = nn.Dropout(hidden_dropout)

        # 选择激活函数
        if activation == "gelu":
            self.act_fn = F.gelu
        elif activation == "relu":
            self.act_fn = F.relu
        elif activation == "silu" or activation == "swish":
            self.act_fn = F.silu
        elif activation == "tanh":
            self.act_fn = torch.tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return self.dropout(x)


class Dropout(nn.Module):
    """
    增强的Dropout模块
    支持多种dropout模式
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__()
        self.p = p
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0:
            return F.dropout(x, self.p, self.training, self.inplace)
        return x


class LayerNorm(nn.Module):
    """
    标准LayerNorm实现
    作为RMSNorm的备选
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.weight * (x - mean) / torch.sqrt(var + self.eps) + self.bias


class ResidualConnection(nn.Module):
    """
    残差连接模块
    支持Pre-Norm和Post-Norm
    """

    def __init__(self, hidden_size: int, norm_eps: float = 1e-6, pre_norm: bool = True):
        super().__init__()
        self.norm = RMSNorm(hidden_size, norm_eps)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        if self.pre_norm:
            # Pre-Norm: Norm -> Sublayer -> Residual
            return x + sublayer(self.norm(x))
        else:
            # Post-Norm: Sublayer -> Residual -> Norm
            return self.norm(x + sublayer(x))


class TransformerMLP(nn.Module):
    """
    Transformer的完整MLP块
    包含LayerNorm和残差连接
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.1,
        norm_eps: float = 1e-6,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.norm = RMSNorm(hidden_size, norm_eps)

        if use_swiglu:
            self.ffn = SwiGLUFFN(hidden_size, intermediate_size, hidden_dropout)
        else:
            self.ffn = FeedForward(hidden_size, intermediate_size, hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Norm架构
        residual = x
        x = self.norm(x)
        x = self.ffn(x)
        return residual + x
