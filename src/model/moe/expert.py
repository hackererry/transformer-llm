"""
MoE 专家模块
实现单个 SwiGLU 专家
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUExpert(nn.Module):
    """
    单个 SwiGLU 专家

    与标准 SwiGLU FFN 结构相同，但可配置独立的 intermediate_size
    用于 MoE 中的专家网络
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # SwiGLU 结构：gate_proj, up_proj, down_proj
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(hidden_dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, hidden_size] 或 [num_tokens, hidden_size]

        Returns:
            output: 与输入相同形状
        """
        # SwiGLU: Swish(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        output = self.down_proj(hidden)
        return self.dropout(output)


class SharedExpert(nn.Module):
    """
    共享专家

    DeepSeek-V3 风格：共享专家始终激活，处理通用知识
    与普通专家结构相同，但不会参与路由
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.1,
    ):
        super().__init__()
        self.expert = SwiGLUExpert(hidden_size, intermediate_size, hidden_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: [batch_size, seq_len, hidden_size]

        Returns:
            output: 与输入相同形状
        """
        return self.expert(x)
