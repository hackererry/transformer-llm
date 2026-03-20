"""
语言模型头模块
包含各种类型的输出层实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LMHead(nn.Module):
    """
    标准语言模型头
    将隐藏状态映射到词表大小的logits
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        bias: bool = False,
        layer_norm: bool = False,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # 可选的LayerNorm
        if layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.layer_norm = None

        # 输出投影
        self.fc = nn.Linear(hidden_size, vocab_size, bias=bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
        Returns:
            logits: 输出logits [batch, seq_len, vocab_size]
        """
        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        return self.fc(hidden_states)


class TiedLMHead(nn.Module):
    """
    权重共享的语言模型头
    与输入嵌入层共享权重，节省参数
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        # 不存储权重，使用外部嵌入权重

    def forward(
        self,
        hidden_states: torch.Tensor,
        embedding_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: 输入隐藏状态 [batch, seq_len, hidden_size]
            embedding_weight: 嵌入层权重 [vocab_size, hidden_size]
        Returns:
            logits: 输出logits [batch, seq_len, vocab_size]
        """
        return F.linear(hidden_states, embedding_weight)


class AdaptiveLMHead(nn.Module):
    """
    自适应语言模型头
    支持adaptive softmax，可加速大词表的训练
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        cutoffs: list = None,
        div_value: float = 4.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        if cutoffs is None:
            # 默认分割点
            cutoffs = [vocab_size // 4, vocab_size // 2, 3 * vocab_size // 4]

        self.cutoffs = cutoffs + [vocab_size]
        self.div_value = div_value

        # 创建多个cluster
        self.clusters = nn.ModuleList()
        for i in range(len(self.cutoffs)):
            if i == 0:
                # 第一个cluster包含高频词
                self.clusters.append(
                    nn.Linear(hidden_size, self.cutoffs[0])
                )
            else:
                # 后续cluster包含低频词
                dim = hidden_size // (div_value ** i)
                self.clusters.append(nn.Sequential(
                    nn.Linear(hidden_size, dim),
                    nn.Linear(dim, self.cutoffs[i] - self.cutoffs[i - 1])
                ))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        简化版实现: 直接拼接所有cluster的输出
        """
        outputs = []
        for cluster in self.clusters:
            outputs.append(cluster(hidden_states))
        return torch.cat(outputs, dim=-1)


class MLPHead(nn.Module):
    """
    MLP分类头
    用于下游任务的分类
    """

    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        hidden_dropout: float = 0.1,
        intermediate_size: Optional[int] = None,
    ):
        super().__init__()
        intermediate_size = intermediate_size or hidden_size // 2

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(intermediate_size, num_labels),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, hidden_size] 或 [batch, seq_len, hidden_size]
        Returns:
            logits: [batch, num_labels] 或 [batch, seq_len, num_labels]
        """
        return self.mlp(hidden_states)


class Pooler(nn.Module):
    """
    池化层
    将序列表示池化为单个向量
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            pooled: [batch, hidden_size]
        """
        # 取第一个token (通常是[CLS])
        first_token = hidden_states[:, 0]
        pooled = self.dense(first_token)
        return self.activation(pooled)


class SequenceSummary(nn.Module):
    """
    序列摘要层
    支持多种池化方式
    """

    def __init__(
        self,
        hidden_size: int,
        summary_type: str = "last",
        summary_dropout: float = 0.1,
    ):
        super().__init__()
        self.summary_type = summary_type
        self.dropout = nn.Dropout(summary_dropout)

        if summary_type == "first":
            self.summary = self._first
        elif summary_type == "last":
            self.summary = self._last
        elif summary_type == "mean":
            self.summary = self._mean
        elif summary_type == "cls":
            self.summary = self._cls
        else:
            raise ValueError(f"Unknown summary type: {summary_type}")

    def _first(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """取第一个token"""
        return hidden_states[:, 0]

    def _last(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """取最后一个token"""
        return hidden_states[:, -1]

    def _mean(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """平均池化"""
        return hidden_states.mean(dim=1)

    def _cls(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """取[CLS] token (同first)"""
        return hidden_states[:, 0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
        Returns:
            summary: [batch, hidden_size]
        """
        summary = self.summary(hidden_states)
        return self.dropout(summary)
