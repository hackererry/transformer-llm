"""
MoE 路由器模块
实现 Top-K 路由策略
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class TopKRouter(nn.Module):
    """
    Top-K 路由器

    功能：
    - Top-K 专家选择
    - 训练时添加噪声以促进负载均衡
    - 归一化路由权重

    参考：
    - GShard: https://arxiv.org/abs/2006.16668
    - Switch Transformer: https://arxiv.org/abs/2101.03961
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        routing_bias: bool = False,
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            num_experts: 专家数量
            top_k: 每个 token 选择的专家数量
            noise_std: 路由噪声标准差（训练时用于负载均衡）
            routing_bias: 是否使用路由偏置
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_std = noise_std

        # 路由门控
        self.gate = nn.Linear(hidden_size, num_experts, bias=routing_bias)

        # 初始化权重
        nn.init.xavier_uniform_(self.gate.weight)
        if routing_bias:
            nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size] 或 [num_tokens, hidden_size]

        Returns:
            weights: [num_tokens, top_k] 归一化的专家权重
            indices: [num_tokens, top_k] 选中的专家索引
            router_logits: [num_tokens, num_experts] 路由 logits（用于计算辅助损失）
        """
        # 展平输入以便处理
        original_shape = hidden_states.shape
        if len(original_shape) == 3:
            batch_size, seq_len, hidden_size = original_shape
            hidden_states = hidden_states.view(-1, hidden_size)
        else:
            batch_size = 1
            seq_len = original_shape[0]
            hidden_size = original_shape[1]

        # 计算路由 logits
        router_logits = self.gate(hidden_states)  # [num_tokens, num_experts]

        # 训练时添加噪声以促进负载均衡
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(router_logits) * self.noise_std
            router_logits = router_logits + noise

        # Top-K 选择
        top_k_weights, top_k_indices = torch.topk(
            router_logits, self.top_k, dim=-1
        )  # [num_tokens, top_k]

        # Softmax 归一化（只对选中的专家）
        top_k_weights = F.softmax(top_k_weights, dim=-1)

        return top_k_weights, top_k_indices, router_logits

    def get_router_probs(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        获取所有专家的概率分布（用于负载均衡损失）

        Args:
            router_logits: [num_tokens, num_experts]

        Returns:
            router_probs: [num_tokens, num_experts]
        """
        return F.softmax(router_logits, dim=-1)


class TokenChoiceRouter(nn.Module):
    """
    Token Choice 路由器（每个 token 独立选择专家）

    与 TopKRouter 功能类似，但提供更多控制选项
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        noise_std: float = 0.1,
        capacity_factor: Optional[float] = None,
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            num_experts: 专家数量
            top_k: 每个 token 选择的专家数量
            noise_std: 路由噪声标准差
            capacity_factor: 专家容量因子（限制每个专家处理的 token 数）
        """
        super().__init__()
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=noise_std,
        )
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            weights: [num_tokens, top_k] 专家权重
            indices: [num_tokens, top_k] 专家索引
            router_logits: [num_tokens, num_experts] 路由 logits
        """
        return self.router(hidden_states)

    def compute_capacity(self, num_tokens: int) -> int:
        """
        计算每个专家的容量

        Args:
            num_tokens: token 总数

        Returns:
            capacity: 每个专家最多处理的 token 数
        """
        if self.capacity_factor is None:
            return num_tokens  # 不限制容量

        # capacity = (num_tokens * top_k * capacity_factor) / num_experts
        capacity = int((num_tokens * self.top_k * self.capacity_factor) / self.num_experts)
        return max(capacity, 1)
