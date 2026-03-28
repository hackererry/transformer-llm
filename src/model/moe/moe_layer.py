"""
DeepSeek-V3 风格的 MoE 层
支持共享专家 + Top-K 路由专家
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

from .expert import SwiGLUExpert, SharedExpert
from .router import TopKRouter
from .load_balancing import compute_moe_aux_loss


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-V3 风格的 MoE 层

    特点:
    1. 共享专家：始终激活，处理通用知识
    2. 路由专家：Top-K 选择，处理特定领域
    3. 路由缩放因子：可配置路由输出的权重

    参考:
    - DeepSeek-V2: https://arxiv.org/abs/2405.04434
    - DeepSeek-V3: https://arxiv.org/abs/2412.19437
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        num_shared_experts: int = 1,
        top_k: int = 2,
        hidden_dropout: float = 0.1,
        router_noise_std: float = 0.1,
        routed_scaling_factor: float = 1.0,
        aux_loss_alpha: float = 0.01,
        capacity_factor: Optional[float] = None,
    ):
        """
        Args:
            hidden_size: 隐藏层维度
            intermediate_size: 专家的 FFN 中间层维度
            num_experts: 路由专家数量
            num_shared_experts: 共享专家数量
            top_k: 每个 token 激活的专家数
            hidden_dropout: Dropout 概率
            router_noise_std: 路由噪声标准差
            routed_scaling_factor: 路由输出缩放因子
            aux_loss_alpha: 负载均衡损失系数
            capacity_factor: 专家容量因子（None 表示不限制）
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.top_k = top_k
        self.routed_scaling_factor = routed_scaling_factor
        self.aux_loss_alpha = aux_loss_alpha

        # 共享专家（始终激活）
        # 共享专家使用较大的 intermediate_size，通常为 routed experts 的总和
        shared_intermediate_size = intermediate_size * num_shared_experts
        self.shared_experts = nn.ModuleList([
            SharedExpert(hidden_size, shared_intermediate_size, hidden_dropout)
            for _ in range(num_shared_experts)
        ])

        # 路由专家
        self.routed_experts = nn.ModuleList([
            SwiGLUExpert(hidden_size, intermediate_size, hidden_dropout)
            for _ in range(num_experts)
        ])

        # 路由器
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=router_noise_std,
        )

        # 专家容量（用于负载均衡）
        self.capacity_factor = capacity_factor

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        前向传播

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]

        Returns:
            output: [batch_size, seq_len, hidden_size]
            aux_outputs: 包含 router_logits 用于计算辅助损失
        """
        batch_size, seq_len, hidden_size = hidden_states.shape

        # 1. 计算共享专家输出（始终激活）
        shared_output = torch.zeros_like(hidden_states)
        for shared_expert in self.shared_experts:
            shared_output = shared_output + shared_expert(hidden_states)

        # 2. 路由专家计算
        weights, indices, router_logits = self.router(hidden_states)

        # 3. 计算路由专家输出
        routed_output = self._compute_routed_output(
            hidden_states, weights, indices
        )

        # 4. 组合输出: shared + scaled * routed
        output = shared_output + self.routed_scaling_factor * routed_output

        # 5. 计算辅助损失（用于训练）
        aux_loss = None
        if self.training:
            aux_loss = compute_moe_aux_loss(
                router_logits=router_logits,
                num_experts=self.num_experts,
                top_k=self.top_k,
                aux_loss_alpha=self.aux_loss_alpha,
            )

        aux_outputs = {
            "router_logits": router_logits,
            "aux_loss": aux_loss,
        }

        return output, aux_outputs

    def _compute_routed_output(
        self,
        hidden_states: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算路由专家输出（高效实现）

        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            weights: [num_tokens, top_k]
            indices: [num_tokens, top_k]

        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len

        # 展平输入
        hidden_states_flat = hidden_states.view(num_tokens, hidden_size)

        # 初始化输出
        output = torch.zeros_like(hidden_states_flat)

        # 对每个专家计算
        for expert_idx in range(self.num_experts):
            # 找出所有选择了该专家的位置
            # indices: [num_tokens, top_k]
            expert_mask = (indices == expert_idx)  # [num_tokens, top_k]

            if not expert_mask.any():
                continue

            # 获取该专家需要处理的 token 索引
            token_indices = expert_mask.any(dim=1).nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue

            # 获取这些 token 的输入
            expert_input = hidden_states_flat[token_indices]  # [num_selected, hidden_size]

            # 计算专家输出
            expert_output = self.routed_experts[expert_idx](expert_input)

            # 获取对应的权重
            # 找到每个 token 对该专家的权重（可能有多个 top-k 位置）
            expert_weights = torch.zeros(
                len(token_indices),
                device=hidden_states.device,
                dtype=hidden_states.dtype
            )

            # 对每个 top_k 位置检查
            for k in range(self.top_k):
                selected = (indices[token_indices, k] == expert_idx)
                expert_weights += selected.float() * weights[token_indices, k]

            # 加权求和
            weighted_output = expert_output * expert_weights.unsqueeze(-1)

            # 累加到输出
            output[token_indices] += weighted_output

        # 恢复形状
        output = output.view(batch_size, seq_len, hidden_size)

        return output

    def get_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        计算辅助损失（用于外部调用）

        Args:
            router_logits: [num_tokens, num_experts]

        Returns:
            aux_loss: 标量损失
        """
        return compute_moe_aux_loss(
            router_logits=router_logits,
            num_experts=self.num_experts,
            top_k=self.top_k,
            aux_loss_alpha=self.aux_loss_alpha,
        )


class MoEMLP(nn.Module):
    """
    简化的 MoE MLP 层

    不包含共享专家，仅用于兼容性测试
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dropout: float = 0.1,
        router_noise_std: float = 0.1,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # 专家
        self.experts = nn.ModuleList([
            SwiGLUExpert(hidden_size, intermediate_size, hidden_dropout)
            for _ in range(num_experts)
        ])

        # 路由器
        self.router = TopKRouter(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            noise_std=router_noise_std,
        )

        self.aux_loss_alpha = aux_loss_alpha

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """前向传播"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        num_tokens = batch_size * seq_len

        # 路由
        weights, indices, router_logits = self.router(hidden_states)

        # 计算输出
        output = torch.zeros_like(hidden_states)
        hidden_states_flat = hidden_states.view(num_tokens, hidden_size)

        for expert_idx in range(self.num_experts):
            expert_mask = (indices == expert_idx)
            if not expert_mask.any():
                continue

            token_indices = expert_mask.any(dim=1).nonzero(as_tuple=True)[0]
            if len(token_indices) == 0:
                continue

            expert_input = hidden_states_flat[token_indices]
            expert_output = self.experts[expert_idx](expert_input)

            expert_weights = torch.zeros(len(token_indices), device=hidden_states.device, dtype=hidden_states.dtype)
            for i, token_idx in enumerate(token_indices):
                k_positions = (indices[token_idx] == expert_idx).nonzero(as_tuple=True)[0]
                for k_pos in k_positions:
                    expert_weights[i] += weights[token_idx, k_pos]

            output.view(num_tokens, hidden_size)[token_indices] += expert_output * expert_weights.unsqueeze(-1)

        aux_loss = None
        if self.training:
            aux_loss = compute_moe_aux_loss(
                router_logits, self.num_experts, self.top_k, self.aux_loss_alpha
            )

        return output, {"router_logits": router_logits, "aux_loss": aux_loss}
