"""
MoE 负载均衡损失
实现 GShard 风格的辅助损失，促进专家负载均衡
"""
import torch
import torch.nn.functional as F
from typing import Optional


def compute_load_balancing_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    aux_loss_alpha: float = 0.01,
) -> torch.Tensor:
    """
    计算负载均衡辅助损失 (GShard 风格)

    L_aux = alpha * num_experts * sum(f_i * P_i)
    其中:
    - f_i: 专家 i 被分配的 token 比例
    - P_i: 专家 i 的平均路由概率

    这个损失鼓励所有专家均匀地处理 token

    Args:
        router_logits: [batch, seq, num_experts] 或 [num_tokens, num_experts]
        num_experts: 专家数量
        top_k: 每个 token 选择的专家数
        aux_loss_alpha: 损失系数

    Returns:
        aux_loss: 标量损失
    """
    # 展平输入
    if len(router_logits.shape) == 3:
        router_logits = router_logits.view(-1, num_experts)

    num_tokens = router_logits.shape[0]

    # 计算路由概率
    router_probs = F.softmax(router_logits, dim=-1)  # [num_tokens, num_experts]

    # 计算 Top-K 掩码
    _, top_k_indices = torch.topk(router_probs, top_k, dim=-1)  # [num_tokens, top_k]

    # 创建 one-hot 掩码表示每个 token 选择了哪些专家
    expert_mask = F.one_hot(top_k_indices, num_experts).float()  # [num_tokens, top_k, num_experts]
    expert_mask = expert_mask.sum(dim=1)  # [num_tokens, num_experts]

    # f_i: 每个专家被分配的 token 比例
    tokens_per_expert = expert_mask.mean(dim=0)  # [num_experts]

    # P_i: 每个专家的平均路由概率
    router_prob_per_expert = router_probs.mean(dim=0)  # [num_experts]

    # 负载均衡损失
    aux_loss = aux_loss_alpha * num_experts * torch.sum(
        tokens_per_expert * router_prob_per_expert
    )

    return aux_loss


def compute_z_loss(
    router_logits: torch.Tensor,
    aux_loss_alpha: float = 0.01,
) -> torch.Tensor:
    """
    计算 Z-loss (路由 logits 的平方和)
    用于防止路由 logits 过大

    Args:
        router_logits: [num_tokens, num_experts]
        aux_loss_alpha: 损失系数

    Returns:
        z_loss: 标量损失
    """
    # 展平输入
    if len(router_logits.shape) == 3:
        router_logits = router_logits.view(-1, -1)

    # Z-loss = mean(logit^2)
    z_loss = aux_loss_alpha * torch.mean(router_logits ** 2)

    return z_loss


def compute_moe_aux_loss(
    router_logits: torch.Tensor,
    num_experts: int,
    top_k: int,
    aux_loss_alpha: float = 0.01,
    z_loss_alpha: float = 0.001,
) -> torch.Tensor:
    """
    计算完整的 MoE 辅助损失

    包括：
    1. 负载均衡损失
    2. Z-loss（可选）

    Args:
        router_logits: [num_tokens, num_experts]
        num_experts: 专家数量
        top_k: 每个 token 选择的专家数
        aux_loss_alpha: 负载均衡损失系数
        z_loss_alpha: Z-loss 系数

    Returns:
        total_aux_loss: 总辅助损失
    """
    # 负载均衡损失
    load_balance_loss = compute_load_balancing_loss(
        router_logits, num_experts, top_k, aux_loss_alpha
    )

    # Z-loss
    if z_loss_alpha > 0:
        z_loss = compute_z_loss(router_logits, z_loss_alpha)
        total_aux_loss = load_balance_loss + z_loss
    else:
        total_aux_loss = load_balance_loss

    return total_aux_loss
