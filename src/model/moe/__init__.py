"""
MoE (Mixture of Experts) 模块

实现 DeepSeek-V3 风格的 MoE 层：
- 共享专家：始终激活，处理通用知识
- 路由专家：Top-K 选择，处理特定领域
- 负载均衡：辅助损失促进专家均衡使用
"""

from .expert import SwiGLUExpert, SharedExpert
from .router import TopKRouter
from .moe_layer import DeepSeekMoE, MoEMLP
from .load_balancing import (
    compute_load_balancing_loss,
    compute_z_loss,
    compute_moe_aux_loss,
)

__all__ = [
    "SwiGLUExpert",
    "SharedExpert",
    "TopKRouter",
    "DeepSeekMoE",
    "MoEMLP",
    "compute_load_balancing_loss",
    "compute_z_loss",
    "compute_moe_aux_loss",
]
