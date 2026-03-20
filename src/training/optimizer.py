"""
优化器模块
支持多种优化器配置
"""
import torch
from torch.optim import Optimizer, AdamW, Adam, SGD
from typing import Optional, List, Union, Callable
import math


def create_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    momentum: float = 0.9,
    **kwargs,
) -> Optimizer:
    """
    创建优化器的工厂函数

    Args:
        model: 要优化的模型
        optimizer_type: 优化器类型 (adamw, adam, sgd)
        learning_rate: 学习率
        weight_decay: 权重衰减
        beta1, beta2: Adam的beta参数
        eps: Adam的epsilon参数
        momentum: SGD的动量参数
    """
    # 分离需要权重衰减和不需要权重衰减的参数
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # bias和LayerNorm权重不进行权重衰减
        if "bias" in name or "norm" in name.lower() or "layernorm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if optimizer_type.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_type.lower() == "adam":
        optimizer = Adam(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps,
        )
    elif optimizer_type.lower() == "sgd":
        optimizer = SGD(
            optimizer_grouped_parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    return optimizer


class AdamWOptimizer(Optimizer):
    """
    自定义AdamW优化器
    实现解耦权重衰减
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        执行一步优化

        Args:
            closure: 计算loss的闭包函数
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # 偏差校正
                if group["correct_bias"]:
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                else:
                    bias_correction1 = bias_correction2 = 1.0

                # 更新动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 计算步长
                step_size = group["lr"] / bias_correction1

                # 更新参数
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                p.addcdiv_(exp_avg, denom, value=-step_size)

                # 权重衰减
                if group["weight_decay"] > 0:
                    p.add_(p, alpha=-group["lr"] * group["weight_decay"])

        return loss


class LAMB(Optimizer):
    """
    LAMB优化器
    适用于大batch训练
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        correct_bias: bool = True,
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "correct_bias": correct_bias,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Adam更新
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                adam_step = exp_avg / bias_correction1 / (exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + group["eps"])

                # 权重衰减
                if group["weight_decay"] > 0:
                    adam_step.add_(p, alpha=group["weight_decay"])

                # LAMB信任比率
                weight_norm = p.norm()
                adam_norm = adam_step.norm()

                if weight_norm > 0 and adam_norm > 0:
                    trust_ratio = weight_norm / adam_norm
                else:
                    trust_ratio = 1.0

                p.add_(adam_step, alpha=-group["lr"] * trust_ratio)

        return loss


def get_optimizer(
    model: torch.nn.Module,
    optimizer_type: str = "adamw",
    **kwargs,
) -> Optimizer:
    """
    获取优化器的便捷函数
    """
    return create_optimizer(model, optimizer_type, **kwargs)
