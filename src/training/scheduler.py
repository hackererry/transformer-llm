"""
学习率调度器模块
支持多种学习率调度策略
"""
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    LambdaLR,
    CosineAnnealingLR,
    LinearLR,
    ConstantLR,
)
import math
from typing import Optional, List, Union


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 0,
    warmup_ratio: float = 0.0,
    min_lr: float = 0.0,
    **kwargs,
) -> LRScheduler:
    """
    创建学习率调度器的工厂函数

    Args:
        optimizer: 优化器
        scheduler_type: 调度器类型
        num_training_steps: 总训练步数
        num_warmup_steps: 预热步数
        warmup_ratio: 预热比例
        min_lr: 最小学习率
    """
    # 计算warmup步数
    if warmup_ratio > 0:
        num_warmup_steps = int(num_training_steps * warmup_ratio)

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    elif scheduler_type == "constant":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
        )
    elif scheduler_type == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            min_lr=min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    min_lr: float = 0.0,
) -> LambdaLR:
    """
    带预热的余弦退火学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        num_cycles: 余弦周期的数量
        min_lr: 最小学习率比例
    """
    base_lr = optimizer.defaults["lr"]

    def lr_lambda(current_step: int) -> float:
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

        # 确保不低于min_lr
        return max(min_lr / base_lr, cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    带预热的线性衰减学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
    """
    def lr_lambda(current_step: int) -> float:
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 线性衰减阶段
        return max(
            0.0,
            float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            ),
        )

    return LambdaLR(optimizer, lr_lambda)


def get_constant_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
) -> LambdaLR:
    """
    带预热的恒定学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def get_polynomial_decay_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float = 1e-7,
    power: float = 1.0,
) -> LambdaLR:
    """
    带预热的多项式衰减学习率调度

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        lr_end: 最终学习率
        power: 多项式幂次
    """
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int) -> float:
        # 预热阶段
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # 多项式衰减阶段
        if current_step > num_training_steps:
            return lr_end / lr_init

        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining ** power + lr_end
        return decay / lr_init

    return LambdaLR(optimizer, lr_lambda)


class CosineAnnealingWarmRestarts(LRScheduler):
    """
    带热重启的余弦退火
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 2,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: 优化器
            T_0: 第一次重启的周期
            T_mult: 重启后周期增加的倍数
            eta_min: 最小学习率
            last_epoch: 上一个epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        return [
            self.eta_min + (base_lr - self.eta_min) * (
                1 + math.cos(math.pi * self.T_cur / self.T_0)
            ) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_0:
                self.T_cur = 0
                self.T_0 = self.T_0 * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
            self.T_cur = epoch % self.T_0

        super().step(epoch)


class OneCycleLR(LRScheduler):
    """
    1Cycle学习率策略
    先增后减，可加速收敛
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        div_factor: float = 25.0,
        final_div_factor: float = 1e4,
    ):
        """
        Args:
            optimizer: 优化器
            max_lr: 最大学习率
            total_steps: 总步数
            pct_start: 上升阶段的比例
            anneal_strategy: 衰减策略 ("cos" or "linear")
            div_factor: 初始学习率 = max_lr / div_factor
            final_div_factor: 最终学习率 = max_lr / final_div_factor
        """
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor

        self.initial_lr = max_lr / div_factor
        self.final_lr = max_lr / final_div_factor
        self.step_up = int(total_steps * pct_start)

        super().__init__(optimizer)

    def get_lr(self) -> List[float]:
        current_step = self.last_epoch

        if current_step <= self.step_up:
            # 上升阶段
            pct = current_step / self.step_up
            if self.anneal_strategy == "cos":
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (1 - math.cos(math.pi * pct)) / 2
            else:
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * pct
        else:
            # 下降阶段
            pct = (current_step - self.step_up) / (self.total_steps - self.step_up)
            if self.anneal_strategy == "cos":
                lr = self.max_lr - (self.max_lr - self.final_lr) * (1 - math.cos(math.pi * pct)) / 2
            else:
                lr = self.max_lr - (self.max_lr - self.final_lr) * pct

        return [lr for _ in self.base_lrs]


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    **kwargs,
) -> LRScheduler:
    """
    获取学习率调度器的便捷函数
    """
    return create_scheduler(optimizer, scheduler_type, **kwargs)
