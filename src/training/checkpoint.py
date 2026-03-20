"""
检查点管理模块
支持模型保存、加载和恢复训练
"""
import os
import json
import torch
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import glob
import shutil


class CheckpointManager:
    """
    检查点管理器
    处理模型保存、加载和自动清理
    """

    def __init__(
        self,
        output_dir: str,
        max_checkpoints: int = 3,
        save_best_only: bool = False,
        metric_name: str = "loss",
        metric_mode: str = "min",
    ):
        """
        Args:
            output_dir: 检查点保存目录
            max_checkpoints: 最多保存的检查点数量
            save_best_only: 是否只保存最佳模型
            metric_name: 用于判断最佳模型的指标名称
            metric_mode: "min"或"max"，指标越小/越大越好
        """
        self.output_dir = output_dir
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.metric_mode = metric_mode

        self.best_metric = float("inf") if metric_mode == "min" else float("-inf")
        self.checkpoints: List[str] = []

        os.makedirs(output_dir, exist_ok=True)

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """
        保存检查点

        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            step: 当前步数
            epoch: 当前轮数
            metrics: 评估指标字典
            extra_state: 额外状态
            is_best: 是否是最佳模型

        Returns:
            保存的检查点路径
        """
        metrics = metrics or {}

        # 检查是否是最佳模型
        if self.metric_name in metrics:
            current_metric = metrics[self.metric_name]
            if self.metric_mode == "min":
                is_best = current_metric < self.best_metric
                if is_best:
                    self.best_metric = current_metric
            else:
                is_best = current_metric > self.best_metric
                if is_best:
                    self.best_metric = current_metric

        # 构建状态字典
        state = {
            "step": step,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }

        if scheduler is not None:
            state["scheduler_state_dict"] = scheduler.state_dict()

        if extra_state:
            state["extra_state"] = extra_state

        # 保存检查点
        checkpoint_name = f"checkpoint-step-{step}"
        checkpoint_path = os.path.join(self.output_dir, checkpoint_name)

        torch.save(state, f"{checkpoint_path}.pt")

        # 保存模型配置
        if hasattr(model, "config"):
            config_path = f"{checkpoint_path}-config.json"
            with open(config_path, "w") as f:
                json.dump(model.config.to_dict() if hasattr(model.config, "to_dict") else model.config, f, indent=2)

        self.checkpoints.append(checkpoint_path)

        # 如果是最佳模型，额外保存一份
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model")
            shutil.copy(f"{checkpoint_path}.pt", f"{best_path}.pt")
            if hasattr(model, "config"):
                shutil.copy(f"{checkpoint_path}-config.json", f"{best_path}-config.json")

        # 清理旧检查点
        if not self.save_best_only:
            self._cleanup_old_checkpoints()

        return checkpoint_path

    def load(
        self,
        checkpoint_path: str,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        load_best: bool = False,
    ) -> Dict[str, Any]:
        """
        加载检查点

        Args:
            checkpoint_path: 检查点路径或目录
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            load_best: 是否加载最佳模型

        Returns:
            加载的状态字典
        """
        if load_best or checkpoint_path == "best":
            checkpoint_path = os.path.join(self.output_dir, "best_model.pt")
        elif os.path.isdir(checkpoint_path):
            # 如果是目录，找最新的检查点
            checkpoints = glob.glob(os.path.join(checkpoint_path, "checkpoint-step-*.pt"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=os.path.getctime)
            else:
                raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        state = torch.load(checkpoint_path, map_location="cpu")

        # 加载模型权重
        model.load_state_dict(state["model_state_dict"])

        # 加载优化器状态
        if optimizer is not None and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])

        # 加载调度器状态
        if scheduler is not None and "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])

        return state

    def load_latest(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        加载最新的检查点

        Returns:
            加载的状态字典，如果没有检查点则返回None
        """
        checkpoints = glob.glob(os.path.join(self.output_dir, "checkpoint-step-*.pt"))
        if not checkpoints:
            return None

        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1].replace(".pt", "")))
        return self.load(latest_checkpoint, model, optimizer, scheduler)

    def _cleanup_old_checkpoints(self):
        """清理旧检查点，只保留最新的几个"""
        while len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if os.path.exists(f"{old_checkpoint}.pt"):
                os.remove(f"{old_checkpoint}.pt")
            if os.path.exists(f"{old_checkpoint}-config.json"):
                os.remove(f"{old_checkpoint}-config.json")

    def list_checkpoints(self) -> List[str]:
        """列出所有检查点"""
        return glob.glob(os.path.join(self.output_dir, "checkpoint-step-*.pt"))

    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """获取检查点信息"""
        state = torch.load(checkpoint_path, map_location="cpu")
        return {
            "step": state.get("step"),
            "epoch": state.get("epoch"),
            "metrics": state.get("metrics"),
            "timestamp": state.get("timestamp"),
        }


def save_model(
    model: torch.nn.Module,
    output_path: str,
    save_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
):
    """
    保存模型

    Args:
        model: 模型
        output_path: 输出路径
        save_optimizer: 是否保存优化器
        optimizer: 优化器
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    state = {"model_state_dict": model.state_dict()}

    if save_optimizer and optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(state, f"{output_path}.pt")

    # 保存配置
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        with open(f"{output_path}-config.json", "w") as f:
            json.dump(config, f, indent=2)


def load_model(
    model: torch.nn.Module,
    checkpoint_path: str,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    加载模型

    Args:
        model: 模型
        checkpoint_path: 检查点路径
        load_optimizer: 是否加载优化器
        optimizer: 优化器
        strict: 是否严格匹配权重

    Returns:
        加载的状态字典
    """
    state = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(state["model_state_dict"], strict=strict)

    if load_optimizer and optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state


def save_pretrained(
    model: torch.nn.Module,
    output_dir: str,
    tokenizer=None,
    tokenizer_format: str = "both",
):
    """
    以HuggingFace格式保存模型

    Args:
        model: 模型
        output_dir: 输出目录
        tokenizer: tokenizer (可选)
        tokenizer_format: tokenizer保存格式
            - "legacy": 仅旧版格式 (vocab.json + merges.txt + tokenizer_config.json)
            - "unified": 仅新版一站式格式 (tokenizer.json)
            - "both": 同时保存两种格式（默认）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存模型权重
    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # 保存配置
    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "to_dict"):
            config = config.to_dict()
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    # 保存tokenizer
    if tokenizer is not None:
        if hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(output_dir)
        elif hasattr(tokenizer, "save"):
            # 支持新版的format参数
            try:
                tokenizer.save(output_dir, format=tokenizer_format)
            except TypeError:
                # 兼容旧版save方法
                tokenizer.save(output_dir)
