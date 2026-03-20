"""
训练器模块
完整的训练循环实现
"""
import os
import time
import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from src.utils import get_device
from typing import Optional, Dict, Any, Callable, Union, Tuple
import math
from tqdm import tqdm

from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .checkpoint import CheckpointManager, save_model


class TrainingInterrupted(Exception):
    """训练被手动中断的异常"""
    pass


class Trainer:
    """
    训练器主类
    处理完整的训练流程
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        output_dir: str = "./output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        per_device_eval_batch_size: int = 4,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        lr_scheduler_type: str = "cosine",
        num_warmup_steps: int = 0,
        warmup_ratio: float = 0.1,
        logging_dir: str = "./logs",
        logging_steps: int = 10,
        save_steps: int = 500,
        save_total_limit: int = 3,
        eval_steps: int = 500,
        bf16: bool = True,
        dataloader_num_workers: int = 0,
        dataloader_drop_last: bool = False,
        seed: int = 42,
        resume_from_checkpoint: Optional[str] = None,
        collate_fn: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
    ):
        """
        Args:
            model: 要训练的模型
            train_dataset: 训练数据集
            eval_dataset: 评估数据集
            tokenizer: tokenizer
            output_dir: 输出目录
            num_train_epochs: 训练轮数
            per_device_train_batch_size: 训练批次大小
            per_device_eval_batch_size: 评估批次大小
            gradient_accumulation_steps: 梯度累积步数
            learning_rate: 学习率
            weight_decay: 权重衰减
            max_grad_norm: 梯度裁剪阈值
            lr_scheduler_type: 学习率调度器类型
            num_warmup_steps: 预热步数
            warmup_ratio: 预热比例
            logging_dir: 日志目录
            logging_steps: 日志记录步数间隔
            save_steps: 保存步数间隔
            save_total_limit: 保存检查点数量限制
            eval_steps: 评估步数间隔
            bf16: 是否使用BF16
            dataloader_num_workers: 数据加载器工作进程数
            dataloader_drop_last: 是否丢弃最后不完整批次
            seed: 随机种子
            resume_from_checkpoint: 恢复训练的检查点路径
            collate_fn: 数据整理函数
            compute_metrics: 指标计算函数
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # 训练配置
        self.output_dir = output_dir
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.lr_scheduler_type = lr_scheduler_type
        self.num_warmup_steps = num_warmup_steps
        self.warmup_ratio = warmup_ratio

        # 日志和保存配置
        self.logging_dir = logging_dir
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.save_total_limit = save_total_limit
        self.eval_steps = eval_steps

        # 精度配置
        self.bf16 = bf16

        # 数据加载配置
        self.dataloader_num_workers = dataloader_num_workers
        self.dataloader_drop_last = dataloader_drop_last

        # 其他配置
        self.seed = seed
        self.resume_from_checkpoint = resume_from_checkpoint
        self.collate_fn = collate_fn
        self.compute_metrics = compute_metrics

        # 设备 - 自动检测GPU/CPU
        self.device = get_device()
        self.model = self.model.to(self.device)
        print(f"Using device: {self.device}")

        # 初始化
        self._setup()

    def _setup(self):
        """初始化设置"""
        # 设置随机种子
        torch.manual_seed(self.seed)

        # 创建数据加载器
        self.train_dataloader = self._create_dataloader(
            self.train_dataset,
            self.per_device_train_batch_size,
            shuffle=True,
        )

        if self.eval_dataset is not None:
            self.eval_dataloader = self._create_dataloader(
                self.eval_dataset,
                self.per_device_eval_batch_size,
                shuffle=False,
            )

        # 创建优化器
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # 计算训练步数
        self.num_training_steps = len(self.train_dataloader) * self.num_train_epochs

        # 计算预热步数
        if self.warmup_ratio > 0:
            self.num_warmup_steps = int(self.num_training_steps * self.warmup_ratio)

        # 创建学习率调度器
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=self.lr_scheduler_type,
            num_training_steps=self.num_training_steps,
            num_warmup_steps=self.num_warmup_steps,
        )

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            self.output_dir,
            max_checkpoints=self.save_total_limit,
        )

        # 混合精度
        self.use_amp = self.bf16 and hasattr(torch, "bfloat16")
        if self.use_amp:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logging_dir, exist_ok=True)

    def _create_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
    ) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.dataloader_num_workers,
            collate_fn=self.collate_fn,
            drop_last=self.dataloader_drop_last,
            pin_memory=False,  # CPU训练不需要pin_memory
        )

    def train(self) -> Dict[str, float]:
        """
        执行训练

        Returns:
            训练结果字典
        """
        # 设置信号处理器
        self.interrupted = False
        self._setup_signal_handlers()

        # 恢复训练
        start_epoch = 0
        start_step = 0
        global_step = 0

        if self.resume_from_checkpoint:
            state = self.checkpoint_manager.load(
                self.resume_from_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler,
            )
            start_epoch = state.get("epoch", 0)
            start_step = state.get("step", 0)
            global_step = start_step

        # 训练循环
        total_loss = 0.0
        best_eval_loss = float("inf")
        avg_epoch_loss = 0.0

        print(f"Starting training from epoch {start_epoch}, step {start_step}")
        print(f"Total training steps: {self.num_training_steps}")

        try:
            for epoch in range(start_epoch, self.num_train_epochs):
                self.model.train()
                epoch_loss = 0.0
                epoch_steps = 0

                progress_bar = tqdm(
                    enumerate(self.train_dataloader),
                    total=len(self.train_dataloader),
                    desc=f"Epoch {epoch + 1}/{self.num_train_epochs}",
                )

                for step, batch in progress_bar:
                    # 检查中断
                    if self.interrupted:
                        break

                    # 跳过已训练的步数
                    if epoch == start_epoch and step < start_step % len(self.train_dataloader):
                        continue

                    # 准备输入
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    # 前向传播
                    with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                        outputs = self.model(**batch)
                        loss = outputs["loss"] / self.gradient_accumulation_steps

                    # 反向传播
                    loss.backward()

                    total_loss += loss.item()
                    epoch_loss += loss.item()

                    # 梯度累积
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        if self.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.max_grad_norm,
                            )

                        # 更新参数
                        self.optimizer.step()
                        self.scheduler.step()
                        self.optimizer.zero_grad()

                        global_step += 1
                        epoch_steps += 1

                        # 日志记录
                        if global_step % self.logging_steps == 0:
                            avg_loss = total_loss / self.logging_steps
                            lr = self.scheduler.get_last_lr()[0]
                            progress_bar.set_postfix(
                                loss=f"{avg_loss:.4f}",
                                lr=f"{lr:.2e}",
                            )
                            self._log({"loss": avg_loss, "learning_rate": lr, "step": global_step})
                            total_loss = 0.0

                        # 保存检查点
                        if global_step % self.save_steps == 0:
                            self._save_checkpoint(epoch, global_step)

                        # 评估
                        if self.eval_dataset is not None and global_step % self.eval_steps == 0:
                            eval_results = self.evaluate()
                            if eval_results["loss"] < best_eval_loss:
                                best_eval_loss = eval_results["loss"]
                                self._save_checkpoint(epoch, global_step, is_best=True)

                # 每个epoch结束后的平均损失
                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

                if self.interrupted:
                    break

        except Exception as e:
            print(f"Training interrupted: {e}")

        finally:
            # 无论是否中断都保存模型
            print("\nSaving model...")
            self._save_final_model()
            if self.interrupted:
                print(f"Training stopped at step {global_step}")
            else:
                print("Training completed successfully!")

        return {"final_loss": avg_epoch_loss, "best_eval_loss": best_eval_loss, "global_step": global_step}

    def _setup_signal_handlers(self):
        """设置信号处理器，支持手动终止"""
        def signal_handler(signum, frame):
            print("\n" + "=" * 50)
            print("Received interrupt signal! Will save model after current step...")
            print("=" * 50)
            self.interrupted = True

        # 注册信号处理器 (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)
        # Windows下也支持Ctrl+Break
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def evaluate(self) -> Dict[str, float]:
        """
        执行评估

        Returns:
            评估结果字典
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]

                total_loss += loss.item() * batch["input_ids"].size(0)
                total_samples += batch["input_ids"].size(0)

        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        results = {"loss": avg_loss, "perplexity": perplexity}

        print(f"Evaluation results: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

        # 计算额外指标
        if self.compute_metrics:
            additional_metrics = self.compute_metrics(results)
            results.update(additional_metrics)

        self._log(results)

        return results

    def _save_checkpoint(self, epoch: int, step: int, is_best: bool = False):
        """保存检查点"""
        metrics = {"epoch": epoch, "step": step}
        self.checkpoint_manager.save(
            self.model,
            self.optimizer,
            self.scheduler,
            step=step,
            epoch=epoch,
            metrics=metrics,
            is_best=is_best,
        )
        print(f"Checkpoint saved at step {step}")

    def _save_final_model(self):
        """保存最终模型"""
        save_path = os.path.join(self.output_dir, "final_model")
        save_model(self.model, save_path)
        print(f"Final model saved to {save_path}")

    def _log(self, metrics: Dict[str, float]):
        """记录日志"""
        log_file = os.path.join(self.logging_dir, "training_log.txt")
        with open(log_file, "a") as f:
            log_line = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            f.write(f"{log_line}\n")


class TrainerState:
    """
    训练状态类
    用于跟踪和恢复训练状态
    """

    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.learning_rate = None
        self.loss_history = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "learning_rate": self.learning_rate,
            "loss_history": self.loss_history,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "TrainerState":
        state = cls()
        state.epoch = state_dict.get("epoch", 0)
        state.global_step = state_dict.get("global_step", 0)
        state.best_metric = state_dict.get("best_metric")
        state.learning_rate = state_dict.get("learning_rate")
        state.loss_history = state_dict.get("loss_history", [])
        return state
