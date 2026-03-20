"""
GPU训练器
支持混合精度、梯度累积、显存优化等
"""
import os
import sys
import time
import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, Tuple, List, Callable, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class TrainingInterrupted(Exception):
    """训练被手动中断的异常"""
    pass


@dataclass
class GPUTrainingConfig:
    """GPU训练配置"""
    # 基础训练参数
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # 学习率参数
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # 混合精度
    use_amp: bool = True  # 自动混合精度
    fp16: bool = False    # FP16
    bf16: bool = True     # BF16 (推荐用于RTX 40系列)

    # 显存优化
    gradient_checkpointing: bool = False
    optim_cpu_offload: bool = False  # 优化器CPU卸载

    # Flash Attention
    use_flash_attention: bool = True

    # 数据加载
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2

    # 日志和保存
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500

    # 其他
    seed: int = 42
    resume_from_checkpoint: str = None
    max_steps: int = -1

    def __post_init__(self):
        # 自动检测BF16支持
        if self.bf16 and torch.cuda.is_available():
            device_capability = torch.cuda.get_device_capability()
            # RTX 40系列 (Ada Lovelace) 支持BF16
            if device_capability[0] >= 8:
                self.bf16 = True
                self.fp16 = False
            else:
                # 旧显卡用FP16
                self.bf16 = False
                self.fp16 = True


class GPUTrainer:
    """
    GPU优化训练器
    针对RTX 4060Ti 8G优化
    """

    def __init__(
        self,
        model: nn.Module,
        config: GPUTrainingConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Any = None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        if torch.cuda.is_available():
            # 打印GPU信息
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")

        # 将模型移到GPU
        self.model = self.model.to(self.device)

        # 混合精度
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.fp16 = config.fp16
        self.bf16 = config.bf16

        # GradScaler用于FP16
        self.scaler = GradScaler() if self.fp16 and self.use_amp else None

        # 优化器
        self.optimizer = self._create_optimizer()

        # 学习率调度器
        self.scheduler = None

        # 数据加载器
        self.train_dataloader = self._create_dataloader(train_dataset, shuffle=True)
        self.eval_dataloader = None
        if eval_dataset:
            self.eval_dataloader = self._create_dataloader(eval_dataset, shuffle=False)

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")

        # 显存监控
        self.memory_stats = []

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """创建优化器"""
        # 分离权重衰减参数
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )

        return optimizer

    def _create_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers if self.device.type == "cuda" else 0,
            pin_memory=self.config.dataloader_pin_memory if self.device.type == "cuda" else False,
            prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
            collate_fn=self.collate_fn,
            drop_last=True,
        )

    def _create_scheduler(self, num_training_steps: int):
        """创建学习率调度器"""
        num_warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps > 0
            else int(num_training_steps * self.config.warmup_ratio)
        )

        if self.config.lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
        elif self.config.lr_scheduler_type == "linear":
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
        else:
            self.scheduler = None

        logger.info(f"Created {self.config.lr_scheduler_type} scheduler with {num_warmup_steps} warmup steps")

    def _get_dtype(self) -> torch.dtype:
        """获取混合精度类型"""
        if self.bf16:
            return torch.bfloat16
        elif self.fp16:
            return torch.float16
        else:
            return torch.float32

    def _log_memory(self, prefix: str = ""):
        """记录显存使用"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            stats = {
                "step": self.global_step,
                "allocated_gb": allocated,
                "reserved_gb": reserved,
                "max_allocated_gb": max_allocated,
            }
            self.memory_stats.append(stats)

            if prefix:
                logger.info(f"{prefix} - Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    def _clear_cache(self):
        """清理GPU缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _setup_signal_handlers(self):
        """设置信号处理器，支持手动终止"""
        self.interrupted = False

        def signal_handler(signum, frame):
            logger.warning("\n" + "=" * 50)
            logger.warning("Received interrupt signal! Saving model...")
            logger.warning("=" * 50)
            self.interrupted = True

        # 注册信号处理器 (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)
        # Windows下也支持Ctrl+Break
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def _check_interrupted(self):
        """检查是否被中断"""
        if self.interrupted:
            raise TrainingInterrupted("Training interrupted by user")

    def train(self) -> Dict[str, Any]:
        """
        主训练循环
        """
        # 设置信号处理器
        self._setup_signal_handlers()

        # 计算总步数
        num_training_steps = len(self.train_dataloader) * self.config.num_train_epochs
        if self.config.max_steps > 0:
            num_training_steps = min(num_training_steps, self.config.max_steps)

        self._create_scheduler(num_training_steps)

        logger.info("=" * 50)
        logger.info("Starting GPU Training")
        logger.info(f"  Num epochs: {self.config.num_train_epochs}")
        logger.info(f"  Batch size: {self.config.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        logger.info(f"  Total steps: {num_training_steps}")
        logger.info(f"  Mixed precision: {'BF16' if self.bf16 else 'FP16' if self.fp16 else 'FP32'}")
        logger.info("=" * 50)

        self._log_memory("Before training")

        # 训练循环
        self.model.train()
        total_loss = 0
        logging_loss = 0
        start_time = time.time()

        try:
            for epoch in range(self.config.num_train_epochs):
                self.epoch = epoch
                epoch_loss = 0

                for step, batch in enumerate(self.train_dataloader):
                    # 检查中断
                    self._check_interrupted()

                    # 移动数据到GPU
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

                    # 混合精度前向传播
                    with autocast(
                        device_type="cuda",
                        dtype=self._get_dtype(),
                        enabled=self.use_amp,
                    ):
                        outputs = self.model(
                            input_ids=batch.get("input_ids"),
                            attention_mask=batch.get("attention_mask"),
                            labels=batch.get("labels"),
                        )
                        loss = outputs["loss"] / self.config.gradient_accumulation_steps

                    # 反向传播
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    total_loss += loss.item()
                    logging_loss += loss.item()

                    # 梯度累积
                    if (step + 1) % self.config.gradient_accumulation_steps == 0:
                        # 梯度裁剪
                        if self.scaler:
                            self.scaler.unscale_(self.optimizer)

                        if self.config.max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.max_grad_norm,
                            )

                        # 优化器步进
                        if self.scaler:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            self.optimizer.step()

                        # 学习率调度
                        if self.scheduler:
                            self.scheduler.step()

                        self.optimizer.zero_grad()
                        self.global_step += 1

                        # 日志记录
                        if self.global_step % self.config.logging_steps == 0:
                            avg_loss = logging_loss / self.config.logging_steps
                            lr = self.optimizer.param_groups[0]["lr"]
                            elapsed = time.time() - start_time

                            logger.info(
                                f"Epoch {epoch+1}/{self.config.num_train_epochs} | "
                                f"Step {self.global_step}/{num_training_steps} | "
                                f"Loss: {avg_loss:.4f} | "
                                f"LR: {lr:.2e} | "
                                f"Time: {elapsed:.1f}s"
                            )

                            self._log_memory(f"Step {self.global_step}")
                            logging_loss = 0

                        # 保存检查点
                        if self.global_step % self.config.save_steps == 0:
                            self._save_checkpoint()

                        # 评估
                        if self.eval_dataloader and self.global_step % self.config.eval_steps == 0:
                            eval_loss = self.evaluate()
                            logger.info(f"Evaluation loss: {eval_loss:.4f}")

                    # 检查最大步数
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        break

                epoch_loss = total_loss / (step + 1)
                logger.info(f"Epoch {epoch+1} completed. Average loss: {epoch_loss:.4f}")

                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break

        except TrainingInterrupted:
            logger.warning("Training interrupted! Saving model...")

        finally:
            # 保存模型（无论正常结束还是中断）
            logger.info("Saving model...")
            self._save_final_model()
            logger.info("Model saved successfully!")

        total_time = time.time() - start_time

        if self.interrupted:
            logger.info(f"Training stopped at step {self.global_step}")
            logger.info(f"Time elapsed: {total_time/3600:.2f} hours")
        else:
            logger.info(f"Training completed in {total_time/3600:.2f} hours")

        return {
            "train_loss": total_loss / self.global_step if self.global_step > 0 else 0,
            "global_step": self.global_step,
            "training_time": total_time,
            "interrupted": self.interrupted,
        }

    @torch.no_grad()
    def evaluate(self) -> float:
        """评估"""
        self.model.eval()
        total_loss = 0

        for batch in self.eval_dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            with autocast(
                device_type="cuda",
                dtype=self._get_dtype(),
                enabled=self.use_amp,
            ):
                outputs = self.model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
                total_loss += outputs["loss"].item()

        avg_loss = total_loss / len(self.eval_dataloader)
        self.model.train()
        return avg_loss

    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            f"checkpoint-{self.global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 保存模型
        torch.save(
            self.model.state_dict(),
            os.path.join(checkpoint_dir, "pytorch_model.bin"),
        )

        # 保存优化器
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(checkpoint_dir, "optimizer.bin"),
        )

        # 保存训练状态
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_loss": self.best_loss,
        }
        torch.save(state, os.path.join(checkpoint_dir, "trainer_state.json"))

        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        self._clear_cache()

    def _save_final_model(self):
        """保存最终模型"""
        output_dir = os.path.join(self.config.output_dir, "final_model")
        os.makedirs(output_dir, exist_ok=True)

        # 保存模型
        torch.save(
            self.model.state_dict(),
            os.path.join(output_dir, "pytorch_model.bin"),
        )

        # 保存配置
        if hasattr(self.model, "config"):
            import json
            config_dict = self.model.config.to_dict() if hasattr(self.model.config, "to_dict") else vars(self.model.config)
            with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

        # 保存tokenizer
        if self.tokenizer and hasattr(self.tokenizer, "save"):
            self.tokenizer.save(output_dir, format="both")

        logger.info(f"Final model saved to {output_dir}")


def get_gpu_memory_info() -> Dict[str, float]:
    """获取GPU显存信息"""
    if not torch.cuda.is_available():
        return {"available": False}

    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(0),
        "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
        "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        "free_gb": (torch.cuda.get_device_properties(0).total_memory -
                   torch.cuda.memory_allocated()) / 1024**3,
    }


def estimate_memory_requirements(
    model_params: int,
    batch_size: int,
    seq_length: int,
    hidden_size: int,
    precision: str = "bf16",
) -> Dict[str, float]:
    """
    估算显存需求

    Args:
        model_params: 模型参数量
        batch_size: 批次大小
        seq_length: 序列长度
        hidden_size: 隐藏层大小
        precision: 精度 ("fp32", "fp16", "bf16")
    """
    # 精度因子
    precision_bytes = {"fp32": 4, "fp16": 2, "bf16": 2}.get(precision, 2)

    # 模型参数
    model_memory = model_params * precision_bytes / 1024**3

    # 梯度 (与参数相同)
    gradient_memory = model_memory

    # 优化器状态 (AdamW: 2倍参数)
    optimizer_memory = model_params * 8 / 1024**3  # fp32的momentum和variance

    # 激活值 (粗略估计)
    # 每层的激活大约是 batch * seq * hidden * num_layers
    activation_memory = batch_size * seq_length * hidden_size * 12 * precision_bytes / 1024**3

    total = model_memory + gradient_memory + optimizer_memory + activation_memory

    return {
        "model_gb": model_memory,
        "gradient_gb": gradient_memory,
        "optimizer_gb": optimizer_memory,
        "activation_gb": activation_memory,
        "total_gb": total,
        "recommended_gpu_gb": total * 1.3,  # 30% 余量
    }
