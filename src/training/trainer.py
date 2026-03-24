"""
统一训练器模块
支持CPU和GPU训练，自动检测最佳设备
"""
import os
import time
import signal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.cuda.amp import GradScaler
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, field
import math
from tqdm import tqdm
from contextlib import contextmanager

from .optimizer import create_optimizer
from .scheduler import create_scheduler
from .checkpoint import CheckpointManager, save_model


@dataclass
class TrainingConfig:
    """统一训练配置"""
    # 基础参数
    output_dir: str = "./output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    max_steps: int = -1

    # 学习率
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    warmup_steps: int = 0

    # 精度和优化
    bf16: bool = False  # 自动检测
    fp16: bool = False
    gradient_checkpointing: bool = False

    # GPU优化（仅GPU有效）
    use_flash_attention: bool = False
    dataloader_num_workers: int = 0
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = 2
    dataloader_drop_last: bool = True

    # 日志和保存
    logging_dir: str = "./logs"
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    eval_steps: int = 500

    # 其他
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    compute_metrics: Optional[Callable] = None

    # 内部状态
    _device_type: str = field(default="cpu", init=False, repr=False)

    def __post_init__(self):
        """自动检测最佳精度"""
        if torch.cuda.is_available():
            self._device_type = "cuda"
            # RTX 40系列 (Ampere+) 支持 BF16
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                self.bf16 = True
                self.fp16 = False
            else:
                self.bf16 = False
                self.fp16 = True
        else:
            self._device_type = "cpu"
            self.bf16 = False
            self.fp16 = False


class PerformanceMonitor:
    """性能监控器 - 记录各阶段耗时"""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.reset()

    def reset(self):
        self.timings: Dict[str, List[float]] = {
            "data_loading": [],
            "forward": [],
            "backward": [],
            "optimizer_step": [],
            "batch_total": [],
        }
        self.step_count = 0

    @contextmanager
    def measure(self, stage: str):
        if not self.enabled:
            yield
            return
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.timings[stage].append(elapsed)

    def record_batch(self, batch_time: float):
        if self.enabled:
            self.timings["batch_total"].append(batch_time)

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for stage, times in self.timings.items():
            if len(times) > 0:
                summary[stage] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
            else:
                summary[stage] = {"count": 0, "total": 0, "mean": 0, "min": 0, "max": 0}
        return summary

    def print_summary(self):
        summary = self.get_summary()
        print("\n" + "=" * 70)
        print("                      Performance Summary")
        print("=" * 70)
        print(f"{'Stage':<20} {'Count':>8} {'Total(s)':>12} {'Mean(ms)':>12} {'Min(ms)':>12} {'Max(ms)':>12}")
        print("-" * 70)

        for stage, stats in summary.items():
            if stats["count"] > 0:
                stage_name = stage.replace("_", " ").title()
                print(f"{stage_name:<20} {stats['count']:>8} {stats['total']:>12.3f} {stats['mean']*1000:>12.2f} {stats['min']*1000:>12.2f} {stats['max']*1000:>12.2f}")

        print("-" * 70)
        total_time = sum(s["total"] for s in summary.values())
        if total_time > 0:
            print(f"\nTime Breakdown:")
            for stage, stats in summary.items():
                if stats["total"] > 0:
                    pct = stats["total"] / total_time * 100
                    stage_name = stage.replace("_", " ").title()
                    print(f"  {stage_name:<18}: {pct:>6.2f}%  ({stats['total']:.3f}s)")
        print("=" * 70 + "\n")

    def log_step(self, step: int, global_step: int, loss: float, lr: float):
        if not self.enabled or self.step_count == 0:
            return
        summary = self.get_summary()
        recent = {
            "data_loading": summary["data_loading"]["mean"] if summary["data_loading"]["count"] > 0 else 0,
            "forward": summary["forward"]["mean"] if summary["forward"]["count"] > 0 else 0,
            "backward": summary["backward"]["mean"] if summary["backward"]["count"] > 0 else 0,
            "batch": summary["batch_total"]["mean"] if summary["batch_total"]["count"] > 0 else 0,
        }
        print(f"[Step {global_step}] Loss: {loss:.4f} | LR: {lr:.2e} | "
              f"Data: {recent['data_loading']*1000:.1f}ms | "
              f"Fwd: {recent['forward']*1000:.1f}ms | "
              f"Bwd: {recent['backward']*1000:.1f}ms | "
              f"Total: {recent['batch']*1000:.1f}ms")


class TrainingInterrupted(Exception):
    """训练被手动中断的异常"""
    pass


class Trainer:
    """
    统一训练器
    自动检测GPU/CPU并使用最佳配置
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        config: TrainingConfig,
        eval_dataset: Optional[Dataset] = None,
        tokenizer=None,
        collate_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.collate_fn = collate_fn
        self.config = config

        # 自动设备检测
        self.device = self._setup_device()
        self.is_gpu = self.device.type == "cuda"

        # 将模型移到设备
        self.model = self.model.to(self.device)

        # 混合精度设置
        self.use_amp = self.is_gpu and (config.bf16 or config.fp16)
        if config.bf16 and self.is_gpu:
            self.dtype = torch.bfloat16
        elif config.fp16 and self.is_gpu:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # GradScaler for FP16
        self.scaler = GradScaler() if self.use_amp and config.fp16 else None

        # 性能监控
        self.perf_monitor = PerformanceMonitor(enabled=True)

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self.interrupted = False

        # 初始化
        self._setup()

    def _setup_device(self) -> torch.device:
        """设置设备并打印信息"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\n{'='*60}")
            print(f"  GPU Training Mode")
            print(f"{'='*60}")
            print(f"  Device: {gpu_name}")
            print(f"  Total Memory: {gpu_memory:.1f} GB")
            print(f"  Precision: {'BF16' if self.config.bf16 else 'FP16' if self.config.fp16 else 'FP32'}")
            print(f"{'='*60}\n")

            # 启用TF32 for Ampere+
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        else:
            device = torch.device("cpu")
            print(f"\n{'='*60}")
            print(f"  CPU Training Mode")
            print(f"{'='*60}")
            print(f"  Device: CPU")
            print(f"{'='*60}\n")

        return device

    def _setup(self):
        """初始化训练组件"""
        # 设置随机种子
        torch.manual_seed(self.config.seed)

        # 创建数据加载器
        self.train_dataloader = self._create_dataloader(
            self.train_dataset,
            self.config.per_device_train_batch_size,
            shuffle=True,
        )

        if self.eval_dataset is not None:
            self.eval_dataloader = self._create_dataloader(
                self.eval_dataset,
                self.config.per_device_eval_batch_size,
                shuffle=False,
            )

        # 创建优化器
        self.optimizer = create_optimizer(
            self.model,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # 计算训练步数
        num_training_steps = len(self.train_dataloader) * self.config.num_train_epochs
        if self.config.max_steps > 0:
            num_training_steps = min(num_training_steps, self.config.max_steps)

        # 计算预热步数
        num_warmup_steps = (
            self.config.warmup_steps
            if self.config.warmup_steps > 0
            else int(num_training_steps * self.config.warmup_ratio)
        )

        # 创建学习率调度器
        self.scheduler = create_scheduler(
            self.optimizer,
            scheduler_type=self.config.lr_scheduler_type,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
        )

        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(
            self.config.output_dir,
            max_checkpoints=self.config.save_total_limit,
        )

        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)

    def _create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        """创建数据加载器"""
        # IterableDataset 不支持 shuffle
        if isinstance(dataset, IterableDataset):
            shuffle = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.dataloader_num_workers if self.is_gpu else 0,
            pin_memory=self.config.dataloader_pin_memory if self.is_gpu else False,
            prefetch_factor=self.config.dataloader_prefetch_factor if self.config.dataloader_num_workers > 0 else None,
            collate_fn=self.collate_fn,
            drop_last=self.config.dataloader_drop_last,
        )

    def _setup_signal_handlers(self):
        """设置信号处理器，支持手动终止"""
        def signal_handler(signum, frame):
            print("\n" + "=" * 50)
            print("Received interrupt signal! Will save model after current step...")
            print("=" * 50)
            self.interrupted = True

        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, signal_handler)

    def _log(self, metrics: Dict[str, float]):
        """记录日志"""
        log_file = os.path.join(self.config.logging_dir, "training_log.txt")
        with open(log_file, "a") as f:
            log_line = " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in metrics.items()])
            f.write(f"{log_line}\n")

    def train(self) -> Dict[str, Any]:
        """执行训练"""
        self._setup_signal_handlers()

        # 恢复训练
        start_epoch = 0
        start_step = 0

        if self.config.resume_from_checkpoint:
            state = self.checkpoint_manager.load(
                self.config.resume_from_checkpoint,
                self.model,
                self.optimizer,
                self.scheduler,
            )
            start_epoch = state.get("epoch", 0)
            start_step = state.get("step", 0)
            self.global_step = start_step

        total_training_steps = len(self.train_dataloader) * self.config.num_train_epochs
        if self.config.max_steps > 0:
            total_training_steps = min(total_training_steps, self.config.max_steps)

        print(f"Starting training from epoch {start_epoch}, step {start_step}")
        print(f"Total training steps: {total_training_steps}")

        total_loss = 0.0
        best_eval_loss = float("inf")
        avg_epoch_loss = 0.0
        start_time = time.time()

        try:
            for epoch in range(start_epoch, self.config.num_train_epochs):
                self.model.train()
                epoch_loss = 0.0
                epoch_steps = 0
                self.perf_monitor.reset()

                progress_bar = tqdm(
                    enumerate(self.train_dataloader),
                    total=len(self.train_dataloader),
                    desc=f"Epoch {epoch + 1}/{self.config.num_train_epochs}",
                )

                for step, batch in progress_bar:
                    if self.interrupted:
                        break

                    # 跳过已训练的步
                    if epoch == start_epoch and step < start_step % len(self.train_dataloader):
                        continue

                    batch_start_time = time.perf_counter()

                    # 移动数据到设备
                    with self.perf_monitor.measure("data_loading"):
                        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                                for k, v in batch.items()}

                    # 前向传播
                    with self.perf_monitor.measure("forward"):
                        if self.is_gpu:
                            with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                                outputs = self.model(**batch)
                                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                        else:
                            outputs = self.model(**batch)
                            loss = outputs["loss"] / self.config.gradient_accumulation_steps

                    # 反向传播
                    with self.perf_monitor.measure("backward"):
                        if self.scaler:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    total_loss += loss.item()
                    epoch_loss += loss.item()

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

                        # 更新参数
                        with self.perf_monitor.measure("optimizer_step"):
                            if self.scaler:
                                self.scaler.step(self.optimizer)
                                self.scaler.update()
                            else:
                                self.optimizer.step()

                            self.scheduler.step()
                            self.optimizer.zero_grad()

                        # 记录批次时间
                        batch_time = time.perf_counter() - batch_start_time
                        self.perf_monitor.record_batch(batch_time)

                        self.global_step += 1
                        epoch_steps += 1

                        # 日志
                        if self.global_step % self.config.logging_steps == 0:
                            avg_loss = total_loss / self.config.logging_steps
                            lr = self.optimizer.param_groups[0]["lr"]
                            progress_bar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")
                            self.perf_monitor.log_step(step, self.global_step, avg_loss, lr)
                            self._log({"loss": avg_loss, "learning_rate": lr, "step": self.global_step})
                            total_loss = 0.0

                        # 保存检查点
                        if self.global_step % self.config.save_steps == 0:
                            self._save_checkpoint(epoch, self.global_step)

                        # 评估
                        if self.eval_dataset is not None and self.global_step % self.config.eval_steps == 0:
                            eval_results = self.evaluate()
                            if eval_results["loss"] < best_eval_loss:
                                best_eval_loss = eval_results["loss"]
                                self._save_checkpoint(epoch, self.global_step, is_best=True)

                        # 检查最大步数
                        if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                            break

                avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
                print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
                self.perf_monitor.print_summary()

                if self.interrupted or self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break

        except Exception as e:
            print(f"Training error: {e}")

        finally:
            print("\nSaving model...")
            self._save_final_model()
            if self.interrupted:
                print(f"Training stopped at step {self.global_step}")
            else:
                print("Training completed successfully!")

        training_time = time.time() - start_time
        return {
            "final_loss": avg_epoch_loss,
            "best_eval_loss": best_eval_loss,
            "global_step": self.global_step,
            "training_time": training_time,
        }

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """执行评估"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            if self.is_gpu:
                with torch.amp.autocast('cuda', dtype=self.dtype, enabled=self.use_amp):
                    outputs = self.model(**batch)
                    loss = outputs["loss"]
            else:
                outputs = self.model(**batch)
                loss = outputs["loss"]

            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)

        avg_loss = total_loss / total_samples
        perplexity = math.exp(avg_loss) if avg_loss < 10 else float("inf")

        results = {"loss": avg_loss, "perplexity": perplexity}
        print(f"Evaluation results: loss={avg_loss:.4f}, perplexity={perplexity:.2f}")

        if self.config.compute_metrics:
            additional_metrics = self.config.compute_metrics(results)
            results.update(additional_metrics)

        self._log(results)
        self.model.train()
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

        # 清理GPU缓存
        if self.is_gpu:
            torch.cuda.empty_cache()

    def _save_final_model(self):
        """保存最终模型"""
        save_path = os.path.join(self.config.output_dir, "final_model")
        save_model(self.model, save_path)
        print(f"Final model saved to {save_path}")


class TrainerState:
    """训练状态类"""
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
