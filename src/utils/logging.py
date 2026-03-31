"""
日志模块
提供训练过程中的日志记录功能
"""
import os
import sys
import logging
import time
import io
from datetime import datetime
from typing import Optional, Dict, Any, List
import json


class UTF8StreamHandler(logging.StreamHandler):
    """支持 UTF-8 编码的 StreamHandler，解决 Windows GBK 编码问题"""

    def __init__(self, stream=None):
        # Windows 下强制使用 UTF-8 编码
        if sys.platform == "win32":
            if stream is None:
                stream = sys.stdout
            # 获取原始二进制流并用 UTF-8 TextIOWrapper 包装
            if hasattr(stream, 'buffer'):
                raw_stream = stream.buffer
            else:
                raw_stream = stream
            wrapped = io.TextIOWrapper(raw_stream, encoding='utf-8', errors='replace')
            super().__init__(wrapped)
        else:
            super().__init__(stream)

    def emit(self, record):
        """重写 emit 方法，确保 UTF-8 编码输出"""
        try:
            # 确保使用 UTF-8 编码
            if sys.platform == "win32" and hasattr(self.stream, 'reconfigure'):
                try:
                    self.stream.reconfigure(encoding='utf-8', errors='replace')
                except Exception:
                    pass
            super().emit(record)
        except UnicodeEncodeError:
            # 备用方案：直接写入字节
            try:
                msg = self.format(record) + self.terminator
                self.stream.buffer.write(msg.encode('utf-8', errors='replace'))
                self.stream.buffer.flush()
            except Exception:
                pass


class Logger:
    """
    训练日志记录器
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        experiment_name: Optional[str] = None,
        log_level: int = logging.INFO,
        console_output: bool = True,
    ):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            log_level: 日志级别
            console_output: 是否输出到控制台
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # 生成实验名称
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name

        # 创建logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # 清除现有handlers

        # 文件handler
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
        # Windows 下使用 UTF-8 编码
        if sys.platform == "win32":
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # 控制台handler
        if console_output:
            console_handler = UTF8StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter("%(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # 指标历史
        self.metrics_history: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def log(self, message: str, level: int = logging.INFO):
        """记录日志消息"""
        self.logger.log(level, message)

    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)

    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self.logger.debug(message)

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ):
        """
        记录指标

        Args:
            metrics: 指标字典
            step: 当前步数
            prefix: 指标前缀
        """
        metric_strs = []
        for key, value in metrics.items():
            full_key = f"{prefix}/{key}" if prefix else key
            metric_strs.append(f"{full_key}={value:.6f}")

        self.info(f"Step {step}: {', '.join(metric_strs)}")

        # 保存到历史
        record = {
            "step": step,
            "timestamp": time.time() - self.start_time,
            **metrics,
        }
        self.metrics_history.append(record)

    def log_hyperparams(self, hyperparams: Dict[str, Any]):
        """记录超参数"""
        self.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.info(f"  {key}: {value}")

        # 保存到文件
        hyperparams_file = os.path.join(
            self.log_dir,
            f"{self.experiment_name}_hyperparams.json",
        )
        with open(hyperparams_file, "w") as f:
            json.dump(hyperparams, f, indent=2)

    def log_model_summary(self, model):
        """记录模型摘要"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.info(f"Model Summary:")
        self.info(f"  Total parameters: {total_params:,}")
        self.info(f"  Trainable parameters: {trainable_params:,}")
        self.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")

        # 估计模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        self.info(f"  Estimated model size: {param_size / (1024 * 1024):.2f} MB")

    def save_metrics(self):
        """保存指标历史到文件"""
        metrics_file = os.path.join(
            self.log_dir,
            f"{self.experiment_name}_metrics.json",
        )
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

    def close(self):
        """关闭logger"""
        self.save_metrics()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


class ProgressTracker:
    """
    训练进度跟踪器
    """

    def __init__(
        self,
        total_steps: int,
        log_interval: int = 10,
        logger: Optional[Logger] = None,
    ):
        """
        Args:
            total_steps: 总步数
            log_interval: 日志间隔
            logger: 日志记录器
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.logger = logger

        self.current_step = 0
        self.start_time = time.time()
        self.step_times: List[float] = []
        self.losses: List[float] = []

    def step(self, loss: Optional[float] = None):
        """更新一步"""
        self.current_step += 1
        current_time = time.time()

        if loss is not None:
            self.losses.append(loss)

        # 计算速度
        elapsed = current_time - self.start_time
        steps_per_sec = self.current_step / elapsed
        eta = (self.total_steps - self.current_step) / steps_per_sec if steps_per_sec > 0 else 0

        # 定期记录
        if self.current_step % self.log_interval == 0:
            avg_loss = sum(self.losses[-self.log_interval:]) / len(self.losses[-self.log_interval:]) if self.losses else 0
            progress = self.current_step / self.total_steps * 100

            msg = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%) | "
            msg += f"Loss: {avg_loss:.4f} | "
            msg += f"Speed: {steps_per_sec:.2f} steps/s | "
            msg += f"ETA: {self._format_time(eta)}"

            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def get_summary(self) -> Dict[str, Any]:
        """获取摘要"""
        elapsed = time.time() - self.start_time
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0

        return {
            "total_steps": self.total_steps,
            "current_step": self.current_step,
            "elapsed_time": elapsed,
            "steps_per_second": self.current_step / elapsed if elapsed > 0 else 0,
            "average_loss": avg_loss,
        }


class TensorBoardLogger:
    """
    TensorBoard日志记录器
    简化版本，支持基本功能
    """

    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            print("TensorBoard not available. Install with: pip install tensorboard")
            self.writer = None
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """记录直方图"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        if self.enabled:
            self.writer.add_text(tag, text, step)

    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """记录超参数"""
        if self.enabled:
            self.writer.add_hparams(hparams, metrics)

    def close(self):
        """关闭writer"""
        if self.enabled:
            self.writer.close()


def setup_logger(
    log_dir: str = "./logs",
    experiment_name: Optional[str] = None,
    console_output: bool = True,
) -> Logger:
    """
    设置日志记录器

    Args:
        log_dir: 日志目录
        experiment_name: 实验名称
        console_output: 是否输出到控制台

    Returns:
        Logger实例
    """
    return Logger(log_dir, experiment_name, console_output=console_output)
