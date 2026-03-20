"""
内存管理模块
监控和优化CPU内存使用
"""
import torch
import gc
import os
import psutil
from typing import Optional, Dict, Any, List, Callable
import time
from contextlib import contextmanager
import warnings


class MemoryMonitor:
    """
    内存监控器
    跟踪内存使用情况
    """

    def __init__(self, interval: float = 1.0):
        """
        Args:
            interval: 监控间隔(秒)
        """
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self.peak_memory = 0
        self.memory_history: List[Dict[str, float]] = []

    def get_memory_info(self) -> Dict[str, float]:
        """获取当前内存信息"""
        mem_info = self.process.memory_info()

        # 获取系统内存信息
        system_mem = psutil.virtual_memory()

        info = {
            "rss_mb": mem_info.rss / (1024 * 1024),  # 常驻内存
            "vms_mb": mem_info.vms / (1024 * 1024),  # 虚拟内存
            "system_total_mb": system_mem.total / (1024 * 1024),
            "system_available_mb": system_mem.available / (1024 * 1024),
            "system_used_percent": system_mem.percent,
            "timestamp": time.time(),
        }

        # 更新峰值
        if info["rss_mb"] > self.peak_memory:
            self.peak_memory = info["rss_mb"]

        return info

    def log_memory(self, tag: str = "") -> Dict[str, float]:
        """记录当前内存使用"""
        info = self.get_memory_info()
        info["tag"] = tag
        self.memory_history.append(info)

        if tag:
            print(f"[Memory {tag}] RSS: {info['rss_mb']:.2f} MB, "
                  f"Available: {info['system_available_mb']:.2f} MB")

        return info

    def get_peak_memory(self) -> float:
        """获取峰值内存使用(MB)"""
        return self.peak_memory

    def get_summary(self) -> Dict[str, Any]:
        """获取内存使用摘要"""
        if not self.memory_history:
            return {}

        rss_values = [m["rss_mb"] for m in self.memory_history]

        return {
            "peak_memory_mb": self.peak_memory,
            "avg_memory_mb": sum(rss_values) / len(rss_values),
            "min_memory_mb": min(rss_values),
            "max_memory_mb": max(rss_values),
            "num_samples": len(self.memory_history),
        }


class MemoryOptimizer:
    """
    内存优化器
    提供各种内存优化策略
    """

    def __init__(self, model: Optional[torch.nn.Module] = None):
        self.model = model
        self.monitor = MemoryMonitor()

    def clear_cache(self):
        """清理缓存"""
        gc.collect()
        # CPU没有torch.cuda.empty_cache()的等效方法
        # 但可以手动触发垃圾回收

    def optimize_tensors(self):
        """优化张量内存布局"""
        if self.model is None:
            return

        for param in self.model.parameters():
            if param.data is not None:
                # 确保张量是连续的
                if not param.data.is_contiguous():
                    param.data = param.data.contiguous()

    def get_model_memory(self) -> Dict[str, float]:
        """获取模型内存使用"""
        if self.model is None:
            return {}

        param_memory = 0
        buffer_memory = 0

        for param in self.model.parameters():
            param_memory += param.numel() * param.element_size()

        for buffer in self.model.buffers():
            buffer_memory += buffer.numel() * buffer.element_size()

        return {
            "param_memory_mb": param_memory / (1024 * 1024),
            "buffer_memory_mb": buffer_memory / (1024 * 1024),
            "total_memory_mb": (param_memory + buffer_memory) / (1024 * 1024),
        }

    def estimate_training_memory(
        self,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
        dtype_bytes: int = 4,
    ) -> Dict[str, float]:
        """
        估计训练所需内存

        Args:
            batch_size: 批次大小
            seq_length: 序列长度
            hidden_size: 隐藏层维度
            num_layers: 层数
            vocab_size: 词表大小
            dtype_bytes: 数据类型字节数

        Returns:
            内存估计字典
        """
        # 模型参数
        # 嵌入层
        embed_params = vocab_size * hidden_size

        # 每层的参数
        # QKV投影: 3 * hidden * hidden
        # 输出投影: hidden * hidden
        # FFN: 2 * hidden * intermediate (假设intermediate = 4 * hidden)
        # LayerNorm: 2 * hidden
        params_per_layer = (
            3 * hidden_size * hidden_size +  # QKV
            hidden_size * hidden_size +       # 输出投影
            2 * hidden_size * 4 * hidden_size +  # FFN (SwiGLU有3个投影，简化为2)
            2 * hidden_size                   # LayerNorm
        )

        total_params = embed_params + num_layers * params_per_layer

        # 参数内存
        param_memory = total_params * dtype_bytes

        # 梯度内存 (与参数相同)
        gradient_memory = param_memory

        # 优化器状态 (AdamW: 2倍参数)
        optimizer_memory = 2 * param_memory

        # 激活值内存
        activation_per_layer = batch_size * seq_length * hidden_size * dtype_bytes
        activation_memory = num_layers * activation_per_layer

        # 总内存
        total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory

        return {
            "param_memory_mb": param_memory / (1024 * 1024),
            "gradient_memory_mb": gradient_memory / (1024 * 1024),
            "optimizer_memory_mb": optimizer_memory / (1024 * 1024),
            "activation_memory_mb": activation_memory / (1024 * 1024),
            "total_memory_mb": total_memory / (1024 * 1024),
            "estimated_params": total_params,
        }


@contextmanager
def memory_context(
    tag: str = "",
    clear_before: bool = True,
    clear_after: bool = True,
):
    """
    内存监控上下文管理器

    Args:
        tag: 标签
        clear_before: 进入前清理
        clear_after: 退出后清理
    """
    monitor = MemoryMonitor()

    if clear_before:
        gc.collect()

    monitor.log_memory(f"{tag}_start")

    try:
        yield monitor
    finally:
        if clear_after:
            gc.collect()

        monitor.log_memory(f"{tag}_end")


def get_memory_usage() -> Dict[str, float]:
    """获取当前内存使用"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()

    return {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
        "available_mb": system_mem.available / (1024 * 1024),
        "percent": system_mem.percent,
    }


def print_memory_usage(tag: str = ""):
    """打印当前内存使用"""
    usage = get_memory_usage()
    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}Memory: RSS={usage['rss_mb']:.2f}MB, "
          f"Available={usage['available_mb']:.2f}MB, "
          f"Used={usage['percent']:.1f}%")


class Offloader:
    """
    模型卸载器
    将不活跃的模型部分卸载到CPU内存或磁盘
    """

    def __init__(self, model: torch.nn.Module, offload_dir: Optional[str] = None):
        self.model = model
        self.offload_dir = offload_dir
        self.offloaded_params: Dict[str, Any] = {}

    def offload_layer(self, layer_name: str):
        """卸载指定层到CPU内存"""
        if self.offload_dir:
            # 卸载到磁盘
            layer = self._get_layer(layer_name)
            if layer is not None:
                path = os.path.join(self.offload_dir, f"{layer_name}.pt")
                torch.save(layer.state_dict(), path)
                self.offloaded_params[layer_name] = path
                # 清空GPU内存
                for param in layer.parameters():
                    param.data = torch.empty(0)
        else:
            # 卸载到CPU内存
            layer = self._get_layer(layer_name)
            if layer is not None:
                for name, param in layer.named_parameters():
                    self.offloaded_params[f"{layer_name}.{name}"] = param.data.cpu()
                    param.data = torch.empty(0)

    def load_layer(self, layer_name: str):
        """从卸载位置加载层"""
        if layer_name in self.offloaded_params:
            if self.offload_dir:
                # 从磁盘加载
                path = self.offloaded_params[layer_name]
                state_dict = torch.load(path)
                layer = self._get_layer(layer_name)
                if layer is not None:
                    layer.load_state_dict(state_dict)
            else:
                # 从CPU内存加载
                layer = self._get_layer(layer_name)
                if layer is not None:
                    for name, param in layer.named_parameters():
                        key = f"{layer_name}.{name}"
                        if key in self.offloaded_params:
                            param.data = self.offloaded_params[key]

    def _get_layer(self, layer_name: str) -> Optional[torch.nn.Module]:
        """根据名称获取层"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        return None


def optimize_for_cpu_training(
    model: torch.nn.Module,
    enable_gradient_checkpointing: bool = True,
    clear_gradients_after_step: bool = True,
) -> torch.nn.Module:
    """
    优化模型以适应CPU训练

    Args:
        model: 模型
        enable_gradient_checkpointing: 是否启用梯度检查点
        clear_gradients_after_step: 是否在每步后清理梯度

    Returns:
        优化后的模型
    """
    if enable_gradient_checkpointing:
        from .gradient_checkpoint import set_gradient_checkpointing
        set_gradient_checkpointing(model, True)

    # 设置为CPU
    model = model.cpu()

    # 确保所有张量是连续的
    for param in model.parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    return model
