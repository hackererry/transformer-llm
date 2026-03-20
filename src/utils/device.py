"""
设备管理模块
处理设备选择和优化
"""
import torch
from typing import Optional, Dict, Any, List
import os


def get_device() -> torch.device:
    """
    获取最佳可用设备

    Returns:
        torch.device对象
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    获取设备信息

    Returns:
        设备信息字典
    """
    info = {
        "device": str(get_device()),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0)
        info["cuda_device_capability"] = torch.cuda.get_device_capability(0)
        info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0) / (1024 ** 3)
        info["cuda_memory_reserved"] = torch.cuda.memory_reserved(0) / (1024 ** 3)

    return info


def to_device(
    data: Any,
    device: Optional[torch.device] = None,
    non_blocking: bool = False,
) -> Any:
    """
    将数据移动到指定设备

    Args:
        data: 要移动的数据(张量、字典、列表等)
        device: 目标设备
        non_blocking: 是否异步

    Returns:
        移动后的数据
    """
    if device is None:
        device = get_device()

    if isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    elif isinstance(data, dict):
        return {k: to_device(v, device, non_blocking) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(v, device, non_blocking) for v in data]
    elif isinstance(data, tuple):
        return tuple(to_device(v, device, non_blocking) for v in data)
    else:
        return data


def set_seed(seed: int):
    """
    设置随机种子

    Args:
        seed: 随机种子
    """
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_memory_info() -> Dict[str, float]:
    """
    获取内存使用信息

    Returns:
        内存信息字典(MB)
    """
    import psutil

    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    result = {
        "rss_mb": mem_info.rss / (1024 * 1024),
        "vms_mb": mem_info.vms / (1024 * 1024),
    }

    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / (1024 * 1024)
        result["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return result


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    优化模型用于推理

    Args:
        model: 模型

    Returns:
        优化后的模型
    """
    # 设置为评估模式
    model.eval()

    # 禁用梯度计算
    for param in model.parameters():
        param.requires_grad = False

    # 如果支持，使用torch.compile
    if hasattr(torch, "compile") and get_device().type != "cpu":
        try:
            model = torch.compile(model)
        except Exception:
            pass

    return model


def enable_tf32():
    """
    启用TF32以加速Ampere GPU上的计算
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def set_num_threads(num_threads: int):
    """
    设置CPU线程数

    Args:
        num_threads: 线程数
    """
    torch.set_num_threads(num_threads)
    if num_threads > 1:
        # 设置OpenMP线程数
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)


def get_optimal_num_threads() -> int:
    """
    获取最优的CPU线程数

    Returns:
        推荐的线程数
    """
    cpu_count = os.cpu_count() or 4
    # 通常使用物理核心数
    return min(cpu_count, 8)


class DeviceManager:
    """
    设备管理器
    统一管理设备和相关设置
    """

    def __init__(self, device: Optional[str] = None):
        """
        Args:
            device: 指定设备，None则自动选择
        """
        if device is None:
            self.device = get_device()
        else:
            self.device = torch.device(device)

        self._setup()

    def _setup(self):
        """初始化设置"""
        if self.device.type == "cpu":
            # CPU优化
            num_threads = get_optimal_num_threads()
            set_num_threads(num_threads)
            print(f"Using CPU with {num_threads} threads")
        elif self.device.type == "cuda":
            # CUDA优化
            enable_tf32()
            print(f"Using CUDA: {torch.cuda.get_device_name()}")

    def to_device(self, data: Any) -> Any:
        """将数据移动到当前设备"""
        return to_device(data, self.device)

    def get_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        return get_device_info()

    def get_memory_info(self) -> Dict[str, float]:
        """获取内存信息"""
        return get_memory_info()

    def __str__(self) -> str:
        return str(self.device)

    def __repr__(self) -> str:
        return f"DeviceManager(device={self.device})"


def print_device_info():
    """打印设备信息"""
    info = get_device_info()
    print("Device Information:")
    print(f"  Device: {info['device']}")
    print(f"  CPU Cores: {info['cpu_count']}")
    print(f"  CUDA Available: {info['cuda_available']}")

    if info["cuda_available"]:
        print(f"  CUDA Device: {info['cuda_device_name']}")
        print(f"  CUDA Memory: {info['cuda_memory_allocated']:.2f} GB allocated")
