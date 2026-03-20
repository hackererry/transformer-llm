"""
CPU优化模块
导出所有CPU优化相关组件
"""
from .gradient_checkpoint import (
    gradient_checkpoint,
    CheckpointedModule,
    CheckpointedSequential,
    SelectiveCheckpoint,
    enable_gradient_checkpointing,
    set_gradient_checkpointing,
    ActivationCheckpointWrapper,
    estimate_memory_savings,
)
from .memory import (
    MemoryMonitor,
    MemoryOptimizer,
    memory_context,
    get_memory_usage,
    print_memory_usage,
    Offloader,
    optimize_for_cpu_training,
)
from .parallel import (
    ParallelDataLoader,
    ChunkedBatchSampler,
    BucketBatchSampler,
    ParallelProcessor,
    get_optimal_num_workers,
    create_dataloader,
)

__all__ = [
    # 梯度检查点
    "gradient_checkpoint",
    "CheckpointedModule",
    "CheckpointedSequential",
    "SelectiveCheckpoint",
    "enable_gradient_checkpointing",
    "set_gradient_checkpointing",
    "ActivationCheckpointWrapper",
    "estimate_memory_savings",
    # 内存管理
    "MemoryMonitor",
    "MemoryOptimizer",
    "memory_context",
    "get_memory_usage",
    "print_memory_usage",
    "Offloader",
    "optimize_for_cpu_training",
    # 并行化
    "ParallelDataLoader",
    "ChunkedBatchSampler",
    "BucketBatchSampler",
    "ParallelProcessor",
    "get_optimal_num_workers",
    "create_dataloader",
]
