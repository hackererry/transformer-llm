"""
训练模块
导出所有训练相关组件
"""
from .optimizer import (
    create_optimizer,
    AdamWOptimizer,
    LAMB,
    get_optimizer,
)
from .scheduler import (
    create_scheduler,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    get_scheduler,
)
from .checkpoint import (
    CheckpointManager,
    save_model,
    load_model,
    save_pretrained,
)
from .trainer import (
    Trainer,
    TrainerState,
)

# GPU训练模块（可选导入）
try:
    from .trainer_gpu import (
        GPUTrainer,
        GPUTrainingConfig,
        get_gpu_memory_info,
        estimate_memory_requirements,
    )
    _gpu_trainer_available = True
except ImportError:
    _gpu_trainer_available = False

__all__ = [
    # 优化器
    "create_optimizer",
    "AdamWOptimizer",
    "LAMB",
    "get_optimizer",
    # 调度器
    "create_scheduler",
    "get_cosine_schedule_with_warmup",
    "get_linear_schedule_with_warmup",
    "get_constant_schedule_with_warmup",
    "get_polynomial_decay_schedule_with_warmup",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "get_scheduler",
    # 检查点
    "CheckpointManager",
    "save_model",
    "load_model",
    "save_pretrained",
    # 训练器
    "Trainer",
    "TrainerState",
]

# GPU训练（可选）
if _gpu_trainer_available:
    __all__.extend([
        "GPUTrainer",
        "GPUTrainingConfig",
        "get_gpu_memory_info",
        "estimate_memory_requirements",
    ])
