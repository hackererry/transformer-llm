"""
CPU大模型训练框架
从零开始的Transformer训练框架，针对CPU优化
"""

__version__ = "0.1.0"

from .model import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    TransformerModel,
    CausalLMModel,
    create_model,
)

from .data import (
    PretrainDataset,
    FinetuneDataset,
    get_tokenizer,
    get_collator,
)

from .training import (
    Trainer,
    create_optimizer,
    create_scheduler,
    CheckpointManager,
)

from .utils import (
    setup_logger,
    set_seed,
    get_device,
)

__all__ = [
    # 版本
    "__version__",
    # 配置
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    # 模型
    "TransformerModel",
    "CausalLMModel",
    "create_model",
    # 数据
    "PretrainDataset",
    "FinetuneDataset",
    "get_tokenizer",
    "get_collator",
    # 训练
    "Trainer",
    "create_optimizer",
    "create_scheduler",
    "CheckpointManager",
    # 工具
    "setup_logger",
    "set_seed",
    "get_device",
]
