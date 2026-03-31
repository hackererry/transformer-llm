# 训练层 (src/training/)

统一训练器、优化器、调度器、检查点管理。

## 目录结构

```
src/training/
├── trainer.py     # Trainer — 统一训练器（自动检测 CPU/GPU）
├── optimizer.py   # AdamW / LAMB 优化器
├── scheduler.py  # CosineAnnealing / Linear 等学习率调度器
└── checkpoint.py  # CheckpointManager / save_model / load_model
```

## 核心组件

### trainer.py — Trainer

统一训练器，自动检测 CPU/GPU 并选择最佳配置。

| 特性 | 说明 |
|------|------|
| **自动设备检测** | CPU/GPU 自动选择 |
| **自动精度选择** | GPU 自动启用 BF16（RTX 30/40 系列）或 FP16 |
| **混合精度训练** | `torch.amp.autocast` 实现 |
| **梯度累积** | 支持多步累积 |
| **梯度裁剪** | `max_grad_norm` |
| **学习率调度** | Cosine / Linear / Constant 等 |
| **检查点管理** | 自动保存、手动终止保存（Ctrl+C） |
| **性能监控** | Data Loading / Forward / Backward / Optimizer Step 耗时 |

**TrainingConfig 关键参数：**

| 类别 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| 基础 | `output_dir` | `./output` | 输出目录 |
| 基础 | `num_train_epochs` | 3 | 训练轮数 |
| 基础 | `per_device_train_batch_size` | 4 | 每设备批次大小 |
| 基础 | `gradient_accumulation_steps` | 1 | 梯度累积步数 |
| 学习率 | `learning_rate` | 5e-5 | 学习率 |
| 学习率 | `weight_decay` | 0.01 | 权重衰减 |
| 学习率 | `lr_scheduler_type` | `cosine` | 调度器类型 |
| 学习率 | `warmup_ratio` | 0.1 | 预热比例 |
| 精度 | `bf16` | True | BF16 混合精度（GPU） |
| 精度 | `gradient_checkpointing` | False | 梯度检查点 |
| 优化 | `use_flash_attention` | 自动 | Flash Attention |
| 日志 | `logging_steps` | 10 | 日志间隔 |
| 日志 | `save_steps` | 500 | 保存间隔 |
| 日志 | `save_total_limit` | 3 | 检查点数量限制 |

### optimizer.py — 优化器

| 类/函数 | 说明 |
|---------|------|
| `AdamWOptimizer` | AdamW 优化器 |
| `LAMB` | LAMB 优化器（适合大 batch 训练） |
| `create_optimizer()` | 工厂函数创建优化器 |
| `get_optimizer()` | 获取优化器实例 |

**使用示例：**

```python
from src.training import create_optimizer, TrainingConfig

config = TrainingConfig(learning_rate=5e-5, weight_decay=0.01)
optimizer = create_optimizer(model, config)
```

### scheduler.py — 学习率调度器

| 类/函数 | 说明 |
|---------|------|
| `CosineAnnealingWarmRestarts` | 余弦退火Warm Restarts |
| `OneCycleLR` | One Cycle 学习率策略 |
| `get_cosine_schedule_with_warmup()` | Cosine Warmup 调度 |
| `get_linear_schedule_with_warmup()` | Linear Warmup 调度 |
| `get_constant_schedule_with_warmup()` | Constant Warmup 调度 |
| `get_polynomial_decay_schedule_with_warmup()` | 多项式衰减 Warmup |
| `create_scheduler()` | 工厂函数创建调度器 |
| `get_scheduler()` | 获取调度器实例 |

### checkpoint.py — 检查点管理

| 类/函数 | 说明 |
|---------|------|
| `CheckpointManager` | 检查点管理器，支持保存/加载/恢复 |
| `save_model()` | 保存模型检查点 |
| `load_model()` | 加载模型检查点 |
| `save_pretrained()` | 保存为 HuggingFace 格式 |

**保存内容：**
- 模型权重（`pytorch_model.bin` / `model.safetensors`）
- 配置文件（`config.json`）
- Tokenizer 文件
- 优化器状态（可选）
- 训练状态（`trainer_state.json`）

## 使用示例

```python
from src.training import Trainer, TrainingConfig
from src.model import ModelConfig, CausalLMModel
from src.data import ShardedPreprocessedDataset, get_collator, get_tokenizer
from torch.utils.data import DataLoader

# 模型
model_config = ModelConfig.small()
model = CausalLMModel(model_config)

# 数据
tokenizer = get_tokenizer("output/preprocessed/tokenizer")
dataset = ShardedPreprocessedDataset("output/preprocessed", split="train")
dataloader = DataLoader(dataset, batch_size=8, collate_fn=get_collator(tokenizer))

# 配置
train_config = TrainingConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5,
    bf16=True,  # 自动检测
)

# 训练
trainer = Trainer(model, train_config, train_loader=dataloader)
trainer.train()
```

## 手动终止保存

训练过程中按 `Ctrl+C` 可安全终止训练，自动保存当前检查点：

```
KeyboardInterrupt received!
Saving checkpoint at step 1234...
Checkpoint saved to: ./output/checkpoint-step-1234
Training stopped.
```

## 设备自动检测

| 设备 | 自动行为 |
|------|---------|
| **GPU** | 启用 BF16/FP16 混合精度，Flash Attention（如可用） |
| **CPU** | FP32 训练，gradient_checkpointing 不支持 |
