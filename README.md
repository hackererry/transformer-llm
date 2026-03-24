# Transformer大模型训练框架

一个基于 PyTorch 的 LLM（大语言模型）训练框架，支持预训练和微调(SFT)，针对 CPU 和 GPU 训练进行了优化。

## 功能特性

| 功能 | 说明 |
|------|------|
| **预训练** | 从零开始训练基础语言模型 |
| **指令微调(SFT)** | 使用监督学习微调模型遵循指令 |
| **文本生成** | 支持多种解码策略的文本生成 |
| **模型续训** | 支持加载已有模型继续训练 |
| **断点续训** | 支持从检查点恢复训练 |
| **CPU优化** | BF16混合精度、梯度检查点、内存优化 |
| **GPU优化** | BF16/FP16混合精度、Flash Attention、梯度累积 |
| **中文支持** | BPE分词器原生支持中文字符 |
| **优雅退出** | Ctrl+C手动终止训练，自动保存模型 |

---

## 项目架构

### 目录结构

```
transformer/
├── src/                          # 核心源代码
│   ├── model/                    # 模型层 - Transformer架构实现
│   │   ├── __init__.py          # 模型模块导出
│   │   ├── config.py            # ModelConfig - 模型配置
│   │   ├── transformer.py        # TransformerModel/CausalLMModel - 主模型
│   │   ├── attention.py          # Attention/FlashAttention - 注意力机制
│   │   ├── attention_gpu.py     # GPU优化注意力（可选导入）
│   │   ├── layers.py            # RMSNorm/SwiGLUFFN/FeedForward - 层定义
│   │   ├── embedding.py         # TokenEmbedding/RotaryEmbedding - 嵌入
│   │   └── lm_head.py           # LMHead - 语言模型头
│   │
│   ├── data/                    # 数据层 - tokenizer、数据集、collator
│   │   ├── __init__.py          # 数据模块导出
│   │   ├── tokenizer.py         # HuggingFaceBPETokenizer - BPE分词器
│   │   ├── dataset.py           # PretrainDataset/FinetuneDataset - 数据集
│   │   ├── preprocessed_dataset.py  # ShardedPreprocessedDataset - 分片数据集
│   │   └── collator.py          # DataCollatorForCausalLM/SFT - 数据整理
│   │
│   ├── training/                # 训练层 - 统一Trainer、optimizer、scheduler
│   │   ├── __init__.py          # 训练模块导出
│   │   ├── trainer.py           # Trainer - 统一训练器（CPU/GPU自动检测）
│   │   ├── optimizer.py        # AdamW/LAMB - 优化器
│   │   ├── scheduler.py        # CosineAnnealing/Linear - 学习率调度
│   │   └── checkpoint.py        # CheckpointManager - 检查点管理
│   │
│   ├── cpu_optim/              # CPU优化模块
│   │   ├── gradient_checkpoint.py  # 梯度检查点
│   │   ├── memory.py           # 内存优化
│   │   └── parallel.py         # 并行化支持
│   │
│   └── utils/                  # 工具层
│       ├── __init__.py         # 工具模块导出
│       ├── device.py           # 设备管理、内存信息
│       ├── logging.py          # 日志记录
│       └── metrics.py          # 性能指标计算
│
├── scripts/                     # 训练脚本
│   ├── preprocess_data.py      # 数据预处理脚本
│   ├── pretrain.py             # 预训练脚本（自动检测CPU/GPU）
│   ├── finetune.py             # 指令微调脚本
│   └── generate.py             # 文本生成脚本
│
├── tests/                       # 测试用例
│   ├── __init__.py
│   ├── test_model.py           # 模型模块测试
│   ├── test_data.py            # 数据模块测试
│   ├── test_training.py        # 训练模块测试
│   └── conftest.py             # pytest配置
│
├── dataset/                     # 数据处理工具
│   ├── epub_to_txt.py         # EPUB转文本
│   └── clean_text.py          # 文本清洗
│
├── logs/                        # 训练日志
├── output/                      # 输出目录
├── CLAUDE.md                   # 项目说明
└── README.md                   # 本文档
```

---

## 核心模块说明

### 1. 模型层 (src/model/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `config.py` | `ModelConfig` | 模型配置（hidden_size, num_layers, num_heads等） |
| `config.py` | `TrainingConfig` | 训练配置（学习率、批次大小等） |
| `transformer.py` | `TransformerModel` | Transformer主体（无LM head） |
| `transformer.py` | `TransformerBlock` | 单个Transformer块 |
| `transformer.py` | `CausalLMModel` | 因果语言模型主类（包含LM head） |
| `attention.py` | `Attention` | 标准多头注意力机制 |
| `attention.py` | `FlashAttention` | Flash Attention实现 |
| `layers.py` | `RMSNorm` | RMS归一化 |
| `layers.py` | `SwiGLUFFN` | SwiGLU激活的前馈网络 |
| `layers.py` | `FeedForward` | 标准前馈网络 |
| `embedding.py` | `RotaryEmbedding` | RoPE旋转位置编码 |
| `embedding.py` | `apply_rotary_pos_emb` | 应用RoPE编码 |
| `lm_head.py` | `LMHead` | 语言模型输出头 |

### 2. 数据层 (src/data/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `tokenizer.py` | `HuggingFaceBPETokenizer` | HuggingFace tokenizers封装（Rust实现，21x加速） |
| `tokenizer.py` | `get_tokenizer` | 获取tokenizer的工厂函数 |
| `dataset.py` | `PretrainDataset` | 原始文本预训练数据集 |
| `dataset.py` | `PretrainIterableDataset` | 可迭代形式预训练数据集 |
| `dataset.py` | `FinetuneDataset` | 指令微调数据集 |
| `dataset.py` | `TextFileDataset` | 文本文件数据集 |
| `preprocessed_dataset.py` | `PreprocessedDataset` | 预处理缓存数据集 |
| `preprocessed_dataset.py` | `ShardedPreprocessedDataset` | 分片预处理数据集（避免OOM） |
| `preprocessed_dataset.py` | `save_preprocessed_data` | 保存预处理数据 |
| `collator.py` | `DataCollatorForCausalLM` | 因果LM数据整理器 |
| `collator.py` | `DataCollatorForSFT` | SFT数据整理器 |
| `collator.py` | `DynamicBatchSampler` | 动态批采样器 |
| `collator.py` | `get_collator` | 获取collator的工厂函数 |

### 3. 训练层 (src/training/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `trainer.py` | `Trainer` | **统一训练器**（自动检测CPU/GPU） |
| `trainer.py` | `TrainingConfig` | 训练配置（统一版本） |
| `trainer.py` | `PerformanceMonitor` | 性能监控器 |
| `trainer.py` | `TrainerState` | 训练状态类 |
| `optimizer.py` | `create_optimizer` | 创建优化器的工厂函数 |
| `optimizer.py` | `AdamWOptimizer` | AdamW优化器封装 |
| `optimizer.py` | `LAMB` | LAMB优化器 |
| `scheduler.py` | `create_scheduler` | 创建学习率调度器 |
| `scheduler.py` | `CosineAnnealingWarmRestarts` | 余弦退火预热 |
| `scheduler.py` | `OneCycleLR` | 单周期学习率 |
| `checkpoint.py` | `CheckpointManager` | 检查点管理器 |
| `checkpoint.py` | `save_model` | 保存模型 |
| `checkpoint.py` | `load_model` | 加载模型 |
| `checkpoint.py` | `save_pretrained` | 以HuggingFace格式保存 |

### 4. 工具层 (src/utils/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `device.py` | `get_device` | 获取设备 |
| `device.py` | `set_seed` | 设置随机种子 |
| `device.py` | `get_memory_info` | 获取内存信息 |
| `device.py` | `print_device_info` | 打印设备信息 |
| `logging.py` | `setup_logger` | 设置日志记录 |
| `metrics.py` | `compute_perplexity` | 计算困惑度 |
| `metrics.py` | `compute_accuracy` | 计算准确率 |

---

## 预设模型配置

| 配置 | 参数量 | hidden_size | layers | heads | intermediate_size |
|------|--------|-------------|--------|-------|-------------------|
| tiny | ~10M | 256 | 6 | 8 | 512 |
| small | ~100M | 512 | 12 | 8 | 1024 |
| medium | ~500M | 1024 | 24 | 16 | 2048 |

---

## 常见命令

### 1. 数据预处理

```bash
# 方式1: 目录输入（自动扫描所有txt文件）
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --max_seq_length 512 \
    --vocab_size 32000

# 方式2: 文件列表输入
python scripts/preprocess_data.py \
    --train_files dataset/data/train1.txt dataset/data/train2.txt \
    --output_dir output/preprocessed \
    --max_seq_length 512

# 方式3: 带验证集
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --validation_file dataset/data/val.txt \
    --output_dir output/preprocessed

# 强制重新处理
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --force_reprocess
```

### 2. 预训练（自动检测GPU/CPU）

```bash
# 使用预处理数据训练
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config small \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4

# 继续训练已有模型
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_path output/final_model \
    --num_train_epochs 1

# 从检查点恢复
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --resume_from_checkpoint output/checkpoint-step-1000 \
    --num_train_epochs 1

# GPU上使用BF16
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config small \
    --bf16

# CPU训练（自动检测，无需额外参数）
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config tiny \
    --num_train_epochs 1
```

### 3. 指令微调

```bash
python scripts/finetune.py \
    --train_file data/instructions.json \
    --model_path output/final_model \
    --output_dir output/sft \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

### 4. 文本生成

```bash
# 单次生成
python scripts/generate.py \
    --model_path output/sft/final_model \
    --prompt "今天天气真好，" \
    --max_new_tokens 100

# 交互模式
python scripts/generate.py \
    --model_path output/sft/final_model \
    --interactive

# 调整采样参数
python scripts/generate.py \
    --model_path output/sft/final_model \
    --prompt "你好" \
    --max_new_tokens 50 \
    --temperature 0.7 \
    --top_p 0.9 \
    --do_sample
```

---

## 参数详解

### preprocess_data.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_dir` | None | 训练数据目录（自动扫描.txt文件） |
| `--train_files` | None | 训练数据文件列表 |
| `--validation_file` | None | 验证数据文件路径 |
| `--output_dir` | ./preprocessed_data | 输出目录 |
| `--max_seq_length` | 512 | 最大序列长度 |
| `--vocab_size` | 32000 | BPE词表大小 |
| `--min_frequency` | 2 | BPE最小频率 |
| `--shard_size` | 10000 | 每个分片的样本数量 |
| `--tokenizer_sample_bytes` | 100MB | 用于训练tokenizer的采样字节数 |
| `--force_reprocess` | False | 强制重新处理 |

### pretrain.py 参数

#### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--preprocessed_data` | **必填** | 预处理数据目录路径 |

#### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | None | 预训练模型路径（用于继续训练） |
| `--model_config` | tiny | 模型规模: tiny/small/medium |

#### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | ./output | 输出目录 |
| `--num_train_epochs` | 3 | 训练轮数 |
| `--per_device_train_batch_size` | 4 | 每设备批次大小 |
| `--gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `--max_steps` | -1 | 最大步数(-1=不限制) |
| `--learning_rate` | 5e-5 | 学习率 |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `--lr_scheduler_type` | cosine | 调度器: cosine/linear/constant |
| `--warmup_ratio` | 0.1 | 预热比例 |

#### 精度与优化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bf16` | 自动检测 | BF16混合精度（GPU自动启用） |
| `--fp16` | False | FP16混合精度 |
| `--gradient_checkpointing` | False | 梯度检查点 |
| `--use_flash_attention` | False | Flash Attention（GPU） |
| `--num_workers` | 0 | 数据加载进程数 |

#### 日志与保存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--logging_steps` | 10 | 日志间隔 |
| `--save_steps` | 500 | 保存间隔 |
| `--save_total_limit` | 3 | 最多保存检查点数 |

#### 恢复训练

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--resume_from_checkpoint` | None | 从检查点恢复 |
| `--seed` | 42 | 随机种子 |

### finetune.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_file` | **必填** | 微调数据文件（JSON格式） |
| `--validation_file` | None | 验证数据文件 |
| `--model_path` | None | 预训练模型路径 |
| `--model_config` | tiny | 模型配置 |
| `--template` | alpaca | 指令模板: alpaca/chat/simple |
| `--output_dir` | ./output_sft | 输出目录 |
| `--num_train_epochs` | 3 | 训练轮数 |
| `--per_device_train_batch_size` | 4 | 批次大小 |
| `--learning_rate` | 2e-5 | 学习率 |
| `--max_seq_length` | 512 | 最大序列长度 |

### generate.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | **必填** | 模型路径 |
| `--tokenizer_path` | None | Tokenizer路径 |
| `--prompt` | "" | 输入提示 |
| `--max_new_tokens` | 100 | 最大生成长度 |
| `--temperature` | 1.0 | 温度参数 |
| `--top_k` | 50 | Top-k采样 |
| `--top_p` | 0.95 | Top-p采样 |
| `--do_sample` | True | 是否采样 |
| `--num_return_sequences` | 1 | 返回序列数 |
| `--seed` | 42 | 随机种子 |
| `--interactive` | False | 交互模式 |

### 生成参数详解

- **temperature**: 控制随机性
  - 0.1-0.5: 确定性输出
  - 0.7-1.0: 平衡
  - >1.0: 高度随机

- **top_k**: 限制采样范围
  - 较小: 保守输出
  - 较大: 更多样性
  - 0: 禁用

- **top_p**: 核采样
  - 较低: 保守
  - 较高: 多样
  - 1.0: 禁用(greedy)

---

## 训练数据格式

### 预训练数据

纯文本文件，每行一个样本：
```
第一段文本内容...
第二段文本内容...
```

### 微调数据

JSON数组格式：
```json
[
  {"instruction": "任务指令", "input": "输入", "output": "输出"},
  {"instruction": "翻译", "input": "Hello", "output": "你好"}
]
```

---

## 预处理数据格式

预处理后的数据保存在 `output/preprocessed/` 目录：
```
output/preprocessed/
├── train_000.pt          # 训练分片
├── train_001.pt
├── val_000.pt            # 验证分片（如果有）
├── tokenizer/            # Tokenizer文件
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── merges.txt
└── dataset_info.json     # 数据集元信息
```

缓存格式（.pt文件）：
```python
{
    "version": "1.0",
    "metadata": {
        "max_seq_length": 512,
        "vocab_size": 32000,
        "num_examples": 10000,
        "original_file": "train.txt",
    },
    "examples": [
        {"input_ids": [...], "labels": [...]},
        ...
    ]
}
```

---

## 测试

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v

# 运行特定测试类
pytest tests/test_model.py::TestModelConfig -v

# 运行特定测试函数
pytest tests/test_model.py::TestModelConfig::test_tiny_config -v

# 显示详细输出
pytest tests/ -v -s

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 测试文件说明

| 文件 | 测试内容 |
|------|----------|
| `tests/test_model.py` | 模型配置、Transformer、注意力机制、RoPE、生成 |
| `tests/test_data.py` | Tokenizer、数据整理器、动态批采样 |
| `tests/test_training.py` | 优化器、调度器、检查点管理、训练器 |
| `tests/conftest.py` | pytest配置和共享fixture |

### 测试用例列表

#### test_model.py

- `TestModelConfig`: 模型配置测试
  - `test_default_config`: 默认配置
  - `test_tiny_config`: Tiny配置
  - `test_small_config`: Small配置
  - `test_medium_config`: Medium配置
  - `test_head_dim_calculation`: head_dim计算
  - `test_invalid_config`: 无效配置校验
  - `test_to_dict`/`test_from_dict`: 配置序列化

- `TestRMSNorm`: RMSNorm测试
- `TestSwiGLUFFN`: SwiGLU FFN测试
- `TestRotaryEmbedding`: RoPE测试
- `TestCausalLMModel`: 因果语言模型测试
  - 模型创建、前向传播、损失计算、文本生成、KV缓存
- `TestCreateModel`: 模型工厂函数测试

#### test_data.py

- `TestBPETokenizer`: BPE Tokenizer测试
  - 创建、编码解码、特殊token、训练、保存加载
- `TestDataCollatorForCausalLM`: 因果LM整理器测试
- `TestDataCollatorForSFT`: SFT整理器测试
- `TestGetCollator`: Collator工厂函数测试

#### test_training.py

- `TestOptimizer`: 优化器测试
  - AdamW/Adam/SGD创建、权重衰减分离
- `TestScheduler`: 调度器测试
  - Cosine/Linear创建、预热阶段
- `TestCheckpointManager`: 检查点管理器测试
  - 保存加载、最佳模型
- `TestTrainer`: 训练器测试
  - 训练器创建

---

## 工作流程

```
步骤1: 数据预处理
  python scripts/preprocess_data.py \
      --train_file dataset/data/train.txt \
      --output_dir output/preprocessed

步骤2: 模型预训练
  python scripts/pretrain.py \
      --preprocessed_data output/preprocessed \
      --model_config small \
      --num_train_epochs 3

步骤3: (可选) 指令微调
  python scripts/finetune.py \
      --train_file data/instructions.json \
      --model_path output/final_model \
      --output_dir output/sft

步骤4: 文本生成
  python scripts/generate.py \
      --model_path output/sft/final_model \
      --prompt "你好" \
      --interactive
```

---

## GPU显存估算 (RTX 4060Ti 8G)

| 模型 | 参数量 | 估算显存 |
|------|--------|----------|
| Tiny | ~10M | ~1 GB |
| Small | ~100M | ~4 GB |
| Medium | ~500M | ~8 GB |

推荐配置 (8G显存):
- Tiny: batch_size=16, gradient_accumulation=2
- Small: batch_size=8, gradient_accumulation=4
- Medium: batch_size=4, gradient_accumulation=8

---

## 安装与依赖

```bash
# 克隆仓库
git clone <repo_url>
cd transformer

# 安装依赖
pip install -r requirements.txt

# GPU额外依赖 (可选)
pip install flash-attn --no-build-isolation
```

核心依赖：
- PyTorch >= 2.0.0
- tokenizers >= 0.13.0 (HuggingFace Rust实现)
- NumPy >= 1.24.0
- tqdm, psutil, PyYAML

---

## 编码规范

1. **所有代码文件使用 UTF-8 编码**
2. **使用 `torch.amp.autocast('cuda', ...)` 而非已废弃的 `autocast(device_type='cuda')`**
3. **HuggingFace tokenizers 保存的 merges 是列表格式，加载时需兼容处理**
