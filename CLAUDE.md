# CLAUDE.md

## 项目概述

这是一个基于 PyTorch 的 LLM（大语言模型）训练框架，支持预训练和微调(SFT)，针对 CPU 和 GPU 训练进行了优化。

## 核心架构

```
src/
├── model/          # 模型层 - Transformer架构实现
├── data/           # 数据层 - tokenizer、数据集、collator
├── training/       # 训练层 - 统一Trainer、optimizer、scheduler
└── utils/          # 工具层 - 设备管理、日志、性能监控
```

## 核心模块说明

### 1. 模型层 (src/model/)

| 文件 | 说明 |
|------|------|
| `config.py` | `ModelConfig` - 模型配置（hidden_size, num_layers, num_heads等） |
| `transformer.py` | `CausalLMModel` - 因果语言模型主类 |
| `attention.py` | `Attention`, `FlashAttention` - 注意力机制 |
| `embedding.py` | `RotaryEmbedding` - RoPE旋转位置编码 |
| `layers.py` | `RMSNorm`, `FeedForward`, `SwiGLUFFN` - 标准化和FFN层 |
| `lm_head.py` | `LMHead` - 语言模型输出头 |
| `attention_gpu.py` | GPU优化的Flash Attention实现 |

### 2. 数据层 (src/data/)

| 文件 | 说明 |
|------|------|
| `tokenizer.py` | `BPETokenizer` - 简单BPE实现<br>`HuggingFaceBPETokenizer` - HuggingFace tokenizers封装（Rust实现，21x加速） |
| `dataset.py` | `PretrainDataset` - 原始文本预训练数据集<br>`FinetuneDataset` - 指令微调数据集 |
| `preprocessed_dataset.py` | `ShardedPreprocessedDataset` - 分片预处理数据集（避免OOM） |
| `collator.py` | `DataCollatorForCausalLM` - 数据整理器 |

### 3. 训练层 (src/training/)

| 文件 | 说明 |
|------|------|
| `trainer.py` | `Trainer` - **统一训练器**（自动检测CPU/GPU） |
| `optimizer.py` | AdamW, LAMB 优化器 |
| `scheduler.py` | CosineAnnealing, Linear 等学习率调度器 |
| `checkpoint.py` | 检查点保存/加载 |

## 重要规则

### 规则1：代码修改后必须执行测试

**任何代码修改后，都必须执行对应的测试程序：**

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v
```

**按以下规则选择测试：**

| 修改内容 | 测试文件 | 命令 |
|---------|---------|------|
| 修改 `src/model/` | `tests/test_model.py` | `pytest tests/test_model.py -v` |
| 修改 `src/data/` | `tests/test_data.py` | `pytest tests/test_data.py -v` |
| 修改 `src/training/` | `tests/test_training.py` | `pytest tests/test_training.py -v` |
| 修改多个模块 | `tests/` | `pytest tests/ -v` |

### 规则2：关联影响检查

**程序如果有大的修改，需要考虑关联影响：**

1. **检查调用关系**：分析修改的函数/类被哪些其他模块调用
2. **影响评估**：评估修改是否会影响其他程序的正常使用
3. **同步修改**：如果影响其他模块，需要同比修改相关代码
4. **更新测试**：如果修改影响到其他模块的接口或行为，需要同步修改测试代码

```python
# 示例：修改 transformer.py 中的 CausalLMModel
# 步骤1：检查谁调用了 CausalLMModel
grep -r "CausalLMModel" src/

# 步骤2：评估影响范围
# 步骤3：如需修改，同步更新
# 步骤4：更新测试代码
```

### 规则3：功能增强需先提交计划

**任何程序功能的增强都需要先给出计划，待用户确认后再执行。**

计划应包含：
- 功能目标：增强要实现什么
- 实现方案：大致的技术路线
- 影响范围：可能影响哪些现有模块
- 测试策略：如何验证增强的正确性

```markdown
## 功能增强计划：XXX

### 目标
[描述要实现的功能]

### 实现方案
[描述大致的技术路线]

### 影响范围
[列出可能影响的模块]

### 测试策略
[描述验证方法]
```

**确认后再执行，实施后必须运行测试验证。**

### 规则4：预处理数据格式

预处理后的数据保存在 `output/preprocessed/` 目录：
```
output/preprocessed/
├── tokenizer/           # 训练好的tokenizer
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── train_000.pt        # 训练分片（.pt格式）
├── train_001.pt
├── val_000.pt          # 验证分片
└── dataset_info.json    # 数据集元信息
```

### 规则5：代码文件、架构或脚本参数大的变动时，需要更新README.md，涉及到引入新库的，需要更新requirements.txt

### 规则6：统一训练器使用规范

`Trainer` 自动检测设备并选择最佳配置：
- **GPU可用时**：自动使用GPU训练，RTX 30/40系列自动选择BF16
- **GPU不可用时**：自动回退到CPU训练

关键参数：
```python
TrainingConfig(
    bf16=True,           # 自动检测
    fp16=False,
    gradient_checkpointing=False,  # CPU不支持
    dataloader_num_workers=0,      # CPU训练建议0
)
```

## 关键特性

### 1. 自动设备检测
```python
# pretrain.py 会自动检测并选择设备
if torch.cuda.is_available():
    # GPU训练，优先BF16
else:
    # CPU训练
```

### 2. 数据预处理 vs 原始数据

| 方式 | 命令 | 优点 |
|------|------|------|
| 预处理数据 | `--preprocessed_data output/preprocessed` | 训练时无需tokenize，更快 |

### 3. 支持的模型配置

```python
ModelConfig.tiny()    # 20M 参数 (6层, 256隐藏, 8头)
ModelConfig.small()   # 85M 参数
ModelConfig.medium()  # 305M 参数
```

## 常见命令

### 数据预处理
```bash
python scripts/preprocess_data.py \
    --train_file dataset/data/train.txt \
    --output_dir output/preprocessed \
    --num_shards 4
```

### 预训练
```bash
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config small \
    --num_train_epochs 3
```

### 继续训练
```bash
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_path output/final_model \
    --num_train_epochs 1
```

### 运行测试
```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v
```

## 编码规范

1. **所有代码文件使用 UTF-8 编码**
2. **使用 `torch.amp.autocast('cuda', ...)` 而非已废弃的 `autocast(device_type='cuda')`**
3. **HuggingFace tokenizers 保存的 merges 是列表格式，加载时需兼容处理**

## 项目状态

- ✅ 统一训练器（CPU/GPU自动检测）
- ✅ 数据预处理 + 分片加载
- ✅ HuggingFace tokenizers 集成
- ✅ 性能监控（Data Loading, Forward, Backward, Optimizer Step）
- ✅ 支持手动终止训练（Ctrl+C）自动保存
