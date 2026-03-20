# CPU大模型训练框架

一个从零开始构建的Transformer大语言模型训练框架，专门针对CPU训练进行优化。

## 特性

- **完整的Transformer实现**: 包含RoPE位置编码、RMSNorm、SwiGLU等现代LLM组件
- **CPU优化**: 梯度检查点、内存管理、多进程数据加载
- **灵活的配置系统**: 支持多种模型规模(tiny/small/medium)
- **预训练+微调**: 支持从头预训练和指令微调(SFT)
- **混合精度训练**: 支持BF16精度(CPU支持)

## 项目结构

```
transformer/
├── configs/                 # 配置文件
│   ├── model/              # 模型配置
│   ├── training/           # 训练配置
│   └── default.yaml
├── src/
│   ├── model/              # 模型实现
│   ├── data/               # 数据处理
│   ├── training/           # 训练逻辑
│   ├── cpu_optim/          # CPU优化
│   └── utils/              # 工具函数
├── scripts/                # 训练脚本
│   ├── pretrain.py         # 预训练
│   ├── finetune.py         # 微调
│   └── generate.py         # 文本生成
├── tests/                  # 测试
├── requirements.txt
└── README.md
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-repo/transformer.git
cd transformer

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

训练数据应该是纯文本文件，每行一个样本:

```
这是一行训练文本。
这是另一行训练文本。
...
```

对于微调，使用JSON/JSONL格式:

```json
[
  {"instruction": "翻译这句话", "input": "Hello", "output": "你好"},
  {"instruction": "回答问题", "input": "什么是AI?", "output": "AI是人工智能..."}
]
```

### 2. 预训练

```bash
python scripts/pretrain.py \
    --train_file data/train.txt \
    --validation_file data/val.txt \
    --model_config tiny \
    --output_dir output/pretrain \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --learning_rate 5e-5
```

### 3. 指令微调

```bash
python scripts/finetune.py \
    --train_file data/instructions.json \
    --model_path output/pretrain/final_model \
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
```

## 模型配置

支持三种预设配置:

| 配置 | 参数量 | hidden_size | layers | heads |
|------|--------|-------------|--------|-------|
| tiny | ~10M   | 256         | 6      | 8     |
| small| ~100M  | 512         | 12     | 8     |
| medium| ~500M | 1024        | 24     | 16    |

也可以通过配置文件自定义:

```yaml
# configs/model/custom.yaml
model:
  vocab_size: 32000
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 2048
```

## CPU优化策略

### 1. 梯度检查点

减少激活值内存，以计算换内存:

```bash
python scripts/pretrain.py \
    --gradient_checkpointing \
    ...
```

### 2. 梯度累积

模拟更大batch size:

```bash
python scripts/pretrain.py \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    ...
```

### 3. BF16混合精度

CPU支持BF16精度:

```bash
python scripts/pretrain.py \
    --bf16 \
    ...
```

## 架构细节

### Transformer组件

- **位置编码**: RoPE (旋转位置编码)
- **归一化**: RMSNorm (比LayerNorm更高效)
- **激活函数**: SwiGLU (LLaMA风格)
- **注意力**: Multi-Head Attention + KV缓存

### 代码示例

```python
from src import ModelConfig, CausalLMModel, Trainer

# 创建模型
config = ModelConfig.tiny()
model = CausalLMModel(config)

# 训练
trainer = Trainer(
    model=model,
    train_dataset=train_dataset,
    output_dir="./output",
    num_train_epochs=3,
)
trainer.train()

# 生成
output = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=0.8,
    top_p=0.95,
)
```

## 验证

运行测试:

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_model.py

# 带覆盖率
pytest --cov=src tests/
```

## 性能优化建议

1. **内存受限**: 使用`gradient_checkpointing`和较小的batch size
2. **速度优先**: 使用`bf16`和多进程数据加载
3. **大模型**: 考虑模型并行或流水线并行(未来支持)

## 依赖说明

核心依赖:
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- tqdm, psutil, PyYAML

可选依赖:
- transformers/tokenizers: HuggingFace tokenizer支持
- tensorboard: TensorBoard日志

---

## 参数说明

本节详细列出所有配置参数及其含义。

### 1. 模型参数 (ModelConfig)

模型配置定义在 `src/model/config.py` 中:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vocab_size` | 32000 | 词表大小，即模型认识的token数量 |
| `hidden_size` | 512 | 隐藏层维度，每个token的向量表示维度 |
| `num_hidden_layers` | 12 | Transformer层数（堆叠的Block数量） |
| `num_attention_heads` | 8 | 注意力头数，用于并行学习不同特征 |
| `intermediate_size` | 1024 | FFN中间层维度，通常为2-4倍hidden_size |
| `max_position_embeddings` | 2048 | 最大序列长度，超过则截断 |
| `rope_theta` | 10000.0 | RoPE位置编码的旋转基数 |
| `rms_norm_eps` | 1e-6 | RMSNorm的epsilon，防止除零 |
| `hidden_dropout` | 0.1 | 隐藏层dropout比率，防止过拟合 |
| `attention_dropout` | 0.1 | 注意力层dropout比率 |
| `use_cache` | True | 是否使用KV缓存（推理加速） |
| `tie_word_embeddings` | False | 是否共享输入输出词嵌入权重 |
| `gradient_checkpointing` | False | 是否启用梯度检查点（节省显存） |
| `head_dim` | (自动计算) | 每个注意力头的维度 = hidden_size / num_heads |

#### 预设模型配置

| 配置 | 参数量 | vocab_size | hidden_size | layers | heads | intermediate_size | max_seq_len |
|------|--------|------------|-------------|--------|-------|-------------------|-------------|
| `tiny` | ~10M | 32000 | 256 | 6 | 8 | 512 | 1024 |
| `small` | ~100M | 32000 | 512 | 12 | 8 | 1024 | 2048 |
| `medium` | ~500M | 32000 | 1024 | 24 | 16 | 2048 | 4096 |

---

### 2. 预训练脚本参数 (pretrain.py)

```bash
python scripts/pretrain.py [参数]
```

#### 数据参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_file` | 是 | - | 训练数据文件路径（每行一段文本） |
| `--validation_file` | 否 | None | 验证数据文件路径 |
| `--vocab_size` | 否 | 32000 | Tokenizer词表大小 |

#### 模型参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_config` | 否 | tiny | 模型规模: tiny, small, medium |

#### 训练参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--output_dir` | 否 | ./output | 模型输出目录 |
| `--num_train_epochs` | 否 | 3 | 训练轮数（完整遍历数据集次数） |
| `--per_device_train_batch_size` | 否 | 4 | 每设备批次大小（越大越稳定但越慢） |
| `--gradient_accumulation_steps` | 否 | 1 | 梯度累积步数（模拟大batch） |
| `--max_steps` | 否 | -1 | 最大训练步数（-1表示不限制） |
| `--max_seq_length` | 否 | 512 | 输入序列最大长度（超过截断） |

#### 优化器参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--learning_rate` | 否 | 5e-5 | 学习率（越大收敛越快但可能不稳定） |
| `--weight_decay` | 否 | 0.01 | 权重衰减（L2正则化，防止过拟合） |
| `--max_grad_norm` | 否 | 1.0 | 梯度裁剪阈值（防止梯度爆炸） |

#### 学习率调度参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--lr_scheduler_type` | 否 | cosine | 调度器类型: cosine, linear, constant |
| `--warmup_ratio` | 否 | 0.1 | 预热比例（前10%的步数线性增加学习率） |
| `--warmup_steps` | 否 | 0 | 预热步数（优先于warmup_ratio） |

#### 日志和保存参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--logging_dir` | 否 | ./logs | 日志输出目录 |
| `--logging_steps` | 否 | 10 | 日志打印间隔步数 |
| `--save_steps` | 否 | 500 | 模型保存间隔步数 |
| `--save_total_limit` | 否 | 3 | 最多保存的检查点数量 |

#### 精度和优化参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--bf16` | 否 | True | 是否使用BF16混合精度（CPU推荐） |
| `--gradient_checkpointing` | 否 | False | 启用梯度检查点（节省显存） |
| `--num_workers` | 否 | 0 | DataLoader工作进程数（多进程加载数据） |

#### 其他参数

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--seed` | 否 | 42 | 随机种子（保证结果可复现） |
| `--resume_from_checkpoint` | 否 | None | 从指定检查点恢复训练 |

---

### 3. 微调脚本参数 (finetune.py)

```bash
python scripts/finetune.py [参数]
```

微调支持预训练模型的大部分参数，以下是微调特有的参数:

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--train_file` | 是 | - | 微调数据文件（JSON/JSONL格式） |
| `--model_path` | 否 | None | 预训练模型路径（从头训练可不指定） |
| `--template` | 否 | alpaca | 指令模板: alpaca, chat, simple |
| `--learning_rate` | 否 | 2e-5 | 微调学习率（通常比预训练小） |
| `--warmup_ratio` | 否 | 0.05 | 微调预热比例（通常较小） |

#### 指令模板格式

**alpaca格式**:
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

**chat格式**:
```
User: {instruction}
{input}
Assistant: {output}
```

**simple格式**:
```
Instruction: {instruction}
Input: {input}
Output: {output}
```

---

### 4. 文本生成参数 (generate.py)

```bash
python scripts/generate.py [参数]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_path` | 是 | - | 模型路径 |
| `--tokenizer_path` | 否 | None | Tokenizer路径（可选） |
| `--prompt` | 否 | "" | 输入提示文本 |
| `--max_new_tokens` | 否 | 100 | 最大生成token数 |
| `--temperature` | 否 | 1.0 | 温度参数（越高越随机，越低越确定） |
| `--top_k` | 否 | 50 | Top-k采样（只从概率最高的k个token中采样） |
| `--top_p` | 否 | 0.95 | Top-p/核采样（从概率累积超过p的token中采样） |
| `--do_sample` | 否 | True | 是否使用采样（False则 greedy decoding） |
| `--num_return_sequences` | 否 | 1 | 返回的序列数量 |
| `--seed` | 否 | 42 | 随机种子 |
| `--interactive` | 否 | False | 交互模式（持续对话） |

#### 生成参数说明

- **temperature**:
  - 0.1-0.5: 输出更确定、更保守
  - 0.7-1.0: 平衡创造性和准确性
  - >1.0: 更随机，可能产生无意义内容

- **top_k**:
  - 较小的k（如10）：更保守的输出
  - 较大的k（如50）：更多样性
  - 0：禁用，等同于不使用top-k

- **top_p**:
  - 较低的p（如0.5）：更保守
  - 较高的p（如0.95）：更多样性
  - 1.0：禁用，等同于greedy

---

### 5. 数据格式说明

#### 预训练数据格式

纯文本文件，每行一个样本:
```
第一段文本内容...
第二段文本内容...
第三段文本内容...
```

#### 微调数据格式

JSON数组:
```json
[
  {
    "instruction": "任务指令",
    "input": "输入内容（可为空）",
    "output": "期望输出"
  }
]
```

JSONL（每行一个JSON）:
```jsonl
{"instruction": "翻译", "input": "Hello", "output": "你好"}
{"instruction": "问答", "input": "1+1=?", "output": "等于2"}
```

---

### 7. Tokenizer格式说明

框架支持两种tokenizer保存格式：

#### 格式对比

| 特性 | Legacy格式（旧版） | Unified格式（新版） |
|------|-------------------|-------------------|
| **文件数量** | 3个文件 | 1个文件 |
| **文件列表** | vocab.json, merges.txt, tokenizer_config.json | tokenizer.json |
| **兼容性** | 完全兼容旧版代码 | 兼容HuggingFace格式 |
| **推荐场景** | 向后兼容 | 新项目、模型分享 |

#### Legacy格式文件结构
```
tokenizer/
├── vocab.json           # 词表映射 {token: id}
├── merges.txt           # BPE合并规则
└── tokenizer_config.json  # 配置（特殊token等）
```

#### Unified格式文件结构
```
tokenizer/
└── tokenizer.json       # 一站式文件，包含所有内容
```

#### 使用方法

**保存tokenizer**:
```python
# 保存为旧版格式
tokenizer.save("output/tokenizer", format="legacy")

# 保存为新版一站式格式
tokenizer.save("output/tokenizer", format="unified")

# 同时保存两种格式（推荐）
tokenizer.save("output/tokenizer", format="both")
```

**加载tokenizer**:
```python
from src.data import BPETokenizer

# 自动检测格式（推荐）
tokenizer = BPETokenizer.load("output/tokenizer", format="auto")

# 强制使用旧版格式
tokenizer = BPETokenizer.load("output/tokenizer", format="legacy")

# 强制使用新版格式
tokenizer = BPETokenizer.load("output/tokenizer", format="unified")
```

#### tokenizer.json 结构示例
```json
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"<unk>": 0, "<pad>": 1, ...},
    "merges": ["a b", "b c", ...]
  },
  "pre_tokenizer": {
    "type": "Regex",
    "pattern": {...}
  },
  "special_tokens_map": {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "bos_token": "<s>",
    "eos_token": "</s>"
  },
  "config": {
    "vocab_size": 32000,
    "model_type": "bpe"
  }
}
```

---

### 6. 重要概念解释

| 概念 | 说明 |
|------|------|
| **Batch Size** | 每次前向计算使用的样本数，越大梯度越稳定 |
| **Gradient Accumulation** | 梯度累积，通过分步计算模拟大batch |
| **Learning Rate** | 学习率，控制参数更新幅度 |
| **Warmup** | 学习率预热，开始时逐渐增加学习率 |
| **Cosine Decay** | 余弦衰减，学习率按余弦曲线下降 |
| **Dropout** | 随机丢弃部分神经元，防止过拟合 |
| **Gradient Clipping** | 梯度裁剪，防止梯度爆炸 |
| **BF16** | Brain Float 16，16位浮点，CPU支持 |
| **Gradient Checkpointing** | 以计算换显存，训练时重新计算激活值 |

---

## 继续训练（增量训练）

当需要用新语料继续训练已有模型时，支持以下几种方式：

### 方式1：用新语料继续预训练

加载已有模型，使用新的训练数据进行增量预训练：

```bash
python scripts/pretrain.py \
    --train_file dataset/data/new_corpus.txt \
    --model_path output/test_pretrain/final_model \
    --output_dir output/continued_pretrain \
    --num_train_epochs 2 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 2
```

### 方式2：从中断的检查点恢复

如果训练过程中断，可以从最近的检查点恢复（保留优化器状态）：

```bash
python scripts/pretrain.py \
    --train_file dataset/data/train.txt \
    --resume_from_checkpoint output/test_pretrain/checkpoint-step-500 \
    --output_dir output/test_pretrain \
    --num_train_epochs 5
```

### 方式3：微调预训练模型

使用指令数据对预训练模型进行监督微调(SFT)：

```bash
python scripts/finetune.py \
    --train_file dataset/instructions.json \
    --model_path output/test_pretrain/final_model \
    --output_dir output/sft_model \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

### 关键参数对比

| 参数 | 用途 | 适用场景 |
|------|------|----------|
| `--model_path` | 加载已有模型的权重和tokenizer | 用新语料继续训练、微调 |
| `--resume_from_checkpoint` | 恢复完整训练状态（含优化器、step） | 训练中断后恢复 |
| `--model_config` | 指定模型规模配置 | 从头开始全新训练 |

### 继续训练注意事项

1. **Tokenizer兼容**: 使用 `--model_path` 时会自动加载对应的 tokenizer，确保词表一致
2. **学习率建议**: 继续训练时推荐使用较小的学习率（如 1e-5 ~ 3e-5），避免破坏已学习的知识
3. **输出目录**: 建议使用新的 `--output_dir`，避免覆盖原有的模型文件
4. **数据分布**: 新语料的分布最好与原训练数据相似，避免灾难性遗忘

---

## GPU训练（RTX 4060Ti 8G）

框架支持GPU训练，针对NVIDIA显卡优化。

### 环境要求

```bash
# CUDA环境（RTX 4060Ti需要CUDA 12.x）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 可选：Flash Attention（显著加速）
pip install flash-attn --no-build-isolation
```

### GPU训练脚本

```bash
# 使用GPU预训练
python scripts/pretrain_gpu.py \
    --train_file dataset/data/train.txt \
    --model_config small \
    --output_dir output/gpu_pretrain \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --max_seq_length 512 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --use_flash_attention \
    --gradient_checkpointing
```

### GPU训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `--bf16` | 使用BF16精度（RTX 40系推荐） | 默认开启 |
| `--fp16` | 使用FP16精度 | 旧显卡使用 |
| `--use_flash_attention` | 使用Flash Attention加速 | 推荐开启 |
| `--gradient_checkpointing` | 梯度检查点，省显存 | 大模型开启 |
| `--per_device_train_batch_size` | 批次大小 | 根据显存调整 |
| `--gradient_accumulation_steps` | 梯度累积 | 模拟大batch |

### 显存估算

框架会自动估算训练所需显存：

```
Memory Estimation:
  Model parameters: 60,000,000
  Model memory: 0.12 GB
  Gradient memory: 0.12 GB
  Optimizer memory: 0.48 GB
  Activation memory: 0.20 GB
  Total estimated: 0.92 GB
  Recommended GPU: 1.19 GB
```

### RTX 4060Ti 8G 推荐配置

| 模型规模 | batch_size | seq_length | 梯度累积 |
|----------|------------|------------|----------|
| Tiny (60M) | 16 | 512 | 2 |
| Small (200M) | 8 | 512 | 4 |
| Medium (700M) | 4 | 512 | 8 |

### 训练速度对比

| 配置 | Tiny (60M) | Small (200M) |
|------|------------|--------------|
| CPU (16核) | ~20s/step | ~120s/step |
| RTX 4060Ti | ~0.5s/step | ~3s/step |
| 提升倍数 | 40x | 40x |

### 手动终止训练

训练过程中可以随时按 **Ctrl+C** 终止训练，系统会自动保存：

- ✅ 模型权重 (pytorch_model.bin)
- ✅ 模型配置 (config.json)
- ✅ Tokenizer (vocab.json, merges.txt, tokenizer.json)

保存位置：`output_dir/final_model/`

---

## 许可证

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request!
