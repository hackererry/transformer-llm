# Transformer大模型训练框架

一个从零开始构建的PyTorch大语言模型(LLM)训练框架，支持CPU和GPU训练，针对RTX 4060Ti 8G GPU进行了专门优化。

## 功能描述

### 核心能力

| 功能 | 说明 |
|------|------|
| **预训练** | 从零开始训练基础语言模型 |
| **指令微调(SFT)** | 使用监督学习微调模型遵循指令 |
| **文本生成** | 支持多种解码策略的文本生成 |
| **模型续训** | 支持加载已有模型继续训练 |
| **断点续训** | 支持从检查点恢复训练 |

### 优化特性

- **CPU优化**: 梯度检查点、内存管理、多进程数据加载、BF16混合精度
- **GPU优化**: BF16/FP16混合精度、Flash Attention、梯度检查点、梯度累积
- **中文支持**: BPE分词器原生支持中文字符
- **优雅退出**: Ctrl+C手动终止训练，自动保存模型

---

## 模型算法

### Transformer架构

本框架实现标准的Transformer Decoder架构，包含以下核心组件：

```
输入Token → 词嵌入 → RoPE位置编码 → [×N层] → RMSNorm → LM Head → 输出概率
                                                            ↑
                                                    KV Cache (推理加速)
```

### 核心组件

| 组件 | 实现 | 说明 |
|------|------|------|
| **位置编码** | RoPE (Rotary Position Embedding) | 旋转位置编码，支持长序列 |
| **归一化** | RMSNorm | 比LayerNorm更高效的归一化方式 |
| **激活函数** | SwiGLU | LLaMA风格的门控线性单元 |
| **注意力机制** | Multi-Head Attention | 多头注意力，支持KV缓存 |
| **前馈网络** | SwiGLU FFN | 三层MLP with Gated activation |

### 算法特点

1. **RoPE位置编码**
   - 通过旋转矩阵实现相对位置感知
   - 无需位置嵌入表，支持更长序列外推

2. **RMSNorm归一化**
   - 仅计算RMS（均方根），减少计算量
   - 公式: `y = x * (w / sqrt(mean(x^2) + eps))`

3. **SwiGLU激活**
   - 门控机制 + SiLU激活: `gate(x) * SiLU(x)`
   - 相比ReLU/GELU有更好的表达能力

### 预设模型配置

| 配置 | 参数量 | hidden_size | layers | heads | intermediate_size |
|------|--------|-------------|--------|-------|-------------------|
| tiny | ~10M | 256 | 6 | 8 | 512 |
| small | ~100M | 512 | 12 | 8 | 1024 |
| medium | ~500M | 1024 | 24 | 16 | 2048 |

---

## 代码架构

### 目录结构

```
transformer-llm/
├── src/
│   ├── model/                 # 模型实现
│   │   ├── config.py          # 模型配置类
│   │   ├── transformer.py     # CausalLM主模型
│   │   ├── attention.py       # 注意力机制(CPU)
│   │   ├── attention_gpu.py   # 注意力机制(GPU+Flash)
│   │   ├── layers.py          # FFN/RMSNorm/LayerNorm
│   │   ├── embedding.py       # 词嵌入+RoPE
│   │   └── lm_head.py         # 语言模型头
│   ├── data/                  # 数据处理
│   │   ├── tokenizer.py       # BPE分词器
│   │   ├── dataset.py         # 预训练/微调数据集
│   │   └── collator.py        # 数据整理
│   ├── training/              # 训练逻辑
│   │   ├── trainer.py         # CPU训练器
│   │   ├── trainer_gpu.py     # GPU训练器
│   │   ├── optimizer.py       # AdamW优化器
│   │   ├── scheduler.py      # 学习率调度
│   │   └── checkpoint.py      # 检查点管理
│   ├── cpu_optim/             # CPU优化
│   │   ├── gradient_checkpoint.py
│   │   ├── memory.py
│   │   └── parallel.py
│   └── utils/                 # 工具函数
│       ├── device.py
│       ├── logging.py
│       └── metrics.py
├── scripts/                   # 训练脚本
│   ├── pretrain.py           # CPU预训练
│   ├── pretrain_gpu.py       # GPU预训练
│   ├── finetune.py           # 指令微调
│   └── generate.py           # 文本生成
├── configs/                   # 配置文件
│   ├── model/                # 模型配置
│   └── training/             # 训练配置
├── tests/                     # 测试用例
├── dataset/                   # 数据处理工具
└── setup.py                   # 安装配置
```

### 模块依赖

```
scripts/
    ↓
trainer.py / trainer_gpu.py
    ↓
model/config.py  ←  model/transformer.py  ←  model/layers.py
    ↓                              ↓
data/tokenizer.py              model/embedding.py
    ↓                              ↓
data/dataset.py             model/attention.py
    ↓
training/optimizer.py
training/scheduler.py
training/checkpoint.py
```

### 关键类说明

| 类名 | 位置 | 职责 |
|------|------|------|
| `ModelConfig` | src/model/config.py | 模型超参数配置 |
| `CausalLMModel` | src/model/transformer.py | 主模型类 |
| `BPETokenizer` | src/data/tokenizer.py | 分词器 |
| `Trainer` | src/training/trainer.py | CPU训练引擎 |
| `GPUTrainer` | src/training/trainer_gpu.py | GPU训练引擎 |

---

## 执行脚本

### 1. 预训练 (CPU)

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

### 2. 预训练 (GPU)

```bash
python scripts/pretrain_gpu.py \
    --train_file data/train.txt \
    --model_config small \
    --output_dir output/gpu_pretrain \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --bf16 \
    --use_flash_attention \
    --gradient_checkpointing
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

### 5. 继续训练

```bash
# 加载已有模型继续训练
python scripts/pretrain.py \
    --train_file data/new_corpus.txt \
    --model_path output/pretrain/final_model \
    --output_dir output/continued \
    --num_train_epochs 2

# 从检查点恢复
python scripts/pretrain.py \
    --train_file data/train.txt \
    --resume_from_checkpoint output/pretrain/checkpoint-500 \
    --output_dir output/pretrain
```

---

## 参数解释

### 预训练参数 (pretrain.py / pretrain_gpu.py)

#### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_file` | 必填 | 训练数据路径（每行一段文本） |
| `--validation_file` | None | 验证数据路径 |
| `--vocab_size` | 32000 | Tokenizer词表大小 |

#### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_config` | tiny | 模型规模: tiny/small/medium |
| `--max_seq_length` | 512 | 最大序列长度 |

#### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | ./output | 输出目录 |
| `--num_train_epochs` | 3 | 训练轮数 |
| `--per_device_train_batch_size` | 4 | 每设备批次大小 |
| `--gradient_accumulation_steps` | 1 | 梯度累积步数 |
| `--max_steps` | -1 | 最大步数(-1=不限制) |

#### 优化器参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | 5e-5 | 学习率 |
| `--weight_decay` | 0.01 | 权重衰减 |
| `--max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `--lr_scheduler_type` | cosine | 调度器: cosine/linear/constant |
| `--warmup_ratio` | 0.1 | 预热比例 |

#### 精度与优化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--bf16` | False | BF16混合精度 |
| `--fp16` | False | FP16混合精度 |
| `--gradient_checkpointing` | False | 梯度检查点 |
| `--use_flash_attention` | False | Flash Attention(GPU) |
| `--num_workers` | 0 | 数据加载进程数 |

#### 日志与保存

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--logging_steps` | 10 | 日志间隔 |
| `--save_steps` | 500 | 保存间隔 |
| `--save_total_limit` | 3 | 最多保存检查点数 |

#### 加载与恢复

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | None | 加载已有模型 |
| `--resume_from_checkpoint` | None | 从检查点恢复 |
| `--seed` | 42 | 随机种子 |

### 微调参数 (finetune.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train_file` | 必填 | 微调数据(JSON格式) |
| `--model_path` | None | 预训练模型路径 |
| `--template` | alpaca | 指令模板: alpaca/chat/simple |
| `--learning_rate` | 2e-5 | 微调学习率 |

### 生成参数 (generate.py)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必填 | 模型路径 |
| `--prompt` | "" | 输入提示 |
| `--max_new_tokens` | 100 | 最大生成长度 |
| `--temperature` | 1.0 | 温度(越高越随机) |
| `--top_k` | 50 | Top-k采样 |
| `--top_p` | 0.95 | Top-p/核采样 |
| `--do_sample` | True | 是否采样 |
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

JSON数组：
```json
[
  {"instruction": "任务指令", "input": "输入", "output": "输出"},
  {"instruction": "翻译", "input": "Hello", "output": "你好"}
]
```

---

## GPU显存估算 (RTX 4060Ti 8G)

| 模型 | 参数 | 估算显存 |
|------|------|----------|
| Tiny | 60M | ~1.2 GB |
| Small | 200M | ~4 GB |
| Medium | 700M | ~8 GB |

推荐配置 (8G显存):
- Tiny: batch_size=16, gradient_accumulation=2
- Small: batch_size=8, gradient_accumulation=4
- Medium: batch_size=4, gradient_accumulation=8

---

## 安装与依赖

```bash
# 克隆仓库
git clone https://github.com/hackererry/transformer-llm.git
cd transformer-llm

# 安装依赖
pip install -r requirements.txt

# GPU额外依赖 (可选)
pip install flash-attn --no-build-isolation
```

### 核心依赖

- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- tqdm, psutil, PyYAML

---

## 许可证

MIT License
