# Transformer 大模型训练框架

一个基于 PyTorch 的大语言模型（LLM）训练框架，提供从数据获取到模型部署的完整流程，支持预训练和指令微调（SFT），针对 CPU 和 GPU 进行了深度优化。

## 功能特性

| 功能 | 说明 |
|------|------|
| **数据获取** | 专业网络爬虫系统，支持多站点批量爬取、反反爬虫 |
| **数据清洗** | 文本清洗、文档格式转换（PDF/CSV/JSON/EPUB） |
| **数据预处理** | BPE分词器、增量预处理、分片缓存 |
| **预训练** | 从零训练基础语言模型，自动检测CPU/GPU |
| **指令微调** | SFT微调，支持多种指令模板 |
| **文本生成** | 多种解码策略、交互模式 |
| **推理优化** | Speculative Decoding、Streaming Attention |
| **训练优化** | BF16/FP16混合精度、Flash Attention、GQA、YaRN |

---

## 完整工作流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           数据准备流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  方式一: 从网络爬取                  方式二: 从文档转换                    │
│  ┌─────────────────┐               ┌─────────────────┐                  │
│  │  网络爬虫系统    │               │  文档格式转换    │                  │
│  │  src/crawler/   │               │  PDF/EPUB/CSV   │                  │
│  └────────┬────────┘               └────────┬────────┘                  │
│           │                                 │                           │
│           ▼                                 ▼                           │
│  ┌─────────────────┐               ┌─────────────────┐                  │
│  │  爬取的文本数据  │               │  转换的文本文件  │                  │
│  └────────┬────────┘               └────────┬────────┘                  │
│           │                                 │                           │
│           └──────────────┬──────────────────┘                           │
│                          ▼                                              │
│                 ┌─────────────────┐                                     │
│                 │   文本清洗       │                                     │
│                 │   去除噪声/规范化 │                                     │
│                 └────────┬────────┘                                     │
│                          ▼                                              │
│                 ┌─────────────────┐                                     │
│                 │   数据预处理     │                                     │
│                 │   Tokenize/分片  │                                     │
│                 └────────┬────────┘                                     │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           模型训练流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                 ┌─────────────────┐                                     │
│                 │   预训练         │ ─────────────────────┐              │
│                 │   Pretrain       │                      │              │
│                 └────────┬────────┘                      │              │
│                          │                               │              │
│                          ▼                               │              │
│                 ┌─────────────────┐                      │              │
│                 │   基础模型       │                      │              │
│                 └────────┬────────┘                      │              │
│                          │                               │              │
│                          ▼                               │              │
│                 ┌─────────────────┐                      │              │
│                 │   指令微调       │                      │              │
│                 │   SFT Finetune   │                      │              │
│                 └────────┬────────┘                      │              │
│                          │                               │              │
│                          ▼                               │              │
│                 ┌─────────────────┐                      │              │
│                 │   微调模型       │ ◄────────────────────┘              │
│                 └────────┬────────┘   (可选: 跳过微调)                   │
│                          │                                              │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           推理生成流程                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                 ┌─────────────────┐                                     │
│                 │   文本生成       │                                     │
│                 │   Generate       │                                     │
│                 └─────────────────┘                                     │
│                          │                                              │
│              ┌───────────┴───────────┐                                  │
│              ▼                       ▼                                  │
│     ┌─────────────────┐     ┌─────────────────┐                         │
│     │   标准生成       │     │   推理加速       │                         │
│     │   Standard       │     │   Speculative   │                         │
│     └─────────────────┘     └─────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 项目架构

```
transformer/
├── src/                              # 核心源代码
│   ├── model/                        # 模型层 - Transformer架构
│   │   ├── config.py                # ModelConfig - 模型配置
│   │   ├── transformer.py           # CausalLMModel - 因果语言模型
│   │   ├── attention/               # 注意力机制模块
│   │   │   ├── base.py              # 基础注意力
│   │   │   ├── standard.py          # 标准注意力
│   │   │   ├── flash.py             # Flash Attention
│   │   │   ├── gqa.py               # 分组查询注意力
│   │   │   └── streaming.py         # 流式注意力
│   │   ├── embedding.py             # RoPE旋转位置编码
│   │   ├── layers.py                # RMSNorm, SwiGLU FFN
│   │   └── lm_head.py               # 语言模型头
│   │
│   ├── data/                        # 数据层
│   │   ├── tokenizer.py             # HuggingFaceBPETokenizer
│   │   ├── dataset.py               # PretrainDataset, FinetuneDataset
│   │   ├── preprocessed_dataset.py  # 分片预处理数据集
│   │   └── collator.py              # 数据整理器
│   │
│   ├── training/                    # 训练层
│   │   ├── trainer.py               # 统一训练器（CPU/GPU自动检测）
│   │   ├── optimizer.py             # AdamW, LAMB优化器
│   │   ├── scheduler.py             # 学习率调度器
│   │   └── checkpoint.py            # 检查点管理
│   │
│   ├── data_processing/             # 数据处理
│   │   ├── clean_text.py            # 文本清洗
│   │   └── document_converter.py    # 文档格式转换
│   │
│   ├── crawler/                     # 网络爬虫系统
│   │   ├── cli.py                   # 命令行接口
│   │   ├── engine.py                # 爬虫引擎
│   │   ├── config.py                # 配置管理
│   │   ├── browser/                 # 浏览器自动化
│   │   ├── anti_crawler/            # 反反爬虫
│   │   └── storage/                 # 数据存储
│   │
│   ├── inference/                   # 推理模块
│   │   └── speculative.py           # Speculative Decoding
│   │
│   ├── cpu_optim/                   # CPU优化
│   │   ├── gradient_checkpoint.py   # 梯度检查点
│   │   ├── memory.py                # 内存优化
│   │   └── parallel.py              # 并行化支持
│   │
│   └── utils/                       # 工具层
│       ├── device.py                # 设备管理
│       ├── logging.py               # 日志记录
│       ├── metrics.py               # 性能指标
│       └── profiling.py             # 性能分析
│
├── scripts/                         # 训练脚本
│   ├── preprocess_data.py           # 数据预处理
│   ├── pretrain.py                  # 预训练
│   ├── finetune.py                  # 指令微调
│   └── generate.py                  # 文本生成
│
├── configs/                         # 配置文件
│   └── crawler/                     # 爬虫配置
│
├── tests/                           # 测试用例
└── dataset/                         # 数据目录
```

---

## 一、数据获取

### 1.1 网络爬虫系统

专业的网络爬虫系统，支持多站点批量爬取、反反爬虫、自动文本清洗。

**功能特性：**
- 多站点批量爬取
- 自动文本清洗
- robots.txt 合规检查
- 异步爬取支持
- SQLite数据库存储
- 断点续爬支持

**使用方法：**

```bash
# 批量爬取（使用默认配置）
python -m src.crawler.cli run

# 指定配置文件和并行数
python -m src.crawler.cli run --config configs/crawler/crawler_config.yaml --parallel 3

# 指定输出目录
python -m src.crawler.cli run --output-dir ./output/crawled --parallel 2

# 查看爬虫状态
python -m src.crawler.cli status
```

**配置文件格式（YAML）：**

```yaml
global:
  delay: 2.0          # 请求间隔（秒）
  max_pages: 30       # 每个站点最多爬取页数
  depth: 3            # 爬取深度
  max_concurrent: 2   # 每个站点最大并发数

sites:
  - name: example
    start_url: https://example.com/
    max_pages: 30
    content_type: news
    delay: 2.0
```

**内容类型说明：**

| 类型 | 说明 |
|------|------|
| `news` | 新闻资讯类，自动提取正文 |
| `novel` | 小说/文学类，保留段落格式 |
| `wiki` | 百科/知识类，提取主要内容 |
| `auto` | 自动检测（默认） |

---

## 二、数据清洗

### 2.1 文本清洗工具

对转换后的文本进行清洗和预处理。

**功能特性：**
- 移除多余空白
- 修复编码问题
- 移除页码
- 移除URL/邮箱
- 分割大文件

**使用方法：**

```bash
# 清洗单个文件
python -m src.data_processing.clean_text clean input.txt -o cleaned.txt

# 清洗并移除URL和邮箱
python -m src.data_processing.clean_text clean input.txt --remove-urls --remove-emails

# 批量清洗目录
python -m src.data_processing.clean_text clean -d ./raw_data -o ./cleaned_data

# 分割大文件
python -m src.data_processing.clean_text split large.txt -o ./chunks --max-chars 1000000
```

### 2.2 文档格式转换

支持 PDF、CSV、JSON、EPUB 格式转换为纯文本。

**功能特性：**
- 支持多种文档格式（PDF, CSV, JSON, EPUB）
- 支持单个文件转换
- 支持批量目录转换
- **支持并发转换**（自动使用 CPU 核数）
- 自动提取元数据
- 支持合并多个文件

**使用方法：**

```bash
# 转换单个文件
python -m src.data_processing.document_converter book.epub
python -m src.data_processing.document_converter document.pdf
python -m src.data_processing.document_converter data.csv
python -m src.data_processing.document_converter data.json

# 指定输出路径
python -m src.data_processing.document_converter book.epub -o output.txt

# 批量转换目录（默认使用 CPU 核数并发）
python -m src.data_processing.document_converter -d ./documents -o ./txt_output

# 指定并发线程数
python -m src.data_processing.document_converter -d ./documents -w 4

# 批量转换并合并为一个文件
python -m src.data_processing.document_converter -d ./documents --merge merged_training.txt
```

---

## 三、数据预处理

### 3.1 预处理脚本

将原始文本转换为模型可用的训练数据。

**功能特性：**
- BPE分词器训练
- 增量处理模式
- 分片缓存（避免OOM）
- 多文件支持

**使用方法：**

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
    --output_dir output/preprocessed

# 方式3: 带验证集
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --validation_file dataset/data/val.txt \
    --output_dir output/preprocessed

# 方式4: 增量添加新数据（默认模式）
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed

# 方式5: 查看变化但不处理（dry-run）
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --dry-run

# 方式6: 完全重新处理
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --full

# 方式7: Tokenizer 扩展模式
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --tokenizer-mode extend
```

### 3.2 增量预处理功能

数据预处理脚本支持增量处理，避免重复处理未变化的数据：

| 场景 | 行为 |
|------|------|
| **新增文件** | 只处理新文件，追加分片 |
| **修改文件** | 删除旧分片，重新处理该文件 |
| **删除文件** | 保留已处理分片，标记为 `orphaned` |
| **文件未变化** | 跳过处理 |

**Tokenizer 处理模式：**

| 模式 | 说明 |
|------|------|
| `frozen` | 使用现有 tokenizer（默认） |
| `extend` | 检测新词汇，必要时扩展词表（触发全量重处理） |
| `retrain` | 重新训练 tokenizer（触发全量重处理） |

### 3.3 预处理参数

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
| `--incremental` | True | 启用增量处理模式（默认） |
| `--full` | False | 完全重新处理所有数据 |
| `--dry-run` | False | 仅检测变化，不实际处理 |
| `--tokenizer-mode` | frozen | Tokenizer 模式: frozen/extend/retrain |

### 3.4 输出格式

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

---

## 四、模型预训练

### 4.1 预训练脚本

从零开始训练基础语言模型，自动检测 CPU/GPU。

**使用方法：**

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

# GPU上使用BF16（自动检测）
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config small

# CPU训练（自动检测）
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config tiny \
    --num_train_epochs 1
```

### 4.2 预设模型配置

| 配置 | 参数量 | hidden_size | layers | heads | intermediate_size |
|------|--------|-------------|--------|-------|-------------------|
| tiny | ~10M | 256 | 6 | 8 | 512 |
| small | ~100M | 512 | 12 | 8 | 1024 |
| medium | ~500M | 1024 | 24 | 16 | 2048 |

### 4.3 训练优化选项

**GPU优化（自动启用）：**
- BF16/FP16 混合精度（根据GPU自动选择）
- Flash Attention（默认启用）
- GQA 分组查询注意力（默认启用）
- YaRN 长度外推（支持4倍外推）
- 梯度检查点（可选）

**使用示例：**

```bash
# 禁用 Flash Attention
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --no_flash_attention

# 禁用 GQA
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --no_gqa

# 自定义 YaRN 外推因子
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --rope_scaling_factor 8.0

# 启用梯度检查点（节省显存）
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --gradient_checkpointing
```

### 4.4 预训练参数

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
| `--no_flash_attention` | False | 禁用Flash Attention |
| `--no_gqa` | False | 禁用GQA |
| `--rope_scaling_factor` | 4.0 | YaRN长度外推因子 |
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

---

## 五、模型微调

### 5.1 微调脚本

使用监督学习微调模型遵循指令。

**使用方法：**

```bash
python scripts/finetune.py \
    --train_file data/instructions.json \
    --model_path output/final_model \
    --output_dir output/sft \
    --num_train_epochs 3 \
    --learning_rate 2e-5
```

### 5.2 微调数据格式

JSON数组格式：

```json
[
  {"instruction": "任务指令", "input": "输入", "output": "输出"},
  {"instruction": "翻译", "input": "Hello", "output": "你好"}
]
```

### 5.3 指令模板

| 模板 | 说明 |
|------|------|
| `alpaca` | Alpaca格式（默认） |
| `chat` | 对话格式 |
| `simple` | 简单格式 |

### 5.4 微调参数

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

---

## 六、模型生成

### 6.1 生成脚本

使用训练好的模型生成文本。

**使用方法：**

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

### 6.2 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | **必填** | 模型路径 |
| `--tokenizer_path` | None | Tokenizer路径 |
| `--prompt` | "" | 输入提示 |
| `--max_new_tokens` | 100 | 最大生成token数 |
| `--temperature` | 1.0 | 温度参数 |
| `--top_k` | 50 | Top-k采样 |
| `--top_p` | 0.95 | Top-p采样 |
| `--do_sample` | True | 是否采样 |
| `--num_return_sequences` | 1 | 返回序列数 |
| `--seed` | 42 | 随机种子 |
| `--interactive` | False | 交互模式 |

### 6.3 采样参数详解

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
  - 1.0: 禁用

### 6.4 推理加速（Speculative Decoding）

使用 Speculative Decoding 进行推理加速，可实现 2-3x 加速。

```python
from src.inference.speculative import speculative_generate

# 使用 Speculative Decoding 生成
output = speculative_generate(
    draft_model=draft_model,    # 小模型（快速生成候选）
    target_model=target_model,  # 大模型（验证）
    input_ids=input_ids,
    num_draft_tokens=4,         # 每次生成的候选token数
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
)
```

---

## 七、完整数据准备流程

### 方式一：从文档准备数据

```bash
# 1. 转换文档为TXT
python -m src.data_processing.document_converter -d ./raw_documents --merge raw_data.txt

# 2. 清洗文本
python -m src.data_processing.clean_text clean raw_data.txt -o cleaned_data.txt

# 3. (可选) 分割大文件
python -m src.data_processing.clean_text split cleaned_data.txt -o ./train_chunks

# 4. 数据预处理
python scripts/preprocess_data.py --train_dir ./train_chunks --output_dir output/preprocessed

# 5. 开始预训练
python scripts/pretrain.py --preprocessed_data output/preprocessed --model_config small
```

### 方式二：从网络爬取数据

```bash
# 1. 爬取网站内容（使用配置文件批量爬取）
python -m src.crawler.cli run --config configs/crawler/crawler_config.yaml --parallel 2

# 2. 清洗爬取结果（爬虫已自动清洗，可视需要进一步处理）
python -m src.data_processing.clean_text clean ./crawled/example_com.txt -o cleaned_data.txt

# 3. 数据预处理
python scripts/preprocess_data.py --train_file cleaned_data.txt --output_dir output/preprocessed

# 4. 开始预训练
python scripts/pretrain.py --preprocessed_data output/preprocessed --model_config small
```

---

## 八、测试

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v
pytest tests/test_crawler.py -v

# 生成覆盖率报告
pytest tests/ --cov=src --cov-report=html
```

### 测试文件说明

| 文件 | 测试内容 |
|------|----------|
| `tests/test_model.py` | 模型配置、Transformer、注意力机制、RoPE、生成 |
| `tests/test_data.py` | Tokenizer、数据整理器、动态批采样 |
| `tests/test_training.py` | 优化器、调度器、检查点管理、训练器 |
| `tests/test_crawler.py` | 爬虫配置、解析器、文本提取、数据库 |

---

## 九、安装与依赖

### 安装

```bash
# 克隆仓库
git clone <repo_url>
cd transformer

# 安装依赖
pip install -r requirements.txt

# GPU额外依赖 (可选)
pip install flash-attn --no-build-isolation
```

### 核心依赖

```
# 核心依赖
torch>=2.0.0
numpy>=1.24.0
tqdm>=4.65.0
psutil>=5.9.0

# HuggingFace tokenizers (21x faster BPE)
tokenizers>=0.13.0
transformers>=4.30.0

# 网络爬虫依赖
requests>=2.28.0
beautifulsoup4>=4.12.0
lxml>=4.9.0
httpx>=0.25.0
playwright>=1.40.0

# 文本提取
pdfplumber>=0.10.0
pandas>=2.0.0
```

---

## 十、GPU显存估算

### RTX 4060Ti 8G 显存估算

| 模型 | 参数量 | 估算显存 |
|------|--------|----------|
| Tiny | ~10M | ~1 GB |
| Small | ~100M | ~4 GB |
| Medium | ~500M | ~8 GB |

### 推荐配置 (8G显存)

| 模型 | batch_size | gradient_accumulation |
|------|------------|----------------------|
| Tiny | 16 | 2 |
| Small | 8 | 4 |
| Medium | 4 | 8 |

---

## 十一、编码规范

1. **所有代码文件使用 UTF-8 编码**
2. **使用 `torch.amp.autocast('cuda', ...)` 而非已废弃的 `autocast(device_type='cuda')`**
3. **HuggingFace tokenizers 保存的 merges 是列表格式，加载时需兼容处理**

---

## 十二、注意事项

1. **EPUB文件**: 确保EPUB文件没有DRM保护
2. **编码**: 工具会自动处理UTF-8编码
3. **大文件**: 建议分割成小于1GB的文件
4. **版权**: 仅转换您有权使用的内容
5. **网络爬虫**:
   - 配置文件位于 `configs/crawler/crawler_config.yaml`
   - 遵守 robots.txt 规则（默认开启）
   - 设置合理的请求延迟，避免对服务器造成压力
   - 仅爬取公开、允许爬取的内容
   - 确保有权使用爬取的内容进行训练

---

## 项目状态

- ✅ 统一训练器（CPU/GPU自动检测）
- ✅ 数据预处理 + 分片加载
- ✅ HuggingFace tokenizers 集成
- ✅ 性能监控（Data Loading, Forward, Backward, Optimizer Step）
- ✅ 支持手动终止训练（Ctrl+C）自动保存
- ✅ 增量预处理
- ✅ Flash Attention
- ✅ GQA 分组查询注意力
- ✅ YaRN 长度外推
- ✅ Speculative Decoding
- ✅ 网络爬虫系统
- ✅ 文档格式转换
