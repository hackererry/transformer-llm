# 数据层 (src/data/)

Tokenizer、数据集、DataLoader 封装，支持预训练和微调。

## 目录结构

```
src/data/
├── tokenizer.py             # HuggingFaceBPETokenizer — Rust 实现 BPE（21x 加速）
├── dataset.py               # PretrainDataset / FinetuneDataset / TextFileDataset / MemoryMappedDataset
├── preprocessed_dataset.py  # PreprocessedDataset / ShardedPreprocessedDataset
└── collator.py             # DataCollatorForCausalLM / DataCollatorForSFT / DynamicBatchSampler
```

## 核心组件

### tokenizer.py — HuggingFaceBPETokenizer

基于 HuggingFace tokenizers 库的 Rust 实现 BPE Tokenizer，比纯 Python 快 100 倍以上。

| 类/函数 | 说明 |
|---------|------|
| `HuggingFaceBPETokenizer` | BPE 分词器，支持中英文 |
| `get_tokenizer()` | 工厂函数，自动加载或创建 Tokenizer |

**关键方法：**

| 方法 | 说明 |
|------|------|
| `encode(text)` | 单句编码 → token IDs |
| `decode(ids)` | token IDs → 文本 |
| `encode_batch(texts)` | 批量编码 |
| `decode_batch(ids)` | 批量解码 |
| `save(path)` | 保存到文件 |
| `load(path)` | 从文件加载 |
| `train_from_files(files)` | 从文件训练词表 |
| `get_vocab_size()` | 获取词表大小 |

### dataset.py — 数据集

| 类 | 说明 |
|-----|------|
| `PretrainDataset` | 预训练数据集，支持大文件高效加载和缓存 |
| `PretrainIterableDataset` | 流式预训练数据集（避免全量加载） |
| `FinetuneDataset` | 指令微调数据集，支持 Alpaca/ChatML 等模板 |
| `TextFileDataset` | 简单文本文件数据集 |
| `MemoryMappedDataset` | 内存映射数据集（超大文件） |
| `create_dataset()` | 工厂函数，自动选择数据集类型 |

**数据格式（FinetuneDataset）：**

```json
[
  {"instruction": "任务指令", "input": "输入", "output": "输出"},
  {"instruction": "翻译", "input": "Hello", "output": "你好"}
]
```

**指令模板：**

| 模板 | 格式 |
|------|------|
| alpaca | `Instruction: {instruction}\nInput: {input}\nOutput: {output}` |
| chat | ChatML 格式的对话 |
| simple | `{instruction}\n{input}\n{output}` |

### preprocessed_dataset.py — 分片预处理数据集

| 类 | 说明 |
|-----|------|
| `PreprocessedDataset` | 从 .pt 缓存加载，验证元数据兼容性 |
| `ShardedPreprocessedDataset` | 分片加载，避免 OOM |
| `save_preprocessed_data()` | 保存预处理数据到 .pt |
| `create_sharded_dataset()` | 工厂函数创建分片数据集 |

**缓存格式：**

```python
{
    "version": "1.0",
    "metadata": {
        "max_seq_length": 512,
        "vocab_size": 32000,
        "num_examples": 10000,
    },
    "examples": [
        {"input_ids": [...], "labels": [...]},
        ...
    ]
}
```

### collator.py — 数据整理器

| 类 | 说明 |
|-----|------|
| `DataCollatorForLanguageModeling` | 语言模型数据整理器，动态 padding |
| `DataCollatorForCausalLM` | 因果 LM 整理器（兼容新旧格式） |
| `DataCollatorForSFT` | SFT 微调整理器 |
| `DynamicBatchSampler` | 动态批次采样器 |
| `get_collator()` | 工厂函数 |

**动态 padding 策略：**

- `padding="longest"` — 批次内最长序列为基准
- `padding="max_length"` — 使用固定 max_length
- `truncation=True` — 超过 max_length 的序列截断
- Labels padding 使用 `-100`（交叉熵损失会忽略）

## 使用示例

```python
from src.data import (
    get_tokenizer,
    PretrainDataset,
    FinetuneDataset,
    ShardedPreprocessedDataset,
    get_collator,
)
from torch.utils.data import DataLoader

# Tokenizer
tokenizer = get_tokenizer("output/preprocessed/tokenizer")

# 方式1：从原始文本（训练时在线 tokenize）
dataset = PretrainDataset(
    data_path="dataset/data/train.txt",
    tokenizer=tokenizer,
    max_seq_length=512,
)

# 方式2：从预处理缓存（推荐，训练更快）
dataset = ShardedPreprocessedDataset(
    data_dir="output/preprocessed",
    split="train",
)

# Collator
collator = get_collator(tokenizer, max_seq_length=512)

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=8,
    collate_fn=collator,
    shuffle=True,
)
```

## 性能优化

- **预处理 + 分片** — 训练时无需 tokenize，20x 加速
- **ShardedPreprocessedDataset** — 分片懒加载，避免 OOM
- **MemoryMappedDataset** — 超大文件的内存映射
- **动态 padding** — 减少无效计算
