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
| **增量预处理** | 只处理变化的文件，支持新增/修改/删除检测 |
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
│   ├── data_processing/         # 数据清洗层 - 文本清洗和文档转换
│   │   ├── __init__.py          # 数据清洗模块导出
│   │   ├── clean_text.py        # TextCleaner - 文本清洗工具
│   │   └── document_converter.py # 文档格式转换（PDF/CSV/JSON/EPUB）
│   │
│   ├── cpu_optim/              # CPU优化模块
│   │   ├── gradient_checkpoint.py  # 梯度检查点
│   │   ├── memory.py           # 内存优化
│   │   └── parallel.py         # 并行化支持
│   │
│   ├── utils/                  # 工具层
│   │   ├── __init__.py         # 工具模块导出
│   │   ├── device.py           # 设备管理、内存信息
│   │   ├── logging.py          # 日志记录
│   │   └── metrics.py          # 性能指标计算
│   │
│   └── crawler/                # 专业爬虫系统
│       ├── __init__.py         # 爬虫模块导出
│       ├── cli.py              # 命令行接口（run/status）
│       ├── engine.py           # 爬虫引擎
│       ├── config.py           # 配置管理
│       ├── browser/            # 浏览器自动化
│       ├── anti_crawler/      # 反反爬虫
│       └── storage/           # 数据存储
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
│   ├── test_crawler.py         # 爬虫模块测试
│   └── conftest.py             # pytest配置
│
├── dataset/                     # 数据处理工具
│   ├── document_converter.py  # 文档格式转换（PDF/CSV/JSON/EPUB）
│   ├── clean_text.py          # 文本清洗
│   ├── finetuneData/          # 微调数据样例
│   └── data/                  # 训练数据
│
├── configs/crawler/           # 爬虫配置
│   └── crawler_config.yaml    # 批量爬取配置
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
| `attention.py` | `GroupedQueryAttention` | 分组查询注意力（GQA） |
| `attention.py` | `CrossAttention` | 交叉注意力（解码器） |
| `layers.py` | `RMSNorm` | RMS归一化 |
| `layers.py` | `SwiGLUFFN` | SwiGLU激活的前馈网络 |
| `layers.py` | `FeedForward` | 标准前馈网络 |
| `layers.py` | `TransformerMLP` | Transformer MLP层 |
| `layers.py` | `MLP` | 通用MLP层 |
| `layers.py` | `LayerNorm` | 层归一化 |
| `embedding.py` | `RotaryEmbedding` | RoPE旋转位置编码 |
| `embedding.py` | `apply_rotary_pos_emb` | 应用RoPE编码 |
| `lm_head.py` | `LMHead` | 语言模型输出头 |
| `lm_head.py` | `TiedLMHead` | 权重绑定的LM头 |
| `lm_head.py` | `AdaptiveLMHead` | 自适应LM头 |
| `lm_head.py` | `MLPHead` | MLP型LM头 |
| `lm_head.py` | `Pooler` | 池化层 |
| `lm_head.py` | `SequenceSummary` | 序列摘要层 |

### 2. 数据层 (src/data/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `tokenizer.py` | `HuggingFaceBPETokenizer` | HuggingFace tokenizers封装（Rust实现，21x加速） |
| `tokenizer.py` | `get_tokenizer` | 获取tokenizer的工厂函数 |
| `dataset.py` | `PretrainDataset` | 原始文本预训练数据集 |
| `dataset.py` | `PretrainIterableDataset` | 可迭代形式预训练数据集 |
| `dataset.py` | `FinetuneDataset` | 指令微调数据集 |
| `dataset.py` | `TextFileDataset` | 文本文件数据集 |
| `dataset.py` | `MemoryMappedDataset` | 内存映射数据集（节省内存） |
| `dataset.py` | `create_dataset` | 数据集工厂函数 |
| `preprocessed_dataset.py` | `PreprocessedDataset` | 预处理缓存数据集 |
| `preprocessed_dataset.py` | `ShardedPreprocessedDataset` | 分片预处理数据集（避免OOM） |
| `preprocessed_dataset.py` | `save_preprocessed_data` | 保存预处理数据 |
| `preprocessed_dataset.py` | `create_sharded_dataset` | 创建分片数据集 |
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

### 4. 数据清洗层 (src/data_processing/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `clean_text.py` | `TextCleaner` | 文本清洗器类 |
| `clean_text.py` | `clean_file()` | 清洗单个文件 |
| `clean_text.py` | `split_large_file()` | 分割大文件 |
| `clean_text.py` | `batch_clean_directory()` | 批量清洗目录 |
| `document_converter.py` | `convert_to_txt()` | 单文件格式转换 |
| `document_converter.py` | `batch_convert()` | 批量转换（支持并发） |
| `document_converter.py` | `merge_txt_files()` | 合并多个TXT文件 |
| `document_converter.py` | `PDFExtractor` | PDF文本提取器 |
| `document_converter.py` | `CSVExtractor` | CSV文本提取器 |
| `document_converter.py` | `JSONExtractor` | JSON文本提取器 |
| `document_converter.py` | `EPUBExtractor` | EPUB电子书提取器 |

### 5. 工具层 (src/utils/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `device.py` | `get_device` | 获取设备 |
| `device.py` | `set_seed` | 设置随机种子 |
| `device.py` | `get_memory_info` | 获取内存信息 |
| `device.py` | `print_device_info` | 打印设备信息 |
| `device.py` | `DeviceManager` | 设备管理器 |
| `device.py` | `get_device_info` | 获取设备详细信息 |
| `device.py` | `to_device` | 数据迁移到设备 |
| `device.py` | `enable_tf32` | 启用TF32计算 |
| `device.py` | `set_num_threads` | 设置CPU线程数 |
| `device.py` | `get_optimal_num_threads` | 获取最优线程数 |
| `logging.py` | `setup_logger` | 设置日志记录 |
| `logging.py` | `Logger` | 日志类 |
| `logging.py` | `TensorBoardLogger` | TensorBoard日志 |
| `logging.py` | `ProgressTracker` | 进度跟踪器 |
| `metrics.py` | `compute_perplexity` | 计算困惑度 |
| `metrics.py` | `compute_accuracy` | 计算准确率 |
| `metrics.py` | `compute_bleu_score` | 计算BLEU分数 |
| `metrics.py` | `compute_f1_score` | 计算F1分数 |
| `metrics.py` | `MetricsTracker` | 指标跟踪器 |

### 6. CPU优化层 (src/cpu_optim/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `gradient_checkpoint.py` | `gradient_checkpoint` | 梯度检查点装饰器 |
| `gradient_checkpoint.py` | `CheckpointedModule` | 检查点模块封装 |
| `gradient_checkpoint.py` | `CheckpointedSequential` | 顺序模块检查点封装 |
| `gradient_checkpoint.py` | `SelectiveCheckpoint` | 选择性检查点 |
| `gradient_checkpoint.py` | `enable_gradient_checkpointing` | 启用梯度检查点 |
| `gradient_checkpoint.py` | `estimate_memory_savings` | 估算内存节省 |
| `memory.py` | `MemoryMonitor` | 内存监控器 |
| `memory.py` | `MemoryOptimizer` | 内存优化器 |
| `memory.py` | `Offloader` | CPU卸载器 |
| `memory.py` | `optimize_for_cpu_training` | CPU训练内存优化 |
| `memory.py` | `memory_context` | 内存上下文管理器 |
| `parallel.py` | `ParallelDataLoader` | 并行数据加载器 |
| `parallel.py` | `ChunkedBatchSampler` | 分块批采样器 |
| `parallel.py` | `BucketBatchSampler` | 桶式批采样器 |
| `parallel.py` | `ParallelProcessor` | 并行处理器 |
| `parallel.py` | `get_optimal_num_workers` | 获取最优工作进程数 |
| `parallel.py` | `create_dataloader` | 数据加载器工厂函数 |

### 7. 爬虫系统 (src/crawler/)

| 文件 | 类/函数 | 说明 |
|------|---------|------|
| `cli.py` | `main` | 命令行入口（run/status） |
| `engine.py` | `crawl_sites` | 批量爬取站点 |
| `config.py` | `CrawlerConfig` | 爬虫配置类 |
| `config.py` | `CrawlerSettings` | 爬虫设置类 |
| `browser/playwright_manager.py` | `PlaywrightManager` | Playwright浏览器管理器 |
| `browser/page_interactions.py` | `PageInteractions` | 页面交互操作 |
| `anti_crawler/fingerprint.py` | `FingerprintGenerator` | 浏览器指纹生成 |
| `anti_crawler/proxy_pool.py` | `ProxyPool` | 代理池管理 |
| `anti_crawler/user_agent_pool.py` | `UserAgentPool` | User-Agent池 |
| `storage/database.py` | `Database` | SQLite数据库存储 |
| `storage/file_storage.py` | `FileStorage` | 文件存储管理器 |

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

#### 增量预处理功能 (v3.0)

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

## 数据预处理与爬虫工具

### 目录结构

| 目录 | 说明 |
|------|------|
| `dataset/` | 数据处理工具 |
| `src/crawler/` | 专业爬虫系统 |

### 1. 文档格式转换 (`dataset/document_converter.py`)

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
# 串行模式（单线程）
python -m src.data_processing.document_converter -d ./documents -w 1
# 批量转换并合并为一个文件
python -m src.data_processing.document_converter -d ./documents --merge merged_training.txt
```

### 2. 文本清洗 (`src/data_processing/clean_text.py`)

对转换后的文本进行清洗和预处理。

**功能特性：**
- 移除多余空白
- 修复编码问题
- 移除页码
- 移除URL/邮箱
- 分割大文件

**使用方法：**

```bash
# 清洗文本文件
python -m src.data_processing.clean_text clean input.txt -o cleaned.txt

# 清洗并移除URL和邮箱
python -m src.data_processing.clean_text clean input.txt --remove-urls --remove-emails

# 分割大文件
python -m src.data_processing.clean_text split large.txt -o ./chunks --max-chars 1000000
```

### 3. 网络爬虫 (`src/crawler/cli.py`)

从互联网批量爬取文本数据用于大模型预训练。

**功能特性：**
- 多站点批量爬取
- 自动文本清洗
- robots.txt 合规检查
- 异步爬取支持（高性能）
- 统一SQLite数据库存储
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

# 指定输出目录查看状态
python -m src.crawler.cli status --output-dir ./output/crawled
```

**配置文件格式（YAML）：**

```yaml
global:
  delay: 2.0          # 请求间隔（秒）
  max_pages: 30       # 每个站点最多爬取页数
  depth: 3             # 爬取深度
  max_concurrent: 2    # 每个站点最大并发数

sites:
  - name: cnfin
    start_url: https://www.cnfin.com/
    max_pages: 30
    content_type: news
    delay: 2.0

  - name: nbd
    start_url: https://finance.nbd.com.cn/
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

### 完整数据准备流程

```bash
# 方式一：从文档准备数据
# 1. 转换文档为TXT
python -m src.data_processing.document_converter -d ./raw_documents --merge raw_data.txt

# 2. 清洗文本
python -m src.data_processing.clean_text clean raw_data.txt -o cleaned_data.txt

# 3. (可选) 分割大文件
python -m src.data_processing.clean_text split cleaned_data.txt -o ./train_chunks

# 方式二：从网络爬取数据
# 1. 爬取网站内容（使用配置文件批量爬取）
python -m src.crawler.cli run --config configs/crawler/crawler_config.yaml --parallel 2

# 2. 清洗爬取结果（爬虫已自动清洗，可视需要进一步处理）
python -m src.data_processing.clean_text clean ./crawled/example_com.txt -o cleaned_data.txt

# 开始预训练
python scripts/preprocess_data.py --train_file cleaned_data.txt --output_dir output/preprocessed
```

### 支持的输入格式

| 格式 | 工具 | 说明 |
|------|------|------|
| `.pdf`, `.csv`, `.json`, `.epub` | src.data_processing.document_converter | 文档格式转换 |
| `.txt` | src.data_processing.clean_text | 纯文本（需清洗） |
| YAML配置 | src.crawler.cli | 批量网络爬取 |

### 输出格式

所有工具输出的都是纯文本格式 (`.txt`)，每行一段文本，可直接用于预训练：

```
这是第一段内容。

这是第二段内容。

...
```

### 注意事项

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
| `--incremental` | True | 启用增量处理模式（默认） |
| `--full` | False | 完全重新处理所有数据 |
| `--dry-run` | False | 仅检测变化，不实际处理 |
| `--tokenizer-mode` | frozen | Tokenizer 模式: frozen/extend/retrain |
| `--force_reprocess` | False | 强制重新处理（等同于 --full） |

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

### dataset_info.json v3.0 格式

```json
{
  "version": "3.0",
  "config": {
    "max_seq_length": 512,
    "vocab_size": 32000,
    "min_frequency": 2,
    "shard_size": 10000
  },
  "files": {
    "file1.txt": {
      "hash": "md5...",
      "status": "processed",
      "shards": ["train_000.pt"],
      "num_examples": 5000
    }
  },
  "shards": {
    "train_000.pt": {
      "index": 0,
      "source_files": ["file1.txt"],
      "num_examples": 5000
    }
  },
  "summary": {
    "total_files": 1,
    "total_shards": 1,
    "total_examples": 5000,
    "next_shard_index": 1
  }
}
```

**文件状态说明：**

| 状态 | 说明 |
|------|------|
| `processed` | 已正常处理 |
| `legacy` | 从 v2.0 迁移的遗留数据 |
| `orphaned` | 源文件已删除，但分片保留 |

### 缓存格式（.pt文件）

```python
{
    "version": "1.0",
    "metadata": {
        "max_seq_length": 512,
        "vocab_size": 32000,
        "num_examples": 10000,
        "source_files": ["train.txt"],
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
pytest tests/test_crawler.py -v

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
| `tests/test_crawler.py` | 爬虫配置、解析器、文本提取、数据库 |
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

#### test_crawler.py

- `TestCrawlerConfig`: 爬虫配置测试
- `TestSiteConfig`: 站点配置测试
- `TestRobotChecker`: robots.txt检查测试
- `TestTextExtractor`: 文本提取测试
  - 标题提取、正文提取、噪声移除
- `TestAsyncCrawler`: 异步爬虫测试
  - URL验证、链接提取、文本清洗
- `TestCrawlResult`: 爬取结果测试
- `TestCrawlerIntegration`: 爬虫集成测试

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
