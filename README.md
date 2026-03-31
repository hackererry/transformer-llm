# Transformer 大模型训练框架

基于 PyTorch 的大语言模型（LLM）训练框架，支持**预训练**和**指令微调（SFT）**，针对 CPU 和 GPU 进行了深度优化。核心架构采用模块化设计，覆盖从数据获取到模型部署的完整流程。

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              完整工作流程                                     │
│                                                                            │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │   数据获取    │ ──▶ │   数据清洗    │ ──▶ │   数据预处理  │              │
│  │  (爬虫/文档)  │     │  (去噪/脱敏)  │     │ (分词/分片)   │              │
│  └──────────────┘     └──────────────┘     └──────────────┘              │
│                                                            │               │
│                                                            ▼               │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐              │
│  │   模型推理    │ ◀── │   模型微调    │ ◀── │   模型预训练  │              │
│  │  (Generate)   │     │   (SFT)       │     │  (Pretrain)   │              │
│  └──────────────┘     └──────────────┘     └──────────────┘              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 项目目录结构

```
transformer/
├── src/
│   ├── model/                  # 模型层 - Transformer 架构实现
│   │   ├── config.py           # ModelConfig — 模型配置
│   │   ├── transformer.py      # TransformerBlock / CausalLMModel — 核心模型
│   │   ├── attention/          # 注意力机制子模块
│   │   │   ├── base.py         # 基础接口、RoPE、causal mask
│   │   │   ├── standard.py      # 标准注意力
│   │   │   ├── flash.py         # Flash Attention
│   │   │   ├── gqa.py           # 分组查询注意力 (GQA)
│   │   │   ├── streaming.py     # StreamingLLM 注意力
│   │   │   └── mla.py           # 多头潜在注意力 (MLA)
│   │   ├── embedding.py         # TokenEmbedding / RotaryEmbedding — 词嵌入 + RoPE
│   │   ├── layers.py           # RMSNorm / SwiGLUFFN / FeedForward — 归一化与 FFN 层
│   │   ├── lm_head.py          # LMHead — 语言模型输出头
│   │   └── moe/                # MoE 子模块
│   │       ├── expert.py        # 专家模块（SwiGLU / Shared Expert）
│   │       ├── router.py        # Top-K 路由器
│   │       ├── moe_layer.py     # DeepSeekMoE 层
│   │       └── load_balancing.py # 负载均衡损失
│   │
│   ├── data/                   # 数据层 - tokenizer、数据集、collator
│   │   ├── tokenizer.py        # HuggingFaceBPETokenizer — Rust 实现 BPE（21x 加速）
│   │   ├── dataset.py          # PretrainDataset / FinetuneDataset / TextFileDataset
│   │   ├── preprocessed_dataset.py  # 分片预处理数据集（避免 OOM）
│   │   └── collator.py         # DataCollatorForCausalLM / DynamicBatchSampler
│   │
│   ├── training/               # 训练层 - Trainer、优化器、调度器、检查点
│   │   ├── trainer.py          # Trainer — 统一训练器（自动检测 CPU/GPU）
│   │   ├── optimizer.py        # AdamW / LAMB 优化器
│   │   ├── scheduler.py        # CosineAnnealing / Linear 等学习率调度器
│   │   └── checkpoint.py       # 检查点保存/加载
│   │
│   ├── data_processing/        # 数据处理层 - 清洗、转换、去重、脱敏
│   │   ├── clean_text.py        # TextCleaner — 文本清洗
│   │   ├── document_converter.py  # PDF / EPUB / CSV / JSON → TXT
│   │   ├── deduplicate.py       # 精确去重 / 近似去重
│   │   ├── pii_remover.py       # PII 脱敏（手机号、邮箱、身份证等）
│   │   ├── quality_filter.py    # 质量评分与过滤
│   │   ├── pipeline.py          # CleaningPipeline — 可组合清洗流水线
│   │   └── cleaning_db.py      # 清洗数据数据库存储
│   │
│   ├── crawler/                 # 爬虫层 - 异步爬取、浏览器自动化、反爬
│   │   ├── cli.py              # 命令行接口
│   │   ├── engine.py           # 爬虫引擎（httpx + BeautifulSoup）
│   │   ├── config.py           # CrawlerConfig — 爬虫配置
│   │   ├── browser/            # Playwright 浏览器自动化
│   │   ├── anti_crawler/       # 反反爬虫（代理池、UA 池、指纹）
│   │   └── storage/            # 数据存储（SQLite / 文件 / Redis）
│   │
│   └── utils/                   # 工具层 - 设备管理、日志、指标、性能分析、数据库
│       ├── device.py            # get_device / DeviceManager — 设备检测与优化
│       ├── logging.py           # Logger / ProgressTracker — 日志记录
│       ├── metrics.py           # compute_perplexity / MetricsTracker — 指标计算
│       ├── profiling.py         # OptimizationProfiler — 性能分析
│       ├── database.py          # DatabaseManager — SQLite 管理器
│       ├── database_schema.py   # 数据库表结构定义
│       └── repository.py        # CrawlerPageRepository / CleanedDocumentRepository
│
├── scripts/                      # 训练脚本入口
│   ├── preprocess_data.py       # 数据预处理（tokenize + 分片）
│   ├── pretrain.py              # 预训练（CPU/GPU 自动检测）
│   ├── finetune.py              # 指令微调 (SFT)
│   └── generate.py              # 文本生成
│
├── configs/                      # 配置文件
│   └── crawler/                  # 爬虫配置
│
├── tests/                        # 测试用例
└── dataset/                      # 数据目录
```

---

## 核心模块概览

| 模块 | 说明 | 关键文件 |
|------|------|---------|
| **model** | Transformer 架构实现，支持标准 FFN 与 MoE | `transformer.py`, `config.py` |
| **data** | Tokenizer、数据集、DataLoader | `tokenizer.py`, `dataset.py`, `collator.py` |
| **training** | 统一训练器，优化器与调度器 | `trainer.py`, `optimizer.py`, `scheduler.py` |
| **data_processing** | 文本清洗、文档转换、去重、脱敏 | `clean_text.py`, `pipeline.py`, `deduplicate.py` |
| **crawler** | 异步爬虫、浏览器自动化、反反爬虫 | `engine.py`, `cli.py` |
| **utils** | 设备管理、日志、指标、数据库 | `device.py`, `logging.py`, `metrics.py` |

---

## 快速开始

### 安装依赖

```bash
git clone <repo_url>
cd transformer
pip install -r requirements.txt

# GPU 额外依赖（可选）
pip install flash-attn --no-build-isolation
```

### 完整数据准备 + 预训练流程

```bash
# 1. 数据预处理（tokenize + 分片）
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --max_seq_length 512 \
    --vocab_size 32000

# 2. 预训练
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config small \
    --num_train_epochs 3

# 3. 指令微调（可选）
python scripts/finetune.py \
    --train_file data/instructions.json \
    --model_path output/final_model \
    --num_train_epochs 3

# 4. 文本生成
python scripts/generate.py \
    --model_path output/sft/final_model \
    --prompt "今天天气真好，"
```

---

## 数据准备

### 方式一：从文档转换

```bash
# PDF / EPUB / CSV / JSON → TXT（并发转换）
python -m src.data_processing.document_converter -d ./documents --merge raw_data.txt

# 清洗文本
python -m src.data_processing.clean_text clean raw_data.txt -o cleaned.txt

# 预处理
python scripts/preprocess_data.py --train_file cleaned.txt --output_dir output/preprocessed
```

### 方式二：从网络爬取

```bash
python -m src.crawler.cli run --config configs/crawler/crawler_config.yaml --parallel 2
```

---

## 模型配置

| 配置 | 参数量 | hidden_size | layers | heads | FFN/MoE |
|------|--------|-------------|--------|-------|---------|
| tiny | ~10M | 256 | 6 | 8 | FFN |
| small | ~100M | 512 | 12 | 8 | FFN |
| medium | ~500M | 1024 | 24 | 16 | FFN |
| moe_small | ~20M active | 512 | 12 | 8 | MoE (8 experts) |
| moe_medium | ~100M active | 1024 | 24 | 16 | MoE (8 experts) |

---

## 训练特性

- **自动设备检测** — CPU/GPU 自动选择，GPU 自动启用 BF16
- **Flash Attention** — GPU 自动启用（显存优化）
- **GQA 分组查询注意力** — 减少 KV 缓存开销
- **YaRN 长度外推** — 支持 4 倍序列长度外推
- **StreamingLLM** — 无限长度推理
- **MoE 混合专家** — DeepSeek-V3 风格，共享专家 + Top-K 路由
- **MLA 多头潜在注意力** — DeepSeek-V2 风格，KV 压缩
- **梯度检查点** — 显存节省（可选）
- **手动终止保存** — Ctrl+C 自动保存检查点

---

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_model.py -v
pytest tests/test_data.py -v
pytest tests/test_training.py -v
pytest tests/test_crawler.py -v
```

---

## 编码规范

1. 所有代码文件使用 **UTF-8** 编码
2. 使用 `torch.amp.autocast('cuda', ...)` 而非已废弃的 `autocast(device_type='cuda')`
3. HuggingFace tokenizers 保存的 merges 是列表格式，加载时需兼容处理

---

## 项目状态

- ✅ 统一训练器（CPU/GPU 自动检测）
- ✅ 数据预处理 + 分片加载
- ✅ HuggingFace tokenizers 集成（21x 加速）
- ✅ Flash Attention / GQA / YaRN / StreamingLLM
- ✅ MoE (DeepSeek-V3 风格) / MLA (DeepSeek-V2 风格)
- ✅ 增量预处理
- ✅ 性能监控（Data Loading / Forward / Backward / Optimizer Step）
- ✅ 支持手动终止训练（Ctrl+C）自动保存
- ✅ 网络爬虫系统
- ✅ 文档格式转换（PDF / EPUB / CSV / JSON）
- ✅ 文本清洗流水线（PII 脱敏 / 去重 / 质量过滤）
- ✅ 清洗数据数据库存储
