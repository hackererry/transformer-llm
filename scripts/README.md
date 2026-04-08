# Scripts 目录

训练与数据处理脚本的命令行入口。

## 脚本列表

| 脚本 | 用途 |
|------|------|
| `crawl.py` | 网络爬虫批量爬取 |
| `convert_document.py` | 文档格式转换（PDF/CSV/JSON/EPUB → TXT） |
| `clean_data.py` | 数据清洗（PII 脱敏 / 去重 / 质量过滤） |
| `preprocess_data.py` | 数据预处理（tokenize + 分片） |
| `pretrain.py` | 预训练语言模型（CPU/GPU 自动检测） |
| `finetune.py` | 指令微调 (SFT) |
| `generate.py` | 文本生成 |

---

## 用法说明

### convert_document.py — 文档格式转换

```bash
# 单文件转换
python scripts/convert_document.py -i input.pdf -o output.txt

# 批量转换目录
python scripts/convert_document.py -I input_dir -O output_dir

# 指定并发线程数
python scripts/convert_document.py -I input_dir -O output_dir --workers 8

# 合并多个TXT文件
python scripts/convert_document.py --merge file1.txt file2.txt -o merged.txt

# 合并目录下所有TXT文件
python scripts/convert_document.py --merge_dir input_dir -o merged.txt
```

### crawl.py — 网络爬虫

```bash
# 从配置文件批量爬取
python scripts/crawl.py run --config configs/crawler/crawler_config.yaml --parallel 2

# 指定输出目录
python scripts/crawl.py run --config configs/crawler/crawler_config.yaml --output-dir ./output/crawled

# 查看爬虫状态（最近7天）
python scripts/crawl.py status

# 查看最近30天统计
python scripts/crawl.py status --days 30
```

### clean_data.py — 数据清洗

```bash
# 单文件清洗（默认存储到数据库 + 文档级去重）
python scripts/clean_data.py -i data/raw.txt -o data/clean.txt

# 目录批量清洗
python scripts/clean_data.py -I data/raw -O data/clean

# 禁用文档级去重
python scripts/clean_data.py -I data/raw -O data/clean --no-doc-dedup

# 禁用数据库存储
python scripts/clean_data.py -I data/raw -O data/clean --no-db
```

### preprocess_data.py — 数据预处理

```bash
python scripts/preprocess_data.py \
    --train_dir dataset/data \
    --output_dir output/preprocessed \
    --max_seq_length 512 \
    --vocab_size 32000
```

### pretrain.py — 预训练

```bash
python scripts/pretrain.py \
    --preprocessed_data output/preprocessed \
    --model_config small \
    --num_train_epochs 3
```

### finetune.py — 指令微调

```bash
python scripts/finetune.py \
    --train_file data/instructions.json \
    --model_path output/final_model \
    --num_train_epochs 3
```

### generate.py — 文本生成

```bash
python scripts/generate.py \
    --model_path output/sft/final_model \
    --prompt "今天天气真好，"
```
