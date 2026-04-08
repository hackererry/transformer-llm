# 数据处理层 (src/data_processing/)

文本清洗、文档格式转换、去重、脱敏、质量过滤，提供独立工具和可组合流水线。

## 目录结构

```
src/data_processing/
├── clean_text.py        # TextCleaner — 文本清洗
├── document_converter.py # PDF / EPUB / CSV / JSON / Parquet → TXT
├── deduplicate.py        # 精确去重 / 近似去重 / 行级去重
├── pii_remover.py       # PII 脱敏（手机号、邮箱、身份证等）
├── quality_filter.py    # 质量评分与过滤
├── pipeline.py           # CleaningPipeline — 可组合清洗流水线
└── cleaning_db.py        # CleaningDatabase — 清洗数据数据库存储
```

## 核心组件

### clean_text.py — 文本清洗

`TextCleaner` 类提供多种清洗操作：

| 方法 | 说明 |
|------|------|
| `fix_encoding_issues()` | 修复编码问题（乱码、非法字符） |
| `remove_extra_whitespace()` | 移除多余空白（连续空格/换行） |
| `normalize_unicode()` | Unicode 规范化（NFC） |
| `remove_control_chars()` | 移除控制字符 |
| `remove_extra_punctuation()` | 规范化标点符号 |
| `remove_urls()` | 移除 URL |
| `remove_emails()` | 移除邮箱地址 |
| `remove_page_numbers()` | 移除页码（"第X页"等） |
| `remove_headers_footers()` | 移除页眉页脚 |
| `split_large_file()` | 分割大文件 |

**函数接口：**

| 函数 | 说明 |
|------|------|
| `clean_file(input, output, options)` | 清洗单个文件 |
| `split_large_file(input, output_dir, max_chars)` | 分割大文件 |
| `batch_clean_directory(input_dir, output_dir, options)` | 批量清洗目录 |

**使用示例：**

```bash
# 清洗单个文件
python scripts/clean_data.py -i input.txt -o cleaned.txt

# 批量清洗目录
python scripts/clean_data.py -i ./raw_data -o ./cleaned_data
```

```python
from src.data_processing.clean_text import split_large_file

# 分割大文件
split_large_file("large.txt", "./chunks", max_chars=1000000)
```

### document_converter.py — 文档转换

支持多种格式转换为纯文本：

| 格式 | 提取器 | 说明 |
|------|--------|------|
| EPUB | `EPUBExtractor` | 保留章节结构 |
| PDF | `PDFExtractor` | 文本提取，支持扫描 PDF |
| CSV | `CSVExtractor` | 表格数据拼接 |
| JSON | `JSONExtractor` | JSON 字段提取和拼接 |
| Parquet | `ParquetExtractor` | 大文件流式读取，支持 ~100MB 分片输出 |

| 函数 | 说明 |
|------|------|
| `convert_to_txt(input, output)` | 通用转换（自动检测格式） |
| `convert_parquet_to_txt(input, output, max_file_size)` | Parquet 转 TXT（支持分片） |
| `epub_to_txt(input, output)` | EPUB 转 TXT |
| `batch_convert(input_dir, output_dir, workers)` | 批量转换（并发） |
| `merge_txt_files(input_dir, output_file)` | 合并多个 TXT |

**使用示例：**

```python
from src.data_processing import batch_convert, merge_txt_files, convert_to_txt

# 单个文件转换
convert_to_txt("book.epub", "output.txt")

# 批量转换（自动使用 CPU 核数并发）
txt_files = batch_convert("./documents", "./txt_output")

# 指定并发线程数
txt_files = batch_convert("./documents", "./txt_output", max_workers=4)

# 批量转换并合并
merge_txt_files(txt_files, "merged.txt")
```

### deduplicate.py — 去重

| 函数 | 说明 |
|------|------|
| `exact_deduplicate(texts, threshold)` | 精确去重（MD5/SHA256） |
| `near_deduplicate(texts, threshold)` | 近似去重（MinHash / SimHash） |
| `deduplicate_lines(text_file, output_file)` | 行级去重 |
| `deduplicate_lines_from_text(text, keep_order)` | 字符串行级去重 |

**使用示例：**

```python
from src.data_processing import exact_deduplicate, near_deduplicate

# 精确去重
unique_texts = exact_deduplicate(texts, threshold=0.95)

# 近似去重（MinHash，Jaccard 相似度）
unique_texts = near_deduplicate(texts, threshold=0.85)
```

### pii_remover.py — PII 脱敏

支持的 PII 类型：

| 类型 | 正则模式 | 示例 |
|------|---------|------|
| 手机号 | `1[3-9]\d{9}` | 13812345678 |
| 邮箱 | 标准邮箱格式 | user@example.com |
| 身份证 | 18位身份证 | 110101199001011234 |
| IP地址 | IPv4 | 192.168.1.1 |
| 银行卡 | 16/19位卡号 | 6222021234567890 |
| QQ号 | 5-11位数字 | 123456789 |
| 微信 | 微信号格式 | wxid_xxxxx |

| 函数 | 说明 |
|------|------|
| `remove_pii(text, placeholder)` | 移除 PII 并替换为占位符 |
| `remove_pii_with_count(text)` | 移除并返回统计信息 |
| `has_pii(text)` | 检测是否包含 PII |

### quality_filter.py — 质量过滤

**质量评分维度：**

| 维度 | 说明 |
|------|------|
| 长度分数 | 文本长度是否合理（10-50000字符） |
| 重复分数 | 字符级/词级重复率 |
| 特殊字符比例 | 特殊字符占比是否过高 |
| 标点分数 | 标点符号使用是否合理 |
| 语言分数 | 字符分布是否符合目标语言 |

| 函数 | 说明 |
|------|------|
| `compute_quality_score(text)` | 计算质量分数（0-1） |
| `filter_by_quality(texts, threshold)` | 按阈值过滤 |
| `filter_by_quality_with_stats(texts, threshold)` | 过滤并返回统计信息 |

### pipeline.py — 清洗流水线

`CleaningPipeline` 支持自由组合清洗步骤：

```python
from src.data_processing import CleaningPipeline
from src.data_processing.clean_text import TextCleaner
from src.data_processing.pii_remover import remove_pii
from src.data_processing.deduplicate import deduplicate_lines_from_text

# 方式1：使用预设流水线
from src.data_processing import build_light_pipeline, build_standard_pipeline

# 轻量流水线：编码修复 + 空白规范化
pipeline = build_light_pipeline()

# 标准流水线：编码 + 空白 + URL/邮箱 + 质量过滤
pipeline = build_standard_pipeline(threshold=0.5)

# 方式2：自定义流水线
pipeline = (
    CleaningPipeline()
    .add(TextCleaner.fix_encoding_issues)
    .add(TextCleaner.remove_extra_whitespace)
    .add(TextCleaner.remove_urls)
    .add(TextCleaner.remove_emails)
    .add_quality_filter(threshold=0.5)
    .add_dedup()
)

# 流式处理
for cleaned_line, line_no, out_no in pipeline.process_stream(input_path, output_path):
    print(f"Processed: {out_no} lines")
```

**流水线模式：**

| 模式 | 步骤 | 适用场景 |
|------|------|---------|
| `build_light_pipeline()` | 编码修复 + 空白规范化 | 快速清洗 |
| `build_standard_pipeline()` | 编码 + 空白 + URL/邮箱 + 质量过滤 | 标准流程 |
| `stream_clean_pipeline()` | 逐行处理 + 内存去重 | 超大文件 |

### cleaning_db.py — 清洗数据库

SQLite 数据库存储清洗结果，支持增量清洗：

**数据库表：**

| 表名 | 说明 |
|------|------|
| `cleaned_documents` | 清洗后的文档（原始 MD5、清洗后 MD5、完整内容） |
| `cleaning_runs` | 清洗运行记录 |

**CLI 使用：**

```bash
python scripts/clean_data.py --db db/cleaning.db --input ./raw_data --output ./cleaned
```

## 完整数据清洗流程

```python
from src.data_processing import batch_convert, merge_txt_files
from src.data_processing.clean_text import clean_file, split_large_file

# 1. 文档转换
txt_files = batch_convert("./documents")
merge_txt_files(txt_files, "raw.txt")

# 2. 文本清洗
clean_file("raw.txt", "cleaned.txt")

# 3. 分割大文件
split_large_file("cleaned.txt", "./chunks", max_chars=1000000)
```

```bash
# 4. 数据预处理
python scripts/preprocess_data.py --train_dir ./chunks --output_dir output/preprocessed
```
