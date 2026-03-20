# 数据预处理工具

本目录包含用于准备训练数据的工具脚本。

## 工具列表

### 1. EPUB转TXT (`epub_to_txt.py`)

将EPUB电子书格式转换为纯文本格式。

**功能特性：**
- 支持单个文件转换
- 支持批量目录转换
- 自动提取章节顺序
- 保留元数据（书名、作者等）
- 支持合并多个文件

**使用方法：**

```bash
# 转换单个文件
python dataset/epub_to_txt.py book.epub

# 指定输出路径
python dataset/epub_to_txt.py book.epub -o output.txt

# 批量转换目录
python dataset/epub_to_txt.py -d ./epubs -o ./txt_output

# 批量转换并合并为一个文件
python dataset/epub_to_txt.py -d ./epubs --merge merged_training.txt
```

### 2. 文本清洗 (`clean_text.py`)

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
python dataset/clean_text.py clean input.txt -o cleaned.txt

# 清洗并移除URL和邮箱
python dataset/clean_text.py clean input.txt --remove-urls --remove-emails

# 分割大文件
python dataset/clean_text.py split large.txt -o ./chunks --max-chars 1000000
```

## 完整数据准备流程

```bash
# 1. 转换EPUB为TXT
python dataset/epub_to_txt.py -d ./raw_epubs --merge raw_data.txt

# 2. 清洗文本
python dataset/clean_text.py clean raw_data.txt -o cleaned_data.txt

# 3. (可选) 分割大文件
python dataset/clean_text.py split cleaned_data.txt -o ./train_chunks

# 4. 开始预训练
python scripts/pretrain.py --train_file cleaned_data.txt --model_config tiny
```

## 支持的输入格式

| 格式 | 工具 | 说明 |
|------|------|------|
| `.epub` | epub_to_txt.py | 电子书格式 |
| `.txt` | clean_text.py | 纯文本（需清洗） |

## 输出格式

所有工具输出的都是纯文本格式 (`.txt`)，每行一段文本，可直接用于预训练：

```
这是第一段内容。

这是第二段内容。

...
```

## 注意事项

1. **EPUB文件**: 确保EPUB文件没有DRM保护
2. **编码**: 工具会自动处理UTF-8编码
3. **大文件**: 建议分割成小于1GB的文件
4. **版权**: 仅转换您有权使用的内容
