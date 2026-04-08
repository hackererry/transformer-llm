#!/usr/bin/env python
"""
文本转换工具
支持 PDF, CSV, JSON, EPUB 格式转换为纯文本，用于大模型预训练
"""
import os
import re
import zipfile
import threading
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from html import unescape
from typing import List, Optional, Tuple, Dict
from pathlib import Path


class HTMLTextExtractor(HTMLParser):
    """
    HTML文本提取器
    从HTML标签中提取纯文本内容
    """

    # 需要换行的块级元素
    BLOCK_ELEMENTS = {
        'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
        'br', 'hr', 'li', 'tr', 'section', 'article',
        'header', 'footer', 'nav', 'aside', 'main',
    }

    # 需要忽略的元素
    IGNORE_ELEMENTS = {
        'script', 'style', 'noscript', 'meta', 'link',
        'head', 'title',
    }

    def __init__(self):
        super().__init__()
        self.text_parts: List[str] = []
        self.current_tag_stack: List[str] = []
        self.ignore_depth: int = 0

    def handle_starttag(self, tag: str, attrs):
        tag = tag.lower()

        if tag in self.IGNORE_ELEMENTS:
            self.ignore_depth += 1
            return

        if self.ignore_depth > 0:
            return

        self.current_tag_stack.append(tag)

        # 块级元素前添加换行
        if tag in self.BLOCK_ELEMENTS:
            self.text_parts.append('\n')

    def handle_endtag(self, tag: str):
        tag = tag.lower()

        if tag in self.IGNORE_ELEMENTS:
            self.ignore_depth = max(0, self.ignore_depth - 1)
            return

        if self.ignore_depth > 0:
            return

        if self.current_tag_stack and self.current_tag_stack[-1] == tag:
            self.current_tag_stack.pop()

        # 块级元素后添加换行
        if tag in self.BLOCK_ELEMENTS:
            self.text_parts.append('\n')

    def handle_data(self, data: str):
        if self.ignore_depth > 0:
            return

        # 处理文本内容
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        """获取提取的纯文本"""
        text = ''.join(self.text_parts)
        # 清理多余空白
        text = self._clean_text(text)
        return text

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 合并多个换行为两个换行（段落分隔）
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 合并多个空格为一个
        text = re.sub(r'[ \t]+', ' ', text)
        # 移除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        return text.strip()


class EPUBExtractor:
    """
    EPUB电子书提取器
    解析EPUB文件并提取文本内容
    """

    # EPUB使用的XML命名空间
    NAMESPACES = {
        'container': 'urn:oasis:names:tc:opendocument:xmlns:container',
        'opf': 'http://www.idpf.org/2007/opf',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'dcterms': 'http://purl.org/dc/terms/',
    }

    def __init__(self, epub_path: str):
        self.epub_path = epub_path
        self.zip_file: Optional[zipfile.ZipFile] = None
        self.content_opf_path: Optional[str] = None
        self.spine_items: List[str] = []
        self.metadata: dict = {}

    def extract(self) -> Tuple[str, dict]:
        """
        提取EPUB内容

        Returns:
            (文本内容, 元数据字典)
        """
        try:
            self.zip_file = zipfile.ZipFile(self.epub_path, 'r')

            # 解析容器文件获取content.opf路径
            self._parse_container()

            # 解析content.opf获取阅读顺序和元数据
            self._parse_content_opf()

            # 按顺序提取所有章节内容
            chapters = self._extract_chapters()

            # 组合所有文本
            full_text = self._combine_chapters(chapters)

            return full_text, self.metadata

        finally:
            if self.zip_file:
                self.zip_file.close()

    def _parse_container(self):
        """解析META-INF/container.xml获取content.opf路径"""
        try:
            container_xml = self.zip_file.read('META-INF/container.xml')
            root = ET.fromstring(container_xml)

            # 查找rootfile元素
            for rootfile in root.iter():
                if rootfile.tag.endswith('rootfile'):
                    self.content_opf_path = rootfile.get('full-path')
                    break

            if not self.content_opf_path:
                raise ValueError("无法找到content.opf文件路径")

        except KeyError:
            raise ValueError("无效的EPUB文件：缺少META-INF/container.xml")

    def _parse_content_opf(self):
        """解析content.opf获取spine和metadata"""
        if not self.content_opf_path:
            raise ValueError("content.opf路径未设置")

        try:
            opf_content = self.zip_file.read(self.content_opf_path)
            root = ET.fromstring(opf_content)

            # 获取opf文件所在目录
            opf_dir = os.path.dirname(self.content_opf_path)

            # 解析manifest和spine
            manifest = {}
            for item in root.iter():
                if item.tag.endswith('item'):
                    item_id = item.get('id')
                    href = item.get('href')
                    media_type = item.get('media-type', '')

                    if href:
                        # 处理相对路径
                        if opf_dir:
                            href = os.path.join(opf_dir, href)
                        manifest[item_id] = {
                            'href': href,
                            'media_type': media_type
                        }

            # 解析spine（阅读顺序）
            for itemref in root.iter():
                if itemref.tag.endswith('itemref'):
                    idref = itemref.get('idref')
                    if idref and idref in manifest:
                        self.spine_items.append(manifest[idref]['href'])

            # 如果没有spine，使用manifest中的HTML文件
            if not self.spine_items:
                for item_id, item_info in manifest.items():
                    if 'html' in item_info['media_type'] or \
                       item_info['href'].endswith(('.html', '.htm', '.xhtml')):
                        self.spine_items.append(item_info['href'])

            # 解析元数据
            self._extract_metadata(root)

        except KeyError:
            raise ValueError(f"无法读取content.opf: {self.content_opf_path}")

    def _extract_metadata(self, root):
        """提取元数据"""
        for elem in root.iter():
            tag = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag

            if tag == 'title' and elem.text:
                self.metadata['title'] = elem.text.strip()
            elif tag == 'creator' and elem.text:
                self.metadata['author'] = elem.text.strip()
            elif tag == 'language' and elem.text:
                self.metadata['language'] = elem.text.strip()
            elif tag == 'publisher' and elem.text:
                self.metadata['publisher'] = elem.text.strip()
            elif tag == 'date' and elem.text:
                self.metadata['date'] = elem.text.strip()
            elif tag == 'description' and elem.text:
                self.metadata['description'] = elem.text.strip()

    def _extract_chapters(self) -> List[Tuple[str, str]]:
        """提取所有章节内容"""
        chapters = []

        for idx, href in enumerate(self.spine_items):
            try:
                # 规范化路径
                href = href.replace('\\', '/')

                # 尝试读取文件
                try:
                    content = self.zip_file.read(href)
                except KeyError:
                    # 尝试URL解码
                    from urllib.parse import unquote
                    decoded_href = unquote(href)
                    content = self.zip_file.read(decoded_href)

                # 解析HTML提取文本
                html_content = content.decode('utf-8', errors='ignore')
                text = self._html_to_text(html_content)

                if text.strip():
                    chapters.append((f"第{idx+1}章", text))

            except Exception as e:
                print(f"  警告: 无法处理 {href}: {e}")
                continue

        return chapters

    def _html_to_text(self, html: str) -> str:
        """将HTML转换为纯文本"""
        # 先进行HTML实体解码
        html = unescape(html)

        # 使用自定义解析器
        extractor = HTMLTextExtractor()
        try:
            extractor.feed(html)
            return extractor.get_text()
        except Exception:
            # 备用方案：简单的正则替换
            text = re.sub(r'<[^>]+>', '', html)
            text = unescape(text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

    def _combine_chapters(self, chapters: List[Tuple[str, str]]) -> str:
        """组合所有章节"""
        parts = []

        # 添加标题信息
        if self.metadata.get('title'):
            title = f"《{self.metadata['title']}》"
            parts.append(title)
            parts.append('=' * len(title))

        if self.metadata.get('author'):
            parts.append(f"作者: {self.metadata['author']}")
        if self.metadata.get('publisher'):
            parts.append(f"出版社: {self.metadata['publisher']}")

        parts.append('')  # 空行分隔

        # 添加各章节内容
        for chapter_title, chapter_text in chapters:
            parts.append(chapter_title)
            parts.append('-' * 20)
            parts.append(chapter_text)
            parts.append('')  # 章节间空行

        return '\n'.join(parts)


class PDFExtractor:
    """
    PDF文本提取器
    使用pdfplumber提取PDF文本内容
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.metadata: dict = {}

    def extract(self) -> Tuple[str, dict]:
        """
        提取PDF内容

        Returns:
            (文本内容, 元数据字典)
        """
        import pdfplumber

        all_text = []
        self.metadata = {}

        with pdfplumber.open(self.pdf_path) as pdf:
            # 提取元数据
            if pdf.metadata:
                self.metadata = {
                    'title': pdf.metadata.get('Title', ''),
                    'author': pdf.metadata.get('Author', ''),
                    'subject': pdf.metadata.get('Subject', ''),
                    'creator': pdf.metadata.get('Creator', ''),
                }

            # 提取每页文本
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    # 清理文本
                    text = self._clean_text(text)
                    if text.strip():
                        all_text.append(f"--- 第 {page_num} 页 ---\n{text}")

        return '\n\n'.join(all_text), self.metadata

    def _clean_text(self, text: str) -> str:
        """清理PDF提取的文本"""
        import re
        # 合并多个换行
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 合并多个空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 移除行首行尾空格
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)


class CSVExtractor:
    """
    CSV文本提取器
    将CSV每行转换为"字段名: 值"格式
    """

    def __init__(self, csv_path: str):
        self.csv_path = csv_path

    def extract(self) -> Tuple[str, dict]:
        """
        提取CSV内容

        Returns:
            (文本内容, 元数据字典)
        """
        import csv

        records = []
        fieldnames = []

        with open(self.csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            for row in reader:
                # 将每行转换为 "字段名: 值" 格式
                fields = []
                for key, value in row.items():
                    if value and value.strip():
                        fields.append(f"{key}: {value.strip()}")

                if fields:
                    records.append('\n'.join(fields))

        metadata = {'fieldnames': fieldnames, 'record_count': len(records)}
        return '\n\n'.join(records), metadata


class JSONExtractor:
    """
    JSON文本提取器
    支持finetune格式和纯文本JSON
    """

    def __init__(self, json_path: str):
        self.json_path = json_path

    def extract(self) -> Tuple[str, dict]:
        """
        提取JSON内容

        Returns:
            (文本内容, 元数据字典)
        """
        import json

        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        metadata = {}
        text_parts = []

        if isinstance(data, list):
            # JSON数组格式（finetune格式）
            for idx, item in enumerate(data):
                text = self._extract_from_dict(item)
                if text:
                    text_parts.append(f"--- 记录 {idx + 1} ---\n{text}")
            metadata['record_count'] = len(data)
            metadata['format'] = 'array'

        elif isinstance(data, dict):
            # 纯JSON对象
            text = self._extract_from_dict(data)
            if text:
                text_parts.append(text)
            metadata['format'] = 'object'

            # 尝试提取元数据
            for key in ['title', 'author', 'source', 'date']:
                if key in data and isinstance(data[key], str):
                    metadata[key] = data[key]

        return '\n\n'.join(text_parts), metadata

    def _extract_from_dict(self, obj: dict, prefix: str = '') -> str:
        """递归提取字典中的文本"""
        import json

        text_parts = []

        for key, value in obj.items():
            field_name = f"{prefix}{key}" if prefix else key

            if isinstance(value, str) and value.strip():
                text_parts.append(f"{field_name}: {value.strip()}")
            elif isinstance(value, (int, float, bool)):
                text_parts.append(f"{field_name}: {value}")
            elif isinstance(value, list):
                # 处理列表：提取每个元素
                for item in value:
                    if isinstance(item, str) and item.strip():
                        text_parts.append(f"{field_name}: {item.strip()}")
                    elif isinstance(item, dict):
                        nested = self._extract_from_dict(item, f"{field_name}. ")
                        if nested:
                            text_parts.append(nested)
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                nested = self._extract_from_dict(value, f"{field_name}. ")
                if nested:
                    text_parts.append(nested)

        return '\n'.join(text_parts)


class ParquetExtractor:
    """
    Parquet文本提取器
    将Parquet每行转换为"列名: 值"格式，支持按大小分片
    """

    # 默认分片大小: 100MB
    DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024

    def __init__(self, parquet_path: str):
        self.parquet_path = parquet_path

    @staticmethod
    def _format_row(row: dict, single_col: bool) -> str:
        """将一行数据格式化为文本。单列时直接输出值，多列时用 '列名: 值' 格式。"""
        fields = []
        for col_name, value in row.items():
            if value is None:
                continue
            str_val = str(value).strip()
            if str_val:
                fields.append(f"{col_name}: {str_val}" if not single_col else str_val)
        return '\n'.join(fields)

    def extract(self) -> Tuple[str, dict]:
        """
        提取Parquet内容

        Returns:
            (文本内容, 元数据字典)
        """
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(self.parquet_path)
        metadata = {
            'columns': pf.schema_arrow.names,
            'num_rows': pf.metadata.num_rows,
            'record_count': pf.metadata.num_rows,
        }

        records = []
        col_names = pf.schema_arrow.names
        single_col = len(col_names) == 1
        for batch in pf.iter_batches(batch_size=10000):
            for row in batch.to_pylist():
                line = self._format_row(row, single_col)
                if line:
                    records.append(line)

        return '\n'.join(records), metadata

    def extract_chunks(self, max_file_size: int = None) -> List[Tuple[str, dict]]:
        """
        流式读取Parquet并按大小分片

        Args:
            max_file_size: 每个分片的最大字节数（默认100MB）

        Returns:
            [(文本内容, 元数据), ...] 分片列表
        """
        import pyarrow.parquet as pq

        if max_file_size is None:
            max_file_size = self.DEFAULT_MAX_FILE_SIZE

        pf = pq.ParquetFile(self.parquet_path)
        total_rows = pf.metadata.num_rows

        if total_rows == 0:
            print(f"  警告: Parquet文件无数据行: {self.parquet_path}")
            return []

        chunks = []
        current_lines = []
        current_size = 0
        chunk_idx = 0
        rows_in_chunk = 0
        single_col = len(pf.schema_arrow.names) == 1

        for batch in pf.iter_batches(batch_size=10000):
            for row in batch.to_pylist():
                line = self._format_row(row, single_col)
                if not line:
                    continue

                line_bytes = len(line.encode('utf-8'))

                # 单行超过 max_file_size 时独占一个分片
                if line_bytes > max_file_size:
                    # 先保存当前累积的内容
                    if current_lines:
                        text = '\n'.join(current_lines)
                        meta = {
                            'chunk_index': chunk_idx,
                            'record_count': rows_in_chunk,
                            'columns': pf.schema_arrow.names,
                        }
                        chunks.append((text, meta))
                        chunk_idx += 1
                        current_lines = []
                        current_size = 0
                        rows_in_chunk = 0

                    # 大行独占一个分片
                    meta = {
                        'chunk_index': chunk_idx,
                        'record_count': 1,
                        'columns': pf.schema_arrow.names,
                    }
                    chunks.append((line, meta))
                    chunk_idx += 1
                    continue

                # 检查加入该行是否会超出大小限制
                if current_lines:
                    # 已有内容，新增需要加 \n 分隔符（1字节）
                    new_size = current_size + 1 + line_bytes
                else:
                    new_size = line_bytes

                if new_size > max_file_size and current_lines:
                    # 保存当前分片
                    text = '\n'.join(current_lines)
                    meta = {
                        'chunk_index': chunk_idx,
                        'record_count': rows_in_chunk,
                        'columns': pf.schema_arrow.names,
                    }
                    chunks.append((text, meta))
                    chunk_idx += 1
                    current_lines = [line]
                    current_size = line_bytes
                    rows_in_chunk = 1
                else:
                    current_lines.append(line)
                    current_size = new_size
                    rows_in_chunk += 1

        # 处理剩余内容
        if current_lines:
            text = '\n'.join(current_lines)
            meta = {
                'chunk_index': chunk_idx,
                'record_count': rows_in_chunk,
                'columns': pf.schema_arrow.names,
            }
            chunks.append((text, meta))

        return chunks


def convert_parquet_to_txt(
    input_path: str,
    output_path: str = None,
    max_file_size: int = None,
) -> List[str]:
    """
    将Parquet文件转换为TXT（支持按大小分片）

    Args:
        input_path: 输入parquet文件路径
        output_path: 输出TXT文件路径（可选，仅单分片时生效）
        max_file_size: 每个分片的最大字节数（默认100MB）

    Returns:
        所有输出文件路径列表
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    extractor = ParquetExtractor(input_path)
    chunks = extractor.extract_chunks(max_file_size)

    if not chunks:
        print(f"  警告: Parquet文件无有效数据: {input_path}")
        return []

    # 确定输出基础路径
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
    else:
        base_name = os.path.splitext(output_path)[0]

    output_files = []

    if len(chunks) == 1:
        # 单分片 → base_name.txt
        out_path = f"{base_name}.txt"
        text, meta = chunks[0]
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        output_files.append(out_path)
        print(f"  输出: {out_path} ({len(text):,} 字符)")
    else:
        # 多分片 → base_name_0001.txt, base_name_0002.txt, ...
        for idx, (text, meta) in enumerate(chunks):
            out_path = f"{base_name}_{idx + 1:04d}.txt"
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(text)
            output_files.append(out_path)
            print(f"  分片 {idx + 1}/{len(chunks)}: {out_path} ({len(text):,} 字符)")

    total_records = sum(m['record_count'] for _, m in chunks)
    print(f"  总记录数: {total_records:,}, 输出文件: {len(output_files)} 个")

    return output_files


# 文件扩展名到提取器的映射
EXTRACTORS = {
    '.pdf': PDFExtractor,
    '.csv': CSVExtractor,
    '.json': JSONExtractor,
    '.epub': EPUBExtractor,
    '.parquet': ParquetExtractor,
}


def convert_to_txt(input_path: str, output_path: str = None) -> str:
    """
    根据文件扩展名自动选择提取器进行转换

    Args:
        input_path: 输入文件路径
        output_path: 输出TXT文件路径（可选）

    Returns:
        输出文件路径
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"文件不存在: {input_path}")

    # 确定文件扩展名
    ext = os.path.splitext(input_path)[1].lower()

    if ext not in EXTRACTORS:
        supported = ', '.join(EXTRACTORS.keys())
        raise ValueError(f"不支持的格式: {ext}，支持的格式: {supported}")

    # Parquet 文件走分片逻辑
    if ext == '.parquet':
        output_files = convert_parquet_to_txt(input_path, output_path)
        return output_files[0] if output_files else None

    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.txt"

    print(f"正在处理: {input_path}")

    # 使用对应的提取器
    extractor = EXTRACTORS[ext](input_path)
    text, metadata = extractor.extract()

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # 打印统计信息
    if metadata.get('title'):
        print(f"  标题: {metadata.get('title')}")
    if metadata.get('author'):
        print(f"  作者: {metadata.get('author')}")
    if metadata.get('record_count'):
        print(f"  记录数: {metadata.get('record_count')}")
    print(f"  输出: {output_path}")
    print(f"  字数: {len(text):,}")

    return output_path


def epub_to_txt(epub_path: str, output_path: Optional[str] = None) -> str:
    """
    将EPUB文件转换为TXT

    Args:
        epub_path: EPUB文件路径
        output_path: 输出TXT文件路径（可选）

    Returns:
        输出文件路径
    """
    epub_path = os.path.abspath(epub_path)

    if not os.path.exists(epub_path):
        raise FileNotFoundError(f"EPUB文件不存在: {epub_path}")

    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(epub_path)[0]
        output_path = f"{base_name}.txt"

    print(f"正在处理: {epub_path}")

    # 提取内容
    extractor = EPUBExtractor(epub_path)
    text, metadata = extractor.extract()

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    # 打印统计信息
    print(f"  书名: {metadata.get('title', '未知')}")
    print(f"  作者: {metadata.get('author', '未知')}")
    print(f"  输出: {output_path}")
    print(f"  字数: {len(text):,}")

    return output_path


def _convert_single_file(
    input_path: str,
    output_dir: str,
    file_idx: int,
    total_files: int,
    print_lock: threading.Lock
) -> Tuple[str, bool, str]:
    """
    转换单个文件（用于并发调用）

    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        file_idx: 文件序号
        total_files: 总文件数
        print_lock: 打印锁

    Returns:
        (输入路径, 是否成功, 输出路径或错误信息)
    """
    try:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")

        with print_lock:
            print(f"[{file_idx}/{total_files}] 正在处理: {os.path.basename(input_path)}")

        # Parquet 文件走分片逻辑
        ext = os.path.splitext(input_path)[1].lower()
        if ext == '.parquet':
            output_files = convert_parquet_to_txt(input_path, output_path)
            if not output_files:
                raise ValueError("Parquet文件无有效数据")
            with print_lock:
                print(f"[{file_idx}/{total_files}] 完成: {base_name} -> {len(output_files)} 个文件")
            return (input_path, True, output_files)

        # 使用对应的提取器
        extractor = EXTRACTORS[ext](input_path)
        text, metadata = extractor.extract()

        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        with print_lock:
            info_parts = []
            if metadata.get('title'):
                info_parts.append(f"标题: {metadata.get('title')}")
            if metadata.get('author'):
                info_parts.append(f"作者: {metadata.get('author')}")
            if metadata.get('record_count'):
                info_parts.append(f"记录数: {metadata.get('record_count')}")
            info_parts.append(f"字数: {len(text):,}")
            print(f"[{file_idx}/{total_files}] 完成: {info_parts[0] if info_parts else os.path.basename(input_path)}")

        return (input_path, True, output_path)

    except Exception as e:
        with print_lock:
            print(f"[{file_idx}/{total_files}] 失败: {os.path.basename(input_path)} - {e}")
        return (input_path, False, str(e))


def batch_convert(
    input_dir: str,
    output_dir: Optional[str] = None,
    max_workers: Optional[int] = None
) -> List[str]:
    """
    批量转换目录下的所有支持格式的文件（支持并发）

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）
        max_workers: 最大并发线程数（默认为 CPU 核数）

    Returns:
        转换成功的文件列表
    """
    input_dir = os.path.abspath(input_dir)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = input_dir

    # 默认使用 CPU 核数
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # 查找所有支持的文件
    supported_files = []
    for file in os.listdir(input_dir):
        ext = os.path.splitext(file)[1].lower()
        if ext in EXTRACTORS:
            supported_files.append(os.path.join(input_dir, file))

    if not supported_files:
        supported = ', '.join(EXTRACTORS.keys())
        print(f"在 {input_dir} 中未找到支持格式的文件 ({supported})")
        return []

    total_files = len(supported_files)
    print(f"找到 {total_files} 个支持格式的文件")
    print(f"并发线程数: {max_workers}")
    print("=" * 50)

    success_files = []
    failed_files: Dict[str, str] = {}  # 文件路径 -> 错误信息
    print_lock = threading.Lock()

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for idx, input_path in enumerate(supported_files, 1):
            future = executor.submit(
                _convert_single_file,
                input_path,
                output_dir,
                idx,
                total_files,
                print_lock
            )
            futures[future] = input_path

        # 收集结果
        for future in as_completed(futures):
            input_path, success, result = future.result()
            if success:
                # parquet 返回 List[str]，其他返回 str
                if isinstance(result, list):
                    success_files.extend(result)
                else:
                    success_files.append(result)
            else:
                failed_files[input_path] = result

    # 按文件序号排序输出列表
    success_files.sort()

    # 打印总结
    print("\n" + "=" * 50)
    print(f"转换完成: 成功 {len(success_files)}, 失败 {len(failed_files)}")

    if failed_files:
        print("\n失败的文件:")
        for f, err in failed_files.items():
            print(f"  - {os.path.basename(f)}: {err}")

    return success_files


def merge_txt_files(txt_files: List[str], output_path: str, separator: str = "\n\n" + "="*50 + "\n\n"):
    """
    合并多个TXT文件为一个文件

    Args:
        txt_files: TXT文件列表
        output_path: 输出文件路径
        separator: 文件间分隔符
    """
    print(f"\n正在合并 {len(txt_files)} 个文件到 {output_path}")

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for idx, txt_file in enumerate(txt_files):
            if idx > 0:
                outfile.write(separator)

            with open(txt_file, 'r', encoding='utf-8') as infile:
                content = infile.read()
                outfile.write(content)

    total_chars = sum(os.path.getsize(f) for f in txt_files if os.path.exists(f))
    print(f"合并完成，总大小: {total_chars:,} 字节")
