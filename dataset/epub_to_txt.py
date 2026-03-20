#!/usr/bin/env python
"""
EPUB转TXT工具
将EPUB电子书格式转换为纯文本格式，用于大模型预训练
"""
import os
import re
import sys
import zipfile
import argparse
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from html import unescape
from typing import List, Optional, Tuple
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


def batch_convert(input_dir: str, output_dir: Optional[str] = None) -> List[str]:
    """
    批量转换目录下的所有EPUB文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选）

    Returns:
        转换成功的文件列表
    """
    input_dir = os.path.abspath(input_dir)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = input_dir

    # 查找所有EPUB文件
    epub_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.epub'):
            epub_files.append(os.path.join(input_dir, file))

    if not epub_files:
        print(f"在 {input_dir} 中未找到EPUB文件")
        return []

    print(f"找到 {len(epub_files)} 个EPUB文件")
    print("=" * 50)

    success_files = []
    failed_files = []

    for idx, epub_path in enumerate(epub_files, 1):
        print(f"\n[{idx}/{len(epub_files)}]")
        try:
            base_name = os.path.splitext(os.path.basename(epub_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.txt")
            epub_to_txt(epub_path, output_path)
            success_files.append(output_path)
        except Exception as e:
            print(f"  错误: {e}")
            failed_files.append(epub_path)

    # 打印总结
    print("\n" + "=" * 50)
    print(f"转换完成: 成功 {len(success_files)}, 失败 {len(failed_files)}")

    if failed_files:
        print("\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")

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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="EPUB转TXT工具 - 将电子书转换为训练数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 转换单个文件
  python epub_to_txt.py book.epub

  # 指定输出路径
  python epub_to_txt.py book.epub -o output.txt

  # 批量转换目录
  python epub_to_txt.py -d /path/to/epubs -o /path/to/output

  # 批量转换并合并为一个文件
  python epub_to_txt.py -d /path/to/epubs --merge merged.txt
        """
    )

    parser.add_argument("input", nargs="?", help="EPUB文件路径")
    parser.add_argument("-o", "--output", help="输出TXT文件或目录路径")
    parser.add_argument("-d", "--dir", help="批量转换的输入目录")
    parser.add_argument("--merge", metavar="OUTPUT_FILE", help="合并所有转换结果为一个文件")

    args = parser.parse_args()

    if args.dir:
        # 批量转换模式
        txt_files = batch_convert(args.dir, args.output)

        if args.merge and txt_files:
            merge_txt_files(txt_files, args.merge)

    elif args.input:
        # 单文件转换模式
        epub_to_txt(args.input, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
