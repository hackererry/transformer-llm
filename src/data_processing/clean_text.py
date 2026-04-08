"""
文本清洗工具
对转换后的文本进行清洗和预处理
"""
import re
import os
from typing import List, Optional


class TextCleaner:
    """
    文本清洗器
    清理和规范化文本数据
    """

    def __init__(self):
        self.cleaners = []

    def add_cleaner(self, func):
        """添加清洗函数"""
        self.cleaners.append(func)
        return self

    def clean(self, text: str) -> str:
        """应用所有清洗器"""
        for cleaner in self.cleaners:
            text = cleaner(text)
        return text

    @staticmethod
    def remove_empty_lines(text: str) -> str:
        """移除空行"""
        lines = text.split('\n')
        # 过滤掉空行（只包含空白字符的行也算空行）
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        return '\n'.join(non_empty_lines)

    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        """移除多余空白"""
        # 合并多个空格为一个
        text = re.sub(r'[ \t]+', ' ', text)
        # 合并多个换行为最多两个
        text = re.sub(r'\n{3,}', '\n\n', text)
        # 移除行首行尾空白
        lines = [line.strip() for line in text.split('\n')]
        return '\n'.join(lines)

    @staticmethod
    def remove_special_chars(text: str, keep_pattern: str = None) -> str:
        """
        移除特殊字符

        Args:
            text: 输入文本
            keep_pattern: 要保留的字符正则模式
        """
        if keep_pattern:
            # 只保留匹配的字符
            return re.sub(f'[^{keep_pattern}]', '', text)

        # 默认移除控制字符（保留换行和制表符）
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        return text

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """规范化标点符号"""
        # 英文标点规范化
        text = re.sub(r'\.{3,}', '...', text)  # 省略号
        text = re.sub(r'-{2,}', '——', text)   # 破折号

        return text

    @staticmethod
    def remove_wechat_tags(text: str) -> str:
        """移除微信公众号提取残留的 <WECHAT> 标签

        数据来源为微信公众号文章时，英文/特殊字符会被替换为 <WECHAT>。
        这些标签几乎都出现在括号内的英文原文注释中，如：
          弗朗西斯·赫斯特（<WECHAT> <WECHAT> Hirst）
          《经济学人》（The <WECHAT>）
        策略：删除包含 <WECHAT> 的整个括号内容（中文已有翻译，英文残片无价值），
        然后清除残留的独立标签。
        """
        # 1) 移除包含 <WECHAT> 的中文括号及内容
        text = re.sub(r'（[^）]*<WECHAT>[^）]*）', '', text)
        # 2) 移除包含 <WECHAT> 的英文括号及内容
        text = re.sub(r'\([^)]*<WECHAT>[^)]*\)', '', text)
        # 3) 移除残留的独立 <WECHAT> 标签
        text = re.sub(r'<WECHAT>', '', text)
        # 4) 清理括号移除后产生的多余空格
        text = re.sub(r'[ \t]+', ' ', text)
        # 5) 清理因移除括号变为空的括号
        text = re.sub(r'（\s*）', '', text)
        text = re.sub(r'\(\s*\)', '', text)
        return text

    @staticmethod
    def remove_page_numbers(text: str) -> str:
        """移除页码"""
        # 移除独立的数字行（常见于页码）
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        return text

    @staticmethod
    def remove_urls(text: str) -> str:
        """移除URL"""
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'www\.\S+', '', text)
        return text

    @staticmethod
    def remove_emails(text: str) -> str:
        """移除邮箱"""
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        return text

    @staticmethod
    def fix_encoding_issues(text: str) -> str:
        """修复常见编码问题"""
        # 常见的编码错误修复
        replacements = {
            '�': '',
            ' ': ' ',  # 全角空格
            ' ': ' ',  # en空格
            '\u200b': '',  # 零宽空格
            '\ufeff': '',  # BOM
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def remove_book_metadata(text: str) -> str:
        """
        移除书籍元数据：CIP、ISBN、版权页、出版社信息等

        针对从 PDF/TXT 提取的书籍内容，清洗出版相关的元信息
        """
        patterns = [
            # CIP 数据区（多行）
            r'图书在版编目[^\n]*\n[^\n]*',
            # ISBN 号
            r'ISBN\s*[\d\-]+',
            # 出版社信息（常见格式）
            r'出版社\s*[：:]\s*[^\n]+',
            r'出\s*版\s*[：:]\s*[^\n]+',
            # 作者、译者行
            r'作\s*者\s*[：:]\s*[^\n]+',
            r'译\s*者\s*[：:]\s*[^\n]+',
            r'责\s*任\s*编\s*辑\s*[：:]\s*[^\n]+',
            r'文\s*字\s*编\s*辑\s*[：:]\s*[^\n]+',
            r'美\s*术\s*编\s*辑\s*[：:]\s*[^\n]+',
            # 出版信息
            r'定\s*价\s*[：:]\s*[^\n]+',
            r'版\s*次\s*[：:]\s*[^\n]+',
            r'开\s*本\s*[：:]\s*[^\n]+',
            r'字\s*数\s*[：:]\s*[^\n]+',
            r'书\s*号\s*[：:]\s*[^\n]+',
            # 中国图书馆 CIP 核字
            r'中国版本图书馆CIP数据核字[^\n]+',
            r'京权图字[^\n]+',
            # 版权声明
            r'版权[^\n]*必究[^\n]*',
            r'中青版图书，版权所有，盗版必究[^\n]*',
            # 英文原版版权信息
            r'by\s+\S+[^\n]*Copyright[^\n]+',
            r'Simplified\s+Chinese\s+translation\s+copyright[^\n]+',
            # 购买链接、网址行
            r'购\s*书\s*网\s*址\s*[：:]\s*[^\n]+',
            r'公\s*司\s*网\s*址\s*[：:]\s*[^\n]+',
            r'电\s*话\s*[：:]\s*[^\n]+',
            # 发行信息
            r'发\s*行\s*[：:]\s*[^\n]+',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text)

        return text

    @staticmethod
    def remove_format_markers(text: str) -> str:
        """
        移除格式符号：====、----、第X章----、脚注标记等

        针对书籍排版时添加的分隔线和注释标记
        """
        patterns = [
            # 连续等号分隔线（可能有空格）
            r'={3,}\s*',
            # 连续短横线分隔线
            r'-{3,}\s*',
            # 波浪线分隔线
            r'～{2,}\s*',
            # 第X章后面紧跟的横线（常见于 "第 3 章 ---"）
            r'第\s*\d+\s*章\s*-{2,}',
            # 章节框/引用框（中文书名号内的章节注释）
            r'「[^」]*」',
            r'『[^』]*』',
            # 脚注标记 [1] [2] 等
            r'\[\d+\]',
            # 带内容的脚注 [1] 注释文字
            r'\[\d+\s*[^\]]*\]',
            # 上标数字（如第1章后的[1]）
            r'(?<=\S)\[\d+\](?=\s|$)',
        ]

        for pattern in patterns:
            text = re.sub(pattern, '', text)

        return text

    @staticmethod
    def split_into_sentences(text: str, min_length: int = 10) -> List[str]:
        """
        将文本分割成句子

        Args:
            text: 输入文本
            min_length: 最小句子长度
        """
        # 中英文句子分割
        # 中文句号、问号、感叹号
        pattern = r'([。！？.!?]+[」』】)"\']*\s*)'

        sentences = re.split(pattern, text)

        # 重新组合
        result = []
        current = ""

        for part in sentences:
            current += part
            if re.search(r'[。！？.!?]$', part.strip()):
                if len(current.strip()) >= min_length:
                    result.append(current.strip())
                current = ""

        if current.strip() and len(current.strip()) >= min_length:
            result.append(current.strip())

        return result


def clean_file(
    input_path: str,
    output_path: Optional[str] = None,
    remove_urls: bool = False,
    remove_emails: bool = False,
    remove_page_numbers: bool = True,
) -> str:
    """
    清洗文本文件

    Args:
        input_path: 输入文件路径
        output_path: 输出文件路径
        remove_urls: 是否移除URL
        remove_emails: 是否移除邮箱
        remove_page_numbers: 是否移除页码

    Returns:
        输出文件路径
    """
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_cleaned.txt"

    # 读取文件
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    # 创建清洗器
    cleaner = TextCleaner()

    # 添加基本清洗
    cleaner.add_cleaner(TextCleaner.fix_encoding_issues)
    cleaner.add_cleaner(TextCleaner.remove_empty_lines)
    cleaner.add_cleaner(TextCleaner.remove_extra_whitespace)

    # 添加书籍元数据清洗
    cleaner.add_cleaner(TextCleaner.remove_book_metadata)
    cleaner.add_cleaner(TextCleaner.remove_format_markers)

    if remove_urls:
        cleaner.add_cleaner(TextCleaner.remove_urls)

    if remove_emails:
        cleaner.add_cleaner(TextCleaner.remove_emails)

    if remove_page_numbers:
        cleaner.add_cleaner(TextCleaner.remove_page_numbers)

    # 执行清洗
    cleaned_text = cleaner.clean(text)

    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

    print(f"清洗完成: {input_path} -> {output_path}")
    print(f"  原始字符数: {len(text):,}")
    print(f"  清洗后字符数: {len(cleaned_text):,}")

    return output_path


def split_large_file(
    input_path: str,
    output_dir: str,
    max_chars: int = 1000000,
    overlap: int = 1000,
) -> List[str]:
    """
    将大文件分割成多个小文件

    Args:
        input_path: 输入文件路径
        output_dir: 输出目录
        max_chars: 每个文件最大字符数
        overlap: 文件间重叠字符数

    Returns:
        分割后的文件列表
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 按段落分割
    paragraphs = text.split('\n\n')

    files = []
    current_chunk = []
    current_length = 0
    file_idx = 1

    for para in paragraphs:
        para_len = len(para)

        if current_length + para_len > max_chars and current_chunk:
            # 保存当前块
            output_path = os.path.join(output_dir, f"part_{file_idx:04d}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(current_chunk))
            files.append(output_path)

            # 开始新块（保留一些重叠）
            if overlap > 0 and current_chunk:
                # 保留最后几个段落作为重叠
                overlap_text = '\n\n'.join(current_chunk[-2:])[-overlap:]
                current_chunk = [overlap_text]
                current_length = len(overlap_text)
            else:
                current_chunk = []
                current_length = 0

            file_idx += 1

        current_chunk.append(para)
        current_length += para_len + 2  # +2 for '\n\n'

    # 保存最后一块
    if current_chunk:
        output_path = os.path.join(output_dir, f"part_{file_idx:04d}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(current_chunk))
        files.append(output_path)

    print(f"文件分割完成: {len(files)} 个文件")
    return files


def batch_clean_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    remove_urls: bool = False,
    remove_emails: bool = False,
    remove_page_numbers: bool = True,
) -> List[str]:
    """
    批量清洗目录下的所有TXT文件

    Args:
        input_dir: 输入目录
        output_dir: 输出目录（可选，默认覆盖原文件）
        remove_urls: 是否移除URL
        remove_emails: 是否移除邮箱
        remove_page_numbers: 是否移除页码

    Returns:
        清洗成功的文件列表
    """
    input_dir = os.path.abspath(input_dir)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = input_dir

    # 查找所有TXT文件
    txt_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.txt'):
            txt_files.append(os.path.join(input_dir, file))

    if not txt_files:
        print(f"在 {input_dir} 中未找到TXT文件")
        return []

    print(f"找到 {len(txt_files)} 个TXT文件")
    print("=" * 50)

    success_files = []
    failed_files = []

    for idx, input_path in enumerate(txt_files, 1):
        print(f"\n[{idx}/{len(txt_files)}]")
        try:
            base_name = os.path.basename(input_path)
            output_path = os.path.join(output_dir, base_name)

            clean_file(
                input_path,
                output_path,
                remove_urls=remove_urls,
                remove_emails=remove_emails,
                remove_page_numbers=remove_page_numbers,
            )
            success_files.append(output_path)
        except Exception as e:
            print(f"  错误: {e}")
            failed_files.append(input_path)

    # 打印总结
    print("\n" + "=" * 50)
    print(f"清洗完成: 成功 {len(success_files)}, 失败 {len(failed_files)}")

    if failed_files:
        print("\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")

    return success_files
