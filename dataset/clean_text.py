"""
文本清洗工具
对转换后的文本进行清洗和预处理
"""
import re
import os
import argparse
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="文本清洗工具")

    subparsers = parser.add_subparsers(dest="command", help="命令")

    # clean 命令
    clean_parser = subparsers.add_parser("clean", help="清洗文本文件")
    clean_parser.add_argument("input", nargs="?", help="输入文件路径（与 -d 互斥）")
    clean_parser.add_argument("-o", "--output", help="输出文件路径")
    clean_parser.add_argument("-d", "--dir", help="批量清洗的输入目录")
    clean_parser.add_argument("--remove-urls", action="store_true", help="移除URL")
    clean_parser.add_argument("--remove-emails", action="store_true", help="移除邮箱")

    # split 命令
    split_parser = subparsers.add_parser("split", help="分割大文件")
    split_parser.add_argument("input", help="输入文件路径")
    split_parser.add_argument("-o", "--output-dir", required=True, help="输出目录")
    split_parser.add_argument("--max-chars", type=int, default=1000000, help="每文件最大字符数")
    split_parser.add_argument("--overlap", type=int, default=1000, help="文件间重叠字符数")

    args = parser.parse_args()

    if args.command == "clean":
        if args.dir:
            # 批量清洗模式
            batch_clean_directory(
                args.dir,
                args.output,
                remove_urls=args.remove_urls,
                remove_emails=args.remove_emails,
            )
        elif args.input:
            # 单文件清洗模式
            clean_file(
                args.input,
                args.output,
                remove_urls=args.remove_urls,
                remove_emails=args.remove_emails,
            )
        else:
            clean_parser.print_help()
    elif args.command == "split":
        split_large_file(
            args.input,
            args.output_dir,
            args.max_chars,
            args.overlap,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
