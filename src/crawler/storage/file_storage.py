# -*- coding: utf-8 -*-
"""
文件存储管理器
File Storage Manager for crawled content
"""
import os
from datetime import datetime
from typing import Optional


class FileStorage:
    """文件存储管理器

    按日期分文件，每文件最大1MB，按句号换行
    """

    def __init__(self, data_dir: str, max_file_size: int = 1 * 1024 * 1024):
        """
        初始化文件存储

        Args:
            data_dir: 数据存储目录
            max_file_size: 单文件最大字节数，默认1MB
        """
        self.data_dir = data_dir
        self.max_file_size = max_file_size
        self.current_date = datetime.now().strftime("%Y%m%d")
        self.current_seq = 0
        self.current_file_path: Optional[str] = None
        self._current_file_size = 0
        os.makedirs(data_dir, exist_ok=True)

        # 查找最后一个已有文件，设置起始位置
        self._find_last_file()

    def write_content(self, content: str) -> str:
        """
        写入内容到文件

        按句号分割换行，只保留文本内容（不含URL）
        增量存储：文件未达到1MB时持续写入该文件，超过1MB才创建新文件

        Args:
            content: 要写入的文本内容

        Returns:
            写入的文件路径
        """
        if not content or not content.strip():
            return ""

        # 按句号分割换行
        lines = content.replace('。', '。\n').split('\n')
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            return ""

        # 获取当前文件路径（如果是新的一天，重置序号）
        self._check_date_change()

        # 检查当前文件大小，如果超过限制则创建新文件
        while self._current_file_size >= self.max_file_size:
            self.current_seq += 1
            self._current_file_size = 0

        file_path = self._get_current_file_path()

        # 追加模式打开文件
        with open(file_path, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')

        self._current_file_size = os.path.getsize(file_path)
        self.current_file_path = file_path

        return file_path

    def _check_date_change(self):
        """检查日期是否变化，如果变化则重置序号"""
        today = datetime.now().strftime("%Y%m%d")
        if today != self.current_date:
            self.current_date = today
            self.current_seq = 0
            self._current_file_size = 0

    def _find_last_file(self):
        """查找目录中最后一个文件，设置起始位置"""
        if not os.path.exists(self.data_dir):
            return

        # 查找匹配当前日期的文件
        prefix = f"{self.current_date}-"
        max_seq = -1
        last_file_size = 0

        for filename in os.listdir(self.data_dir):
            if filename.startswith(prefix) and filename.endswith('.txt'):
                try:
                    seq_str = filename[len(prefix):-4]  # 去掉前缀和.txt
                    seq = int(seq_str)
                    if seq > max_seq:
                        max_seq = seq
                        file_path = os.path.join(self.data_dir, filename)
                        last_file_size = os.path.getsize(file_path)
                except ValueError:
                    continue

        if max_seq >= 0:
            self.current_seq = max_seq
            self._current_file_size = last_file_size

    def _get_current_file_path(self) -> str:
        """获取当前文件路径"""
        return os.path.join(self.data_dir, f"{self.current_date}-{self.current_seq}.txt")

    def _get_next_file_path(self) -> str:
        """获取下一个文件路径"""
        self.current_seq += 1
        return os.path.join(self.data_dir, f"{self.current_date}-{self.current_seq}.txt")

    def get_current_file_path(self) -> Optional[str]:
        """获取当前活跃文件路径"""
        return self.current_file_path
