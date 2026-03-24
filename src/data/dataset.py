"""
数据集模块
支持预训练和微调数据集
"""
import os
import json
import mmap
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, List, Dict, Any, Union, Callable, Tuple
import numpy as np


class PretrainDataset(Dataset):
    """
    预训练数据集
    支持大规模文本文件的高效加载
    支持新旧两种缓存格式：
    - 新格式：包含 metadata 和 examples 的字典（支持元数据验证）
    - 旧格式：直接的 examples 列表（向后兼容）
    """

    CACHE_VERSION = "1.0"

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        buffer_size: int = 10000,
        overwrite_cache: bool = False,
        cache_dir: Optional[str] = None,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.buffer_size = buffer_size

        # 缓存路径
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(data_path), ".cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # 加载或处理数据
        self.examples = self._load_or_process_data(overwrite_cache)

    def _get_cache_path(self) -> str:
        """获取缓存文件路径"""
        import hashlib
        file_hash = hashlib.md5(self.data_path.encode()).hexdigest()[:8]
        return os.path.join(
            self.cache_dir,
            f"pretrain_{file_hash}_{self.max_seq_length}.pt",
        )

    def _load_or_process_data(self, overwrite: bool) -> List[Dict]:
        """加载缓存或处理原始数据"""
        cache_path = self._get_cache_path()

        if os.path.exists(cache_path) and not overwrite:
            print(f"Loading cached data from {cache_path}")
            cached_data = torch.load(cache_path, map_location="cpu", weights_only=False)

            # 检测缓存格式：新格式(dict with metadata) vs 旧格式(list)
            if isinstance(cached_data, dict) and "examples" in cached_data:
                print(f"  Detected new cache format (v{cached_data.get('version', 'unknown')})")
                self._metadata = cached_data.get("metadata", {})
                return cached_data["examples"]
            else:
                print(f"  Detected legacy cache format (list)")
                self._metadata = {}
                return cached_data

        print(f"Processing data from {self.data_path}")
        examples = self._process_data()

        # 保存为新格式
        cache_data = {
            "version": self.CACHE_VERSION,
            "metadata": {
                "max_seq_length": self.max_seq_length,
                "num_examples": len(examples),
                "original_file": self.data_path,
            },
            "examples": examples,
        }
        torch.save(cache_data, cache_path)
        print(f"Cached data saved to {cache_path}")

        self._metadata = cache_data["metadata"]
        return examples

    def _process_data(self) -> List[Dict]:
        """处理原始文本数据"""
        examples = []
        current_chunk = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 编码
                tokens = self.tokenizer.encode(line, add_special_tokens=False)

                # 添加到当前chunk
                current_chunk.extend(tokens)

                # 当chunk足够大时，创建样本
                while len(current_chunk) >= self.max_seq_length:
                    # 创建一个样本
                    sample = current_chunk[:self.max_seq_length]
                    examples.append({
                        "input_ids": sample,
                        "labels": sample.copy(),
                    })
                    # 滑动窗口，有一定重叠
                    current_chunk = current_chunk[self.max_seq_length // 2:]

        # 处理剩余数据
        if len(current_chunk) > 0:
            # Padding
            if len(current_chunk) < self.max_seq_length:
                current_chunk = current_chunk + [self.tokenizer.pad_token_id] * (self.max_seq_length - len(current_chunk))
            examples.append({
                "input_ids": current_chunk,
                "labels": current_chunk.copy(),
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


class PretrainIterableDataset(IterableDataset):
    """
    可迭代的预训练数据集
    适用于非常大的数据集，不需要一次性加载到内存
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __iter__(self):
        """迭代生成样本"""
        current_chunk = []

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tokens = self.tokenizer.encode(line, add_special_tokens=False)
                current_chunk.extend(tokens)

                while len(current_chunk) >= self.max_seq_length:
                    sample = current_chunk[:self.max_seq_length]
                    yield {
                        "input_ids": torch.tensor(sample, dtype=torch.long),
                        "labels": torch.tensor(sample.copy(), dtype=torch.long),
                    }
                    current_chunk = current_chunk[self.max_seq_length // 2:]


class FinetuneDataset(Dataset):
    """
    微调数据集
    支持指令微调(instruction tuning)格式
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        instruction_column: str = "instruction",
        input_column: str = "input",
        output_column: str = "output",
        template: str = "alpaca",
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

        self.instruction_column = instruction_column
        self.input_column = input_column
        self.output_column = output_column
        self.template = template

        # 加载数据
        self.data = self._load_data()
        self.examples = self._process_data()

    def _load_data(self) -> List[Dict]:
        """加载数据文件"""
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif self.data_path.endswith(".jsonl"):
            data = []
            with open(self.data_path, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
            return data
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")

    def _format_prompt(self, item: Dict) -> Tuple[str, str]:
        """格式化prompt"""
        instruction = item.get(self.instruction_column, "")
        input_text = item.get(self.input_column, "")
        output = item.get(self.output_column, "")

        if self.template == "alpaca":
            if input_text:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            full_text = prompt + output
        elif self.template == "chat":
            messages = []
            if input_text:
                messages.append({"role": "user", "content": f"{instruction}\n{input_text}"})
            else:
                messages.append({"role": "user", "content": instruction})
            messages.append({"role": "assistant", "content": output})
            full_text = self._format_chat(messages)
            prompt = full_text[:full_text.index(output)]
        else:
            # 简单模板
            prompt = f"Instruction: {instruction}\n"
            if input_text:
                prompt += f"Input: {input_text}\n"
            prompt += "Output: "
            full_text = prompt + output

        return prompt, full_text

    def _format_chat(self, messages: List[Dict]) -> str:
        """格式化对话格式"""
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                formatted += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted += f"<|assistant|)\n{content}\n"
        return formatted

    def _process_data(self) -> List[Dict]:
        """处理数据"""
        examples = []

        for item in self.data:
            prompt, full_text = self._format_prompt(item)

            # 编码
            full_tokens = self.tokenizer.encode(full_text, add_special_tokens=True)
            prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)

            # 截断
            if len(full_tokens) > self.max_seq_length:
                full_tokens = full_tokens[:self.max_seq_length]

            # 创建labels: prompt部分用-100屏蔽
            labels = full_tokens.copy()
            prompt_len = min(len(prompt_tokens), len(full_tokens))
            for i in range(prompt_len):
                labels[i] = -100

            examples.append({
                "input_ids": full_tokens,
                "labels": labels,
                "attention_mask": [1] * len(full_tokens),
            })

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.examples[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
        }


class TextFileDataset(Dataset):
    """
    简单文本文件数据集
    每行一个样本
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
        add_special_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.add_special_tokens = add_special_tokens

        # 读取所有行
        with open(data_path, "r", encoding="utf-8") as f:
            self.lines = [line.strip() for line in f if line.strip()]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        line = self.lines[idx]
        tokens = self.tokenizer.encode(line, add_special_tokens=self.add_special_tokens)

        # 截断或padding
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        attention_mask = [1] * len(tokens)

        # Padding
        padding_length = self.max_seq_length - len(tokens)
        if padding_length > 0:
            tokens = tokens + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens.copy(), dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class MemoryMappedDataset(Dataset):
    """
    内存映射数据集
    使用mmap加载大文件，减少内存占用
    """

    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_seq_length: int = 512,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data_path = data_path

        # 获取文件大小和行数
        self._file = open(data_path, "r", encoding="utf-8")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # 构建行索引
        self._line_offsets = [0]
        for i, byte in enumerate(self._mmap):
            if byte == ord('\n'):
                self._line_offsets.append(i + 1)

    def __len__(self) -> int:
        return len(self._line_offsets) - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self._line_offsets[idx]
        end = self._line_offsets[idx + 1] - 1  # -1 去掉换行符

        line = self._mmap[start:end].decode("utf-8").strip()
        tokens = self.tokenizer.encode(line, add_special_tokens=True)

        # 截断
        if len(tokens) > self.max_seq_length:
            tokens = tokens[:self.max_seq_length]

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(tokens.copy(), dtype=torch.long),
        }

    def __del__(self):
        if hasattr(self, '_mmap'):
            self._mmap.close()
        if hasattr(self, '_file'):
            self._file.close()


def create_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int = 512,
    dataset_type: str = "pretrain",
    **kwargs,
) -> Dataset:
    """
    创建数据集的工厂函数
    """
    if dataset_type == "pretrain":
        return PretrainDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs,
        )
    elif dataset_type == "finetune":
        return FinetuneDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs,
        )
    elif dataset_type == "text":
        return TextFileDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs,
        )
    elif dataset_type == "streaming":
        return PretrainIterableDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
