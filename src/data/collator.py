"""
数据整理器模块
处理动态padding和批处理
"""
import torch
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class DataCollatorForLanguageModeling:
    """
    语言模型数据整理器
    处理动态padding和label创建
    """

    pad_token_id: int = 0
    max_length: Optional[int] = None
    padding: str = "longest"  # "longest" or "max_length"
    truncation: bool = True
    return_tensors: str = "pt"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        整理一批样本

        Args:
            batch: 样本列表，每个样本是包含input_ids等字段的字典
        Returns:
            整理后的批次字典
        """
        # 提取所有字段
        input_ids = [item["input_ids"] for item in batch]

        # 处理labels
        if "labels" in batch[0]:
            labels = [item["labels"] for item in batch]
        else:
            # 如果没有labels，使用input_ids作为labels
            labels = input_ids.copy()

        # 处理attention_mask
        if "attention_mask" in batch[0]:
            attention_masks = [item["attention_mask"] for item in batch]
        else:
            attention_masks = [[1] * len(ids) for ids in input_ids]

        # Padding
        if self.padding == "longest":
            max_len = max(len(ids) for ids in input_ids)
            if self.max_length:
                max_len = min(max_len, self.max_length)
        else:
            max_len = self.max_length or max(len(ids) for ids in input_ids)

        # 应用padding和截断
        input_ids = self._pad_sequence(input_ids, max_len, self.pad_token_id)
        labels = self._pad_sequence(labels, max_len, -100)  # labels用-100 padding
        attention_masks = self._pad_sequence(attention_masks, max_len, 0)

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks),
        }

    def _pad_sequence(
        self,
        sequences: List[List[int]],
        max_length: int,
        pad_value: int,
    ) -> List[torch.Tensor]:
        """对序列列表进行padding"""
        result = []
        for seq in sequences:
            # 转换为tensor
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()

            # 截断
            if self.truncation and len(seq) > max_length:
                seq = seq[:max_length]

            # Padding
            padding_length = max_length - len(seq)
            if padding_length > 0:
                seq = seq + [pad_value] * padding_length

            result.append(torch.tensor(seq, dtype=torch.long))

        return result


@dataclass
class DataCollatorForCausalLM:
    """
    因果语言模型数据整理器
    自动创建causal mask
    """

    pad_token_id: int = 0
    max_length: Optional[int] = None
    padding: str = "longest"

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """整理批次"""
        input_ids = [item["input_ids"] for item in batch]

        # 确定最大长度
        if self.padding == "longest":
            max_len = max(len(ids) for ids in input_ids)
            if self.max_length:
                max_len = min(max_len, self.max_length)
        else:
            max_len = self.max_length or max(len(ids) for ids in input_ids)

        batch_size = len(input_ids)

        # 预分配 batch tensor（避免逐样本创建再 stack）
        input_ids_tensor = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask_tensor = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels_tensor = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, ids in enumerate(input_ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            # 截断
            seq_len = min(len(ids), max_len)
            input_ids_tensor[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)
            attention_mask_tensor[i, :seq_len] = 1
            labels_tensor[i, :seq_len] = torch.tensor(ids[:seq_len], dtype=torch.long)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
        }


@dataclass
class DataCollatorForSFT:
    """
    监督微调数据整理器
    处理instruction-input-output格式
    """

    pad_token_id: int = 0
    max_length: Optional[int] = None
    padding: str = "longest"
    ignore_index: int = -100

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """整理批次"""
        input_ids = [item["input_ids"] for item in batch]
        labels = [item.get("labels", item["input_ids"]) for item in batch]
        attention_masks = [item.get("attention_mask", [1] * len(input_ids[i])) for i, item in enumerate(batch)]

        # 确定最大长度
        if self.padding == "longest":
            max_len = max(len(ids) for ids in input_ids)
            if self.max_length:
                max_len = min(max_len, self.max_length)
        else:
            max_len = self.max_length or max(len(ids) for ids in input_ids)

        # 处理每个样本
        padded_inputs = []
        padded_labels = []
        padded_masks = []

        for ids, lbl, mask in zip(input_ids, labels, attention_masks):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(lbl, torch.Tensor):
                lbl = lbl.tolist()
            if isinstance(mask, torch.Tensor):
                mask = mask.tolist()

            # 截断
            if len(ids) > max_len:
                ids = ids[:max_len]
                lbl = lbl[:max_len]
                mask = mask[:max_len]

            # Padding
            padding_length = max_len - len(ids)
            if padding_length > 0:
                ids = ids + [self.pad_token_id] * padding_length
                lbl = lbl + [self.ignore_index] * padding_length
                mask = mask + [0] * padding_length

            padded_inputs.append(torch.tensor(ids, dtype=torch.long))
            padded_labels.append(torch.tensor(lbl, dtype=torch.long))
            padded_masks.append(torch.tensor(mask, dtype=torch.long))

        return {
            "input_ids": torch.stack(padded_inputs),
            "labels": torch.stack(padded_labels),
            "attention_mask": torch.stack(padded_masks),
        }


class DynamicBatchSampler:
    """
    动态批采样器
    根据序列长度动态调整批次大小，提高训练效率
    """

    def __init__(
        self,
        dataset,
        max_tokens: int = 8192,
        max_batch_size: int = 32,
        drop_last: bool = False,
        length_fn: Optional[callable] = None,
    ):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.max_batch_size = max_batch_size
        self.drop_last = drop_last

        # 获取每个样本的长度
        if length_fn is None:
            self.lengths = [len(item["input_ids"]) for item in dataset]
        else:
            self.lengths = [length_fn(item) for item in dataset]

    def __iter__(self):
        """迭代生成批次索引"""
        # 按长度排序
        indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])

        batches = []
        current_batch = []
        current_max_len = 0

        for idx in indices:
            sample_len = self.lengths[idx]

            # 检查是否可以添加到当前批次
            new_max_len = max(current_max_len, sample_len)
            new_tokens = new_max_len * (len(current_batch) + 1)

            if new_tokens <= self.max_tokens and len(current_batch) < self.max_batch_size:
                current_batch.append(idx)
                current_max_len = new_max_len
            else:
                # 开始新批次
                if current_batch:
                    batches.append(current_batch)
                current_batch = [idx]
                current_max_len = sample_len

        # 处理最后一个批次
        if current_batch and not self.drop_last:
            batches.append(current_batch)

        for batch in batches:
            yield batch

    def __len__(self):
        """估算批次数"""
        # 粗略估计
        avg_len = sum(self.lengths) / len(self.lengths)
        avg_batch_size = min(self.max_tokens // int(avg_len), self.max_batch_size)
        return len(self.lengths) // max(1, avg_batch_size)


def get_collator(
    collator_type: str = "causal_lm",
    pad_token_id: int = 0,
    max_length: Optional[int] = None,
    **kwargs,
):
    """
    获取数据整理器的工厂函数
    """
    if collator_type == "language_modeling":
        return DataCollatorForLanguageModeling(
            pad_token_id=pad_token_id,
            max_length=max_length,
            **kwargs,
        )
    elif collator_type == "causal_lm":
        return DataCollatorForCausalLM(
            pad_token_id=pad_token_id,
            max_length=max_length,
            **kwargs,
        )
    elif collator_type == "sft":
        return DataCollatorForSFT(
            pad_token_id=pad_token_id,
            max_length=max_length,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown collator type: {collator_type}")
