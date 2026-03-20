"""
评估指标模块
提供常用的NLP和训练评估指标
"""
import torch
import math
from typing import List, Dict, Optional, Any, Callable
import numpy as np


def compute_perplexity(loss: float) -> float:
    """
    计算困惑度

    Args:
        loss: 交叉熵损失

    Returns:
        困惑度
    """
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def compute_accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> float:
    """
    计算准确率

    Args:
        predictions: 预测logits [batch, seq_len, vocab_size] 或 [batch, seq_len]
        labels: 标签 [batch, seq_len]
        ignore_index: 忽略的标签索引

    Returns:
        准确率
    """
    if predictions.dim() == 3:
        predictions = predictions.argmax(dim=-1)

    # 创建有效标签掩码
    mask = labels != ignore_index

    # 计算正确预测数
    correct = (predictions == labels) & mask
    total = mask.sum()

    if total == 0:
        return 0.0

    return correct.sum().float() / total.float()


def compute_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    """
    计算token级别的准确率指标

    Args:
        logits: 模型输出logits [batch, seq_len, vocab_size]
        labels: 标签 [batch, seq_len]
        ignore_index: 忽略的标签索引

    Returns:
        指标字典
    """
    predictions = logits.argmax(dim=-1)

    # 创建掩码
    mask = labels != ignore_index

    # 总体准确率
    total_correct = ((predictions == labels) & mask).sum().float()
    total_tokens = mask.sum().float()
    overall_accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    # Top-5准确率
    _, top5_predictions = logits.topk(5, dim=-1)
    top5_correct = top5_predictions.eq(labels.unsqueeze(-1)).any(dim=-1)
    top5_correct = (top5_correct & mask).sum().float()
    top5_accuracy = top5_correct / total_tokens if total_tokens > 0 else 0.0

    return {
        "accuracy": overall_accuracy.item(),
        "top5_accuracy": top5_accuracy.item(),
    }


def compute_bleu_score(
    predictions: List[str],
    references: List[str],
    max_n: int = 4,
) -> Dict[str, float]:
    """
    计算BLEU分数 (简化版)

    Args:
        predictions: 预测文本列表
        references: 参考文本列表
        max_n: 最大n-gram

    Returns:
        BLEU分数字典
    """
    from collections import Counter

    def get_ngrams(text: str, n: int) -> Counter:
        tokens = text.split()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i + n]))
        return Counter(ngrams)

    scores = {}

    for n in range(1, max_n + 1):
        total_pred_ngrams = 0
        total_match_ngrams = 0

        for pred, ref in zip(predictions, references):
            pred_ngrams = get_ngrams(pred, n)
            ref_ngrams = get_ngrams(ref, n)

            for ngram, count in pred_ngrams.items():
                total_pred_ngrams += count
                total_match_ngrams += min(count, ref_ngrams.get(ngram, 0))

        if total_pred_ngrams > 0:
            scores[f"bleu_{n}"] = total_match_ngrams / total_pred_ngrams
        else:
            scores[f"bleu_{n}"] = 0.0

    # 计算几何平均
    if all(s > 0 for s in scores.values()):
        bleu = np.exp(np.mean([np.log(s) for s in scores.values()]))
    else:
        bleu = 0.0

    scores["bleu"] = bleu

    return scores


def compute_f1_score(
    predictions: List[int],
    labels: List[int],
    average: str = "micro",
) -> Dict[str, float]:
    """
    计算F1分数

    Args:
        predictions: 预测标签列表
        labels: 真实标签列表
        average: 平均方式 ("micro", "macro", "weighted")

    Returns:
        F1分数字典
    """
    predictions = np.array(predictions)
    labels = np.array(labels)

    # 获取所有类别
    classes = np.unique(np.concatenate([predictions, labels]))

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for cls in classes:
        true_positive = np.sum((predictions == cls) & (labels == cls))
        false_positive = np.sum((predictions == cls) & (labels != cls))
        false_negative = np.sum((predictions != cls) & (labels == cls))

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(labels == cls)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1s = np.array(f1s)
    supports = np.array(supports)

    if average == "micro":
        # 全局计算
        true_positives = np.sum((predictions == labels))
        false_positives = np.sum((predictions != labels))
        false_negatives = np.sum((predictions != labels))

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    elif average == "macro":
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        f1 = np.mean(f1s)

    elif average == "weighted":
        total_support = np.sum(supports)
        precision = np.sum(precisions * supports) / total_support if total_support > 0 else 0
        recall = np.sum(recalls * supports) / total_support if total_support > 0 else 0
        f1 = np.sum(f1s * supports) / total_support if total_support > 0 else 0

    else:
        raise ValueError(f"Unknown average type: {average}")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


class MetricsTracker:
    """
    指标跟踪器
    跟踪和聚合多个指标
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], count: int = 1):
        """
        更新指标

        Args:
            metrics: 指标字典
            count: 样本数量
        """
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
                self.counts[key] = 0

            self.metrics[key].append(value * count)
            self.counts[key] += count

    def average(self) -> Dict[str, float]:
        """计算平均值"""
        result = {}
        for key in self.metrics:
            if self.counts[key] > 0:
                result[key] = sum(self.metrics[key]) / self.counts[key]
            else:
                result[key] = 0.0
        return result

    def reset(self):
        """重置所有指标"""
        self.metrics = {}
        self.counts = {}

    def get_last(self) -> Dict[str, float]:
        """获取最近的指标值"""
        result = {}
        for key, values in self.metrics.items():
            if values:
                result[key] = values[-1] / self.counts[key] if self.counts[key] > 0 else 0.0
        return result


class PerplexityCalculator:
    """
    困惑度计算器
    支持累积计算
    """

    def __init__(self):
        self.total_loss = 0.0
        self.total_tokens = 0

    def update(self, loss: float, num_tokens: int):
        """更新损失"""
        self.total_loss += loss * num_tokens
        self.total_tokens += num_tokens

    def compute(self) -> float:
        """计算困惑度"""
        if self.total_tokens == 0:
            return float("inf")

        avg_loss = self.total_loss / self.total_tokens
        return compute_perplexity(avg_loss)

    def reset(self):
        """重置"""
        self.total_loss = 0.0
        self.total_tokens = 0


def compute_generation_metrics(
    generated_texts: List[str],
    reference_texts: List[str],
) -> Dict[str, float]:
    """
    计算文本生成指标

    Args:
        generated_texts: 生成的文本列表
        reference_texts: 参考文本列表

    Returns:
        指标字典
    """
    metrics = {}

    # BLEU分数
    bleu_scores = compute_bleu_score(generated_texts, reference_texts)
    metrics.update({f"bleu_{k}": v for k, v in bleu_scores.items()})

    # 平均长度
    gen_lengths = [len(text.split()) for text in generated_texts]
    ref_lengths = [len(text.split()) for text in reference_texts]

    metrics["avg_gen_length"] = np.mean(gen_lengths)
    metrics["avg_ref_length"] = np.mean(ref_lengths)
    metrics["length_ratio"] = metrics["avg_gen_length"] / metrics["avg_ref_length"] if metrics["avg_ref_length"] > 0 else 0

    return metrics
