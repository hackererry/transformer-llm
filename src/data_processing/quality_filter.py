"""
文本质量过滤模块
基于启发式规则计算文本质量分数，过滤低质量文本
"""
import re
from typing import List, Tuple


# 中文停用词表（高频常用词）
CHINESE_STOP_WORDS: set = {
    '的', '了', '是', '在', '和', '有', '我', '也', '就', '人', '都',
    '一个', '上', '不', '与', '为', '这', '那', '中', '来', '到', '对',
    '以', '可', '会', '被', '用', '他', '她', '它', '们', '而', '但',
    '或', '如', '从', '说', '要', '去', '你', '会', '还', '把', '让',
    '给', '想', '能', '没', '很', '都', '说', '只', '这个', '自己',
}

# 模板占位符关键词
TEMPLATE_KEYWORDS: List[str] = [
    '此处填写', '待填写', 'TODO', '待定', '空白', '无内容',
    '[此处省略', '（以下空白）', '未完待续', '略', '......',
    '（略）', '[  ]', '□', '■', '★', '☆',
    '请填写', '请输入', '暂无', '暂无信息',
]


def compute_quality_score(text: str) -> float:
    """
    基于启发式规则计算文本质量分数 (0.0 ~ 1.0)

    分数越高表示质量越好，低于 0.5 通常表示低质量文本。

    扣分维度：
    1. 字符异常：连续重复字符、乱码混搭、非中文超长串
    2. 结构异常：标点密度过低、行长度异常
    3. 内容异常：模板占位符、停用词密度过低、句子比例低

    Args:
        text: 输入文本

    Returns:
        质量分数 (0.0 ~ 1.0)
    """
    score = 1.0
    total_chars = len(text)

    # 空文本直接返回 0
    if total_chars == 0:
        return 0.0

    words = text.split()

    # =====================
    # 1. 字符级异常（最高扣 0.35）
    # =====================

    # 1.1 连续重复字符（如 "啊啊啊啊啊啊啊"）
    if re.search(r'(.)\1{4,}', text):
        score -= 0.25

    # 1.2 乱码混搭（中韩/中日混合）
    if re.search(r'[\u4e00-\u9fff]{1,3}[\uac00-\ud7af]{1,3}', text):
        score -= 0.2

    # 1.3 纯非中文超长串（如 base64、代码片段、哈希值）
    if re.search(r'^[^\u4e00-\u9fff\u4e00-\u9fff]{30,}$', text, re.MULTILINE):
        score -= 0.15

    # 1.4 异常 Unicode 字符比例过高
    unusual_chars = len(re.findall(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', text))
    if total_chars > 0 and unusual_chars / total_chars > 0.05:
        score -= 0.15

    # =====================
    # 2. 结构级异常（最高扣 0.25）
    # =====================

    # 2.1 标点密度过低（几乎没有句号/逗号 = 可能是乱码）
    punct_count = len(re.findall(r'[。！？.!?;,:，、]', text))
    punct_ratio = punct_count / total_chars
    if punct_ratio < 0.01:
        score -= 0.2

    # 2.2 行长度异常
    lines = text.split('\n')
    if lines:
        non_empty_lines = [l for l in lines if l.strip()]
        if non_empty_lines:
            avg_line_len = sum(len(l) for l in non_empty_lines) / len(non_empty_lines)
            if avg_line_len < 5:
                score -= 0.15
            elif avg_line_len > 1000:
                score -= 0.1

    # 2.3 整篇文本过长（可能是日志、爬取残留）
    if total_chars > 100000:
        score -= 0.1

    # =====================
    # 3. 内容级异常（最高扣 0.3）
    # =====================

    # 3.1 模板占位符关键词
    for kw in TEMPLATE_KEYWORDS:
        if kw in text:
            score -= 0.25
            break

    # 3.2 停用词密度过低（无意义片段，如随机字符串）
    if words:
        stop_ratio = sum(1 for w in words if w in CHINESE_STOP_WORDS) / len(words)
        if stop_ratio < 0.03 and len(words) > 10:
            score -= 0.2

    # 3.3 句子比例过低（大量 token 但几乎无完整句子 = 被截断的片段）
    if words and len(words) > 10:
        sentence_count = len(re.findall(r'[。！？.!?]', text))
        if sentence_count / len(words) < 0.03:
            score -= 0.15

    # 3.4 数字/字母比例异常高（可能是表格、编码数据）
    alpha_count = len(re.findall(r'[a-zA-Z0-9]', text))
    alpha_ratio = alpha_count / total_chars
    if alpha_ratio > 0.8:
        score -= 0.15

    # 3.5 包含明显的爬虫/系统残留
    system_keywords = ['http://', 'https://', 'www.', '.com', '.cn',
                       'Copyright', '版权所有', 'All Rights Reserved']
    for kw in system_keywords:
        if kw in text:
            score -= 0.15
            break

    # 确保分数在 [0.0, 1.0] 范围内
    return max(0.0, min(1.0, score))


def filter_by_quality(
    texts: List[str],
    min_score: float = 0.5,
) -> Tuple[List[str], List[float]]:
    """
    过滤低质量文本

    Args:
        texts: 文本列表
        min_score: 最低质量分数阈值，低于此值的文本被过滤，默认 0.5

    Returns:
        (通过过滤的文本列表, 对应的质量分数列表)
    """
    results: List[str] = []
    scores: List[float] = []

    for text in texts:
        score = compute_quality_score(text)
        if score >= min_score:
            results.append(text)
            scores.append(score)

    return results, scores


def filter_by_quality_with_stats(
    texts: List[str],
    min_score: float = 0.5,
) -> Tuple[List[str], List[float], dict]:
    """
    过滤低质量文本，并返回统计信息

    Args:
        texts: 文本列表
        min_score: 最低质量分数阈值

    Returns:
        (通过过滤的文本列表, 分数列表, 统计信息字典)
    """
    results, scores = filter_by_quality(texts, min_score)

    stats = {
        'total': len(texts),
        'passed': len(results),
        'filtered': len(texts) - len(results),
        'pass_rate': len(results) / len(texts) if texts else 0,
        'avg_score': sum(scores) / len(scores) if scores else 0,
        'min_score': min(scores) if scores else 0,
        'max_score': max(scores) if scores else 0,
    }

    return results, scores, stats
