# -*- coding: utf-8 -*-
"""
PII个人信息移除模块
通过正则表达式识别并替换个人敏感信息
"""
import re
from typing import Dict, Tuple


# =====================
# 无上下文要求的 PII 模式（格式本身足够特异性）
# =====================
PII_PATTERNS: Dict[str, str] = {
    "phone_cn": r"1[3-9]\d{9}",                    # 手机号（1开头11位）
    "id_card": r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]",
    "email": r"\S+@\S+\.\S+",                       # 邮箱（含 @）
    "ip_address": r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",  # IP地址（四段点分）
    "bank_card": r"\d{16,19}",                      # 银行卡号（16-19位连续数字）
}

PII_PLACEHOLDERS: Dict[str, str] = {
    "phone_cn": "<PHONE>",
    "phone_fix": "<PHONE>",
    "id_card": "<ID>",
    "email": "<EMAIL>",
    "ip_address": "<IP>",
    "bank_card": "<BANK>",
    "qq": "<QQ>",
    "passport": "<PASSPORT>",
}

# =====================
# 需要上下文关键词的 PII 模式（仅当前面有特定关键词时才匹配）
# 格式: {name: (keyword_prefix_regex, number_regex, placeholder)}
# =====================
CONTEXT_PII_PATTERNS: Dict[str, Tuple[str, str, str]] = {
    "qq": (
        r"(?:QQ|qq|Q号|企鹅号)\s*[：:]*\s*",
        r"[1-9]\d{4,10}\b",
        "<QQ>",
    ),
    "passport": (
        r"(?:护照|passport|通行证)\s*[：:]*\s*",
        r"[A-Z]\d{8,9}",
        "<PASSPORT>",
    ),
    "phone_fix": (
        r"(?:电话|tel|fax|座机|固话|传真)\s*[：:]*\s*",
        r"\d{3,4}[-\s]?\d{7,8}",
        "<PHONE>",
    ),
}

# =====================
# 复合 PII 模式（多字段组合）
# =====================
COMPOUND_PATTERNS: Dict[str, str] = {
    "address": r"(?:地址|住址|家庭地址|通讯地址|收货地址)[：:\s]*[^\n]{10,60}",
    "name_phone": r"[\u4e00-\u9fff]{2,4}[：:\s]*1[3-9]\d{9}",
}


def remove_pii(
    text: str,
    patterns: Dict[str, str] = PII_PATTERNS,
    placeholders: Dict[str, str] = PII_PLACEHOLDERS,
) -> str:
    """
    移除文本中的个人敏感信息（PII）

    使用正则表达式匹配常见 PII 模式，并替换为占位符。

    Args:
        text: 输入文本
        patterns: PII 匹配模式集，默认使用 PII_PATTERNS
        placeholders: PII 替换占位符，默认使用 PII_PLACEHOLDERS

    Returns:
        PII 被替换为占位符后的文本
    """
    # 1. 无上下文要求的模式：直接替换
    for pii_type, pattern in patterns.items():
        placeholder = placeholders.get(pii_type, "<PII>")
        text = re.sub(pattern, placeholder, text)

    # 2. 上下文关键词模式：仅当关键词存在时替换数字部分
    for _name, (keyword_pat, num_pat, placeholder) in CONTEXT_PII_PATTERNS.items():
        # 用捕获组保留关键词，只替换后面的数字/号码
        text = re.sub(
            f"({keyword_pat})({num_pat})",
            rf"\g<1>{placeholder}",
            text,
        )

    # 3. 复合模式
    for pii_type, pattern in COMPOUND_PATTERNS.items():
        placeholder = "<ADDR>" if pii_type == "address" else "<PII>"
        text = re.sub(pattern, placeholder, text)

    return text


def remove_pii_with_count(text: str) -> Tuple[str, Dict[str, int]]:
    """
    移除文本中的 PII，并返回被移除的 PII 数量统计

    Args:
        text: 输入文本

    Returns:
        (替换后的文本, PII 数量统计字典)
    """
    counts = {k: 0 for k in PII_PATTERNS}
    counts.update({k: 0 for k in CONTEXT_PII_PATTERNS})
    counts.update({k: 0 for k in COMPOUND_PATTERNS})

    # 统计无上下文模式
    for pii_type, pattern in PII_PATTERNS.items():
        matches = re.findall(pattern, text)
        counts[pii_type] = len(matches)

    # 统计上下文关键词模式
    for name, (keyword_pat, num_pat, _placeholder) in CONTEXT_PII_PATTERNS.items():
        full_pattern = f"{keyword_pat}{num_pat}"
        matches = re.findall(full_pattern, text)
        counts[name] = len(matches)

    # 统计复合模式
    for pii_type, pattern in COMPOUND_PATTERNS.items():
        matches = re.findall(pattern, text)
        counts[pii_type] = len(matches)

    result = remove_pii(text)
    return result, counts


def has_pii(text: str) -> bool:
    """
    检查文本是否包含 PII

    Args:
        text: 输入文本

    Returns:
        是否包含 PII
    """
    # 检查无上下文模式
    for pattern in PII_PATTERNS.values():
        if re.search(pattern, text):
            return True

    # 检查上下文关键词模式
    for _name, (keyword_pat, num_pat, _placeholder) in CONTEXT_PII_PATTERNS.items():
        full_pattern = f"{keyword_pat}{num_pat}"
        if re.search(full_pattern, text):
            return True

    # 检查复合模式
    for pattern in COMPOUND_PATTERNS.values():
        if re.search(pattern, text):
            return True

    return False
