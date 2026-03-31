"""
PII（个人信息）移除模块
通过正则表达式识别并替换常见个人信息
"""
import re
from typing import Dict


# PII 正则模式库
PII_PATTERNS: Dict[str, str] = {
    # 手机号（中国大陆 11 位，以 1 开头）
    'phone_cn': r'1[3-9]\d{9}',

    # 固定电话（3/4 位区号 + 7/8 位号码）
    'phone_fix': r'\d{3,4}[-\s]?\d{7,8}',

    # 身份证号（中国 18 位）
    'id_card': r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]',

    # 邮箱地址
    'email': r'\S+@\S+\.\S+',

    # IP 地址
    'ip_address': r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',

    # 银行卡号（16-19 位）
    'bank_card': r'\d{16,19}',

    # QQ 号（5-11 位数字）
    'qq': r'\b[1-9]\d{4,10}\b',

    # 微信号（字母开头，5-20 位）
    'wechat': r'[a-zA-Z][a-zA-Z0-9_-]{5,19}',

    # 护照号（字母 + 8-9 位数字）
    'passport': r'[A-Z]\d{8,9}',

    # 国内军警电话
    'military_phone': r'\d{5,7}',
}

# PII 替换占位符
PII_PLACEHOLDERS: Dict[str, str] = {
    'phone_cn': '<PHONE>',
    'phone_fix': '<PHONE>',
    'id_card': '<ID>',
    'email': '<EMAIL>',
    'ip_address': '<IP>',
    'bank_card': '<BANK>',
    'qq': '<QQ>',
    'wechat': '<WECHAT>',
    'passport': '<PASSPORT>',
    'military_phone': '<PHONE>',
}

# 复合 PII 模式（多行或复杂格式）
COMPOUND_PATTERNS: Dict[str, str] = {
    # 地址行
    'address': r'(?:地址|住址|家庭地址|通讯地址|收货地址)[：:\s]*[^\n]{10,60}',
    # 姓名+电话组合
    'name_phone': r'[\u4e00-\u9fff]{2,4}[：:\s]*1[3-9]\d{9}',
}


def remove_pii(
    text: str,
    patterns: Dict[str, str] = None,
    placeholders: Dict[str, str] = None,
) -> str:
    """
    移除文本中的个人信息（PII）

    使用正则表达式匹配常见 PII 模式并替换为占位符。

    Args:
        text: 输入文本
        patterns: PII 正则模式库，默认使用 PII_PATTERNS
        placeholders: PII 替换占位符，默认使用 PII_PLACEHOLDERS

    Returns:
        PII 已替换为占位符的文本
    """
    if patterns is None:
        patterns = PII_PATTERNS
    if placeholders is None:
        placeholders = PII_PLACEHOLDERS

    # 简单模式替换
    for pii_type, pattern in patterns.items():
        placeholder = placeholders.get(pii_type, '<PII>')
        text = re.sub(pattern, placeholder, text)

    # 复合模式替换
    for pii_type, pattern in COMPOUND_PATTERNS.items():
        text = re.sub(pattern, '<ADDR>', text)

    return text


def remove_pii_with_count(text: str) -> tuple:
    """
    移除文本中的 PII，并返回被移除的 PII 数量统计

    Args:
        text: 输入文本

    Returns:
        (替换后的文本, PII 数量统计字典)
    """
    counts: Dict[str, int] = {k: 0 for k in list(PII_PATTERNS.keys()) + list(COMPOUND_PATTERNS.keys())}

    result = text

    # 统计各类型 PII 数量
    for pii_type, pattern in {**PII_PATTERNS, **COMPOUND_PATTERNS}.items():
        matches = re.findall(pattern, result)
        counts[pii_type] = len(matches)

    # 执行替换
    result = remove_pii(result)

    return result, counts


def has_pii(text: str) -> bool:
    """
    检查文本是否包含 PII

    Args:
        text: 输入文本

    Returns:
        是否包含 PII
    """
    all_patterns = {**PII_PATTERNS, **COMPOUND_PATTERNS}

    for pattern in all_patterns.values():
        if re.search(pattern, text):
            return True

    return False
