"""
文档级/段落级去重模块
提供精确哈希去重和 N-gram 近似去重功能
"""
import hashlib
from typing import List, Set, Iterator, Dict


def exact_deduplicate(documents: List[str]) -> List[str]:
    """
    精确哈希去重（SHA-256）

    Args:
        documents: 文档列表

    Returns:
        去重后的文档列表
    """
    seen_hashes: Set[str] = set()
    unique_docs: List[str] = []

    for doc in documents:
        content_hash = hashlib.sha256(doc.encode('utf-8')).hexdigest()
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_docs.append(doc)

    return unique_docs


def ngram_signature(text: str, n: int = 5) -> Set[str]:
    """
    生成 N-gram 集合签名

    Args:
        text: 输入文本
        n: N-gram 大小，默认 5-gram

    Returns:
        N-gram 集合
    """
    tokens = text.split()
    ngrams: Set[str] = set()

    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i+n])
        ngrams.add(ngram)

    return ngrams


def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """
    计算 Jaccard 相似度

    Args:
        set1: 集合1
        set2: 集合2

    Returns:
        Jaccard 相似度 (0.0 ~ 1.0)
    """
    if not set1 or not set2:
        return 0.0

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0.0


def near_deduplicate(
    documents: List[str],
    threshold: float = 0.8,
    n: int = 5
) -> List[str]:
    """
    N-gram 近似去重

    使用 Jaccard 相似度检测近似重复的文档。
    Jaccard 相似度超过 threshold 的文档只保留第一个。

    Args:
        documents: 文档列表
        threshold: Jaccard 相似度阈值，默认 0.8
        n: N-gram 大小，默认 5

    Returns:
        去重后的文档列表
    """
    signatures: List[Set[str]] = [ngram_signature(doc, n) for doc in documents]
    keep_indices: List[int] = []

    for i, sig in enumerate(signatures):
        is_duplicate = False
        for j in keep_indices:
            if jaccard_similarity(sig, signatures[j]) > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            keep_indices.append(i)

    return [documents[i] for i in keep_indices]


def deduplicate_lines(file_path: str) -> Iterator[str]:
    """
    流式去重每行文本

    避免重复段落进入预处理流水线，节省 tokenize 时间。

    Args:
        file_path: 文本文件路径

    Yields:
        去重后的每行文本
    """
    seen: Set[str] = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
            if line_hash not in seen:
                seen.add(line_hash)
                yield line


def deduplicate_lines_from_text(lines: List[str]) -> List[str]:
    """
    对文本列表进行行级去重

    Args:
        lines: 文本行列表

    Returns:
        去重后的文本行列表
    """
    seen: Set[str] = set()
    unique_lines: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
        if line_hash not in seen:
            seen.add(line_hash)
            unique_lines.append(line)

    return unique_lines


def deduplicate_lines_with_threshold(
    lines: List[str],
    threshold: int = 3,
) -> List[str]:
    """
    带阈值的行级去重

    统计每行出现的次数，只有出现次数 >= threshold 时才去重。
    这样可以避免误删文档中只出现一次的模板文本。

    例如 threshold=3:
    - 某行出现 1-2 次 -> 全部保留
    - 某行出现 3+ 次 -> 只保留第一次出现的

    Args:
        lines: 文本行列表
        threshold: 去重阈值，出现次数 >= threshold 时去重（默认 3）

    Returns:
        去重后的文本行列表
    """
    if threshold <= 1:
        # 阈值 <= 1 等同于无阈值去重
        return deduplicate_lines_from_text(lines)

    # 统计每行出现次数
    line_counts: Dict[str, int] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
        line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    # 找出需要去重的行（出现次数 >= threshold）
    dedup_hashes: Set[str] = {
        h for h, count in line_counts.items() if count >= threshold
    }

    # 保留结果
    seen: Set[str] = set()
    unique_lines: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()

        # 如果该行需要去重，检查是否已见过
        if line_hash in dedup_hashes:
            if line_hash not in seen:
                seen.add(line_hash)
                unique_lines.append(line)
            # 重复行直接跳过
        else:
            # 不需要去重的行，直接保留
            unique_lines.append(line)

    return unique_lines


def deduplicate_stream_with_threshold(
    file_path: str,
    threshold: int = 3,
) -> Iterator[str]:
    """
    带阈值的流式行级去重

    Args:
        file_path: 文本文件路径
        threshold: 去重阈值（默认 3）

    Yields:
        去重后的每行文本
    """
    if threshold <= 1:
        yield from deduplicate_lines(file_path)
        return

    # 第一遍：统计每行出现次数
    line_counts: Dict[str, int] = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()
            line_counts[line_hash] = line_counts.get(line_hash, 0) + 1

    # 找出需要去重的行
    dedup_hashes: Set[str] = {
        h for h, count in line_counts.items() if count >= threshold
    }

    # 第二遍：输出结果
    seen: Set[str] = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line_hash = hashlib.md5(line.encode('utf-8')).hexdigest()

            if line_hash in dedup_hashes:
                if line_hash not in seen:
                    seen.add(line_hash)
                    yield line
            else:
                yield line
