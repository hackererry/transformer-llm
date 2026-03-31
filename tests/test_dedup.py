"""
去重模块测试
测试精确去重、近似去重、行级去重等功能
"""
import pytest
import sys
import os
import tempfile
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.deduplicate import (
    exact_deduplicate,
    near_deduplicate,
    ngram_signature,
    jaccard_similarity,
    deduplicate_lines,
    deduplicate_lines_from_text,
    deduplicate_lines_with_threshold,
    deduplicate_stream_with_threshold,
)


class TestExactDeduplicate:
    """精确哈希去重测试（SHA-256）"""

    def test_removes_duplicate_documents(self):
        """测试去除完全重复的文档"""
        documents = [
            "这是第一篇文档",
            "这是第二篇文档",
            "这是第一篇文档",  # 重复
            "这是第三篇文档",
        ]
        result = exact_deduplicate(documents)
        assert len(result) == 3
        assert result[0] == "这是第一篇文档"

    def test_keeps_all_unique_documents(self):
        """测试保留所有唯一文档"""
        documents = [
            "文档A",
            "文档B",
            "文档C",
        ]
        result = exact_deduplicate(documents)
        assert len(result) == 3
        assert result == documents

    def test_empty_list(self):
        """测试空列表"""
        result = exact_deduplicate([])
        assert result == []

    def test_single_document(self):
        """测试单文档"""
        result = exact_deduplicate(["唯一的文档"])
        assert len(result) == 1
        assert result[0] == "唯一的文档"

    def test_all_duplicates(self):
        """测试全部重复"""
        documents = ["相同内容"] * 5
        result = exact_deduplicate(documents)
        assert len(result) == 1
        assert result[0] == "相同内容"


class TestNearDeduplicate:
    """近似去重测试（N-gram + Jaccard）

    注意：ngram_signature 使用 text.split() 分割文本，
    因此对于中文文本（无空格）需要调整 n 参数。
    """

    def test_removes_highly_similar_documents(self):
        """测试去除高度相似的文档"""
        # 使用空格分隔的文本以便 N-gram 正确分词
        documents = [
            "Machine learning is an important branch of AI",
            "Machine learning is an important branch of AI",
            "Deep learning is a branch of machine learning",
        ]
        result = near_deduplicate(documents, threshold=0.8)
        assert len(result) == 2  # 前两个应被合并

    def test_keeps_dissimilar_documents(self):
        """测试保留不相似文档"""
        documents = [
            "Today the weather is very nice",
            "It might rain tomorrow",
            "I watched a movie last night",
        ]
        result = near_deduplicate(documents, threshold=0.8)
        assert len(result) == 3

    def test_empty_list(self):
        """测试空列表"""
        result = near_deduplicate([])
        assert result == []

    def test_threshold_0_removes_nothing(self):
        """测试阈值为0时保留所有"""
        documents = ["document one", "document two", "document three"]
        result = near_deduplicate(documents, threshold=0.0)
        assert len(result) == 3

    def test_identical_documents_at_high_threshold(self):
        """测试阈值为0.99时去除完全相同的文档"""
        documents = [
            "This is completely identical text content here",
            "This is completely identical text content here",
            "This is completely different text content here",
        ]
        result = near_deduplicate(documents, threshold=0.99)
        assert len(result) == 2


class TestNgramSignature:
    """N-gram 签名测试"""

    def test_generates_correct_ngrams(self):
        """测试生成正确的 N-gram"""
        text = "机器 学习 是 很 有趣 的"
        sig = ngram_signature(text, n=2)
        expected = {
            "机器 学习",
            "学习 是",
            "是 很",
            "很 有趣",
            "有趣 的",
        }
        assert sig == expected

    def test_short_text(self):
        """测试短文本"""
        text = "短文本"
        sig = ngram_signature(text, n=3)
        assert len(sig) == 0  # 不足3个词

    def test_empty_text(self):
        """测试空文本"""
        sig = ngram_signature("", n=2)
        assert len(sig) == 0


class TestJaccardSimilarity:
    """Jaccard 相似度测试"""

    def test_identical_sets(self):
        """测试相同集合"""
        set1 = {"a", "b", "c"}
        set2 = {"a", "b", "c"}
        assert jaccard_similarity(set1, set2) == 1.0

    def test_disjoint_sets(self):
        """测试不相交集合"""
        set1 = {"a", "b"}
        set2 = {"c", "d"}
        assert jaccard_similarity(set1, set2) == 0.0

    def test_partial_overlap(self):
        """测试部分重叠"""
        set1 = {"a", "b", "c"}
        set2 = {"b", "c", "d"}
        # intersection = 2, union = 4
        assert jaccard_similarity(set1, set2) == 0.5

    def test_empty_set(self):
        """测试空集合"""
        assert jaccard_similarity(set(), {"a", "b"}) == 0.0
        assert jaccard_similarity({"a"}, set()) == 0.0
        assert jaccard_similarity(set(), set()) == 0.0


class TestDeduplicateLinesFromText:
    """行级去重测试（List 输入）"""

    def test_removes_duplicate_lines(self):
        """测试去除重复行"""
        lines = [
            "第一行",
            "第二行",
            "第一行",  # 重复
            "第三行",
        ]
        result = deduplicate_lines_from_text(lines)
        assert len(result) == 3
        assert result == ["第一行", "第二行", "第三行"]

    def test_preserves_order(self):
        """测试保持原始顺序"""
        lines = ["a", "b", "c", "a", "b", "d"]
        result = deduplicate_lines_from_text(lines)
        assert result == ["a", "b", "c", "d"]

    def test_empty_list(self):
        """测试空列表"""
        result = deduplicate_lines_from_text([])
        assert result == []

    def test_strips_whitespace(self):
        """测试自动去除空白"""
        lines = ["  内容  ", "内容", "内容  "]
        result = deduplicate_lines_from_text(lines)
        assert len(result) == 1

    def test_skips_empty_lines(self):
        """测试跳过空行"""
        lines = ["内容1", "", "  ", "内容2"]
        result = deduplicate_lines_from_text(lines)
        assert "" not in result
        assert "内容1" in result
        assert "内容2" in result


class TestDeduplicateLines:
    """行级去重测试（文件输入）"""

    def test_streaming_deduplication(self, tmp_path):
        """测试流式去重"""
        # 创建测试文件
        test_file = tmp_path / "test.txt"
        test_file.write_text("第一行\n第二行\n第一行\n第三行\n", encoding="utf-8")

        result = list(deduplicate_lines(str(test_file)))
        assert len(result) == 3
        assert result == ["第一行", "第二行", "第三行"]

    def test_empty_file(self, tmp_path):
        """测试空文件"""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        result = list(deduplicate_lines(str(test_file)))
        assert result == []

    def test_skips_empty_lines(self, tmp_path):
        """测试跳过空行"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("内容1\n\n内容2\n  \n内容1\n", encoding="utf-8")

        result = list(deduplicate_lines(str(test_file)))
        assert len(result) == 2
        assert "内容1" in result
        assert "内容2" in result
        assert "" not in result


class TestDeduplicateWithThreshold:
    """带阈值的行级去重测试"""

    def test_threshold_3_keeps_rare_lines(self):
        """阈值=3 时，出现 1-2 次的行应保留"""
        lines = [
            "只出现一次的行",
            "出现两次的同一行",
            "出现两次的同一行",
            "另一个只出现一次的行",
        ]
        result = deduplicate_lines_with_threshold(lines, threshold=3)
        assert len(result) == 4  # 全部保留，因为没有任何行出现 3+ 次

    def test_threshold_3_removes_frequent_lines(self):
        """阈值=3 时，出现 3+ 次的行只保留第一个"""
        lines = [
            "常见行A",  # 3 次
            "常见行A",
            "常见行A",
            "常见行B",  # 4 次
            "常见行B",
            "常见行B",
            "常见行B",
            "稀有行",   # 1 次
        ]
        result = deduplicate_lines_with_threshold(lines, threshold=3)
        assert len(result) == 3  # 常见行A 1个 + 常见行B 1个 + 稀有行 1个 = 3
        assert result.count("常见行A") == 1
        assert result.count("常见行B") == 1
        assert result.count("稀有行") == 1

    def test_threshold_2_deduplicates_frequent_lines(self):
        """阈值=2 时，出现 2+ 次的行只保留第一个"""
        lines = ["a", "b", "a", "c", "b", "d"]
        result = deduplicate_lines_with_threshold(lines, threshold=2)
        # a出现2次保留1个，b出现2次保留1个，c和d各出现1次保留
        assert result == ["a", "b", "c", "d"]

    def test_threshold_1_deduplicates_all(self):
        """阈值=1 时等同于普通去重（所有出现过的行都去重）"""
        lines = ["a", "b", "a", "c"]
        result = deduplicate_lines_with_threshold(lines, threshold=1)
        # 相当于普通去重
        assert result == ["a", "b", "c"]

    def test_preserves_order(self):
        """测试保持原始顺序"""
        lines = [
            "第一",   # 3 次
            "第二",   # 2 次
            "第一",
            "第二",
            "第一",
            "第三",   # 1 次
        ]
        result = deduplicate_lines_with_threshold(lines, threshold=3)
        # "第一"出现3次会去重（保留第一个，后续跳过）
        # "第二"出现2次不满足阈值，全部保留
        # "第三"出现1次，保留
        assert result == ["第一", "第二", "第二", "第三"]

    def test_streaming_dedup_with_threshold(self, tmp_path):
        """测试流式带阈值去重"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("行A\n行B\n行A\n行B\n行A\n行C\n", encoding="utf-8")

        result = list(deduplicate_stream_with_threshold(str(test_file), threshold=3))
        # 行A出现3次保留1个，行B出现2次保留，行C出现1次保留
        assert len(result) == 4  # 行A(1) + 行B(2) + 行C(1) = 4
        assert result.count("行A") == 1
        assert result.count("行B") == 2
        assert result.count("行C") == 1
