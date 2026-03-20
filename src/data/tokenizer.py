"""
Tokenizer封装模块
支持HuggingFace tokenizers和简单BPE实现
"""
import os
import json
from typing import Optional, List, Dict, Union, Tuple
from collections import defaultdict
import re


class BaseTokenizer:
    """
    Tokenizer基类
    定义通用接口
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        self.vocab = vocab or {}
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}

        # 特殊token
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        # 获取特殊token的ID
        self.unk_token_id = self.vocab.get(unk_token, 0)
        self.pad_token_id = self.vocab.get(pad_token, 1)
        self.bos_token_id = self.vocab.get(bos_token, 2)
        self.eos_token_id = self.vocab.get(eos_token, 3)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        raise NotImplementedError

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        raise NotImplementedError

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """
        便捷调用方法
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # 编码
        encodings = [self.encode(t, add_special_tokens) for t in texts]

        # 截断
        if truncation and max_length:
            encodings = [e[:max_length] for e in encodings]

        # Padding
        if padding:
            max_len = max(len(e) for e in encodings)
            if max_length:
                max_len = min(max_len, max_length)
            encodings = [e + [self.pad_token_id] * (max_len - len(e)) for e in encodings]

        # 构建attention mask
        attention_masks = [
            [1] * len([t for t in e if t != self.pad_token_id]) + [0] * (len(e) - len([t for t in e if t != self.pad_token_id]))
            for e in encodings
        ]

        result = {
            "input_ids": encodings,
            "attention_mask": attention_masks,
        }

        # 转换为tensor
        if return_tensors == "pt":
            import torch
            result["input_ids"] = torch.tensor(result["input_ids"])
            result["attention_mask"] = torch.tensor(result["attention_mask"])

        return result

    def save(self, path: str):
        """保存tokenizer"""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "BaseTokenizer":
        """加载tokenizer"""
        with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        with open(os.path.join(path, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        return cls(vocab=vocab, **config)


class BPETokenizer(BaseTokenizer):
    """
    简单的BPE (Byte Pair Encoding) Tokenizer实现
    支持两种保存格式：
    - legacy: 分散文件 (vocab.json + merges.txt + tokenizer_config.json)
    - unified: 一站式文件 (tokenizer.json)
    """

    # 类常量：正则表达式模式
    DEFAULT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+| ?[^\s\w\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        **kwargs,
    ):
        super().__init__(vocab=vocab, **kwargs)
        self.merges = merges or []
        self.bpe_ranks = {merge: i for i, merge in enumerate(self.merges)}

        # 预编译正则表达式用于分词（兼容Python标准re模块）
        # 支持中英文混合文本
        # \u4e00-\u9fff: CJK统一汉字
        # \u3400-\u4dbf: CJK扩展A
        # \uf900-\ufaff: CJK兼容汉字
        self.pattern = re.compile(self.DEFAULT_PATTERN)

    def train(self, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2):
        """
        训练BPE tokenizer
        """
        # 1. 初始化词汇表为所有字符
        word_freqs = defaultdict(int)
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_freqs[" ".join(list(word))] += 1

        # 2. 添加特殊token
        self.vocab = {
            self.unk_token: 0,
            self.pad_token: 1,
            self.bos_token: 2,
            self.eos_token: 3,
        }

        # 3. 添加所有字符到词汇表
        chars = set()
        for word in word_freqs:
            for char in word.split():
                chars.add(char)

        for char in sorted(chars):
            self.vocab[char] = len(self.vocab)

        # 4. BPE合并
        self.merges = []
        while len(self.vocab) < vocab_size:
            # 计算所有相邻pair的频率
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                symbols = word.split()
                for i in range(len(symbols) - 1):
                    pair = (symbols[i], symbols[i + 1])
                    pair_freqs[pair] += freq

            # 找最高频pair
            if not pair_freqs:
                break

            best_pair = max(pair_freqs, key=pair_freqs.get)
            if pair_freqs[best_pair] < min_frequency:
                break

            # 合并
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = len(self.vocab)

            # 更新word_freqs
            new_word_freqs = {}
            for word, freq in word_freqs.items():
                new_word = word.replace(f"{best_pair[0]} {best_pair[1]}", new_token)
                new_word_freqs[new_word] = freq
            word_freqs = new_word_freqs

        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {merge: i for i, merge in enumerate(self.merges)}

    def _tokenize(self, text: str) -> List[str]:
        """基础分词"""
        return self.pattern.findall(text)

    def _bpe(self, token: str) -> str:
        """对单个token应用BPE"""
        word = " ".join(list(token))

        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break

            # 找到优先级最高的pair
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            if bigram not in self.bpe_ranks:
                break

            # 合并
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word.split()):
                symbols = word.split()
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(symbols[i])
                    i += 1
            word = " ".join(new_word)

        return word

    def _get_pairs(self, word: str) -> List[Tuple[str, str]]:
        """获取word中的所有相邻pair"""
        symbols = word.split()
        return [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        编码文本为token ID列表
        """
        tokens = []

        if add_special_tokens:
            tokens.append(self.bos_token_id)

        # 分词
        words = self._tokenize(text)

        for word in words:
            # 应用BPE
            bpe_tokens = self._bpe(word).split()
            for token in bpe_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.unk_token_id)

        if add_special_tokens:
            tokens.append(self.eos_token_id)

        return tokens

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        解码token ID列表为文本
        """
        tokens = []
        for id in ids:
            if id in self.ids_to_tokens:
                token = self.ids_to_tokens[id]
                if skip_special_tokens and token in [self.unk_token, self.pad_token, self.bos_token, self.eos_token]:
                    continue
                tokens.append(token)
            else:
                tokens.append(self.unk_token)

        return "".join(tokens).replace(" ", "")

    def save(self, path: str, format: str = "legacy"):
        """
        保存tokenizer

        Args:
            path: 保存路径
            format: 保存格式
                - "legacy": 旧版格式 (vocab.json + merges.txt + tokenizer_config.json)
                - "unified": 新版一站式格式 (tokenizer.json)
                - "both": 同时保存两种格式
        """
        if format == "legacy":
            self._save_legacy(path)
        elif format == "unified":
            self._save_unified(path)
        elif format == "both":
            self._save_legacy(path)
            self._save_unified(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'legacy', 'unified', or 'both'")

    def _save_legacy(self, path: str):
        """保存为旧版格式（多个文件）"""
        os.makedirs(path, exist_ok=True)

        # 保存词表
        with open(os.path.join(path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)

        # 保存合并规则
        with open(os.path.join(path, "merges.txt"), "w", encoding="utf-8") as f:
            for merge in self.merges:
                f.write(f"{merge[0]} {merge[1]}\n")

        # 保存配置
        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    def _save_unified(self, path: str):
        """
        保存为一站式 tokenizer.json 格式
        参考最新 HuggingFace tokenizers 格式
        """
        os.makedirs(path, exist_ok=True)

        # 构建 tokenizer.json 结构
        tokenizer_data = {
            "version": "1.0",
            "model": {
                "type": "BPE",
                "dropout": None,
                "unk_token": self.unk_token,
                "continuing_subword_prefix": None,
                "end_of_word_suffix": None,
                "fuse_unk": False,
                "byte_fallback": False,
                "vocab": self.vocab,
                "merges": [f"{m[0]} {m[1]}" for m in self.merges],
            },
            "normalizer": None,
            "pre_tokenizer": {
                "type": "Regex",
                "pattern": {
                    "Regex": self.pattern.pattern
                }
            },
            "post_processor": {
                "type": "TemplateProcessing",
                "single": [
                    {"SpecialToken": {"id": self.bos_token, "type_id": 0}},
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": self.eos_token, "type_id": 0}}
                ],
                "pair": [
                    {"Sequence": {"id": "A", "type_id": 0}},
                    {"SpecialToken": {"id": self.eos_token, "type_id": 0}},
                    {"Sequence": {"id": "B", "type_id": 1}},
                    {"SpecialToken": {"id": self.eos_token, "type_id": 1}}
                ],
                "special_tokens": {
                    self.bos_token: {"id": self.bos_token_id, "ids": [self.bos_token_id], "tokens": [self.bos_token]},
                    self.eos_token: {"id": self.eos_token_id, "ids": [self.eos_token_id], "tokens": [self.eos_token]}
                }
            },
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": True,
                "use_regex": True
            },
            "model_max_length": 512,
            "added_tokens": [
                {"id": self.unk_token_id, "content": self.unk_token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": self.pad_token_id, "content": self.pad_token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": self.bos_token_id, "content": self.bos_token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
                {"id": self.eos_token_id, "content": self.eos_token, "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            ],
            "special_tokens_map": {
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token
            },
            "special_tokens": {
                self.unk_token: {"id": self.unk_token_id, "ids": [self.unk_token_id], "tokens": [self.unk_token]},
                self.pad_token: {"id": self.pad_token_id, "ids": [self.pad_token_id], "tokens": [self.pad_token]},
                self.bos_token: {"id": self.bos_token_id, "ids": [self.bos_token_id], "tokens": [self.bos_token]},
                self.eos_token: {"id": self.eos_token_id, "ids": [self.eos_token_id], "tokens": [self.eos_token]}
            },
            "config": {
                "unk_token": self.unk_token,
                "pad_token": self.pad_token,
                "bos_token": self.bos_token,
                "eos_token": self.eos_token,
                "vocab_size": self.vocab_size,
                "model_type": "bpe"
            }
        }

        # 保存 tokenizer.json
        with open(os.path.join(path, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, format: str = "auto") -> "BPETokenizer":
        """
        加载tokenizer

        Args:
            path: tokenizer路径
            format: 加载格式
                - "auto": 自动检测（优先使用 tokenizer.json）
                - "legacy": 强制使用旧版格式
                - "unified": 强制使用新版 tokenizer.json
        """
        tokenizer_json_path = os.path.join(path, "tokenizer.json")
        vocab_json_path = os.path.join(path, "vocab.json")

        if format == "auto":
            # 自动检测：优先使用 tokenizer.json
            if os.path.exists(tokenizer_json_path):
                return cls._load_unified(path)
            elif os.path.exists(vocab_json_path):
                return cls._load_legacy(path)
            else:
                raise FileNotFoundError(f"No tokenizer files found in {path}")
        elif format == "unified":
            return cls._load_unified(path)
        elif format == "legacy":
            return cls._load_legacy(path)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'auto', 'legacy', or 'unified'")

    @classmethod
    def _load_legacy(cls, path: str) -> "BPETokenizer":
        """从旧版格式加载（多个文件）"""
        with open(os.path.join(path, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)

        with open(os.path.join(path, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        merges = []
        merges_path = os.path.join(path, "merges.txt")
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        merges.append((parts[0], parts[1]))

        return cls(vocab=vocab, merges=merges, **config)

    @classmethod
    def _load_unified(cls, path: str) -> "BPETokenizer":
        """从一站式 tokenizer.json 加载"""
        tokenizer_json_path = os.path.join(path, "tokenizer.json")

        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 提取词表和合并规则
        model = data.get("model", {})
        vocab = model.get("vocab", {})

        # 解析合并规则
        merges = []
        for merge_str in model.get("merges", []):
            parts = merge_str.split()
            if len(parts) == 2:
                merges.append((parts[0], parts[1]))

        # 提取配置
        config = data.get("config", {})
        special_tokens = data.get("special_tokens_map", {})

        return cls(
            vocab=vocab,
            merges=merges,
            unk_token=config.get("unk_token", special_tokens.get("unk_token", "<unk>")),
            pad_token=config.get("pad_token", special_tokens.get("pad_token", "<pad>")),
            bos_token=config.get("bos_token", special_tokens.get("bos_token", "<s>")),
            eos_token=config.get("eos_token", special_tokens.get("eos_token", "</s>")),
        )


class HFTokenizer:
    """
    HuggingFace tokenizers封装
    使用tokenizers库或transformers库
    """

    def __init__(
        self,
        tokenizer_name_or_path: str,
        use_fast: bool = True,
    ):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.use_fast = use_fast
        self._tokenizer = None

    @property
    def tokenizer(self):
        """延迟加载tokenizer"""
        if self._tokenizer is None:
            try:
                # 尝试使用transformers
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_name_or_path,
                    use_fast=self.use_fast,
                )
            except ImportError:
                # 尝试使用tokenizers
                try:
                    from tokenizers import Tokenizer
                    self._tokenizer = Tokenizer.from_file(self.tokenizer_name_or_path)
                except ImportError:
                    raise ImportError(
                        "Please install transformers or tokenizers: "
                        "pip install transformers or pip install tokenizers"
                    )
        return self._tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)


def get_tokenizer(
    tokenizer_name_or_path: str = "",
    tokenizer_type: str = "bpe",
    vocab_size: int = 32000,
    **kwargs,
) -> Union[BaseTokenizer, HFTokenizer]:
    """
    获取tokenizer的工厂函数
    """
    if tokenizer_name_or_path and os.path.exists(tokenizer_name_or_path):
        # 从本地路径加载
        if tokenizer_type == "bpe":
            return BPETokenizer.load(tokenizer_name_or_path)
        else:
            return HFTokenizer(tokenizer_name_or_path, **kwargs)
    elif tokenizer_name_or_path:
        # 从HuggingFace加载
        return HFTokenizer(tokenizer_name_or_path, **kwargs)
    else:
        # 返回空tokenizer，需要后续训练
        if tokenizer_type == "bpe":
            return BPETokenizer(**kwargs)
        else:
            raise ValueError("Cannot create HF tokenizer without name_or_path")
