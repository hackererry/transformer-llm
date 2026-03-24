"""
Tokenizer封装模块
支持HuggingFace tokenizers
"""
import os
import json
from typing import Optional, List, Dict, Tuple, Union


def get_tokenizer(
    tokenizer_name_or_path: str = "",
    tokenizer_type: str = "bpe",
    vocab_size: int = 32000,
    **kwargs,
) -> "HuggingFaceBPETokenizer":
    """
    获取tokenizer的工厂函数

    Args:
        tokenizer_name_or_path: tokenizer路径或名称
        tokenizer_type: 类型 (bpe)
        vocab_size: 词表大小
    """
    if tokenizer_name_or_path and os.path.exists(tokenizer_name_or_path):
        # 从本地路径加载
        return HuggingFaceBPETokenizer.load(tokenizer_name_or_path)
    else:
        # 返回空tokenizer，需要后续训练
        return HuggingFaceBPETokenizer()


class HuggingFaceBPETokenizer:
    """
    基于HuggingFace tokenizers库的高效BPE Tokenizer
    使用Rust实现，比纯Python版本快100倍+
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>",
    ):
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self._tokenizer = None

        if vocab is not None and merges is not None:
            # 从已有的vocab和merges创建
            self._from_vocab_merges(vocab, merges)

    @classmethod
    def load(cls, path: str) -> "HuggingFaceBPETokenizer":
        """从路径加载tokenizer"""
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "Please install tokenizers: pip install tokenizers"
            )

        import json
        tokenizer_json_path = os.path.join(path, "tokenizer.json")

        # 先读取json获取merges
        with open(tokenizer_json_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        merges = tok_data.get("model", {}).get("merges", [])

        # 加载tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_json_path)

        instance = cls()
        instance._tokenizer = tokenizer
        instance.vocab = tokenizer.get_vocab()
        instance.merges = merges

        # 获取特殊token ID
        instance.unk_token_id = tokenizer.token_to_id(instance.unk_token) or 0
        instance.pad_token_id = tokenizer.token_to_id(instance.pad_token) or 1
        instance.bos_token_id = tokenizer.token_to_id(instance.bos_token) or 2
        instance.eos_token_id = tokenizer.token_to_id(instance.eos_token) or 3

        return instance

    def _from_vocab_merges(self, vocab, merges):
        """从vocab和merges创建tokenizer"""
        try:
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import Sequence, Regex, Metaspace
        except ImportError:
            raise ImportError("Please install tokenizers: pip install tokenizers")

        # 创建模型
        model = BPE(vocab, merges, unk_token=self.unk_token)

        # 创建tokenizer
        self._tokenizer = Tokenizer(model)
        self.vocab = vocab
        self.merges = merges

    def train(self, texts: List[str], vocab_size: int = 32000, min_frequency: int = 2):
        """
        使用HuggingFace tokenizers训练BPE

        Args:
            texts: 训练文本列表
            vocab_size: 目标词表大小
            min_frequency: 最小频率
        """
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import UnicodeScripts
        except ImportError:
            raise ImportError(
                "HuggingFace tokenizers not found. Install with: pip install tokenizers"
            )

        # 创建tokenizer (BPE模型)
        tokenizer = Tokenizer(BPE())

        # 配置预分词器 - 使用UnicodeScripts，支持中文等多语言
        tokenizer.pre_tokenizer = UnicodeScripts()

        # 创建训练器
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=[
                self.unk_token,
                self.pad_token,
                self.bos_token,
                self.eos_token,
            ],
        )

        # 训练
        tokenizer.train_from_iterator(texts, trainer=trainer)

        self._tokenizer = tokenizer

        # 保存词汇表信息
        self.vocab = dict(tokenizer.get_vocab())
        # merges 存储在 model.merges 中
        self.merges = list(tokenizer.model.merges) if hasattr(tokenizer.model, 'merges') else []

        # 特殊token ID
        self.unk_token_id = tokenizer.token_to_id(self.unk_token) or 0
        self.pad_token_id = tokenizer.token_to_id(self.pad_token) or 1
        self.bos_token_id = tokenizer.token_to_id(self.bos_token) or 2
        self.eos_token_id = tokenizer.token_to_id(self.eos_token) or 3

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) if self.vocab else 0

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本"""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained yet")

        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """解码"""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained yet")

        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str, format: str = "unified"):
        """保存tokenizer"""
        if self._tokenizer is None:
            raise ValueError("Tokenizer not trained yet")

        os.makedirs(path, exist_ok=True)

        # 使用HuggingFace格式保存
        self._tokenizer.save(os.path.join(path, "tokenizer.json"))

        # 同时保存特殊token信息供原生实现使用
        config = {
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "unk_token_id": self.unk_token_id,
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
        }
        with open(os.path.join(path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 保存merges (兼容两种格式: List[Tuple[str,str]] 或 List[str])
        with open(os.path.join(path, "merges.txt"), "w", encoding="utf-8") as f:
            for merge in self.merges:
                if isinstance(merge, str):
                    # 格式: "a b" -> 直接写入
                    f.write(merge + "\n")
                else:
                    # 格式: ("a", "b") -> 转换为字符串
                    f.write(f"{merge[0]} {merge[1]}\n")

    def __call__(
        self,
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None,
    ) -> Dict:
        """便捷调用方法"""
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

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
