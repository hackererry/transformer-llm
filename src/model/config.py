"""
模型配置模块
定义Transformer模型的各种配置参数
"""
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ModelConfig:
    """Transformer模型配置类"""

    # 基础配置
    vocab_size: int = 32000  # 词表大小
    hidden_size: int = 512  # 隐藏层维度
    num_hidden_layers: int = 12  # Transformer层数
    num_attention_heads: int = 8  # 注意力头数
    intermediate_size: int = 1024  # FFN中间层维度

    # GQA配置（Grouped Query Attention）
    num_key_value_heads: Optional[int] = None  # KV头数，None表示与num_attention_heads相同
    use_flash_attention: bool = True  # 是否使用Flash Attention

    # RoPE位置编码配置
    max_position_embeddings: int = 2048  # 最大序列长度
    rope_theta: float = 10000.0  # RoPE基数
    rope_scaling: dict = field(default_factory=lambda: {"type": "yarn", "factor": 4.0})  # YaRN默认启用，4倍外推

    # 归一化配置
    rms_norm_eps: float = 1e-6  # RMSNorm epsilon

    # Dropout配置
    hidden_dropout: float = 0.1  # 隐藏层dropout
    attention_dropout: float = 0.1  # 注意力dropout

    # 其他配置
    use_cache: bool = True  # 是否使用KV缓存(推理时)
    tie_word_embeddings: bool = False  # 是否共享输入输出嵌入

    # 梯度检查点
    gradient_checkpointing: bool = False  # 是否启用梯度检查点

    # StreamingLLM 配置（无限长度推理）
    use_streaming_llm: bool = False       # 是否启用 StreamingLLM
    sink_size: int = 4                    # Attention Sink 数量（锚点）
    streaming_window_size: int = 4096     # 滑动窗口大小

    def __post_init__(self):
        """配置验证和自动计算"""
        # 确保hidden_size能被num_attention_heads整除
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"

        # 计算每个注意力头的维度
        self.head_dim = self.hidden_size // self.num_attention_heads

        # 如果intermediate_size未指定，默认为4倍hidden_size
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        # 处理GQA配置
        if self.num_key_value_heads is None:
            # 默认使用 GQA，KV 头数为 Q 头数的 1/4（减少 75% KV 缓存）
            self.num_key_value_heads = max(1, self.num_attention_heads // 4)

        # 验证num_attention_heads能被num_key_value_heads整除
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

    @property
    def use_gqa(self) -> bool:
        """是否使用GQA"""
        return self.num_key_value_heads != self.num_attention_heads

    @classmethod
    def tiny(cls) -> "ModelConfig":
        """Tiny配置: ~10M参数"""
        return cls(
            vocab_size=32000,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            intermediate_size=512,
            max_position_embeddings=1024,
        )

    @classmethod
    def small(cls) -> "ModelConfig":
        """Small配置: ~100M参数"""
        return cls(
            vocab_size=32000,
            hidden_size=512,
            num_hidden_layers=12,
            num_attention_heads=8,
            intermediate_size=1024,
            max_position_embeddings=2048,
        )

    @classmethod
    def medium(cls) -> "ModelConfig":
        """Medium配置: ~500M参数"""
        return cls(
            vocab_size=32000,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=2048,
            max_position_embeddings=4096,
        )

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "use_flash_attention": self.use_flash_attention,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "rms_norm_eps": self.rms_norm_eps,
            "hidden_dropout": self.hidden_dropout,
            "attention_dropout": self.attention_dropout,
            "use_cache": self.use_cache,
            "tie_word_embeddings": self.tie_word_embeddings,
            "gradient_checkpointing": self.gradient_checkpointing,
            "head_dim": self.head_dim,
            "use_streaming_llm": self.use_streaming_llm,
            "sink_size": self.sink_size,
            "streaming_window_size": self.streaming_window_size,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ModelConfig":
        """从字典创建配置"""
        # 移除计算得到的字段
        config_dict = {k: v for k, v in config_dict.items() if k not in ["head_dim", "use_gqa"]}
        return cls(**config_dict)

    def get_rope_scaling_factor(self) -> float:
        """获取 RoPE 缩放因子"""
        if self.rope_scaling is not None and self.rope_scaling.get("type") == "yarn":
            return self.rope_scaling.get("factor", 1.0)
        return 1.0


@dataclass
class TrainingConfig:
    """训练配置类"""

    # 基础训练配置
    output_dir: str = "./output"  # 输出目录
    num_train_epochs: int = 3  # 训练轮数
    per_device_train_batch_size: int = 4  # 每设备批次大小
    gradient_accumulation_steps: int = 1  # 梯度累积步数

    # 学习率配置
    learning_rate: float = 5e-5  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    max_grad_norm: float = 1.0  # 梯度裁剪

    # 学习率调度
    lr_scheduler_type: str = "cosine"  # 调度器类型: cosine, linear, constant
    warmup_ratio: float = 0.1  # 预热比例
    warmup_steps: int = 0  # 预热步数(优先于warmup_ratio)

    # 日志配置
    logging_dir: str = "./logs"  # 日志目录
    logging_steps: int = 10  # 日志记录步数间隔
    save_steps: int = 500  # 保存步数间隔
    save_total_limit: int = 3  # 保存检查点数量限制

    # 精度配置
    fp16: bool = False  # 是否使用FP16(不支持CPU)
    bf16: bool = True  # 是否使用BF16(CPU支持)

    # 数据配置
    dataloader_num_workers: int = 0  # 数据加载器工作进程数
    dataloader_drop_last: bool = False  # 是否丢弃最后不完整批次

    # 其他配置
    seed: int = 42  # 随机种子
    resume_from_checkpoint: Optional[str] = None  # 恢复训练的检查点路径

    def __post_init__(self):
        """配置验证"""
        # CPU不支持FP16
        if self.fp16:
            print("Warning: FP16 is not supported on CPU, switching to BF16")
            self.fp16 = False
            self.bf16 = True

    @property
    def effective_batch_size(self) -> int:
        """有效批次大小(考虑梯度累积)"""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


@dataclass
class DataConfig:
    """数据配置类"""

    # 数据路径
    train_file: str = ""  # 训练数据路径
    validation_file: str = ""  # 验证数据路径

    # Tokenizer配置
    tokenizer_name_or_path: str = ""  # Tokenizer路径
    tokenizer_type: str = "bpe"  # Tokenizer类型: bpe, sentencepiece

    # 数据处理配置
    max_seq_length: int = 512  # 最大序列长度
    truncation: bool = True  # 是否截断
    padding: str = "max_length"  # padding策略

    # 预处理配置
    preprocessing_num_workers: int = 4  # 预处理工作进程数
    overwrite_cache: bool = False  # 是否覆盖缓存

    # 微调数据格式
    instruction_column: str = "instruction"  # 指令列名
    input_column: str = "input"  # 输入列名
    output_column: str = "output"  # 输出列名
