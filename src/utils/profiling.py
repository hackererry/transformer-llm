"""
性能监控模块
针对模型优化项的细粒度性能监控

支持的优化项:
- GQA (Grouped Query Attention): KV缓存大小、减少比例
- Flash Attention: 注意力计算时间、显存占用
- StreamingLLM: 序列长度、缓存利用率
- Speculative Decoding: 接受率、加速比
"""
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from contextlib import contextmanager
import time
import os
import json
from datetime import datetime


@dataclass
class GQAMetrics:
    """
    GQA 性能指标

    用法:
        metrics = GQAMetrics(
            num_heads=8, num_kv_heads=2, seq_len=512,
            batch_size=4, head_dim=64, num_layers=12
        )
        print(f"KV Cache: {metrics.kv_cache_size_mb:.2f} MB")
        print(f"Reduction: {metrics.kv_cache_reduction * 100:.1f}%")
    """
    num_heads: int
    num_kv_heads: int
    seq_len: int
    batch_size: int
    head_dim: int
    num_layers: int
    dtype_size: int = 2  # FP16/BF16 = 2 bytes

    @property
    def kv_cache_size_mb(self) -> float:
        """
        KV缓存大小（MB）

        计算: num_layers * 2 (K+V) * batch * kv_heads * seq_len * head_dim * dtype_size
        """
        bytes_size = (
            self.num_layers * 2 * self.batch_size *
            self.num_kv_heads * self.seq_len * self.head_dim * self.dtype_size
        )
        return bytes_size / (1024 ** 2)

    @property
    def kv_cache_reduction(self) -> float:
        """KV缓存减少比例（相比MHA）"""
        if self.num_heads == 0:
            return 0.0
        return 1.0 - (self.num_kv_heads / self.num_heads)

    @property
    def mha_cache_size_mb(self) -> float:
        """MHA需要的缓存大小（MB）"""
        bytes_size = (
            self.num_layers * 2 * self.batch_size *
            self.num_heads * self.seq_len * self.head_dim * self.dtype_size
        )
        return bytes_size / (1024 ** 2)

    @property
    def savings_mb(self) -> float:
        """节省的显存（MB）"""
        return self.mha_cache_size_mb - self.kv_cache_size_mb


@dataclass
class StreamingMetrics:
    """
    StreamingLLM 性能指标

    用法:
        metrics = StreamingMetrics(
            current_seq_len=10000, max_cache_len=4100,
            sink_size=4, window_size=4096
        )
        print(f"Utilization: {metrics.cache_utilization * 100:.1f}%")
    """
    current_seq_len: int = 0
    max_cache_len: int = 0
    sink_size: int = 4
    window_size: int = 4096

    @property
    def cache_utilization(self) -> float:
        """缓存利用率"""
        if self.max_cache_len == 0:
            return 0.0
        return min(1.0, self.current_seq_len / self.max_cache_len)

    @property
    def is_streaming(self) -> bool:
        """是否处于流式模式（超过最大缓存）"""
        return self.current_seq_len > self.max_cache_len

    @property
    def effective_window(self) -> int:
        """有效窗口大小"""
        return min(self.current_seq_len, self.max_cache_len)


@dataclass
class SpeculativeMetrics:
    """
    Speculative Decoding 性能指标

    用法:
        metrics = SpeculativeMetrics()
        metrics.record_draft(4, 0.01)  # 4 tokens, 10ms
        metrics.record_accept(3)       # 3 accepted
        print(f"Acceptance: {metrics.acceptance_rate * 100:.1f}%")
    """
    total_draft_tokens: int = 0
    accepted_tokens: int = 0
    draft_time: float = 0.0
    verify_time: float = 0.0
    standard_time: float = 0.0  # 基准时间（用于计算加速比）
    num_iterations: int = 0

    @property
    def acceptance_rate(self) -> float:
        """接受率"""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_step(self) -> float:
        """
        每步平均生成 token 数

        包括验证后额外生成的 1 个 token
        """
        if self.num_iterations == 0:
            return 1.0
        return (self.accepted_tokens + self.num_iterations) / self.num_iterations

    @property
    def total_time(self) -> float:
        """总时间"""
        return self.draft_time + self.verify_time

    @property
    def speedup(self) -> float:
        """
        加速比

        需要 standard_time 作为基准
        """
        if self.total_time == 0 or self.standard_time == 0:
            return 1.0
        return self.standard_time / self.total_time

    def record_draft(self, num_tokens: int, elapsed: float):
        """记录 draft 阶段"""
        self.total_draft_tokens += num_tokens
        self.draft_time += elapsed
        self.num_iterations += 1

    def record_accept(self, num_accepted: int):
        """记录接受的 token 数"""
        self.accepted_tokens += num_accepted

    def record_verify(self, elapsed: float):
        """记录 verify 阶段"""
        self.verify_time += elapsed

    def reset(self):
        """重置统计"""
        self.total_draft_tokens = 0
        self.accepted_tokens = 0
        self.draft_time = 0.0
        self.verify_time = 0.0
        self.standard_time = 0.0
        self.num_iterations = 0


@dataclass
class FlashAttentionMetrics:
    """
    Flash Attention 性能指标

    支持两种模式：
    1. 实际测量：通过 record() 记录每次调用的实际性能
    2. 理论估算：通过 estimate_theoretical() 估算理论性能提升
    """
    call_count: int = 0
    total_time: float = 0.0
    peak_memory_mb: float = 0.0
    standard_time: float = 0.0  # 标准注意力基准时间

    # 理论估算字段
    seq_len: int = 0
    batch_size: int = 0
    num_heads: int = 0
    head_dim: int = 0
    num_layers: int = 0
    dtype_size: int = 2  # FP16/BF16
    theoretical_mode: bool = False  # 是否使用理论模式

    @property
    def avg_time_ms(self) -> float:
        """平均时间（毫秒）"""
        if self.call_count == 0:
            return 0.0
        return (self.total_time / self.call_count) * 1000

    @property
    def speedup(self) -> float:
        """加速比"""
        if self.standard_time == 0 or self.total_time == 0:
            return 1.0
        return self.standard_time / self.total_time

    @property
    def standard_memory_mb(self) -> float:
        """标准注意力的显存占用（MB）"""
        if not self.theoretical_mode:
            return 0.0
        # 标准注意力需要存储完整的注意力矩阵: batch * heads * seq * seq
        bytes_size = (
            self.batch_size * self.num_heads *
            self.seq_len * self.seq_len * self.dtype_size
        ) * self.num_layers
        return bytes_size / (1024 ** 2)

    @property
    def flash_memory_mb(self) -> float:
        """Flash Attention 的显存占用（MB）"""
        if not self.theoretical_mode:
            return 0.0
        # Flash Attention 只需要存储 Q, K, V 和输出，不需要完整注意力矩阵
        # 约为标准注意力的 1/(seq_len) + O(1)
        # 简化估算：标准显存 / seq_len * 4 (常数因子)
        bytes_size = (
            self.batch_size * self.num_heads *
            self.seq_len * self.head_dim * self.dtype_size * 4
        ) * self.num_layers
        return bytes_size / (1024 ** 2)

    @property
    def memory_reduction(self) -> float:
        """显存减少比例"""
        if not self.theoretical_mode or self.standard_memory_mb == 0:
            return 0.0
        return 1.0 - (self.flash_memory_mb / self.standard_memory_mb)

    @property
    def estimated_speedup(self) -> float:
        """估算的加速比（基于IO复杂度）"""
        if not self.theoretical_mode:
            return self.speedup
        # Flash Attention 的理论加速比约为 2-4x，取决于序列长度
        # 序列越长，加速越明显
        if self.seq_len <= 512:
            return 2.0
        elif self.seq_len <= 1024:
            return 2.5
        elif self.seq_len <= 2048:
            return 3.0
        else:
            return 4.0

    def record(self, elapsed: float, memory_mb: float = 0.0):
        """记录一次调用（实际测量模式）"""
        self.call_count += 1
        self.total_time += elapsed
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)

    def estimate_theoretical(
        self,
        seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype_size: int = 2,
    ):
        """
        设置理论估算参数

        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            num_heads: 注意力头数
            head_dim: 头维度
            num_layers: 层数
            dtype_size: 数据类型字节数（FP16/BF16=2, FP32=4）
        """
        self.theoretical_mode = True
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype_size = dtype_size


@dataclass
class ModelOptimizationConfig:
    """
    模型优化配置记录
    记录所有启用的优化项及其配置
    """
    use_gqa: bool = False
    use_flash_attention: bool = False
    use_yarn: bool = False
    yarn_factor: float = 1.0
    use_streaming_llm: bool = False
    use_speculative_decoding: bool = False
    gradient_checkpointing: bool = False
    mixed_precision: str = "fp32"  # fp32, fp16, bf16

    def to_dict(self) -> Dict:
        return {
            "use_gqa": self.use_gqa,
            "use_flash_attention": self.use_flash_attention,
            "use_yarn": self.use_yarn,
            "yarn_factor": self.yarn_factor,
            "use_streaming_llm": self.use_streaming_llm,
            "use_speculative_decoding": self.use_speculative_decoding,
            "gradient_checkpointing": self.gradient_checkpointing,
            "mixed_precision": self.mixed_precision,
        }


class OptimizationProfiler:
    """
    优化项性能分析器

    统一管理所有优化项的性能监控

    用法:
        profiler = OptimizationProfiler()

        # 记录优化配置
        profiler.record_optimization_config(
            use_gqa=True,
            use_flash_attention=True,
            use_yarn=True,
            yarn_factor=4.0,
        )

        # 监控 GQA
        profiler.record_gqa_metrics(
            num_heads=8, num_kv_heads=2, seq_len=512,
            batch_size=4, head_dim=64, num_layers=12
        )

        # 监控 Speculative Decoding
        with profiler.measure_speculative_verify():
            output = target_model(...)

        # 打印报告
        profiler.print_report()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.gqa_metrics: Optional[GQAMetrics] = None
        self.streaming_metrics: Optional[StreamingMetrics] = None
        self.speculative_metrics = SpeculativeMetrics()
        self.flash_metrics = FlashAttentionMetrics()
        self.model_opt_config = ModelOptimizationConfig()

    def record_optimization_config(
        self,
        use_gqa: bool = False,
        use_flash_attention: bool = False,
        use_yarn: bool = False,
        yarn_factor: float = 1.0,
        use_streaming_llm: bool = False,
        use_speculative_decoding: bool = False,
        gradient_checkpointing: bool = False,
        mixed_precision: str = "fp32",
    ):
        """记录模型优化配置"""
        if not self.enabled:
            return
        self.model_opt_config = ModelOptimizationConfig(
            use_gqa=use_gqa,
            use_flash_attention=use_flash_attention,
            use_yarn=use_yarn,
            yarn_factor=yarn_factor,
            use_streaming_llm=use_streaming_llm,
            use_speculative_decoding=use_speculative_decoding,
            gradient_checkpointing=gradient_checkpointing,
            mixed_precision=mixed_precision,
        )

    def record_gqa_metrics(
        self,
        num_heads: int,
        num_kv_heads: int,
        seq_len: int,
        batch_size: int,
        head_dim: int,
        num_layers: int,
    ):
        """记录 GQA 指标"""
        if not self.enabled:
            return
        self.gqa_metrics = GQAMetrics(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            seq_len=seq_len,
            batch_size=batch_size,
            head_dim=head_dim,
            num_layers=num_layers,
        )

    def record_streaming_metrics(
        self,
        current_seq_len: int,
        max_cache_len: int,
        sink_size: int = 4,
        window_size: int = 4096,
    ):
        """记录 StreamingLLM 指标"""
        if not self.enabled:
            return
        self.streaming_metrics = StreamingMetrics(
            current_seq_len=current_seq_len,
            max_cache_len=max_cache_len,
            sink_size=sink_size,
            window_size=window_size,
        )

    def record_flash_attention_metrics(
        self,
        seq_len: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        num_layers: int,
        dtype_size: int = 2,
    ):
        """
        记录 Flash Attention 理论性能估算参数

        Args:
            seq_len: 序列长度
            batch_size: 批次大小
            num_heads: 注意力头数
            head_dim: 头维度
            num_layers: 层数
            dtype_size: 数据类型字节数（FP16/BF16=2, FP32=4）
        """
        if not self.enabled:
            return
        self.flash_metrics.estimate_theoretical(
            seq_len=seq_len,
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            num_layers=num_layers,
            dtype_size=dtype_size,
        )

    @contextmanager
    def measure_flash_attention(self):
        """
        测量 Flash Attention 时间

        用法:
            with profiler.measure_flash_attention():
                output = flash_attn_func(q, k, v)
        """
        if not self.enabled:
            yield
            return

        sync_cuda = torch.cuda.is_available()
        if sync_cuda:
            torch.cuda.synchronize()

        start = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            memory_mb = 0.0
            if sync_cuda:
                memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

            self.flash_metrics.record(elapsed, memory_mb)

    @contextmanager
    def measure_speculative_draft(self, num_tokens: int):
        """
        测量 Speculative Decoding draft 阶段

        用法:
            with profiler.measure_speculative_draft(4):
                tokens = draft_model.generate(...)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.speculative_metrics.record_draft(num_tokens, elapsed)

    @contextmanager
    def measure_speculative_verify(self):
        """
        测量 Speculative Decoding verify 阶段

        用法:
            with profiler.measure_speculative_verify():
                output = target_model(...)
        """
        if not self.enabled:
            yield
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.speculative_metrics.record_verify(elapsed)

    def record_speculative_accept(self, num_accepted: int):
        """记录 Speculative Decoding 接受的 token 数"""
        if not self.enabled:
            return
        self.speculative_metrics.record_accept(num_accepted)

    def print_report(self):
        """打印性能报告"""
        print("\n" + "=" * 70)
        print("              Optimization Performance Report")
        print("=" * 70)

        # 优化配置概览
        print("\n[Optimization Configuration]")
        opt = self.model_opt_config
        print(f"  GQA (Grouped Query Attention): {'ENABLED' if opt.use_gqa else 'disabled'}")
        print(f"  Flash Attention: {'ENABLED' if opt.use_flash_attention else 'disabled'}")
        print(f"  YaRN Length Extrapolation: {'ENABLED' if opt.use_yarn else 'disabled'}" + (f" (factor={opt.yarn_factor}x)" if opt.use_yarn else ""))
        print(f"  StreamingLLM: {'ENABLED' if opt.use_streaming_llm else 'disabled'}")
        print(f"  Speculative Decoding: {'ENABLED' if opt.use_speculative_decoding else 'disabled'}")
        print(f"  Gradient Checkpointing: {'ENABLED' if opt.gradient_checkpointing else 'disabled'}")
        print(f"  Mixed Precision: {opt.mixed_precision.upper()}")

        # GQA 报告
        if self.gqa_metrics:
            print("\n[GQA - Grouped Query Attention]")
            print(f"  Query Heads: {self.gqa_metrics.num_heads}")
            print(f"  KV Heads: {self.gqa_metrics.num_kv_heads}")
            print(f"  KV Cache Reduction: {self.gqa_metrics.kv_cache_reduction * 100:.1f}%")
            print(f"  Current KV Cache: {self.gqa_metrics.kv_cache_size_mb:.2f} MB")
            print(f"  MHA Would Need: {self.gqa_metrics.mha_cache_size_mb:.2f} MB")
            print(f"  Memory Saved: {self.gqa_metrics.savings_mb:.2f} MB")

        # StreamingLLM 报告
        if self.streaming_metrics:
            print("\n[StreamingLLM - Infinite Length Inference]")
            print(f"  Current Sequence Length: {self.streaming_metrics.current_seq_len}")
            print(f"  Fixed Cache Size: {self.streaming_metrics.max_cache_len}")
            print(f"  Sink Size: {self.streaming_metrics.sink_size}")
            print(f"  Window Size: {self.streaming_metrics.window_size}")
            print(f"  Cache Utilization: {self.streaming_metrics.cache_utilization * 100:.1f}%")
            if self.streaming_metrics.is_streaming:
                print(f"  Status: STREAMING (beyond fixed cache)")

        # Speculative Decoding 报告
        if self.speculative_metrics.num_iterations > 0:
            print("\n[Speculative Decoding]")
            print(f"  Iterations: {self.speculative_metrics.num_iterations}")
            print(f"  Total Draft Tokens: {self.speculative_metrics.total_draft_tokens}")
            print(f"  Accepted Tokens: {self.speculative_metrics.accepted_tokens}")
            print(f"  Acceptance Rate: {self.speculative_metrics.acceptance_rate * 100:.1f}%")
            print(f"  Avg Tokens/Step: {self.speculative_metrics.tokens_per_step:.2f}")
            print(f"  Draft Time: {self.speculative_metrics.draft_time * 1000:.1f} ms")
            print(f"  Verify Time: {self.speculative_metrics.verify_time * 1000:.1f} ms")
            print(f"  Total Time: {self.speculative_metrics.total_time * 1000:.1f} ms")
            if self.speculative_metrics.standard_time > 0:
                print(f"  Speedup: {self.speculative_metrics.speedup:.2f}x")

        # Flash Attention 报告
        # 显示条件：启用了 Flash Attention（配置中开启）或有实际测量数据
        if self.model_opt_config.use_flash_attention or self.flash_metrics.call_count > 0:
            print("\n[Flash Attention]")
            # 理论估算模式
            if self.flash_metrics.theoretical_mode:
                print(f"  Mode: Theoretical Estimation")
                print(f"  Sequence Length: {self.flash_metrics.seq_len}")
                print(f"  Batch Size: {self.flash_metrics.batch_size}")
                print(f"  Attention Heads: {self.flash_metrics.num_heads}")
                print(f"  Head Dimension: {self.flash_metrics.head_dim}")
                print(f"  Num Layers: {self.flash_metrics.num_layers}")
                print(f"  Standard Attention Memory: {self.flash_metrics.standard_memory_mb:.2f} MB")
                print(f"  Flash Attention Memory: {self.flash_metrics.flash_memory_mb:.2f} MB")
                print(f"  Memory Reduction: {self.flash_metrics.memory_reduction * 100:.1f}%")
                print(f"  Estimated Speedup: {self.flash_metrics.estimated_speedup:.1f}x")
            # 实际测量模式
            elif self.flash_metrics.call_count > 0:
                print(f"  Mode: Measured")
                print(f"  Calls: {self.flash_metrics.call_count}")
                print(f"  Total Time: {self.flash_metrics.total_time * 1000:.2f} ms")
                print(f"  Avg Time: {self.flash_metrics.avg_time_ms:.2f} ms")
                if self.flash_metrics.peak_memory_mb > 0:
                    print(f"  Peak Memory: {self.flash_metrics.peak_memory_mb:.2f} MB")

        print("=" * 70 + "\n")

    def get_summary(self) -> Dict:
        """
        获取性能摘要（用于日志）

        Returns:
            包含各项指标的字典
        """
        summary = {}

        # 添加优化配置
        summary["optimization_config"] = self.model_opt_config.to_dict()

        if self.gqa_metrics:
            summary["gqa"] = {
                "kv_cache_mb": round(self.gqa_metrics.kv_cache_size_mb, 2),
                "reduction_pct": round(self.gqa_metrics.kv_cache_reduction * 100, 1),
                "saved_mb": round(self.gqa_metrics.savings_mb, 2),
            }

        if self.streaming_metrics:
            summary["streaming"] = {
                "seq_len": self.streaming_metrics.current_seq_len,
                "utilization": round(self.streaming_metrics.cache_utilization, 2),
                "is_streaming": self.streaming_metrics.is_streaming,
            }

        if self.speculative_metrics.num_iterations > 0:
            summary["speculative"] = {
                "acceptance_rate": round(self.speculative_metrics.acceptance_rate, 3),
                "tokens_per_step": round(self.speculative_metrics.tokens_per_step, 2),
                "total_time_ms": round(self.speculative_metrics.total_time * 1000, 1),
            }

        if self.flash_metrics.call_count > 0 or self.flash_metrics.theoretical_mode:
            flash_summary = {}
            if self.flash_metrics.theoretical_mode:
                flash_summary = {
                    "mode": "theoretical",
                    "seq_len": self.flash_metrics.seq_len,
                    "memory_reduction_pct": round(self.flash_metrics.memory_reduction * 100, 1),
                    "estimated_speedup": self.flash_metrics.estimated_speedup,
                }
            else:
                flash_summary = {
                    "mode": "measured",
                    "calls": self.flash_metrics.call_count,
                    "avg_time_ms": round(self.flash_metrics.avg_time_ms, 2),
                    "peak_memory_mb": round(self.flash_metrics.peak_memory_mb, 2),
                }
            summary["flash_attention"] = flash_summary

        return summary

    def reset(self):
        """重置所有统计"""
        self.gqa_metrics = None
        self.streaming_metrics = None
        self.speculative_metrics = SpeculativeMetrics()
        self.flash_metrics = FlashAttentionMetrics()

    def save_report(
        self,
        output_dir: str = "logs/perf",
        prefix: str = "perf",
        training_results: Optional[Dict] = None,
        model_config: Optional[Dict] = None,
    ) -> str:
        """
        保存性能报告到文件（纯文本格式）

        Args:
            output_dir: 输出目录，默认 logs/perf
            prefix: 文件名前缀，默认 perf
            training_results: 训练结果（可选）
            model_config: 模型配置（可选）

        Returns:
            保存的文件路径
        """
        if not self.enabled:
            return ""

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.log"
        filepath = os.path.join(output_dir, filename)

        # 生成报告文本
        lines = []
        lines.append("=" * 70)
        lines.append("              Optimization Performance Report")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # 优化配置概览
        lines.append("[Optimization Configuration]")
        opt = self.model_opt_config
        lines.append(f"  GQA (Grouped Query Attention): {'ENABLED' if opt.use_gqa else 'disabled'}")
        lines.append(f"  Flash Attention: {'ENABLED' if opt.use_flash_attention else 'disabled'}")
        yarn_status = f"ENABLED (factor={opt.yarn_factor}x)" if opt.use_yarn else "disabled"
        lines.append(f"  YaRN Length Extrapolation: {yarn_status}")
        lines.append(f"  StreamingLLM: {'ENABLED' if opt.use_streaming_llm else 'disabled'}")
        lines.append(f"  Speculative Decoding: {'ENABLED' if opt.use_speculative_decoding else 'disabled'}")
        lines.append(f"  Gradient Checkpointing: {'ENABLED' if opt.gradient_checkpointing else 'disabled'}")
        lines.append(f"  Mixed Precision: {opt.mixed_precision.upper()}")

        # GQA 报告
        if self.gqa_metrics:
            lines.append("")
            lines.append("[GQA - Grouped Query Attention]")
            lines.append(f"  Query Heads: {self.gqa_metrics.num_heads}")
            lines.append(f"  KV Heads: {self.gqa_metrics.num_kv_heads}")
            lines.append(f"  KV Cache Reduction: {self.gqa_metrics.kv_cache_reduction * 100:.1f}%")
            lines.append(f"  Current KV Cache: {self.gqa_metrics.kv_cache_size_mb:.2f} MB")
            lines.append(f"  MHA Would Need: {self.gqa_metrics.mha_cache_size_mb:.2f} MB")
            lines.append(f"  Memory Saved: {self.gqa_metrics.savings_mb:.2f} MB")

        # StreamingLLM 报告
        if self.streaming_metrics:
            lines.append("")
            lines.append("[StreamingLLM - Infinite Length Inference]")
            lines.append(f"  Current Sequence Length: {self.streaming_metrics.current_seq_len}")
            lines.append(f"  Fixed Cache Size: {self.streaming_metrics.max_cache_len}")
            lines.append(f"  Sink Size: {self.streaming_metrics.sink_size}")
            lines.append(f"  Window Size: {self.streaming_metrics.window_size}")
            lines.append(f"  Cache Utilization: {self.streaming_metrics.cache_utilization * 100:.1f}%")
            if self.streaming_metrics.is_streaming:
                lines.append(f"  Status: STREAMING (beyond fixed cache)")

        # Speculative Decoding 报告
        if self.speculative_metrics.num_iterations > 0:
            lines.append("")
            lines.append("[Speculative Decoding]")
            lines.append(f"  Iterations: {self.speculative_metrics.num_iterations}")
            lines.append(f"  Total Draft Tokens: {self.speculative_metrics.total_draft_tokens}")
            lines.append(f"  Accepted Tokens: {self.speculative_metrics.accepted_tokens}")
            lines.append(f"  Acceptance Rate: {self.speculative_metrics.acceptance_rate * 100:.1f}%")
            lines.append(f"  Avg Tokens/Step: {self.speculative_metrics.tokens_per_step:.2f}")
            lines.append(f"  Draft Time: {self.speculative_metrics.draft_time * 1000:.1f} ms")
            lines.append(f"  Verify Time: {self.speculative_metrics.verify_time * 1000:.1f} ms")
            lines.append(f"  Total Time: {self.speculative_metrics.total_time * 1000:.1f} ms")
            if self.speculative_metrics.standard_time > 0:
                lines.append(f"  Speedup: {self.speculative_metrics.speedup:.2f}x")

        # Flash Attention 报告
        # 显示条件：启用了 Flash Attention（配置中开启）或有实际测量数据
        if self.model_opt_config.use_flash_attention or self.flash_metrics.call_count > 0:
            lines.append("")
            lines.append("[Flash Attention]")
            # 理论估算模式
            if self.flash_metrics.theoretical_mode:
                lines.append(f"  Mode: Theoretical Estimation")
                lines.append(f"  Sequence Length: {self.flash_metrics.seq_len}")
                lines.append(f"  Batch Size: {self.flash_metrics.batch_size}")
                lines.append(f"  Attention Heads: {self.flash_metrics.num_heads}")
                lines.append(f"  Head Dimension: {self.flash_metrics.head_dim}")
                lines.append(f"  Num Layers: {self.flash_metrics.num_layers}")
                lines.append(f"  Standard Attention Memory: {self.flash_metrics.standard_memory_mb:.2f} MB")
                lines.append(f"  Flash Attention Memory: {self.flash_metrics.flash_memory_mb:.2f} MB")
                lines.append(f"  Memory Reduction: {self.flash_metrics.memory_reduction * 100:.1f}%")
                lines.append(f"  Estimated Speedup: {self.flash_metrics.estimated_speedup:.1f}x")
            # 实际测量模式
            elif self.flash_metrics.call_count > 0:
                lines.append(f"  Mode: Measured")
                lines.append(f"  Calls: {self.flash_metrics.call_count}")
                lines.append(f"  Total Time: {self.flash_metrics.total_time * 1000:.2f} ms")
                lines.append(f"  Avg Time: {self.flash_metrics.avg_time_ms:.2f} ms")
                if self.flash_metrics.peak_memory_mb > 0:
                    lines.append(f"  Peak Memory: {self.flash_metrics.peak_memory_mb:.2f} MB")

        # 训练结果
        if training_results:
            lines.append("")
            lines.append("[Training Results]")
            for key, value in training_results.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        # 模型配置
        if model_config:
            lines.append("")
            lines.append("[Model Configuration]")
            lines.append(f"  vocab_size: {model_config.get('vocab_size', 'N/A')}")
            lines.append(f"  hidden_size: {model_config.get('hidden_size', 'N/A')}")
            lines.append(f"  num_hidden_layers: {model_config.get('num_hidden_layers', 'N/A')}")
            lines.append(f"  num_attention_heads: {model_config.get('num_attention_heads', 'N/A')}")
            lines.append(f"  num_key_value_heads: {model_config.get('num_key_value_heads', 'N/A')}")
            lines.append(f"  max_position_embeddings: {model_config.get('max_position_embeddings', 'N/A')}")

        lines.append("")
        lines.append("=" * 70)

        # 写入文件
        report_text = "\n".join(lines)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"Performance report saved to: {filepath}")
        return filepath


def format_memory_size(bytes_size: int) -> str:
    """格式化内存大小"""
    if bytes_size < 1024:
        return f"{bytes_size} B"
    elif bytes_size < 1024 ** 2:
        return f"{bytes_size / 1024:.2f} KB"
    elif bytes_size < 1024 ** 3:
        return f"{bytes_size / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_size / (1024 ** 3):.2f} GB"


def get_memory_info() -> Dict[str, float]:
    """
    获取当前内存信息

    Returns:
        包含 CPU 和 GPU 内存使用情况的字典
    """
    import psutil

    info = {
        "cpu_memory_gb": psutil.virtual_memory().used / (1024 ** 3),
        "cpu_memory_percent": psutil.virtual_memory().percent,
    }

    if torch.cuda.is_available():
        info["gpu_memory_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
        info["gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
        info["gpu_memory_percent"] = (
            torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
        )

    return info
