"""
性能监控模块测试
"""
import pytest
import torch
from src.utils.profiling import (
    OptimizationProfiler,
    GQAMetrics,
    StreamingMetrics,
    SpeculativeMetrics,
    FlashAttentionMetrics,
    format_memory_size,
)


class TestGQAMetrics:
    """GQA 指标测试"""

    def test_kv_cache_size_calculation(self):
        """测试 KV 缓存大小计算"""
        metrics = GQAMetrics(
            num_heads=8,
            num_kv_heads=2,
            seq_len=512,
            batch_size=4,
            head_dim=64,
            num_layers=12,
            dtype_size=2,  # FP16
        )

        # 计算: 12 * 2 * 4 * 2 * 512 * 64 * 2 = 12,582,912 bytes = 12.0 MB
        expected_bytes = 12 * 2 * 4 * 2 * 512 * 64 * 2
        expected_mb = expected_bytes / (1024 ** 2)

        assert abs(metrics.kv_cache_size_mb - expected_mb) < 0.01

    def test_kv_cache_reduction(self):
        """测试 KV 缓存减少比例"""
        # 8 heads -> 2 kv heads: 减少 75%
        metrics = GQAMetrics(
            num_heads=8,
            num_kv_heads=2,
            seq_len=512,
            batch_size=4,
            head_dim=64,
            num_layers=12,
        )
        assert metrics.kv_cache_reduction == 0.75

        # 8 heads -> 8 kv heads (MHA): 减少 0%
        metrics_mha = GQAMetrics(
            num_heads=8,
            num_kv_heads=8,
            seq_len=512,
            batch_size=4,
            head_dim=64,
            num_layers=12,
        )
        assert metrics_mha.kv_cache_reduction == 0.0

    def test_mha_comparison(self):
        """测试与 MHA 的对比"""
        metrics = GQAMetrics(
            num_heads=8,
            num_kv_heads=2,
            seq_len=512,
            batch_size=4,
            head_dim=64,
            num_layers=12,
        )

        # MHA 需要 4 倍的缓存
        assert abs(metrics.mha_cache_size_mb - metrics.kv_cache_size_mb * 4) < 0.01
        assert abs(metrics.savings_mb - metrics.kv_cache_size_mb * 3) < 0.01


class TestStreamingMetrics:
    """StreamingLLM 指标测试"""

    def test_cache_utilization_below_max(self):
        """测试缓存利用率（未满）"""
        metrics = StreamingMetrics(
            current_seq_len=1000,
            max_cache_len=4100,
            sink_size=4,
            window_size=4096,
        )
        assert metrics.cache_utilization == 1000 / 4100
        assert not metrics.is_streaming

    def test_cache_utilization_at_max(self):
        """测试缓存利用率（刚好满）"""
        metrics = StreamingMetrics(
            current_seq_len=4100,
            max_cache_len=4100,
            sink_size=4,
            window_size=4096,
        )
        assert metrics.cache_utilization == 1.0
        assert not metrics.is_streaming

    def test_cache_utilization_streaming(self):
        """测试缓存利用率（流式模式）"""
        metrics = StreamingMetrics(
            current_seq_len=10000,
            max_cache_len=4100,
            sink_size=4,
            window_size=4096,
        )
        assert metrics.cache_utilization == 1.0  # 超过最大值后固定为 1.0
        assert metrics.is_streaming
        assert metrics.effective_window == 4100


class TestSpeculativeMetrics:
    """Speculative Decoding 指标测试"""

    def test_acceptance_rate(self):
        """测试接受率计算"""
        metrics = SpeculativeMetrics()
        metrics.record_draft(4, 0.01)
        metrics.record_accept(3)

        assert metrics.acceptance_rate == 0.75

    def test_tokens_per_step(self):
        """测试每步生成 token 数"""
        metrics = SpeculativeMetrics()

        # 第一次: 4 个候选，3 个接受，+1 个额外 = 4 个
        metrics.record_draft(4, 0.01)
        metrics.record_accept(3)

        # 第二次: 4 个候选，2 个接受，+1 个额外 = 3 个
        metrics.record_draft(4, 0.01)
        metrics.record_accept(2)

        # 平均: (4 + 3) / 2 = 3.5
        assert metrics.tokens_per_step == 3.5

    def test_total_time(self):
        """测试总时间计算"""
        metrics = SpeculativeMetrics()
        metrics.record_draft(4, 0.01)
        metrics.record_verify(0.05)

        assert metrics.total_time == pytest.approx(0.06, rel=0.01)

    def test_speedup(self):
        """测试加速比计算"""
        metrics = SpeculativeMetrics()
        metrics.standard_time = 0.12  # 标准生成需要 120ms
        metrics.record_draft(4, 0.01)
        metrics.record_verify(0.05)

        # 加速比 = 0.12 / 0.06 = 2.0
        assert metrics.speedup == pytest.approx(2.0, rel=0.01)

    def test_reset(self):
        """测试重置"""
        metrics = SpeculativeMetrics()
        metrics.record_draft(4, 0.01)
        metrics.record_accept(3)

        metrics.reset()

        assert metrics.total_draft_tokens == 0
        assert metrics.accepted_tokens == 0
        assert metrics.num_iterations == 0


class TestOptimizationProfiler:
    """优化项分析器测试"""

    def test_gqa_metrics_recording(self):
        """测试 GQA 指标记录"""
        profiler = OptimizationProfiler()
        profiler.record_gqa_metrics(
            num_heads=8,
            num_kv_heads=2,
            seq_len=512,
            batch_size=4,
            head_dim=64,
            num_layers=12,
        )

        assert profiler.gqa_metrics is not None
        assert profiler.gqa_metrics.num_heads == 8
        assert profiler.gqa_metrics.kv_cache_reduction == 0.75

    def test_streaming_metrics_recording(self):
        """测试 StreamingLLM 指标记录"""
        profiler = OptimizationProfiler()
        profiler.record_streaming_metrics(
            current_seq_len=10000,
            max_cache_len=4100,
            sink_size=4,
            window_size=4096,
        )

        assert profiler.streaming_metrics is not None
        assert profiler.streaming_metrics.is_streaming

    def test_speculative_metrics_recording(self):
        """测试 Speculative Decoding 指标记录"""
        profiler = OptimizationProfiler()

        profiler.record_speculative_accept(3)
        profiler.speculative_metrics.record_draft(4, 0.01)
        profiler.speculative_metrics.record_verify(0.05)

        assert profiler.speculative_metrics.acceptance_rate == 0.75
        assert profiler.speculative_metrics.total_time == pytest.approx(0.06, rel=0.01)

    def test_disabled_profiler(self):
        """测试禁用的分析器"""
        profiler = OptimizationProfiler(enabled=False)

        profiler.record_gqa_metrics(8, 2, 512, 4, 64, 12)
        profiler.record_streaming_metrics(10000, 4100, 4, 4096)

        assert profiler.gqa_metrics is None
        assert profiler.streaming_metrics is None

    def test_get_summary(self):
        """测试获取摘要"""
        profiler = OptimizationProfiler()
        profiler.record_gqa_metrics(8, 2, 512, 4, 64, 12)
        profiler.record_streaming_metrics(10000, 4100, 4, 4096)
        profiler.speculative_metrics.record_draft(4, 0.01)
        profiler.speculative_metrics.record_verify(0.05)
        profiler.record_speculative_accept(3)

        summary = profiler.get_summary()

        assert "gqa" in summary
        assert "streaming" in summary
        assert "speculative" in summary
        assert summary["gqa"]["reduction_pct"] == 75.0

    def test_reset(self):
        """测试重置"""
        profiler = OptimizationProfiler()
        profiler.record_gqa_metrics(8, 2, 512, 4, 64, 12)

        profiler.reset()

        assert profiler.gqa_metrics is None
        assert profiler.streaming_metrics is None
        assert profiler.speculative_metrics.num_iterations == 0


class TestFormatMemorySize:
    """内存大小格式化测试"""

    def test_bytes(self):
        assert format_memory_size(100) == "100 B"

    def test_kilobytes(self):
        assert format_memory_size(1024) == "1.00 KB"
        assert format_memory_size(2048) == "2.00 KB"

    def test_megabytes(self):
        assert format_memory_size(1024 ** 2) == "1.00 MB"
        assert format_memory_size(1024 ** 2 * 12.5) == "12.50 MB"

    def test_gigabytes(self):
        assert format_memory_size(1024 ** 3) == "1.00 GB"


class TestFlashAttentionMetrics:
    """Flash Attention 指标测试"""

    def test_avg_time_calculation(self):
        """测试平均时间计算"""
        metrics = FlashAttentionMetrics()
        metrics.record(0.01, 100)
        metrics.record(0.02, 150)
        metrics.record(0.015, 120)

        assert metrics.call_count == 3
        assert metrics.avg_time_ms == pytest.approx(15.0, rel=0.01)  # (10+20+15)/3 = 15ms
        assert metrics.peak_memory_mb == 150  # 最大值

    def test_speedup_calculation(self):
        """测试加速比计算"""
        metrics = FlashAttentionMetrics()
        metrics.standard_time = 0.1  # 标准注意力需要 100ms
        metrics.record(0.025, 100)  # Flash Attention 只需要 25ms

        assert metrics.speedup == pytest.approx(4.0, rel=0.01)

    def test_theoretical_mode_memory_calculation(self):
        """测试理论模式的显存计算"""
        metrics = FlashAttentionMetrics()
        metrics.estimate_theoretical(
            seq_len=512,
            batch_size=4,
            num_heads=8,
            head_dim=64,
            num_layers=12,
            dtype_size=2,  # FP16
        )

        assert metrics.theoretical_mode
        assert metrics.seq_len == 512
        # 标准注意力: 4 * 8 * 512 * 512 * 2 * 12 bytes
        expected_standard_mb = (4 * 8 * 512 * 512 * 2 * 12) / (1024 ** 2)
        assert abs(metrics.standard_memory_mb - expected_standard_mb) < 0.1

    def test_theoretical_mode_memory_reduction(self):
        """测试理论模式的显存减少比例"""
        metrics = FlashAttentionMetrics()
        metrics.estimate_theoretical(
            seq_len=512,
            batch_size=4,
            num_heads=8,
            head_dim=64,
            num_layers=12,
        )

        # Flash Attention 应该显著减少显存（约 50% 左右）
        # 因为标准注意力需要 N*N 的注意力矩阵，Flash Attention 只需要 O(N) 的存储
        assert metrics.memory_reduction > 0.3  # 至少减少 30%
        assert metrics.memory_reduction < 0.7  # 最多减少 70%

    def test_theoretical_mode_speedup_estimation(self):
        """测试理论模式的加速比估算"""
        metrics_short = FlashAttentionMetrics()
        metrics_short.estimate_theoretical(
            seq_len=256,
            batch_size=4,
            num_heads=8,
            head_dim=64,
            num_layers=12,
        )

        metrics_long = FlashAttentionMetrics()
        metrics_long.estimate_theoretical(
            seq_len=2048,
            batch_size=4,
            num_heads=8,
            head_dim=64,
            num_layers=12,
        )

        # 序列越长，加速比越大
        assert metrics_long.estimated_speedup > metrics_short.estimated_speedup

    def test_profiler_flash_attention_theoretical(self):
        """测试 profiler 记录 Flash Attention 理论估算"""
        profiler = OptimizationProfiler()
        profiler.record_optimization_config(use_flash_attention=True)
        profiler.record_flash_attention_metrics(
            seq_len=512,
            batch_size=4,
            num_heads=8,
            head_dim=64,
            num_layers=12,
        )

        assert profiler.flash_metrics.theoretical_mode
        # Flash Attention 显存减少约 50%
        assert profiler.flash_metrics.memory_reduction > 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
