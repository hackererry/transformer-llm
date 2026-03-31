# -*- coding: utf-8 -*-
"""
工具模块
导出所有工具函数
"""
from .logging import (
    Logger,
    ProgressTracker,
    TensorBoardLogger,
    setup_logger,
)
from .metrics import (
    compute_perplexity,
    compute_accuracy,
    compute_token_accuracy,
    compute_bleu_score,
    compute_f1_score,
    MetricsTracker,
    PerplexityCalculator,
    compute_generation_metrics,
)
from .device import (
    get_device,
    get_device_info,
    to_device,
    set_seed,
    get_memory_info,
    optimize_for_inference,
    enable_tf32,
    set_num_threads,
    get_optimal_num_threads,
    DeviceManager,
    print_device_info,
)
from .profiling import (
    OptimizationProfiler,
    GQAMetrics,
    StreamingMetrics,
    SpeculativeMetrics,
    FlashAttentionMetrics,
    ModelOptimizationConfig,
    format_memory_size,
)
from .database import (
    DatabaseManager,
    get_default_db_path,
    init_database,
    get_connection,
    backup_database,
)
from .database_schema import (
    init_all_tables,
    drop_all_tables,
    get_all_schemas,
)
from .repository import (
    CrawlerPageRepository,
    CrawlStatsRepository,
    CleanedDocumentRepository,
    CleaningRunRepository,
    get_crawler_page_repo,
    get_crawl_stats_repo,
    get_cleaned_document_repo,
    get_cleaning_run_repo,
)

__all__ = [
    # 日志
    "Logger",
    "ProgressTracker",
    "TensorBoardLogger",
    "setup_logger",
    # 指标
    "compute_perplexity",
    "compute_accuracy",
    "compute_token_accuracy",
    "compute_bleu_score",
    "compute_f1_score",
    "MetricsTracker",
    "PerplexityCalculator",
    "compute_generation_metrics",
    # 设备
    "get_device",
    "get_device_info",
    "to_device",
    "set_seed",
    "get_memory_info",
    "optimize_for_inference",
    "enable_tf32",
    "set_num_threads",
    "get_optimal_num_threads",
    "DeviceManager",
    "print_device_info",
    # 性能分析
    "OptimizationProfiler",
    "GQAMetrics",
    "StreamingMetrics",
    "SpeculativeMetrics",
    "FlashAttentionMetrics",
    "ModelOptimizationConfig",
    "format_memory_size",
    # 数据库管理器
    "DatabaseManager",
    "get_default_db_path",
    "init_database",
    "get_connection",
    "backup_database",
    # 数据库表
    "init_all_tables",
    "drop_all_tables",
    "get_all_schemas",
    # Repository
    "CrawlerPageRepository",
    "CrawlStatsRepository",
    "CleanedDocumentRepository",
    "CleaningRunRepository",
    "get_crawler_page_repo",
    "get_crawl_stats_repo",
    "get_cleaned_document_repo",
    "get_cleaning_run_repo",
]
