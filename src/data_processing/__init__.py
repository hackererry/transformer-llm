"""
数据预处理模块
提供文本清洗和文档格式转换工具
"""
from .clean_text import (
    TextCleaner,
    clean_file,
    split_large_file,
    batch_clean_directory,
)

from .document_converter import (
    convert_to_txt,
    epub_to_txt,
    batch_convert,
    merge_txt_files,
    EPUBExtractor,
    PDFExtractor,
    CSVExtractor,
    JSONExtractor,
)

from .deduplicate import (
    exact_deduplicate,
    near_deduplicate,
    deduplicate_lines,
    deduplicate_lines_from_text,
)

from .pii_remover import (
    remove_pii,
    remove_pii_with_count,
    has_pii,
)

from .quality_filter import (
    compute_quality_score,
    filter_by_quality,
    filter_by_quality_with_stats,
)

from .pipeline import (
    CleaningPipeline,
    build_light_pipeline,
    build_standard_pipeline,
    stream_clean_pipeline,
)

__all__ = [
    # 文本清洗
    "TextCleaner",
    "clean_file",
    "split_large_file",
    "batch_clean_directory",
    # 文档转换
    "convert_to_txt",
    "epub_to_txt",
    "batch_convert",
    "merge_txt_files",
    "EPUBExtractor",
    "PDFExtractor",
    "CSVExtractor",
    "JSONExtractor",
    # 去重
    "exact_deduplicate",
    "near_deduplicate",
    "deduplicate_lines",
    "deduplicate_lines_from_text",
    # PII 移除
    "remove_pii",
    "remove_pii_with_count",
    "has_pii",
    # 质量过滤
    "compute_quality_score",
    "filter_by_quality",
    "filter_by_quality_with_stats",
    # 流水线
    "CleaningPipeline",
    "build_light_pipeline",
    "build_standard_pipeline",
    "stream_clean_pipeline",
]
