"""
推理模块
包含各种推理优化技术
"""
from .speculative import (
    SpeculativeConfig,
    SpeculativeDecoder,
    speculative_generate,
)

__all__ = [
    "SpeculativeConfig",
    "SpeculativeDecoder",
    "speculative_generate",
]
