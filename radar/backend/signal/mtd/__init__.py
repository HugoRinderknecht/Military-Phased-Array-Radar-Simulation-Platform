# mtd - MTD处理模块
"""
完整的动目标检测(MTD)处理实现

包括:
- 多脉冲多普勒处理
- Range-Doppler矩阵生成
- 多普勒滤波器组
- 杂波抑制
"""

from .mtd_processor import (
    MTDProcessor,
    MTDResult,
    DopplerProcessingType,
    DopplerFilterBank,
    ClutterSuppressor,
    create_mtd_processor,
    process_mtd,
)

__all__ = [
    "MTDProcessor",
    "MTDResult",
    "DopplerProcessingType",
    "DopplerFilterBank",
    "ClutterSuppressor",
    "create_mtd_processor",
    "process_mtd",
]
