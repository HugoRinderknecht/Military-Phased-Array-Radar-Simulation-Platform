# cfar - CFAR检测模块
"""
完整的CFAR检测算法实现

包括:
- CA-CFAR: 单元平均CFAR
- GO-CFAR: 最大选择CFAR
- SO-CFAR: 最小选择CFAR
- OS-CFAR: 有序统计CFAR
- TM-CFAR: 削减平均CFAR
- Adaptive CFAR: 自适应CFAR
"""

from .cfar_algorithms import (
    BaseCFAR,
    CACFAR,
    GOCFAR,
    SOCFAR,
    OSCFAR,
    TMCFAR,
    AdaptiveCFAR,
    CFARType,
    CFARDetection,
    create_cfar,
    detect_with_cfar,
)

__all__ = [
    "BaseCFAR",
    "CACFAR",
    "GOCFAR",
    "SOCFAR",
    "OSCFAR",
    "TMCFAR",
    "AdaptiveCFAR",
    "CFARType",
    "CFARDetection",
    "create_cfar",
    "detect_with_cfar",
]
