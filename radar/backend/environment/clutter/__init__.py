# clutter - 杂波模型模块
"""
完整的雷达杂波模型实现

包括:
- Rayleigh分布杂波
- Log-Normal分布杂波
- Weibull分布杂波
- K分布杂波
- 相关杂波生成
- 杂波抑制
"""

from .clutter_models import (
    ClutterType,
    ClutterParameters,
    ClutterGenerator,
    CorrelatedClutterGenerator,
    ClutterSuppressor,
    create_clutter_generator,
    generate_sea_clutter,
    generate_land_clutter,
)

__all__ = [
    "ClutterType",
    "ClutterParameters",
    "ClutterGenerator",
    "CorrelatedClutterGenerator",
    "ClutterSuppressor",
    "create_clutter_generator",
    "generate_sea_clutter",
    "generate_land_clutter",
]
