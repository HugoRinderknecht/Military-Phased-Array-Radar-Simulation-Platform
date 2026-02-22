# rcs - RCS起伏模型模块
"""
完整的目标RCS起伏模型实现

包括:
- Swerling I: 慢起伏，瑞利分布
- Swerling II: 快起伏，瑞利分布
- Swerling III: 慢起伏，4自由度chi-square分布
- Swerling IV: 快起伏，4自由度chi-square分布
- 复杂目标RCS模型
"""

from .swerling_models import (
    SwerlingModel,
    SwerlingGenerator,
    RCSSample,
    RCSGenerator,
    ComplexTargetRCS,
    RCSProbability,
    create_swerling_model,
    generate_rcs_samples,
)

__all__ = [
    "SwerlingModel",
    "SwerlingGenerator",
    "RCSSample",
    "RCSGenerator",
    "ComplexTargetRCS",
    "RCSProbability",
    "create_swerling_model",
    "generate_rcs_samples",
]
