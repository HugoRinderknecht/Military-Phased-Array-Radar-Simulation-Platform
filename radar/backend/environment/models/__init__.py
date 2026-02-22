# models - 运动模型模块
"""
完整的运动模型实现

包括:
- 6DOF运动模型
- 协调转弯模型
- 运动学模型
- 姿态表示
"""

from .6dof_model import (
    AttitudeRepresentation,
    AttitudeState,
    State6DOF,
    SixDOFMotionModel,
    CoordinatedTurnModel6DOF,
    Kinematic6DOF,
    create_6dof_model,
)

__all__ = [
    "AttitudeRepresentation",
    "AttitudeState",
    "State6DOF",
    "SixDOFMotionModel",
    "CoordinatedTurnModel6DOF",
    "Kinematic6DOF",
    "create_6dof_model",
]
