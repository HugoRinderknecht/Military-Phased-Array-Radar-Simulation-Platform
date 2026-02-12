# common - 公共模块
"""
公共模块提供系统级别的通用功能，包括：
- 类型定义：系统中使用的所有数据类型
- 物理常数：雷达系统中使用的物理常数
- 数学工具：通用数学运算函数
- 坐标变换：各种坐标系之间的转换
- 信号工具：信号处理相关的工具函数
- 容器类：环形缓冲区、对象池等数据结构
- 配置管理：系统配置加载和管理
- 日志系统：统一的日志记录接口
- 异常定义：系统自定义异常类
"""

from radar.common.types import (
    # 基础类型
    TargetType,
    MotionModel,
    SwerlingModel,
    WorkMode,
    TaskType,
    CoordinateSystem,

    # 数据类
    Vector3D,
    Attitude,
    Position,
    Velocity,

    # 系统状态
    SimulationState,
    SystemState,
)

from radar.common.constants import (
    # 物理常数
    LIGHT_SPEED,
    BOLTZMANN_CONSTANT,
    STANDARD_TEMPERATURE,

    # 雷达常数
    EARTH_RADIUS,
    RADAR_CONSTANT,

    # 数学常数
    PI,
    TWO_PI,
    DEG_TO_RAD,
    RAD_TO_DEG,
)

__all__ = [
    # 类型
    "TargetType",
    "MotionModel",
    "SwerlingModel",
    "WorkMode",
    "TaskType",
    "CoordinateSystem",
    "Vector3D",
    "Attitude",
    "Position",
    "Velocity",
    "SimulationState",
    "SystemState",

    # 常数
    "LIGHT_SPEED",
    "BOLTZMANN_CONSTANT",
    "STANDARD_TEMPERATURE",
    "EARTH_RADIUS",
    "RADAR_CONSTANT",
    "PI",
    "TWO_PI",
    "DEG_TO_RAD",
    "RAD_TO_DEG",
]
