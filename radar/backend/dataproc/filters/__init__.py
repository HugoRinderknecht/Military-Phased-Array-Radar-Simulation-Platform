# filters - 滤波器模块
"""
完整的滤波器算法实现

包括:
- 标准卡尔曼滤波器 (KF)
- 扩展卡尔曼滤波器 (EKF)
- 无迹卡尔曼滤波器 (UKF)
- 交互多模型滤波器 (IMM)
"""

from .kalman_filter import (
    KalmanFilter,
    KalmanState,
    ProcessModel,
    MeasurementModel,
    ConstantVelocityModel,
    ConstantAccelerationModel,
    CoordinatedTurnModel,
    PositionMeasurement,
    PositionVelocityMeasurement,
    AdaptiveKalmanFilter,
    create_cv_filter,
    create_ca_filter,
)

from .extended_kalman_filter import (
    ExtendedKalmanFilter,
    spherical_to_cartesian_ekf,
)

from .unscented_kalman_filter import (
    UnscentedKalmanFilter,
    UnscentedTransform,
    create_cv_ukf,
)

from .imm_filter import (
    InteractingMultipleModel,
    ModelFilter,
    create_imm_cv_ca,
    create_imm_cv_ct,
)

__all__ = [
    # 基础类
    "KalmanState",
    "ModelFilter",

    # KF
    "KalmanFilter",
    "ProcessModel",
    "MeasurementModel",
    "ConstantVelocityModel",
    "ConstantAccelerationModel",
    "CoordinatedTurnModel",
    "PositionMeasurement",
    "PositionVelocityMeasurement",
    "AdaptiveKalmanFilter",
    "create_cv_filter",
    "create_ca_filter",

    # EKF
    "ExtendedKalmanFilter",
    "spherical_to_cartesian_ekf",

    # UKF
    "UnscentedKalmanFilter",
    "UnscentedTransform",
    "create_cv_ukf",

    # IMM
    "InteractingMultipleModel",
    "create_imm_cv_ca",
    "create_imm_cv_ct",
]
