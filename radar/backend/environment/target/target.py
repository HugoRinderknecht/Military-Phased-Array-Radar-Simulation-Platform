# target.py - 目标模型
"""
本模块实现雷达目标的各种运动模型。

支持的运动模型：
- CV (Constant Velocity): 匀速直线运动
- CA (Constant Acceleration): 匀加速直线运动
- CT (Coordinated Turn): 协调转弯运动
- 6DOF: 六自由度运动
- Waypoint: 航点跟随
"""

import numpy as np
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from radar.common.logger import get_logger
from radar.common.types import (
    Position3D, Velocity3D, TargetType, MotionModel
)
from radar.common.constants import MathConstants


class TargetState(Enum):
    """目标状态"""
    ACTIVE = "active"           # 活动中
    INACTIVE = "inactive"       # 非活动
    LOST = "lost"             # 丢失
    DESTROYED = "destroyed"     # 毁灭


@dataclass
class TargetStateEstimate:
    """
    目标状态估计

    用于滤波器的状态向量。

    Attributes:
        position: 位置 [x, y, z] [米]
        velocity: 速度 [vx, vy, vz] [米/秒]
        acceleration: 加速度 [ax, ay, az] [米/秒²]
        timestamp: 时间戳 [秒]
    """
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0


class MotionModelBase:
    """
    运动模型基类

    定义运动模型的通用接口。
    """

    def predict(self, state: TargetStateEstimate,
              dt: float) -> TargetStateEstimate:
        """
        预测下一时刻状态

        Args:
            state: 当前状态
            dt: 时间步长 [秒]

        Returns:
            预测状态
        """
        raise NotImplementedError


class ConstantVelocityModel(MotionModelBase):
    """
    匀速直线运动模型 (CV)

    状态方程：
    x(k+1) = x(k) + vx(k) * dt
    y(k+1) = y(k) + vy(k) * dt
    z(k+1) = z(k) + vz(k) * dt
    """

    def __init__(self):
        self._logger = get_logger("cv_model")

    def predict(self, state: TargetStateEstimate,
              dt: float) -> TargetStateEstimate:
        """
        CV模型预测

        状态转移矩阵:
        F = [1  dt  0]
            [0  1   0]
            [0  0   1]
        """
        # 位置更新
        new_position = state.position + state.velocity * dt

        # 速度保持不变
        new_velocity = state.velocity.copy()

        # 加速度为0
        new_acceleration = np.zeros(3)

        return TargetStateEstimate(
            position=new_position,
            velocity=new_velocity,
            acceleration=new_acceleration,
            timestamp=state.timestamp + dt
        )


class ConstantAccelerationModel(MotionModelBase):
    """
    匀加速直线运动模型 (CA)

    状态方程包含加速度项。
    """

    def __init__(self):
        self._logger = get_logger("ca_model")

    def predict(self, state: TargetStateEstimate,
              dt: float) -> TargetStateEstimate:
        """
        CA模型预测

        状态转移:
        x(k+1) = x(k) + vx(k)*dt + 0.5*ax(k)*dt²
        v(k+1) = v(k) + a(k)*dt
        """
        # 位置更新
        new_position = (state.position +
                     state.velocity * dt +
                     0.5 * state.acceleration * dt**2)

        # 速度更新
        new_velocity = state.velocity + state.acceleration * dt

        # 加速度保持不变
        new_acceleration = state.acceleration.copy()

        return TargetStateEstimate(
            position=new_position,
            velocity=new_velocity,
            acceleration=new_acceleration,
            timestamp=state.timestamp + dt
        )


class CoordinatedTurnModel(MotionModelBase):
    """
    协调转弯模型 (CT)

    目标以恒定速率和转弯率运动。
    """

    def __init__(self, omega: float = 0.0):
        """
        初始化CT模型

        Args:
            omega: 转弯角速度 [弧度/秒]
        """
        self._logger = get_logger("ct_model")
        self.omega = omega  # 转弯率

    def predict(self, state: TargetStateEstimate,
              dt: float) -> TargetStateEstimate:
        """
        CT模型预测

        使用协调转弯运动学方程。
        """
        x, y, z = state.position
        vx, vy, vz = state.velocity

        if abs(self.omega) < 1e-6:
            # 接近直线运动
            return ConstantVelocityModel().predict(state, dt)

        # 协调转弯方程
        # x(k+1) = x(k) + (vx*sin(ω*dt) + vy*(cos(ω*dt)-1))/ω
        # y(k+1) = y(k) + (vy*sin(ω*dt) - vx*(cos(ω*dt)-1))/ω

        omega_dt = self.omega * dt
        cos_omega_dt = np.cos(omega_dt)
        sin_omega_dt = np.sin(omega_dt)

        new_x = x + (vx * sin_omega_dt +
                     vy * (cos_omega_dt - 1)) / self.omega
        new_y = y + (vy * sin_omega_dt -
                     vx * (cos_omega_dt - 1)) / self.omega

        # z方向匀速
        new_z = z + vz * dt

        new_position = np.array([new_x, new_y, new_z])

        # 速度更新
        new_vx = vx * cos_omega_dt - vy * sin_omega_dt
        new_vy = vx * sin_omega_dt + vy * cos_omega_dt
        new_vz = vz

        new_velocity = np.array([new_vx, new_vy, new_vz])

        # 加速度（向心加速度）
        new_ax = -self.omega * vy
        new_ay = self.omega * vx
        new_az = 0.0

        new_acceleration = np.array([new_ax, new_ay, new_az])

        return TargetStateEstimate(
            position=new_position,
            velocity=new_velocity,
            acceleration=new_acceleration,
            timestamp=state.timestamp + dt
        )


class SixDOFModel(MotionModelBase):
    """
    六自由度运动模型 (6DOF)

    完整的3D刚体运动模型，考虑姿态和角速度。
    """

    def __init__(self):
        self._logger = get_logger("6dof_model")

    def predict(self, state: TargetStateEstimate,
              dt: float) -> TargetStateEstimate:
        """
        6DOF模型预测

        使用完整的刚体运动方程。
        """
        # 简化实现：使用CA模型
        # 完整6DOF需要考虑姿态矩阵、角速度等
        return ConstantAccelerationModel().predict(state, dt)


class WaypointModel(MotionModelBase):
    """
    航点跟随模型

    目标依次飞向一系列航点。
    """

    def __init__(self, waypoints: List[Position3D],
                 speed: float = 200.0):
        """
        初始化航点模型

        Args:
            waypoints: 航点列表
            speed: 巡航速度 [米/秒]
        """
        self._logger = get_logger("waypoint_model")
        self.waypoints = waypoints
        self.speed = speed
        self.current_waypoint_index = 0

    def predict(self, state: TargetStateEstimate,
              dt: float) -> TargetStateEstimate:
        """
        航点模型预测

        飞向当前目标航点。
        """
        if self.current_waypoint_index >= len(self.waypoints):
            # 所有航点已完成，保持当前速度
            return ConstantVelocityModel().predict(state, dt)

        # 获取目标航点
        target = self.waypoints[self.current_waypoint_index]
        target_pos = np.array([target.x, target.y, target.z])
        current_pos = state.position

        # 计算到航点的方向
        direction = target_pos - current_pos
        distance = np.linalg.norm(direction)

        if distance < 100:  # 100米内到达
            self.current_waypoint_index += 1
            return self.predict(state, dt)

        # 期望速度
        desired_velocity = (direction / distance) * self.speed

        # 使用期望速度
        new_state = TargetStateEstimate(
            position=state.position,
            velocity=desired_velocity,
            acceleration=np.zeros(3),
            timestamp=state.timestamp + dt
        )

        return ConstantVelocityModel().predict(new_state, dt)


class Target:
    """
    目标类

    表示雷达环境中的一个目标。
    """

    def __init__(self,
                 target_id: int,
                 target_type: TargetType,
                 initial_position: Position3D,
                 initial_velocity: Velocity3D,
                 motion_model: MotionModel = MotionModel.CONSTANT_VELOCITY):
        """
        初始化目标

        Args:
            target_id: 目标ID
            target_type: 目标类型
            initial_position: 初始位置
            initial_velocity: 初始速度
            motion_model: 运动模型类型
        """
        self._logger = get_logger(f"target_{target_id}")

        self.id = target_id
        self.type = target_type
        self.state = TargetState.ACTIVE

        # 状态估计
        pos_array = np.array([initial_position.x,
                           initial_position.y,
                           initial_position.z])
        vel_array = np.array([initial_velocity.vx,
                           initial_velocity.vy,
                           initial_velocity.vz])

        self._state_estimate = TargetStateEstimate(
            position=pos_array,
            velocity=vel_array,
            timestamp=0.0
        )

        # 运动模型
        self._motion_model = self._create_motion_model(motion_model)

        # RCS属性
        self._mean_rcs = self._get_default_rcs(target_type)
        self._current_rcs = self._mean_rcs

        self._logger.info(
            f"目标创建: type={target_type.value}, "
            f"pos={initial_position}, vel={initial_velocity}"
        )

    def _create_motion_model(self, model_type: MotionModel) -> MotionModelBase:
        """创建运动模型实例"""
        if model_type == MotionModel.CONSTANT_VELOCITY:
            return ConstantVelocityModel()
        elif model_type == MotionModel.CONSTANT_ACCELERATION:
            return ConstantAccelerationModel()
        elif model_type == MotionModel.COORDINATED_TURN:
            return CoordinatedTurnModel()
        elif model_type == MotionModel.SIX_DOF:
            return SixDOFModel()
        elif model_type == MotionModel.WAYPOINT:
            return WaypointModel([])
        else:
            self._logger.warning(f"未知运动模型，使用CV模型: {model_type}")
            return ConstantVelocityModel()

    def _get_default_rcs(self, target_type: TargetType) -> float:
        """获取目标类型的默认RCS"""
        rcs_map = {
            TargetType.AIRCRAFT: 10.0,
            TargetType.MISSILE: 0.1,
            TargetType.SHIP: 1000.0,
            TargetType.HELICOPTER: 3.0,
            TargetType.UAV: 0.5,
        }
        return rcs_map.get(target_type, 5.0)

    def update(self, dt: float) -> None:
        """
        更新目标状态

        Args:
            dt: 时间步长 [秒]
        """
        # 预测新状态
        new_state = self._motion_model.predict(self._state_estimate, dt)
        self._state_estimate = new_state

        # 更新RCS（起伏）
        self._update_rcs()

    def _update_rcs(self) -> None:
        """更新RCS（Swerling起伏模型）"""
        # Swerling I模型：指数分布
        u = np.random.uniform(0, 1)
        self._current_rcs = -self._mean_rcs * np.log(u)

    @property
    def position(self) -> Position3D:
        """获取当前位置"""
        pos = self._state_estimate.position
        return Position3D(pos[0], pos[1], pos[2])

    @property
    def velocity(self) -> Velocity3D:
        """获取当前速度"""
        vel = self._state_estimate.velocity
        return Velocity3D(vel[0], vel[1], vel[2])

    @property
    def rcs(self) -> float:
        """获取当前RCS"""
        return self._current_rcs

    def get_state_estimate(self) -> TargetStateEstimate:
        """获取状态估计"""
        return self._state_estimate

    def set_state(self, state: TargetState) -> None:
        """设置目标状态"""
        self.state = state
        self._logger.info(f"目标状态: {state.value}")


__all__ = [
    "TargetState",
    "TargetStateEstimate",
    "MotionModelBase",
    "ConstantVelocityModel",
    "ConstantAccelerationModel",
    "CoordinatedTurnModel",
    "SixDOFModel",
    "WaypointModel",
    "Target",
]
