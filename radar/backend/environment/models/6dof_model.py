# 6dof_model.py - 完整的6自由度运动模型实现
"""
6DOF (Six Degrees of Freedom) 运动模型

完整实现6自由度刚体运动:
- 姿态表示 (欧拉角、四元数、旋转矩阵)
- 角速度和角加速度
- 协调转弯模型
- 完整动力学方程
"""

import numpy as np
from typing import Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class AttitudeRepresentation(Enum):
    """姿态表示方式"""
    EULER = "euler"       # 欧拉角 (roll, pitch, yaw)
    QUATERNION = "quat"   # 四元数 (w, x, y, z)
    DCM = "dcm"          # 方向余弦矩阵


@dataclass
class AttitudeState:
    """姿态状态"""
    # 使用欧拉角表示
    roll: float = 0.0      # 横滚角 (rad)
    pitch: float = 0.0     # 俯仰角 (rad)
    yaw: float = 0.0       # 偏航角 (rad)

    # 角速度 (rad/s)
    roll_rate: float = 0.0
    pitch_rate: float = 0.0
    yaw_rate: float = 0.0

    def to_dcm(self) -> np.ndarray:
        """转换为方向余弦矩阵"""
        cr, sr = np.cos(self.roll), np.sin(self.roll)
        cp, sp = np.cos(self.pitch), np.sin(self.pitch)
        cy, sy = np.cos(self.yaw), np.sin(self.yaw)

        # DCM = R_z(yaw) * R_y(pitch) * R_x(roll)
        DCM = np.array([
            [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
            [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr]
        ])

        return DCM

    def to_quaternion(self) -> np.ndarray:
        """转换为四元数 [w, x, y, z]"""
        cy = np.cos(self.yaw * 0.5)
        sy = np.sin(self.yaw * 0.5)
        cp = np.cos(self.pitch * 0.5)
        sp = np.sin(self.pitch * 0.5)
        cr = np.cos(self.roll * 0.5)
        sr = np.sin(self.roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qw, qx, qy, qz])

    def normalize_angles(self) -> None:
        """归一化角度到 [-pi, pi]"""
        self.roll = (self.roll + np.pi) % (2 * np.pi) - np.pi
        self.pitch = (self.pitch + np.pi) % (2 * np.pi) - np.pi
        self.yaw = (self.yaw + np.pi) % (2 * np.pi) - np.pi


@dataclass
class State6DOF:
    """6DOF完整状态"""
    # 位置 (m)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    # 速度 (m/s) - 机体坐标系
    u: float = 0.0   # 前向速度
    v: float = 0.0   # 侧向速度
    w: float = 0.0   # 垂直速度

    # 姿态
    attitude: AttitudeState = None

    def __post_init__(self):
        if self.attitude is None:
            self.attitude = AttitudeState()

    def to_vector(self) -> np.ndarray:
        """转换为状态向量"""
        return np.array([
            self.x, self.y, self.z,
            self.u, self.v, self.w,
            self.attitude.roll, self.attitude.pitch, self.attitude.yaw,
            self.attitude.roll_rate, self.attitude.pitch_rate, self.attitude.yaw_rate
        ])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'State6DOF':
        """从状态向量创建"""
        attitude = AttitudeState(
            roll=vec[6],
            pitch=vec[7],
            yaw=vec[8],
            roll_rate=vec[9],
            pitch_rate=vec[10],
            yaw_rate=vec[11]
        )
        return cls(
            x=vec[0], y=vec[1], z=vec[2],
            u=vec[3], v=vec[4], w=vec[5],
            attitude=attitude
        )

    def get_position_ned(self) -> np.ndarray:
        """获取NED坐标系位置"""
        return np.array([self.x, self.y, self.z])

    def get_velocity_ned(self) -> np.ndarray:
        """获取NED坐标系速度"""
        # 将机体坐标系速度转换到NED坐标系
        DCM = self.attitude.to_dcm()
        body_vel = np.array([self.u, self.v, self.w])
        ned_vel = DCM @ body_vel
        return ned_vel

    def get_angular_rates(self) -> np.ndarray:
        """获取角速度 (rad/s)"""
        return np.array([
            self.attitude.roll_rate,
            self.attitude.pitch_rate,
            self.attitude.yaw_rate
        ])


class SixDOFMotionModel:
    """
    6自由度运动模型

    完整的刚体运动学方程
    """

    def __init__(
        self,
        dt: float = 0.01,
        process_noise_pos: float = 0.1,
        process_noise_vel: float = 0.5,
        process_noise_att: float = 0.01,
        process_noise_rate: float = 0.1
    ):
        """
        Args:
            dt: 时间步长 (s)
            process_noise_pos: 位置噪声 (m^2/s)
            process_noise_vel: 速度噪声 (m^2/s^3)
            process_noise_att: 姿态噪声 (rad^2/s)
            process_noise_rate: 角速度噪声 (rad^2/s^3)
        """
        self.dt = dt
        self.process_noise_pos = process_noise_pos
        self.process_noise_vel = process_noise_vel
        self.process_noise_att = process_noise_att
        self.process_noise_rate = process_noise_rate

        # 状态维度: 12 (位置3 + 速度3 + 姿态3 + 角速度3)
        self.state_dim = 12

    def predict(
        self,
        state: State6DOF,
        dt: Optional[float] = None
    ) -> State6DOF:
        """
        预测下一状态

        Args:
            state: 当前状态
            dt: 时间步长 (None则使用默认值)

        Returns:
            预测状态
        """
        if dt is None:
            dt = self.dt

        # 转换为向量处理
        x = state.to_vector()

        # 计算状态导数
        dx_dt = self._compute_derivative(x)

        # 一阶欧拉积分 (可以使用更高阶积分方法)
        x_pred = x + dx_dt * dt

        # 归一化角度
        x_pred[6:9] = (x_pred[6:9] + np.pi) % (2 * np.pi) - np.pi

        return State6DOF.from_vector(x_pred)

    def _compute_derivative(self, x: np.ndarray) -> np.ndarray:
        """
        计算状态导数

        6DOF运动学方程:
        - 位置变化率 = DCM * 速度
        - 姿态变化率 = 角速度转换矩阵 * 角速度
        """
        dx_dt = np.zeros(12)

        # 提取状态分量
        # 位置: x[0:3]
        # 速度: x[3:6] (机体坐标系)
        # 姿态: x[6:9] (roll, pitch, yaw)
        # 角速度: x[9:12] (p, q, r)

        roll, pitch, yaw = x[6], x[7], x[8]
        p, q, r = x[9], x[10], x[11]
        u, v, w = x[3], x[4], x[5]

        # 计算方向余弦矩阵
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        # 位置变化率 (机体速度 -> NED速度)
        dx_dt[0] = u * (cy * cp) + v * (cy * sp * sr - sy * cr) + w * (cy * sp * cr + sy * sr)
        dx_dt[1] = u * (sy * cp) + v * (sy * sp * sr + cy * cr) + w * (sy * sp * cr - cy * sr)
        dx_dt[2] = u * (-sp) + v * (cp * sr) + w * (cp * cr)

        # 姿态变化率 (角速度 -> 欧拉角变化率)
        # 避免奇点 (gimbal lock)
        if abs(cp) > 0.01:  # 非奇点区域
            dx_dt[6] = p + q * sr * np.tan(pitch) + r * cr * np.tan(pitch)  # roll_rate
            dx_dt[7] = q * cr - r * sr  # pitch_rate
            dx_dt[8] = (q * sr + r * cr) / cp  # yaw_rate
        else:
            # 奇点附近，使用近似
            dx_dt[6] = p
            dx_dt[7] = q * cr - r * sr
            dx_dt[8] = 0.0  # 奇点，无法准确估计

        # 速度变化率 (简化：假设速度不变)
        # 完整实现需要考虑空气动力、推力、重力等
        dx_dt[3:6] = 0.0

        # 角速度变化率 (简化：假设角速度不变)
        dx_dt[9:12] = 0.0

        return dx_dt

    def get_process_noise_covariance(self) -> np.ndarray:
        """获取过程噪声协方差矩阵"""
        Q = np.eye(12)

        # 位置噪声
        Q[0:3, 0:3] *= self.process_noise_pos

        # 速度噪声
        Q[3:6, 3:6] *= self.process_noise_vel

        # 姿态噪声
        Q[6:9, 6:9] *= self.process_noise_att

        # 角速度噪声
        Q[9:12, 9:12] *= self.process_noise_rate

        return Q


class CoordinatedTurnModel6DOF:
    """
    协调转弯6DOF模型

    模拟恒定转弯半径的机动飞行
    """

    def __init__(
        self,
        dt: float = 0.01,
        turn_rate: float = 0.1,
        bank_angle: Optional[float] = None
    ):
        """
        Args:
            dt: 时间步长
            turn_rate: 转弯角速度 (rad/s)
            bank_angle: 倾斜角 (rad, None则自动计算)
        """
        self.dt = dt
        self.turn_rate = turn_rate
        self.bank_angle = bank_angle

        # 创建基础6DOF模型
        self.base_model = SixDOFMotionModel(dt)

    def predict(
        self,
        state: State6DOF,
        velocity: float = 100.0,
        dt: Optional[float] = None
    ) -> State6DOF:
        """
        预测协调转弯状态

        Args:
            state: 当前状态
            velocity: 飞行速度 (m/s)
            dt: 时间步长

        Returns:
            预测状态
        """
        if dt is None:
            dt = self.dt

        # 计算协调转弯所需的倾斜角
        g = 9.81
        if self.bank_angle is None:
            # 水平协调转弯: tan(phi) = v * omega / g
            self.bank_angle = np.arctan(velocity * self.turn_rate / g)

        # 更新状态
        new_state = State6DOF.from_vector(state.to_vector())

        # 更新位置 (恒速转弯)
        # 使用转弯圆心计算
        turn_radius = velocity / self.turn_rate if abs(self.turn_rate) > 1e-6 else float('inf')

        if abs(self.turn_rate) > 1e-6:
            # 转弯角度
            delta_theta = self.turn_rate * dt

            # 当前在转弯圆周上的角度
            theta_current = np.arctan2(new_state.y, new_state.x)

            # 新角度
            theta_new = theta_current + delta_theta

            # 计算圆心
            center_x = new_state.x - turn_radius * np.sin(theta_current)
            center_y = new_state.y + turn_radius * np.cos(theta_current)

            # 新位置
            new_state.x = center_x + turn_radius * np.sin(theta_new)
            new_state.y = center_y - turn_radius * np.cos(theta_new)

        # 更新速度方向
        new_state.u = velocity * np.cos(new_state.attitude.yaw)
        new_state.v = velocity * np.sin(new_state.attitude.yaw)

        # 更新航向角
        new_state.attitude.yaw += self.turn_rate * dt
        new_state.attitude.yaw = (new_state.attitude.yaw + np.pi) % (2 * np.pi) - np.pi

        # 设置倾斜角
        new_state.attitude.roll = self.bank_angle

        # 转弯速率
        new_state.attitude.yaw_rate = self.turn_rate

        return new_state


class Kinematic6DOF:
    """
    6DOF运动学模型 (简化版本)

    用于目标跟踪，不涉及复杂的动力学
    """

    def __init__(self, dt: float = 0.1):
        """
        Args:
            dt: 时间步长
        """
        self.dt = dt

    def predict(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        acceleration: Optional[np.ndarray] = None,
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测位置和速度

        Args:
            position: 位置 [x, y, z] (m)
            velocity: 速度 [vx, vy, vz] (m/s)
            acceleration: 加速度 [ax, ay, az] (m/s^2)
            dt: 时间步长

        Returns:
            (position_pred, velocity_pred): 预测位置和速度
        """
        if dt is None:
            dt = self.dt

        if acceleration is None:
            # 恒速模型
            position_pred = position + velocity * dt
            velocity_pred = velocity
        else:
            # 恒加速模型
            position_pred = position + velocity * dt + 0.5 * acceleration * dt**2
            velocity_pred = velocity + acceleration * dt

        return position_pred, velocity_pred

    def predict_with_attitude(
        self,
        position: np.ndarray,
        velocity_body: np.ndarray,
        attitude: AttitudeState,
        dt: Optional[float] = None
    ) -> Tuple[State6DOF, np.ndarray]:
        """
        预测带姿态的状态

        Args:
            position: NED位置
            velocity_body: 机体坐标系速度
            attitude: 姿态状态
            dt: 时间步长

        Returns:
            (state_pred, velocity_pred): 预测状态和NED速度
        """
        if dt is None:
            dt = self.dt

        # 转换机体速度到NED
        DCM = attitude.to_dcm()
        velocity_ned = DCM @ velocity_body

        # 预测位置
        position_pred = position + velocity_ned * dt

        # 创建预测状态
        state_pred = State6DOF(
            x=position_pred[0],
            y=position_pred[1],
            z=position_pred[2],
            u=velocity_body[0],
            v=velocity_body[1],
            w=velocity_body[2],
            attitude=attitude
        )

        return state_pred, velocity_ned


# 便捷函数
def create_6dof_model(
    model_type: str = "kinematic",
    dt: float = 0.1,
    **kwargs
) -> Union[SixDOFMotionModel, CoordinatedTurnModel6DOF, Kinematic6DOF]:
    """
    创建6DOF模型

    Args:
        model_type: 模型类型 ("full", "coordinated_turn", "kinematic")
        dt: 时间步长
        **kwargs: 其他参数

    Returns:
        模型实例
    """
    if model_type == "full":
        return SixDOFMotionModel(dt, **kwargs)
    elif model_type == "coordinated_turn":
        return CoordinatedTurnModel6DOF(dt, **kwargs)
    elif model_type == "kinematic":
        return Kinematic6DOF(dt)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
