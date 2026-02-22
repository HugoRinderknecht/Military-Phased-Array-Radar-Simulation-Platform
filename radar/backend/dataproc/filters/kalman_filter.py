# kalman_filter.py - 完整的卡尔曼滤波器实现
"""
标准卡尔曼滤波器 (KF) - 线性系统状态估计

适用于线性高斯系统的状态估计，提供：
- 状态预测
- 测量更新
- 协方差矩阵传播
- 马氏距离计算
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class KalmanState:
    """卡尔曼滤波器状态"""
    x: np.ndarray  # 状态向量 [n x 1]
    P: np.ndarray  # 协方差矩阵 [n x n]
    timestamp: float = 0.0  # 时间戳 (秒)

    def __post_init__(self):
        """确保维度正确"""
        if self.x.ndim == 1:
            self.x = self.x.reshape(-1, 1)
        if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
            raise ValueError(f"协方差矩阵必须是方阵，得到形状: {self.P.shape}")


class ProcessModel:
    """过程模型基类"""

    def __init__(self, dt: float, Q: np.ndarray):
        """
        Args:
            dt: 时间步长 (秒)
            Q: 过程噪声协方差矩阵 [n x n]
        """
        self.dt = dt
        self.Q = Q

    def F(self) -> np.ndarray:
        """状态转移矩阵"""
        raise NotImplementedError

    def predict(self, state: KalmanState) -> KalmanState:
        """状态预测"""
        F = self.F()
        x_pred = F @ state.x
        P_pred = F @ state.P @ F.T + self.Q
        return KalmanState(x_pred, P_pred, state.timestamp + self.dt)


class ConstantVelocityModel(ProcessModel):
    """恒速 (CV) 模型 - 适用于匀速运动目标"""

    def __init__(self, dt: float, q: float = 1.0):
        """
        Args:
            dt: 时间步长 (秒)
            q: 过程噪声强度 (m^2/s^3)
        """
        # 3D CV模型状态: [x, y, z, vx, vy, vz]
        n = 6
        Q = np.eye(n) * q
        # 速度位置的噪声耦合
        Q[0:3, 3:6] = np.eye(3) * q * dt / 2
        Q[3:6, 0:3] = np.eye(3) * q * dt / 2
        super().__init__(dt, Q)

    def F(self) -> np.ndarray:
        """CV模型状态转移矩阵"""
        dt = self.dt
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt
        return F


class ConstantAccelerationModel(ProcessModel):
    """恒定加速度 (CA) 模型 - 适用于匀加速运动目标"""

    def __init__(self, dt: float, q: float = 1.0):
        """
        Args:
            dt: 时间步长 (秒)
            q: 过程噪声强度 (m^2/s^5)
        """
        # 3D CA模型状态: [x, y, z, vx, vy, vz, ax, ay, az]
        n = 9
        Q = np.eye(n) * q
        super().__init__(dt, Q)

    def F(self) -> np.ndarray:
        """CA模型状态转移矩阵"""
        dt = self.dt
        F = np.eye(9)
        # 位置 = 位置 + 速度*dt + 0.5*加速度*dt^2
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = np.eye(3) * 0.5 * dt**2
        # 速度 = 速度 + 加速度*dt
        F[3:6, 6:9] = np.eye(3) * dt
        return F


class CoordinatedTurnModel(ProcessModel):
    """协调转弯 (CT) 模型 - 适用于匀速转弯目标"""

    def __init__(self, dt: float, q: float = 1.0, omega: float = 0.0):
        """
        Args:
            dt: 时间步长 (秒)
            q: 过程噪声强度 (m^2/s^3)
            omega: 转弯角速度 (rad/s)
        """
        self.omega = omega
        n = 5  # [x, y, vx, vy, omega]
        Q = np.eye(n) * q
        super().__init__(dt, Q)

    def F(self) -> np.ndarray:
        """CT模型状态转移矩阵"""
        dt = self.dt
        omega = self.omega

        if abs(omega) < 1e-6:
            # 小角度近似
            return np.eye(5)

        F = np.eye(5)
        c = np.cos(omega * dt)
        s = np.sin(omega * dt)

        # 位置更新 (考虑转弯)
        F[0, 2] = s / omega
        F[0, 3] = (1 - c) / omega
        F[1, 2] = (1 - c) / omega
        F[1, 3] = s / omega

        # 速度更新
        F[2, 2] = c
        F[2, 3] = s
        F[3, 2] = -s
        F[3, 3] = c

        return F


class MeasurementModel:
    """测量模型基类"""

    def __init__(self, R: np.ndarray):
        """
        Args:
            R: 测量噪声协方差矩阵 [m x m]
        """
        self.R = R

    def H(self, state_dim: int) -> np.ndarray:
        """测量矩阵"""
        raise NotImplementedError

    def h(self, x: np.ndarray) -> np.ndarray:
        """测量函数 (非线性)"""
        raise NotImplementedError

    def innovate(self, state: KalmanState, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算新息

        Returns:
            (y, S): 新息向量, 新息协方差
        """
        H = self.H(state.x.shape[0])
        z_pred = H @ state.x
        y = z - z_pred  # 新息
        S = H @ state.P @ H.T + self.R  # 新息协方差
        return y, S


class PositionMeasurement(MeasurementModel):
    """位置测量模型 - 测量目标位置"""

    def __init__(self, sigma_r: float = 10.0):
        """
        Args:
            sigma_r: 测距误差标准差 (m)
        """
        R = np.eye(3) * sigma_r**2
        super().__init__(R)

    def H(self, state_dim: int) -> np.ndarray:
        """测量矩阵 - 提取位置分量"""
        H = np.zeros((3, state_dim))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z
        return H


class PositionVelocityMeasurement(MeasurementModel):
    """位置-速度测量模型 - 测量位置和速度"""

    def __init__(self, sigma_r: float = 10.0, sigma_v: float = 1.0):
        """
        Args:
            sigma_r: 测距误差标准差 (m)
            sigma_v: 测速误差标准差 (m/s)
        """
        R = np.eye(6)
        R[0:3, 0:3] = np.eye(3) * sigma_r**2
        R[3:6, 3:6] = np.eye(3) * sigma_v**2
        super().__init__(R)

    def H(self, state_dim: int) -> np.ndarray:
        """测量矩阵 - 提取位置和速度分量"""
        H = np.zeros((6, state_dim))
        min_dim = min(6, state_dim)
        for i in range(min_dim):
            H[i, i] = 1.0
        return H


class KalmanFilter:
    """
    标准卡尔曼滤波器

    完整实现包括：
    - 状态预测
    - 测量更新
    - 马氏距离计算
    - 似然计算
    """

    def __init__(
        self,
        process_model: ProcessModel,
        measurement_model: MeasurementModel,
        initial_state: Optional[KalmanState] = None
    ):
        """
        Args:
            process_model: 过程模型
            measurement_model: 测量模型
            initial_state: 初始状态
        """
        self.process_model = process_model
        self.measurement_model = measurement_model
        self.state = initial_state
        self.predicted_state: Optional[KalmanState] = None

    def init(self, x0: np.ndarray, P0: np.ndarray, timestamp: float = 0.0) -> None:
        """初始化滤波器"""
        self.state = KalmanState(x0, P0, timestamp)

    def predict(self, dt: Optional[float] = None) -> KalmanState:
        """状态预测"""
        if self.state is None:
            raise ValueError("滤波器未初始化")

        if dt is not None:
            self.process_model.dt = dt

        self.predicted_state = self.process_model.predict(self.state)
        return self.predicted_state

    def update(self, z: np.ndarray) -> Tuple[KalmanState, Dict[str, Any]]:
        """
        测量更新

        Args:
            z: 测量向量

        Returns:
            (updated_state, info): 更新后的状态和辅助信息
        """
        if self.predicted_state is None:
            raise ValueError("需要先调用 predict()")

        # 获取测量矩阵
        H = self.measurement_model.H(self.predicted_state.x.shape[0])

        # 计算新息
        z_pred = H @ self.predicted_state.x
        y = z - z_pred  # 新息向量

        # 新息协方差
        S = H @ self.predicted_state.P @ H.T + self.measurement_model.R

        # 卡尔曼增益
        K = self.predicted_state.P @ H.T @ np.linalg.inv(S)

        # 状态更新
        x_updated = self.predicted_state.x + K @ y

        # 协方差更新 (Joseph形式，数值更稳定)
        I_KH = np.eye(self.predicted_state.x.shape[0]) - K @ H
        P_updated = I_KH @ self.predicted_state.P @ I_KH.T + K @ self.measurement_model.R @ K.T

        # 创建更新后的状态
        self.state = KalmanState(x_updated, P_updated, self.predicted_state.timestamp)
        self.predicted_state = None

        # 计算统计信息
        info = {
            'innovation': y.flatten(),
            'innovation_covariance': S,
            'kalman_gain': K,
            'mahalanobis_distance': self._mahalanobis_distance(y, S),
            'likelihood': self._likelihood(y, S),
            'NEES': self._nees(y, S)  # 归一化估计误差平方
        }

        return self.state, info

    def _mahalanobis_distance(self, y: np.ndarray, S: np.ndarray) -> float:
        """计算马氏距离"""
        try:
            return float(np.sqrt(y.T @ np.linalg.inv(S) @ y))
        except np.linalg.LinAlgError:
            return float('inf')

    def _likelihood(self, y: np.ndarray, S: np.ndarray) -> float:
        """计算似然"""
        try:
            n = len(y)
            det_S = np.linalg.det(S)
            if det_S <= 0:
                return 0.0
            exp_term = -0.5 * y.T @ np.linalg.inv(S) @ y
            return float((2 * np.pi) ** (-n/2) * det_S ** (-0.5) * np.exp(exp_term))
        except:
            return 0.0

    def _nees(self, y: np.ndarray, S: np.ndarray) -> float:
        """计算归一化估计误差平方 (Normalized Estimation Error Squared)"""
        try:
            return float(y.T @ np.linalg.inv(S) @ y)
        except np.linalg.LinAlgError:
            return float('inf')

    def step(self, z: np.ndarray, dt: float) -> Tuple[KalmanState, Dict[str, Any]]:
        """
        完整的预测-更新步骤

        Args:
            z: 测量向量
            dt: 时间步长

        Returns:
            (updated_state, info): 更新后的状态和辅助信息
        """
        self.predict(dt)
        return self.update(z)


class AdaptiveKalmanFilter(KalmanFilter):
    """自适应卡尔曼滤波器 - 根据新息调整噪声参数"""

    def __init__(
        self,
        process_model: ProcessModel,
        measurement_model: MeasurementModel,
        initial_state: Optional[KalmanState] = None,
        alpha: float = 0.95
    ):
        """
        Args:
            process_model: 过程模型
            measurement_model: 测量模型
            initial_state: 初始状态
            alpha: 遗忘因子 (0-1)
        """
        super().__init__(process_model, measurement_model, initial_state)
        self.alpha = alpha
        self.innovation_history: list = []

    def update(self, z: np.ndarray) -> Tuple[KalmanState, Dict[str, Any]]:
        """自适应测量更新"""
        if self.predicted_state is None:
            raise ValueError("需要先调用 predict()")

        # 标准更新
        updated_state, info = super().update(z)

        # 自适应调整测量噪声协方差
        y = info['innovation']
        self.innovation_history.append(y)

        # 保持最近N个新息
        window = 20
        if len(self.innovation_history) > window:
            self.innovation_history.pop(0)

        # 估计新息协方差
        if len(self.innovation_history) >= 5:
            innovations = np.array(self.innovation_history)
            S_empirical = np.cov(innovations.T)

            # 平滑更新
            self.measurement_model.R = (
                self.alpha * self.measurement_model.R +
                (1 - self.alpha) * S_empirical
            )

        return updated_state, info


def create_cv_filter(
    dt: float,
    sigma_q: float = 1.0,
    sigma_r: float = 10.0,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None
) -> KalmanFilter:
    """
    创建CV模型的卡尔曼滤波器

    Args:
        dt: 时间步长 (秒)
        sigma_q: 过程噪声强度 (m^2/s^3)
        sigma_r: 测量噪声标准差 (m)
        x0: 初始状态 [x, y, z, vx, vy, vz]
        P0: 初始协方差矩阵

    Returns:
        配置好的卡尔曼滤波器
    """
    process_model = ConstantVelocityModel(dt, sigma_q)
    measurement_model = PositionMeasurement(sigma_r)

    kf = KalmanFilter(process_model, measurement_model)

    if x0 is not None:
        if P0 is None:
            P0 = np.eye(6) * 100.0  # 默认初始不确定度
        kf.init(x0, P0)

    return kf


def create_ca_filter(
    dt: float,
    sigma_q: float = 1.0,
    sigma_r: float = 10.0,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None
) -> KalmanFilter:
    """
    创建CA模型的卡尔曼滤波器

    Args:
        dt: 时间步长 (秒)
        sigma_q: 过程噪声强度 (m^2/s^5)
        sigma_r: 测量噪声标准差 (m)
        x0: 初始状态 [x, y, z, vx, vy, vz, ax, ay, az]
        P0: 初始协方差矩阵

    Returns:
        配置好的卡尔曼滤波器
    """
    process_model = ConstantAccelerationModel(dt, sigma_q)
    measurement_model = PositionMeasurement(sigma_r)

    kf = KalmanFilter(process_model, measurement_model)

    if x0 is not None:
        if P0 is None:
            P0 = np.eye(9) * 100.0
        kf.init(x0, P0)

    return kf
