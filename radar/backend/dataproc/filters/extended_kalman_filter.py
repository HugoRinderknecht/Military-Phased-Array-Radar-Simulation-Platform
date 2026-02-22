# extended_kalman_filter.py - 扩展卡尔曼滤波器实现
"""
扩展卡尔曼滤波器 (EKF) - 非线性系统状态估计

适用于非线性系统，通过雅可比矩阵线性化处理：
- 非线性状态转移
- 非线性测量函数
- 数值雅可比计算
- 鲁棒的协方差更新
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from .kalman_filter import KalmanState, ProcessModel, MeasurementModel


class ExtendedKalmanFilter:
    """
    扩展卡尔曼滤波器 (EKF)

    通过雅可比矩阵线性化处理非线性系统
    """

    def __init__(
        self,
        f: Callable[[np.ndarray, float], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        state_dim: int,
        meas_dim: int,
        compute_jacobian: bool = True
    ):
        """
        Args:
            f: 状态转移函数 f(x, dt) -> x_pred
            h: 测量函数 h(x) -> z_pred
            Q: 过程噪声协方差 [n x n]
            R: 测量噪声协方差 [m x m]
            state_dim: 状态维度
            meas_dim: 测量维度
            compute_jacobian: 是否数值计算雅可比矩阵
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.compute_jacobian = compute_jacobian
        self.state: Optional[KalmanState] = None
        self.predicted_state: Optional[KalmanState] = None

    def init(self, x0: np.ndarray, P0: np.ndarray, timestamp: float = 0.0) -> None:
        """初始化滤波器"""
        self.state = KalmanState(x0, P0, timestamp)

    def _compute_jacobian_f(self, x: np.ndarray, dt: float, epsilon: float = 1e-6) -> np.ndarray:
        """数值计算状态转移函数的雅可比矩阵"""
        n = len(x)
        F = np.zeros((n, n))

        # 标称状态
        f_nominal = self.f(x, dt)

        for i in range(n):
            # 扰动
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon
            f_perturbed = self.f(x_perturbed, dt)

            # 有限差分
            F[:, i] = (f_perturbed - f_nominal) / epsilon

        return F

    def _compute_jacobian_h(self, x: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
        """数值计算测量函数的雅可比矩阵"""
        n = len(x)
        m = self.meas_dim
        H = np.zeros((m, n))

        # 标称测量
        h_nominal = self.h(x)

        for i in range(n):
            # 扰动
            x_perturbed = x.copy()
            x_perturbed[i] += epsilon
            h_perturbed = self.h(x_perturbed)

            # 有限差分
            H[:, i] = (h_perturbed - h_nominal) / epsilon

        return H

    def predict(self, dt: float) -> KalmanState:
        """状态预测"""
        if self.state is None:
            raise ValueError("滤波器未初始化")

        # 非线性状态预测
        x_pred = self.f(self.state.x, dt)

        # 计算雅可比矩阵
        if self.compute_jacobian:
            F = self._compute_jacobian_f(self.state.x, dt)
        else:
            # 如果不计算雅可比，使用单位矩阵（简化）
            F = np.eye(self.state_dim)

        # 协方差传播
        P_pred = F @ self.state.P @ F.T + self.Q

        self.predicted_state = KalmanState(x_pred, P_pred, self.state.timestamp + dt)
        return self.predicted_state

    def update(self, z: np.ndarray) -> Tuple[KalmanState, Dict[str, Any]]:
        """测量更新"""
        if self.predicted_state is None:
            raise ValueError("需要先调用 predict()")

        # 预测测量
        z_pred = self.h(self.predicted_state.x)

        # 新息
        y = z - z_pred

        # 计算测量雅可比
        if self.compute_jacobian:
            H = self._compute_jacobian_h(self.predicted_state.x)
        else:
            H = np.eye(self.meas_dim, self.state_dim)

        # 新息协方差
        S = H @ self.predicted_state.P @ H.T + self.R

        # 卡尔曼增益
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.predicted_state.P @ H.T @ S_inv

        # 状态更新
        x_updated = self.predicted_state.x + K @ y

        # 协方差更新 (Joseph形式)
        I_KH = np.eye(self.state_dim) - K @ H
        P_updated = I_KH @ self.predicted_state.P @ I_KH.T + K @ self.R @ K.T

        # 创建更新状态
        self.state = KalmanState(x_updated, P_updated, self.predicted_state.timestamp)
        self.predicted_state = None

        # 统计信息
        info = {
            'innovation': y.flatten(),
            'innovation_covariance': S,
            'kalman_gain': K,
            'predicted_measurement': z_pred.flatten(),
            'mahalanobis_distance': float(np.sqrt(y.T @ S_inv @ y)),
        }

        return self.state, info

    def step(self, z: np.ndarray, dt: float) -> Tuple[KalmanState, Dict[str, Any]]:
        """完整步骤"""
        self.predict(dt)
        return self.update(z)


def spherical_to_cartesian_ekf(
    dt: float = 0.1,
    sigma_r: float = 15.0,
    sigma_theta: float = 0.01,
    sigma_phi: float = 0.01,
    q: float = 1.0
) -> ExtendedKalmanFilter:
    """
    创建球坐标测量的EKF

    将球坐标 (r, theta, phi) 测量转换为笛卡尔坐标

    Args:
        dt: 时间步长
        sigma_r: 测距误差标准差 (m)
        sigma_theta: 方位角误差标准差 (rad)
        sigma_phi: 俯仰角误差标准差 (rad)
        q: 过程噪声强度

    Returns:
        配置好的EKF
    """
    # 状态维度: [x, y, z, vx, vy, vz]
    state_dim = 6
    meas_dim = 3

    # 过程噪声
    Q = np.eye(state_dim) * q

    # 测量噪声 (球坐标)
    R = np.diag([sigma_r**2, sigma_theta**2, sigma_phi**2])

    # 状态转移函数 (CV模型)
    def f(x: np.ndarray, dt: float) -> np.ndarray:
        x_new = x.copy()
        x_new[0:3] += x[3:6] * dt  # 位置更新
        return x_new

    # 测量函数 (笛卡尔 -> 球坐标)
    def h(x: np.ndarray) -> np.ndarray:
        x_pos, y_pos, z_pos = x[0], x[1], x[2]
        r = np.sqrt(x_pos**2 + y_pos**2 + z_pos**2)
        theta = np.arctan2(y_pos, x_pos)
        phi = np.arcsin(z_pos / r) if r > 0 else 0.0
        return np.array([r, theta, phi])

    return ExtendedKalmanFilter(f, h, Q, R, state_dim, meas_dim)
