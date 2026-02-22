# unscented_kalman_filter.py - 无迹卡尔曼滤波器实现
"""
无迹卡尔曼滤波器 (UKF) - 基于无迹变换的非线性估计

UKF不使用雅可比矩阵，而是通过sigma点传播非线性：
- 无迹变换 (Unscented Transform)
- Sigma点采样策略
- 更好的非线性处理能力
- 数值稳定性优于EKF
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Callable
from dataclasses import dataclass
from .kalman_filter import KalmanState


class UnscentedTransform:
    """无迹变换 - 通过sigma点传播均值和协方差"""

    def __init__(self, alpha: float = 1e-3, beta: float = 2.0, kappa: float = 0.0):
        """
        Args:
            alpha: sigma点散布参数 (通常1e-3 <= alpha <= 1)
            beta: 先验分布参数 (高斯分布beta=2)
            kappa: 二阶缩放参数 (通常kappa=0)
        """
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def compute_sigma_points(
        self,
        x: np.ndarray,
        P: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算sigma点

        Args:
            x: 均值向量 [n x 1]
            P: 协方差矩阵 [n x n]

        Returns:
            (sigma_points, weights): sigma点 [n x (2n+1)], 权重
        """
        n = len(x)
        lambda_ = self.alpha**2 * (n + self.kappa) - n

        # 计算权重
        Wm = np.zeros(2 * n + 1)  # 均值权重
        Wc = np.zeros(2 * n + 1)  # 协方差权重

        Wm[0] = lambda_ / (n + lambda_)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        Wm[1:] = Wc[1:] = 1.0 / (2 * (n + lambda_))

        # 计算sqrt(P)
        try:
            # 使用Cholesky分解
            U = np.linalg.cholesky((n + lambda_) * P).T
        except np.linalg.LinAlgError:
            # 如果Cholesky失败，使用SVD
            U = np.linalg.sqrt(n + lambda_ * P)

        # 生成sigma点
        sigma_points = np.zeros((n, 2 * n + 1))
        sigma_points[:, 0] = x.flatten()

        for i in range(n):
            sigma_points[:, i + 1] = x.flatten() + U[:, i]
            sigma_points[:, i + n + 1] = x.flatten() - U[:, i]

        return sigma_points, np.array([Wm, Wc])

    def recover_gaussian(
        self,
        sigma_points: np.ndarray,
        weights: np.ndarray,
        mean_fn: Optional[Callable] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从sigma点恢复均值和协方差

        Args:
            sigma_points: sigma点 [n x (2n+1)]
            weights: 权重 [2 x (2n+1)] 第一行均值权重，第二行协方差权重
            mean_fn: 可选的均值计算函数

        Returns:
            (mean, covariance): 均值 [n x 1], 协方差 [n x n]
        """
        Wm = weights[0, :]
        Wc = weights[1, :]
        n = sigma_points.shape[0]

        # 计算均值
        if mean_fn is not None:
            mean = mean_fn(sigma_points, Wm)
        else:
            mean = np.sum(Wm * sigma_points, axis=1).reshape(-1, 1)

        # 计算协方差
        deviations = sigma_points - mean
        covariance = deviations @ (Wc.reshape(1, -1) * deviations).T

        return mean, covariance


class UnscentedKalmanFilter:
    """
    无迹卡尔曼滤波器 (UKF)

    通过无迹变换处理非线性系统，不需要计算雅可比矩阵
    """

    def __init__(
        self,
        f: Callable[[np.ndarray, float], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        state_dim: int,
        meas_dim: int,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        """
        Args:
            f: 状态转移函数
            h: 测量函数
            Q: 过程噪声协方差
            R: 测量噪声协方差
            state_dim: 状态维度
            meas_dim: 测量维度
            alpha: UKF参数
            beta: UKF参数
            kappa: UKF参数
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.ut = UnscentedTransform(alpha, beta, kappa)
        self.state: Optional[KalmanState] = None
        self.predicted_state: Optional[KalmanState] = None

    def init(self, x0: np.ndarray, P0: np.ndarray, timestamp: float = 0.0) -> None:
        """初始化滤波器"""
        self.state = KalmanState(x0, P0, timestamp)

    def predict(self, dt: float) -> KalmanState:
        """状态预测"""
        if self.state is None:
            raise ValueError("滤波器未初始化")

        # 生成sigma点
        sigma_points, weights = self.ut.compute_sigma_points(self.state.x, self.state.P)

        # 通过状态转移函数传播sigma点
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[1]):
            sigma_points_pred[:, i] = self.f(sigma_points[:, i:i+1], dt).flatten()

        # 恢复预测均值和协方差
        x_pred, P_pred = self.ut.recover_gaussian(sigma_points_pred, weights)

        # 添加过程噪声
        P_pred += self.Q

        self.predicted_state = KalmanState(x_pred, P_pred, self.state.timestamp + dt)
        return self.predicted_state

    def update(self, z: np.ndarray) -> Tuple[KalmanState, Dict[str, Any]]:
        """测量更新"""
        if self.predicted_state is None:
            raise ValueError("需要先调用 predict()")

        # 生成预测sigma点
        sigma_points, weights = self.ut.compute_sigma_points(
            self.predicted_state.x,
            self.predicted_state.P
        )

        # 通过测量函数传播sigma点
        sigma_meas = np.zeros((self.meas_dim, sigma_points.shape[1]))
        for i in range(sigma_points.shape[1]):
            sigma_meas[:, i] = self.h(sigma_points[:, i:i+1]).flatten()

        # 计算预测测量的均值和协方差
        z_pred, Pzz = self.ut.recover_gaussian(sigma_meas, weights)
        Pzz += self.R  # 添加测量噪声

        # 计算互协方差
        deviations_x = sigma_points - self.predicted_state.x
        deviations_z = sigma_meas - z_pred
        Wc = weights[1, :]
        Pxz = deviations_x @ (Wc.reshape(1, -1) * deviations_z).T

        # 卡尔曼增益
        try:
            K = Pxz @ np.linalg.inv(Pzz)
        except np.linalg.LinAlgError:
            K = Pxz @ np.linalg.pinv(Pzz)

        # 新息
        y = z - z_pred

        # 状态更新
        x_updated = self.predicted_state.x + K @ y

        # 协方差更新
        P_updated = self.predicted_state.P - K @ Pzz @ K.T

        # 确保协方差矩阵对称正定
        P_updated = 0.5 * (P_updated + P_updated.T)
        P_updated += np.eye(self.state_dim) * 1e-6

        self.state = KalmanState(x_updated, P_updated, self.predicted_state.timestamp)
        self.predicted_state = None

        # 统计信息
        try:
            mahalanobis_dist = float(np.sqrt(y.T @ np.linalg.inv(Pzz) @ y))
        except:
            mahalanobis_dist = float('inf')

        info = {
            'innovation': y.flatten(),
            'innovation_covariance': Pzz,
            'kalman_gain': K,
            'predicted_measurement': z_pred.flatten(),
            'mahalanobis_distance': mahalanobis_dist,
        }

        return self.state, info

    def step(self, z: np.ndarray, dt: float) -> Tuple[KalmanState, Dict[str, Any]]:
        """完整步骤"""
        self.predict(dt)
        return self.update(z)


def create_cv_ukf(
    dt: float = 0.1,
    sigma_q: float = 1.0,
    sigma_r: float = 10.0,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None
) -> UnscentedKalmanFilter:
    """
    创建CV模型的UKF

    Args:
        dt: 时间步长
        sigma_q: 过程噪声强度
        sigma_r: 测量噪声标准差
        x0: 初始状态
        P0: 初始协方差

    Returns:
        配置好的UKF
    """
    # 状态维度: [x, y, z, vx, vy, vz]
    state_dim = 6
    meas_dim = 3

    # 过程噪声
    Q = np.eye(state_dim) * sigma_q

    # 测量噪声
    R = np.eye(meas_dim) * sigma_r**2

    # 状态转移函数 (CV模型)
    def f(x: np.ndarray, dt: float) -> np.ndarray:
        x_new = x.copy()
        x_new[0:3] += x[3:6] * dt
        return x_new

    # 测量函数 (只观测位置)
    def h(x: np.ndarray) -> np.ndarray:
        return x[0:3]

    ukf = UnscentedKalmanFilter(f, h, Q, R, state_dim, meas_dim)

    if x0 is not None:
        if P0 is None:
            P0 = np.eye(state_dim) * 100.0
        ukf.init(x0, P0)

    return ukf
