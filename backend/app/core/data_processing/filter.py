"""
Kalman滤波模块
完整实现标准KF、EKF、UKF及自适应滤波

参考文档 4.4.10 节
"""
import numpy as np
from typing import Optional, Tuple, Literal
from scipy.linalg import expm, block_diag
from dataclasses import dataclass

from .track_init import Plot, ConfirmedTrack


@dataclass
class FilterState:
    """滤波器状态"""
    state: np.ndarray  # 状态向量
    covariance: np.ndarray  # 协方差矩阵
    time: float  # 时间戳


class KalmanFilter:
    """
    标准Kalman滤波器

    状态模型：匀速(CV)或匀加速(CA)
    """

    def __init__(
        self,
        dt: float,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        motion_model: Literal["CV", "CA"] = "CV",
    ):
        """
        Args:
            dt: 时间步长
            process_noise: 过程噪声强度
            measurement_noise: 测量噪声强度
            motion_model: 运动模型 ("CV"=匀速, "CA"=匀加速)
        """
        self.dt = dt
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.motion_model = motion_model

        # 状态维度: CV=6 (x,y,z,vx,vy,vz), CA=9 (+ax,ay,az)
        self.state_dim = 6 if motion_model == "CV" else 9
        self.meas_dim = 3  # 测量维度 (x,y,z)

        # 初始化状态转移矩阵、过程噪声协方差、测量矩阵
        self.F = self._build_state_transition()
        self.Q = self._build_process_noise_covariance()
        self.H = self._build_measurement_matrix()
        self.R = self._build_measurement_noise_covariance()

    def _build_state_transition(self) -> np.ndarray:
        """构建状态转移矩阵"""
        if self.motion_model == "CV":
            # CV模型: x(k+1) = x(k) + v(k)*dt
            F = np.zeros((6, 6))
            # 位置方程
            F[0, 0] = 1  # x
            F[0, 3] = self.dt  # vx
            F[1, 1] = 1  # y
            F[1, 4] = self.dt  # vy
            F[2, 2] = 1  # z
            F[2, 5] = self.dt  # vz
            # 速度方程（假设常速）
            F[3, 3] = 1
            F[4, 4] = 1
            F[5, 5] = 1
        else:  # CA
            # CA模型: 包含加速度
            F = np.zeros((9, 9))
            # 位置
            F[0, 0] = 1
            F[0, 3] = self.dt
            F[0, 6] = 0.5 * self.dt ** 2
            F[1, 1] = 1
            F[1, 4] = self.dt
            F[1, 7] = 0.5 * self.dt ** 2
            F[2, 2] = 1
            F[2, 5] = self.dt
            F[2, 8] = 0.5 * self.dt ** 2
            # 速度
            F[3, 3] = 1
            F[3, 6] = self.dt
            F[4, 4] = 1
            F[4, 7] = self.dt
            F[5, 5] = 1
            F[5, 8] = self.dt
            # 加速度
            F[6, 6] = 1
            F[7, 7] = 1
            F[8, 8] = 1

        return F

    def _build_process_noise_covariance(self) -> np.ndarray:
        """构建过程噪声协方差矩阵"""
        q = self.process_noise
        dt = self.dt

        if self.motion_model == "CV":
            # CV模型过程噪声（离散白噪声加速度模型）
            Q = np.eye(6) * q
            Q[0, 0] = Q[1, 1] = Q[2, 2] = dt ** 3 / 3
            Q[0, 3] = Q[1, 4] = Q[2, 5] = dt ** 2 / 2
            Q[3, 0] = Q[4, 1] = Q[5, 2] = dt ** 2 / 2
            Q[3, 3] = Q[4, 4] = Q[5, 5] = dt
        else:  # CA
            Q = np.eye(9) * q
            # 简化模型
            Q[:3, :3] *= dt ** 5 / 20
            Q[:3, 3:6] *= dt ** 4 / 8
            Q[:3, 6:] *= dt ** 3 / 6
            Q[3:6, :3] *= dt ** 4 / 8
            Q[3:6, 3:6] *= dt ** 3 / 3
            Q[3:6, 6:] *= dt ** 2 / 2
            Q[6:, :3] *= dt ** 3 / 6
            Q[6:, 3:6] *= dt ** 2 / 2
            Q[6:, 6:] *= dt

        return Q

    def _build_measurement_matrix(self) -> np.ndarray:
        """构建测量矩阵（只观测位置）"""
        H = np.zeros((3, self.state_dim))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # z
        return H

    def _build_measurement_noise_covariance(self) -> np.ndarray:
        """构建测量噪声协方差"""
        r = self.measurement_noise
        R = np.eye(3) * r
        return R

    def predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测步骤

        x̂ₖ₋ = F * x̂ₖ₋₁
        Pₖ₋ = F * Pₖ₋₁ * Fᵀ + Q

        Args:
            state: 上一时刻状态
            covariance: 上一时刻协方差

        Returns:
            (预测状态, 预测协方差)
        """
        # 状态预测
        predicted_state = self.F @ state

        # 协方差预测
        predicted_covariance = self.F @ covariance @ self.F.T + self.Q

        return predicted_state, predicted_covariance

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        更新步骤

        K = Pₖ₋ * Hᵀ * (H * Pₖ₋ * Hᵀ + R)⁻¹
        x̂ₖ = x̂ₖ₋ + K * (z - H * x̂ₖ₋)
        Pₖ = (I - K * H) * Pₖ₋

        Args:
            predicted_state: 预测状态
            predicted_covariance: 预测协方差
            measurement: 测量值 [x, y, z]

        Returns:
            (更新后状态, 更新后协方差, 卡尔曼增益)
        """
        # 新息
        innovation = measurement - self.H @ predicted_state

        # 新息协方差
        innovation_covariance = self.H @ predicted_covariance @ self.H.T + self.R

        # 卡尔曼增益
        try:
            kalman_gain = predicted_covariance @ self.H.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError:
            kalman_gain = np.zeros((self.state_dim, self.meas_dim))

        # 状态更新
        updated_state = predicted_state + kalman_gain @ innovation

        # 协方差更新（Joseph形式，数值更稳定）
        I_KH = np.eye(self.state_dim) - kalman_gain @ self.H
        updated_covariance = I_KH @ predicted_covariance @ I_KH.T + kalman_gain @ self.R @ kalman_gain.T

        return updated_state, updated_covariance, kalman_gain

    def filter_step(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整滤波步骤（预测+更新）

        Args:
            state: 上一时刻状态
            covariance: 上一时刻协方差
            measurement: 当前测量

        Returns:
            (更新后状态, 更新后协方差)
        """
        # 预测
        predicted_state, predicted_covariance = self.predict(state, covariance)

        # 更新
        updated_state, updated_covariance, _ = self.update(
            predicted_state, predicted_covariance, measurement
        )

        return updated_state, updated_covariance


class ExtendedKalmanFilter(KalmanFilter):
    """
    扩展Kalman滤波器(EKF)

    用于非线性测量模型（如极坐标测量）
    """

    def __init__(
        self,
        dt: float,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        motion_model: Literal["CV", "CA"] = "CV",
        measurement_model: Literal["cartesian", "spherical"] = "cartesian",
    ):
        """
        Args:
            dt: 时间步长
            process_noise: 过程噪声强度
            measurement_noise: 测量噪声强度
            motion_model: 运动模型
            measurement_model: 测量模型
                - cartesian: 笛卡尔坐标测量
                - spherical: 球坐标测量（距离、方位、俯仰）
        """
        super().__init__(dt, process_noise, measurement_noise, motion_model)
        self.measurement_model_type = measurement_model

        if measurement_model == "spherical":
            # 球坐标测量维度也是3 (r, az, el)
            self.meas_dim = 3

    def _measurement_function_cartesian_to_spherical(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        笛卡尔坐标到球坐标转换

        z = [r, az, el]ᵀ
        r = sqrt(x² + y² + z²)
        az = atan2(y, x)
        el = atan2(z, sqrt(x² + y²))

        Args:
            state: 状态向量 [x, y, z, ...]

        Returns:
            球坐标测量 [r, az, el]
        """
        x, y, z = state[0], state[1], state[2]

        r = np.sqrt(x**2 + y**2 + z**2)
        az = np.arctan2(y, x)
        el = np.arctan2(z, np.sqrt(x**2 + y**2))

        return np.array([r, az, el])

    def _calculate_jacobian(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        计算测量函数的雅可比矩阵

        H = ∂h/∂x

        Args:
            state: 状态向量

        Returns:
            雅可比矩阵
        """
        x, y, z = state[0], state[1], state[2]
        r = np.sqrt(x**2 + y**2 + z**2)
        rho = np.sqrt(x**2 + y**2)

        if r < 1e-10:
            r = 1e-10
        if rho < 1e-10:
            rho = 1e-10

        # dr/dx, dr/dy, dr/dz
        dr_dx = x / r
        dr_dy = y / r
        dr_dz = z / r

        # daz/dx, daz/dy, daz/dz
        daz_dx = -y / (x**2 + y**2)
        daz_dy = x / (x**2 + y**2)
        daz_dz = 0

        # del/dx, del/dy, del/dz
        del_dx = -x * z / (rho * r**2)
        del_dy = -y * z / (rho * r**2)
        del_dz = rho / r**2

        H = np.zeros((3, self.state_dim))
        H[0, 0] = dr_dx
        H[0, 1] = dr_dy
        H[0, 2] = dr_dz
        H[1, 0] = daz_dx
        H[1, 1] = daz_dy
        H[1, 2] = daz_dz
        H[2, 0] = del_dx
        H[2, 1] = del_dy
        H[2, 2] = del_dz

        return H

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        EKF更新步骤

        Args:
            predicted_state: 预测状态
            predicted_covariance: 预测协方差
            measurement: 测量值

        Returns:
            (更新后状态, 更新后协方差, 卡尔曼增益)
        """
        if self.measurement_model_type == "spherical":
            # 计算预测测量
            predicted_measurement = self._measurement_function_cartesian_to_spherical(predicted_state)

            # 计算雅可比矩阵
            H = self._calculate_jacobian(predicted_state)

            # 新息
            innovation = measurement - predicted_measurement

            # 处理角度环绕
            innovation[1] = (innovation[1] + np.pi) % (2 * np.pi) - np.pi  # 方位角
            innovation[2] = (innovation[2] + np.pi / 2) % np.pi - np.pi / 2  # 俯仰角

            # 新息协方差
            innovation_covariance = H @ predicted_covariance @ H.T + self.R

            # 卡尔曼增益
            try:
                kalman_gain = predicted_covariance @ H.T @ np.linalg.inv(innovation_covariance)
            except np.linalg.LinAlgError:
                kalman_gain = np.zeros((self.state_dim, self.meas_dim))

            # 状态更新
            updated_state = predicted_state + kalman_gain @ innovation

            # 协方差更新
            I_KH = np.eye(self.state_dim) - kalman_gain @ H
            updated_covariance = I_KH @ predicted_covariance @ I_KH.T + kalman_gain @ self.R @ kalman_gain.T

            return updated_state, updated_covariance, kalman_gain
        else:
            # 笛卡尔坐标，使用标准KF更新
            return super().update(predicted_state, predicted_covariance, measurement)


class UnscentedKalmanFilter:
    """
    无迹Kalman滤波器(UKF)

    使用Unscented变换处理非线性
    """

    def __init__(
        self,
        state_dim: int = 6,
        meas_dim: int = 3,
        dt: float = 0.1,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        """
        Args:
            state_dim: 状态维度
            meas_dim: 测量维度
            dt: 时间步长
            process_noise: 过程噪声
            measurement_noise: 测量噪声
            alpha: UKF参数（控制sigma点分布）
            beta: UKF参数（用于高斯分布，beta=2是最优值）
            kappa: UKF参数（通常设为0或3-state_dim）
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim
        self.dt = dt
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # UKF参数
        self.lambda_ = alpha ** 2 * (state_dim + kappa) - state_dim

        # 计算权重
        self._calculate_weights()

        # 噪声协方差
        self.Q = np.eye(state_dim) * process_noise
        self.R = np.eye(meas_dim) * measurement_noise

    def _calculate_weights(self):
        """计算UKF权重"""
        n = self.state_dim

        # 均值权重
        self.Wm = np.zeros(2 * n + 1)
        self.Wm[0] = self.lambda_ / (n + self.lambda_)
        self.Wm[1:] = 1 / (2 * (n + self.lambda_))

        # 协方差权重
        self.Wc = np.zeros(2 * n + 1)
        self.Wc[0] = self.lambda_ / (n + self.lambda_) + (1 - self.alpha ** 2 + self.beta)
        self.Wc[1:] = 1 / (2 * (n + self.lambda_))

    def _generate_sigma_points(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
    ) -> np.ndarray:
        """
        生成Sigma点

        Args:
            state: 状态向量
            covariance: 协方差矩阵

        Returns:
            Sigma点矩阵 (2n+1, n)
        """
        n = self.state_dim
        sigma_points = np.zeros((2 * n + 1, n))

        # 第0个点是均值
        sigma_points[0] = state

        # 计算协方差的平方根
        try:
            # 使用U = sqrt((n+λ)P)
            U = np.linalg.cholesky((n + self.lambda_) * covariance).T
        except np.linalg.LinAlgError:
            # 如果协方差矩阵不正定，使用特征值分解
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            eigenvalues = np.maximum(eigenvalues, 0)  # 确保非负
            U = eigenvectors @ np.diag(np.sqrt((n + self.lambda_) * eigenvalues))

        # 生成剩余的2n个sigma点
        for i in range(n):
            sigma_points[i + 1] = state + U[i]
            sigma_points[n + i + 1] = state - U[i]

        return sigma_points

    def _unscented_transform(
        self,
        sigma_points: np.ndarray,
        noise_covariance: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unscented变换

        Args:
            sigma_points: Sigma点矩阵
            noise_covariance: 噪声协方差（可选）

        Returns:
            (变换后的均值, 变换后的协方差)
        """
        # 计算均值
        mean = np.sum(self.Wm[:, np.newaxis] * sigma_points, axis=0)

        # 计算协方差
        deviations = sigma_points - mean
        covariance = np.sum(
            self.Wc[:, np.newaxis, np.newaxis] *
            deviations[:, :, np.newaxis] *
            deviations[:, np.newaxis, :],
            axis=0
        )

        if noise_covariance is not None:
            covariance += noise_covariance

        return mean, covariance

    def state_transition_function(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """状态转移函数（CV模型）"""
        new_state = state.copy()
        new_state[0] += state[3] * self.dt  # x += vx * dt
        new_state[1] += state[4] * self.dt  # y += vy * dt
        new_state[2] += state[5] * self.dt  # z += vz * dt
        # 速度保持不变
        return new_state

    def measurement_function(
        self,
        state: np.ndarray,
    ) -> np.ndarray:
        """测量函数（只观测位置）"""
        return state[:3]

    def predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """UKF预测步骤"""
        # 生成sigma点
        sigma_points = self._generate_sigma_points(state, covariance)

        # 状态转移
        transformed_points = np.array([
            self.state_transition_function(sp) for sp in sigma_points
        ])

        # Unscented变换得到预测均值和协方差
        predicted_state, predicted_covariance = self._unscented_transform(
            transformed_points, self.Q
        )

        return predicted_state, predicted_covariance

    def update(
        self,
        predicted_state: np.ndarray,
        predicted_covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """UKF更新步骤"""
        # 生成sigma点
        sigma_points = self._generate_sigma_points(predicted_state, predicted_covariance)

        # 测量变换
        measurement_points = np.array([
            self.measurement_function(sp) for sp in sigma_points
        ])

        # Unscented变换得到预测测量
        predicted_measurement, measurement_covariance = self._unscented_transform(
            measurement_points, self.R
        )

        # 计算互协方差
        deviations_state = sigma_points - predicted_state
        deviations_measurement = measurement_points - predicted_measurement

        cross_covariance = np.sum(
            self.Wc[:, np.newaxis, np.newaxis] *
            deviations_state[:, :, np.newaxis] *
            deviations_measurement[:, np.newaxis, :],
            axis=0
        )

        # 卡尔曼增益
        try:
            kalman_gain = cross_covariance @ np.linalg.inv(measurement_covariance)
        except np.linalg.LinAlgError:
            kalman_gain = np.zeros((self.state_dim, self.meas_dim))

        # 更新状态
        innovation = measurement - predicted_measurement
        updated_state = predicted_state + kalman_gain @ innovation

        # 更新协方差
        updated_covariance = predicted_covariance - kalman_gain @ measurement_covariance @ kalman_gain.T

        return updated_state, updated_covariance, kalman_gain


class AdaptiveFilter:
    """
    自适应Kalman滤波器

    根据新息自适应调整Q和R
    """

    def __init__(
        self,
        base_filter: KalmanFilter,
        window_size: int = 10,
        min_q: float = 0.1,
        max_q: float = 10.0,
        min_r: float = 1.0,
        max_r: float = 100.0,
    ):
        """
        Args:
            base_filter: 基础滤波器
            window_size: 自适应窗口大小
            min_q: 最小过程噪声
            max_q: 最大过程噪声
            min_r: 最小测量噪声
            max_r: 最大测量噪声
        """
        self.filter = base_filter
        self.window_size = window_size
        self.min_q = min_q
        self.max_q = max_q
        self.min_r = min_r
        self.max_r = max_r

        self.innovation_history = []

    def update_noise_statistics(
        self,
        innovation: np.ndarray,
        innovation_covariance: np.ndarray,
    ):
        """
        根据新息更新噪声统计

        Args:
            innovation: 新息
            innovation_covariance: 新息协方差
        """
        self.innovation_history.append(innovation)

        if len(self.innovation_history) > self.window_size:
            self.innovation_history.pop(0)

        if len(self.innovation_history) < self.window_size:
            return

        # 计算新息的样本协方差
        innovations = np.array(self.innovation_history)
        sample_cov = np.cov(innovations.T)

        # 调整R
        expected_cov = innovation_covariance
        scale_factor = np.trace(sample_cov) / np.trace(expected_covariance)
        scale_factor = np.clip(scale_factor, 0.1, 10.0)

        # 更新R
        self.filter.R = np.clip(
            self.filter.R * scale_factor,
            self.min_r,
            self.max_r
        )

        # 调整Q（基于新息的幅度）
        innovation_magnitude = np.mean(np.linalg.norm(innovations, axis=1))
        expected_magnitude = np.sqrt(np.trace(expected_cov))

        if innovation_magnitude > expected_magnitude * 1.5:
            # 新息过大，增加Q
            self.filter.Q = np.clip(
                self.filter.Q * 1.1,
                self.min_q,
                self.max_q
            )
        elif innovation_magnitude < expected_magnitude * 0.5:
            # 新息过小，减少Q
            self.filter.Q = np.clip(
                self.filter.Q * 0.9,
                self.min_q,
                self.max_q
            )

    def detect_divergence(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        threshold: float = 100.0,
    ) -> bool:
        """
        检测滤波器发散

        Args:
            state: 状态向量
            covariance: 协方差矩阵
            threshold: 发散检测阈值

        Returns:
            是否发散
        """
        # 检查协方差对角元素
        diag_cov = np.diag(covariance)

        if np.any(diag_cov > threshold):
            return True

        # 检查状态是否异常大
        if np.any(np.abs(state) > 1e6):
            return True

        # 检查协方差矩阵是否正定
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError:
            return True

        return False

    def reset_filter(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
    ):
        """重置滤波器"""
        self.filter.state = initial_state
        self.filter.covariance = initial_covariance
        self.innovation_history = []
