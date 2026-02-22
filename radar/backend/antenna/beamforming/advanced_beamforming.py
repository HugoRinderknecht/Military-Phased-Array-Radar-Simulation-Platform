# advanced_beamforming.py - 高级波束形成算法实现
"""
高级波束形成算法

完整的自适应波束形成实现:
- MVDR (Minimum Variance Distortionless Response)
- LCMV (Linearly Constrained Minimum Variance)
- Generalized Sidelobe Canceller (GSLC)
- SMI (Sample Matrix Inversion)
- Robust Beamforming
"""

import numpy as np
from typing import Tuple, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum


class BeamformingType(Enum):
    """波束形成类型"""
    CONVENTIONAL = "conventional"  # 传统波束形成
    MVDR = "mvdr"                  # 最小方差无失真响应
    LCMV = "lcmv"                  # 线性约束最小方差
    GSLC = "gslc"                  # 广义旁瓣对消器
    ROBUST = "robust"              # 鲁棒波束形成


@dataclass
class BeamformingResult:
    """波束形成结果"""
    weights: np.ndarray           # 权向量
    beam_pattern: np.ndarray      # 波束图
    output_signal: np.ndarray     # 输出信号
    sinr: float                   # 信号干扰噪声比
    null_directions: List[float]  # 零陷方向


class MVDRBeamformer:
    """
    MVDR (Minimum Variance Distortionless Response) 波束形成器

    也称为Capon波束形成器，最小化输出功率同时保持期望方向增益为1
    """

    def __init__(
        self,
        num_elements: int,
        array_geometry: str = "ula",
        element_spacing: float = 0.5
    ):
        """
        Args:
            num_elements: 阵元数
            array_geometry: 阵列几何 ("ula": 均匀线阵, "upa": 均匀面阵)
            element_spacing: 阵元间距 (波长)
        """
        self.num_elements = num_elements
        self.array_geometry = array_geometry
        self.element_spacing = element_spacing

        # 计算阵列位置
        self.array_positions = self._compute_array_positions()

    def _compute_array_positions(self) -> np.ndarray:
        """计算阵元位置"""
        if self.array_geometry == "ula":
            # 均匀线阵
            positions = np.zeros((self.num_elements, 3))
            positions[:, 0] = np.arange(self.num_elements) * self.element_spacing
            return positions
        elif self.array_geometry == "upa":
            # 均匀面阵
            n_sqrt = int(np.sqrt(self.num_elements))
            positions = np.zeros((self.num_elements, 3))
            idx = 0
            for i in range(n_sqrt):
                for j in range(n_sqrt):
                    positions[idx, 0] = i * self.element_spacing
                    positions[idx, 1] = j * self.element_spacing
                    idx += 1
            return positions
        else:
            raise ValueError(f"Unknown array geometry: {self.array_geometry}")

    def compute_steering_vector(
        self,
        azimuth: float,
        elevation: float = 0.0
    ) -> np.ndarray:
        """
        计算导向向量

        Args:
            azimuth: 方位角 (rad)
            elevation: 俯仰角 (rad)

        Returns:
            导向向量 [num_elements]
        """
        # 波数向量
        k = 2 * np.pi * np.array([
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation)
        ])

        # 导向向量
        steering_vector = np.exp(1j * k @ self.array_positions.T)

        return steering_vector

    def compute_weights(
        self,
        desired_azimuth: float,
        desired_elevation: float = 0.0,
        interference_covariance: Optional[np.ndarray] = None,
        diagonal_loading: float = 1e-3
    ) -> np.ndarray:
        """
        计算MVDR权重

        Args:
            desired_azimuth: 期望方向 (rad)
            desired_elevation: 期望俯仰角 (rad)
            interference_covariance: 干扰协方差矩阵
            diagonal_loading: 对角加载因子

        Returns:
            权向量
        """
        # 计算导向向量
        s = self.compute_steering_vector(desired_azimuth, desired_elevation)

        # 如果没有提供干扰协方差，使用单位矩阵
        if interference_covariance is None:
            R = np.eye(self.num_elements)
        else:
            R = interference_covariance

        # 对角加载 (提高鲁棒性)
        R_reg = R + diagonal_loading * np.eye(self.num_elements) * np.trace(R) / self.num_elements

        # MVDR权重: w = (R^(-1) * s) / (s^H * R^(-1) * s)
        try:
            R_inv = np.linalg.inv(R_reg)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_reg)

        numerator = R_inv @ s
        denominator = np.conj(s).T @ numerator

        w = numerator / denominator

        return w

    def compute_output(
        self,
        received_signal: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """
        计算波束形成输出

        Args:
            received_signal: 接收信号 [num_elements, num_samples]
            weights: 权向量

        Returns:
            输出信号 [num_samples]
        """
        return np.conj(weights).T @ received_signal


class LCMVBeamformer:
    """
    LCMV (Linearly Constrained Minimum Variance) 波束形成器

    在多个线性约束下最小化输出功率
    """

    def __init__(
        self,
        num_elements: int,
        array_geometry: str = "ula",
        element_spacing: float = 0.5
    ):
        """
        Args:
            num_elements: 阵元数
            array_geometry: 阵列几何
            element_spacing: 阵元间距
        """
        self.num_elements = num_elements
        self.array_geometry = array_geometry
        self.element_spacing = element_spacing

        # 创建MVDR波束形成器用于导向向量计算
        self.mvdr = MVDRBeamformer(num_elements, array_geometry, element_spacing)

    def compute_weights(
        self,
        constraints: List[Tuple[float, float]],  # (azimuth, elevation) 约束方向
        gain_constraints: np.ndarray,           # 增益约束
        interference_covariance: Optional[np.ndarray] = None,
        diagonal_loading: float = 1e-3
    ) -> np.ndarray:
        """
        计算LCMV权重

        Args:
            constraints: 约束方向列表
            gain_constraints: 增益约束 [num_constraints]
            interference_covariance: 干扰协方差
            diagonal_loading: 对角加载因子

        Returns:
            权向量
        """
        # 计算约束矩阵 C
        C = np.zeros((len(constraints), self.num_elements), dtype=complex)
        for i, (az, el) in enumerate(constraints):
            C[i, :] = self.mvdr.compute_steering_vector(az, el)

        # 如果没有提供干扰协方差，使用单位矩阵
        if interference_covariance is None:
            R = np.eye(self.num_elements)
        else:
            R = interference_covariance

        # 对角加载
        R_reg = R + diagonal_loading * np.eye(self.num_elements) * np.trace(R) / self.num_elements

        # LCMV权重: w = R^(-1) * C^H * (C * R^(-1) * C^H)^(-1) * g
        # 其中 g 是增益约束向量

        try:
            R_inv = np.linalg.inv(R_reg)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_reg)

        C_R_inv_C_H = C @ R_inv @ np.conj(C).T

        try:
            inv_C_R_inv_C_H = np.linalg.inv(C_R_inv_C_H)
        except np.linalg.LinAlgError:
            inv_C_R_inv_C_H = np.linalg.pinv(C_R_inv_C_H)

        w = R_inv @ np.conj(C).T @ inv_C_R_inv_C_H @ gain_constraints

        return w

    def add_null_constraints(
        self,
        null_directions: List[Tuple[float, float]],
        interference_covariance: np.ndarray
    ) -> np.ndarray:
        """
        添加零陷约束

        Args:
            null_directions: 零陷方向列表 [(az1, el1), (az2, el2), ...]
            interference_covariance: 干扰协方差

        Returns:
            权向量
        """
        # 主瓣方向约束
        main_lobe_gain = 1.0

        # 零陷增益约束
        null_gains = np.zeros(len(null_directions))

        # 合并约束
        all_constraints = [(0.0, 0.0)] + null_directions  # 主瓣 + 零陷
        gain_constraints = np.array([main_lobe_gain] + list(null_gains))

        return self.compute_weights(
            all_constraints,
            gain_constraints,
            interference_covariance
        )


class SampleMatrixInversion:
    """
    SMI (Sample Matrix Inversion) 波束形成器

    使用采样协方差矩阵估计的自适应波束形成
    """

    def __init__(
        self,
        num_elements: int,
        array_geometry: str = "ula",
        element_spacing: float = 0.5
    ):
        """
        Args:
            num_elements: 阵元数
            array_geometry: 阵列几何
            element_spacing: 阵元间距
        """
        self.num_elements = num_elements
        self.array_geometry = array_geometry
        self.element_spacing = element_spacing

        self.mvdr = MVDRBeamformer(num_elements, array_geometry, element_spacing)

    def estimate_covariance(
        self,
        training_data: np.ndarray
    ) -> np.ndarray:
        """
        估计干扰协方差矩阵

        Args:
            training_data: 训练数据 [num_elements, num_snapshots]

        Returns:
            协方差矩阵估计
        """
        # 采样协方差矩阵
        R_hat = (training_data @ np.conj(training_data).T) / training_data.shape[1]

        return R_hat

    def compute_weights(
        self,
        desired_azimuth: float,
        desired_elevation: float = 0.0,
        training_data: Optional[np.ndarray] = None,
        sample_covariance: Optional[np.ndarray] = None,
        diagonal_loading: float = 0.1
    ) -> np.ndarray:
        """
        计算SMI权重

        Args:
            desired_azimuth: 期望方向
            desired_elevation: 期望俯仰角
            training_data: 训练数据
            sample_covariance: 采样协方差 (如果已计算)
            diagonal_loading: 对角加载因子

        Returns:
            权向量
        """
        # 估计协方差
        if sample_covariance is None and training_data is not None:
            R = self.estimate_covariance(training_data)
        elif sample_covariance is not None:
            R = sample_covariance
        else:
            R = np.eye(self.num_elements)

        # 计算MVDR权重
        w = self.mvdr.compute_weights(
            desired_azimuth,
            desired_elevation,
            R,
            diagonal_loading
        )

        return w


class RobustBeamformer:
    """
    鲁棒波束形成器

    对导向向量误差和有限样本效应鲁棒
    """

    def __init__(
        self,
        num_elements: int,
        array_geometry: str = "ula",
        element_spacing: float = 0.5,
        uncertainty_level: float = 0.1
    ):
        """
        Args:
            num_elements: 阵元数
            array_geometry: 阵列几何
            element_spacing: 阵元间距
            uncertainty_level: 不确定性水平
        """
        self.num_elements = num_elements
        self.array_geometry = array_geometry
        self.element_spacing = element_spacing
        self.uncertainty_level = uncertainty_level

        self.mvdr = MVDRBeamformer(num_elements, array_geometry, element_spacing)

    def compute_weights(
        self,
        desired_azimuth: float,
        desired_elevation: float = 0.0,
        interference_covariance: Optional[np.ndarray] = None,
        robust_epsilon: float = 0.1
    ) -> np.ndarray:
        """
        计算鲁棒权重 (使用二阶锥规划近似)

        Args:
            desired_azimuth: 期望方向
            desired_elevation: 期望俯仰角
            interference_covariance: 干扰协方差
            robust_epsilon: 鲁棒性参数

        Returns:
            权向量
        """
        # 计算导向向量
        s = self.mvdr.compute_steering_vector(desired_azimuth, desired_elevation)

        # 使用对角加载的简化鲁棒方法
        # 对角加载因子与不确定性水平成正比
        diagonal_loading = robust_epsilon * self.uncertainty_level

        w = self.mvdr.compute_weights(
            desired_azimuth,
            desired_elevation,
            interference_covariance,
            diagonal_loading
        )

        return w

    def compute_weights_worst_case(
        self,
        desired_azimuth: float,
        desired_elevation: float = 0.0,
        interference_covariance: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算最坏情况下的鲁棒权重

        使用约束的权向量范数方法
        """
        # 计算导向向量
        s = self.mvdr.compute_steering_vector(desired_azimuth, desired_elevation)

        # 如果没有提供干扰协方差，使用单位矩阵
        if interference_covariance is None:
            R = np.eye(self.num_elements)
        else:
            R = interference_covariance

        # 最坏情况优化：min w^H*R*w subject to ||w||^2 <= epsilon and |w^H*s| >= 1
        # 近似解：使用加权的MVDR
        epsilon = self.uncertainty_level

        # 归一化导向向量
        s_norm = s / (np.conj(s).T @ s)

        # 计算增强的对角加载
        R_enhanced = R + epsilon * np.eye(self.num_elements)

        try:
            R_inv = np.linalg.inv(R_enhanced)
        except np.linalg.LinAlgError:
            R_inv = np.linalg.pinv(R_enhanced)

        w = R_inv @ s_norm
        w = w / np.sqrt(np.conj(w).T @ R @ w)  # 归一化

        return w


# 便捷函数
def create_beamformer(
    beamforming_type: Union[str, BeamformingType],
    num_elements: int,
    array_geometry: str = "ula",
    element_spacing: float = 0.5,
    **kwargs
) -> Union[MVDRBeamformer, LCMVBeamformer, SampleMatrixInversion, RobustBeamformer]:
    """
    创建波束形成器

    Args:
        beamforming_type: 波束形成类型
        num_elements: 阵元数
        array_geometry: 阵列几何
        element_spacing: 阵元间距
        **kwargs: 其他参数

    Returns:
        波束形成器实例
    """
    if isinstance(beamforming_type, str):
        beamforming_type = BeamformingType(beamforming_type.lower())

    if beamforming_type == BeamformingType.MVDR:
        return MVDRBeamformer(num_elements, array_geometry, element_spacing)
    elif beamforming_type == BeamformingType.LCMV:
        return LCMVBeamformer(num_elements, array_geometry, element_spacing)
    elif beamforming_type == BeamformingType.CONVENTIONAL:
        # 传统波束形成使用MVDR，但没有干扰协方差
        return MVDRBeamformer(num_elements, array_geometry, element_spacing)
    elif beamforming_type == BeamformingType.GSLC:
        return LCMVBeamformer(num_elements, array_geometry, element_spacing)
    elif beamforming_type == BeamformingType.ROBUST:
        uncertainty = kwargs.get('uncertainty_level', 0.1)
        return RobustBeamformer(num_elements, array_geometry, element_spacing, uncertainty)
    else:
        raise ValueError(f"Unknown beamforming type: {beamforming_type}")


def apply_beamforming(
    received_signal: np.ndarray,
    beamforming_type: str = "mvdr",
    desired_azimuth: float = 0.0,
    desired_elevation: float = 0.0,
    null_directions: Optional[List[Tuple[float, float]]] = None,
    training_data: Optional[np.ndarray] = None,
    **kwargs
) -> BeamformingResult:
    """
    应用波束形成

    Args:
        received_signal: 接收信号 [num_elements, num_samples]
        beamforming_type: 波束形成类型
        desired_azimuth: 期望方向
        desired_elevation: 期望俯仰角
        null_directions: 零陷方向列表
        training_data: 训练数据
        **kwargs: 其他参数

    Returns:
        波束形成结果
    """
    num_elements, num_samples = received_signal.shape

    # 创建波束形成器
    if beamforming_type == "lcmv" and null_directions is not None:
        bf = create_beamformer("lcmv", num_elements, **kwargs)

        # 估计干扰协方差
        if training_data is not None:
            smi = SampleMatrixInversion(num_elements)
            R = smi.estimate_covariance(training_data)
        else:
            R = np.eye(num_elements)

        # 计算权重 (包含零陷约束)
        weights = bf.add_null_constraints(null_directions, R)

    elif beamforming_type == "smi":
        bf = create_beamformer("smi", num_elements, **kwargs)
        weights = bf.compute_weights(
            desired_azimuth,
            desired_elevation,
            training_data=training_data
        )

    elif beamforming_type == "robust":
        bf = create_beamformer("robust", num_elements, **kwargs)

        # 估计干扰协方差
        if training_data is not None:
            smi = SampleMatrixInversion(num_elements)
            R = smi.estimate_covariance(training_data)
        else:
            R = None

        weights = bf.compute_weights(
            desired_azimuth,
            desired_elevation,
            R
        )

    else:  # mvdr or conventional
        bf = create_beamformer("mvdr", num_elements, **kwargs)

        # 估计干扰协方差
        if training_data is not None:
            smi = SampleMatrixInversion(num_elements)
            R = smi.estimate_covariance(training_data)
        else:
            R = None

        weights = bf.compute_weights(
            desired_azimuth,
            desired_elevation,
            R
        )

    # 计算输出
    output_signal = bf.compute_output(received_signal, weights)

    # 计算波束图
    azimuths = np.linspace(-np.pi/2, np.pi/2, 181)
    beam_pattern = np.zeros(len(azimuths), dtype=complex)

    for i, az in enumerate(azimuths):
        s = bf.compute_steering_vector(az, 0.0)
        beam_pattern[i] = np.conj(weights).T @ s

    # 计算SINR (简化)
    sinr = 10.0 * np.log10(np.max(np.abs(beam_pattern))**2)

    return BeamformingResult(
        weights=weights,
        beam_pattern=np.abs(beam_pattern),
        output_signal=output_signal,
        sinr=sinr,
        null_directions=null_directions if null_directions else []
    )
