# antenna_system.py - 天线系统
"""
本模块实现相控阵天线系统。

天线系统负责：
- 阵列建模
- 波束形成
- 波束控制
- 方向图计算
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from radar.common.logger import get_logger
from radar.common.types import AzimuthElevation
from radar.common.constants import PhysicsConstants, MathConstants


@dataclass
class ArrayGeometry:
    """
    阵列几何结构

    Attributes:
        shape: 阵列形状 ('linear', 'planar', 'circular')
        size: 阵列尺寸 (M, N) 或 (M,)
        spacing: 阵元间距 [米]
    """
    shape: str = 'planar'
    size: tuple = (32, 32)
    spacing: float = 0.015  # 半波长


@dataclass
class BeamParameters:
    """
    波束参数

    Attributes:
        azimuth: 方位角 [弧度]
        elevation: 俯仰角 [弧度]
        width_az: 方位波束宽度 [弧度]
        width_el: 俯仰波束宽度 [弧度]
        gain: 波束增益 [线性]
        sidelobe_level: 旁瓣电平 [dB]
    """
    azimuth: float = 0.0
    elevation: float = 0.0
    width_az: float = np.deg2rad(5.0)
    width_el: float = np.deg2rad(5.0)
    gain: float = 1.0
    sidelobe_level: float = -25.0


class PhasedArrayAntenna:
    """
    相控阵天线系统

    实现平面相控阵天线的建模和波束形成。
    """

    def __init__(self, geometry: ArrayGeometry,
                 frequency: float = 10e9):
        """
        初始化相控阵天线

        Args:
            geometry: 阵列几何结构
            frequency: 工作频率 [Hz]
        """
        self._logger = get_logger("antenna")
        self.geometry = geometry
        self.frequency = frequency
        self.wavelength = PhysicsConstants.C / frequency

        # 计算阵列位置
        self._element_positions = self._compute_element_positions()

        # 初始化波束参数
        self.current_beam = BeamParameters()

        self._logger.info(
            f"相控阵天线初始化: {geometry.shape}, "
            f"size={geometry.size}, freq={frequency/1e9}GHz"
        )

    def _compute_element_positions(self) -> np.ndarray:
        """
        计算阵元位置

        Returns:
            位置数组 [N_elements, 3] (x, y, z)
        """
        shape = self.geometry.shape
        spacing = self.geometry.spacing

        if shape == 'linear':
            # 线阵：沿y轴排列
            n_elements = self.geometry.size[0]
            positions = np.zeros((n_elements, 3))
            positions[:, 1] = (np.arange(n_elements) -
                             (n_elements - 1) / 2) * spacing
            return positions

        elif shape == 'planar':
            # 面阵：矩形排列
            m, n = self.geometry.size
            n_elements = m * n
            positions = np.zeros((n_elements, 3))

            # y方向（方位）
            y_coords = (np.arange(m) - (m - 1) / 2) * spacing
            # x方向（俯仰）
            x_coords = (np.arange(n) - (n - 1) / 2) * spacing

            idx = 0
            for i in range(m):
                for j in range(n):
                    positions[idx] = [x_coords[j], y_coords[i], 0]
                    idx += 1

            return positions

        elif shape == 'circular':
            # 圆阵
            n_elements = self.geometry.size[0]
            radius = (n_elements * spacing) / (2 * np.pi)
            positions = np.zeros((n_elements, 3))

            angles = np.linspace(0, 2*np.pi, n_elements, endpoint=False)
            positions[:, 0] = radius * np.cos(angles)
            positions[:, 1] = radius * np.sin(angles)
            positions[:, 2] = 0

            return positions

        else:
            raise ValueError(f"未知的阵列形状: {shape}")

    def compute_array_factor(self, az: float, el: float) -> np.ndarray:
        """
        计算阵列因子

        Args:
            az: 目标方位角 [弧度]
            el: 目标俯仰角 [弧度]

        Returns:
            阵列因子（复数）
        """
        # 波数矢量
        k = 2 * np.pi / self.wavelength

        # 波束方向矢量
        u = np.sin(el) * np.cos(az)  # x方向余弦
        v = np.sin(el) * np.sin(az)  # y方向余弦
        w = np.cos(el)              # z方向余弦

        # 计算每个阵元的相位差
        # ψ = k * (x*u + y*v + z*w)
        phase_shifts = k * (
            self._element_positions[:, 0] * u +
            self._element_positions[:, 1] * v +
            self._element_positions[:, 2] * w
        )

        # 计算阵列因子（所有阵元贡献之和）
        # AF = Σ exp(j*ψ)
        array_factor = np.sum(np.exp(1j * phase_shifts))

        return array_factor

    def compute_beam_pattern(self,
                          az_range: Tuple[float, float] = (-np.pi/2, np.pi/2),
                          el_range: Tuple[float, float] = (-np.pi/4, np.pi/4),
                          n_az: int = 181,
                          n_el: int = 91) -> np.ndarray:
        """
        计算波束方向图

        Args:
            az_range: 方位角范围 [弧度]
            el_range: 俯仰角范围 [弧度]
            n_az: 方位角采样数
            n_el: 俯仰角采样数

        Returns:
            方向图 [n_az, n_el] (dB)
        """
        # 生成角度网格
        az_angles = np.linspace(az_range[0], az_range[1], n_az)
        el_angles = np.linspace(el_range[0], el_range[1], n_el)

        # 计算每个方向的阵列因子
        pattern = np.zeros((n_az, n_el))

        for i, az in enumerate(az_angles):
            for j, el in enumerate(el_angles):
                af = self.compute_array_factor(az, el)
                pattern[i, j] = np.abs(af)

        # 转换为dB
        pattern_db = 20 * np.log10(pattern + 1e-10)
        pattern_db = pattern_db - np.max(pattern_db)  # 归一化

        return pattern_db

    def steer_beam(self, az: float, el: float) -> BeamParameters:
        """
        控制波束指向

        Args:
            az: 方位角 [弧度]
            el: 俯仰角 [弧度]

        Returns:
            波束参数
        """
        # 更新当前波束参数
        self.current_beam.azimuth = az
        self.current_beam.elevation = el

        # 计算波束增益
        # 假设均匀加权的矩形阵列
        n_elements = np.prod(self.geometry.size)
        self.current_beam.gain = n_elements  # 理想增益

        # 计算3dB波束宽度
        # θ_3dB ≈ λ/L (弧度)
        if self.geometry.shape == 'planar':
            m, n = self.geometry.size
            az_width = self.wavelength / (n * self.geometry.spacing)
            el_width = self.wavelength / (m * self.geometry.spacing)
        else:
            az_width = self.wavelength / (
                self.geometry.size[0] * self.geometry.spacing
            )
            el_width = az_width

        self.current_beam.width_az = az_width
        self.current_beam.width_el = el_width

        self._logger.debug(
            f"波束控制: az={np.rad2deg(az):.1f}°, "
            f"el={np.rad2deg(el):.1f}°, "
            f"gain={20*np.log10(self.current_beam.gain):.1f}dB"
        )

        return self.current_beam

    def get_beam_gain(self, az: float, el: float) -> float:
        """
        获取指定方向的增益

        Args:
            az: 方位角 [弧度]
            el: 俯仰角 [弧度]

        Returns:
            增益 [线性]
        """
        af = self.compute_array_factor(az, el)
        return np.abs(af)

    def compute_sum_beam(self, az: float, el: float) -> np.ndarray:
        """
        计算和波束权系数

        Args:
            az: 目标方位角 [弧度]
            el: 目标俯仰角 [弧度]

        Returns:
            权系数 [N_elements]
        """
        # 计算期望相位
        k = 2 * np.pi / self.wavelength

        # 波束方向
        u = np.sin(el) * np.cos(az)
        v = np.sin(el) * np.sin(az)
        w = np.cos(el)

        # 期望相位：使波束指向(az, el)
        desired_phase = k * (
            self._element_positions[:, 0] * u +
            self._element_positions[:, 1] * v +
            self._element_positions[:, 2] * w
        )

        # 权系数：w = exp(-j*ψ)
        weights = np.exp(-1j * desired_phase)

        # 归一化
        weights = weights / np.linalg.norm(weights)

        return weights

    def apply_weights(self, weights: np.ndarray,
                   signal: np.ndarray) -> np.ndarray:
        """
        应用权系数到信号

        Args:
            weights: 权系数 [N_elements]
            signal: 阵元信号 [N_elements, ...]

        Returns:
            波束形成后的信号
        """
        # 波束形成：y = w^H * x
        # weights需要共轭
        output = np.dot(np.conj(weights), signal.T)

        return output


__all__ = [
    "ArrayGeometry",
    "BeamParameters",
    "PhasedArrayAntenna",
]
