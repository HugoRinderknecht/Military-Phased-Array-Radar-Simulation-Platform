import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from numba import njit, float64, int32
from models.environment import Environment
from models.target import Target


@dataclass
class RadarSystem:
    radar_area: float
    tr_components: int
    radar_power: float
    frequency: float = 10e9
    antenna_elements: int = 64
    beam_width: float = 2.0
    scan_rate: float = 6.0

    def __post_init__(self):
        self.wavelength = 3e8 / self.frequency
        self.element_spacing = self.wavelength / 2
        self.gain = 10 * np.log10(self.tr_components * 0.7)

    @staticmethod
    @njit(float64(float64, float64, float64, float64, float64, float64))
    def calculate_received_power(transmit_power, transmit_gain, receive_gain, wavelength,
                                 target_rcs, distance):
        """计算雷达接收功率 (Numba优化)"""
        numerator = transmit_power * transmit_gain * receive_gain * wavelength ** 2 * target_rcs
        denominator = (4 * np.pi) ** 3 * distance ** 4
        return numerator / denominator

    @staticmethod
    @njit(float64[:](float64[:], int32, float64, float64))
    def calculate_array_factor(angles, num_elements, element_spacing, wavelength):
        """计算阵列天线方向图 (Numba优化)"""
        result = np.zeros_like(angles)
        for i in range(len(angles)):
            theta = np.deg2rad(angles[i])
            psi = 2 * np.pi * element_spacing * np.sin(theta) / wavelength
            if abs(psi) < 1e-10:  # 避免除以零
                result[i] = 1.0
            else:
                result[i] = np.abs(np.sin(num_elements * psi / 2) /
                                   (num_elements * np.sin(psi / 2)))
        return result


@dataclass
class SimulationParameters:
    radar_system: RadarSystem
    environment: Environment
    targets: List[Target]
    simulation_time: float = 10.0
    time_step: float = 0.06
    monte_carlo_runs: int = 1


# 预编译关键函数
RadarSystem.calculate_received_power(1e6, 30, 30, 0.03, 1.0, 1000)
angles = np.linspace(-60, 60, 181)
RadarSystem.calculate_array_factor(angles, 64, 0.015, 0.03)
