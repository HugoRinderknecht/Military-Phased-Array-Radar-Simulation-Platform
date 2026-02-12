# simulator.py - 环境模拟器
"""
本模块实现雷达环境的综合模拟。

环境模拟器负责：
- 管理所有目标
- 计算目标回波
- 生成杂波
- 模拟干扰
- 考虑传播效应
"""

import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from radar.common.logger import get_logger
from radar.common.types import (
    Position3D, Velocity3D, TargetType, MotionModel,
    Plot, BeamStatus, AzimuthElevation
)
from radar.common.constants import PhysicsConstants


@dataclass
class TargetInfo:
    """
    目标信息

    Attributes:
        id: 目标ID
        type: 目标类型
        position: 位置 [米]
        velocity: 速度 [米/秒]
        rcs: RCS [m²]
        is_active: 是否激活
    """
    id: int
    type: TargetType
    position: Position3D
    velocity: Velocity3D
    rcs: float
    is_active: bool = True


@dataclass
class BeamInfo:
    """
    波束信息

    Attributes:
        azimuth: 方位角 [弧度]
        elevation: 俯仰角 [弧度]
        gain: 波束增益 [线性]
        task_type: 任务类型
    """
    azimuth: float
    elevation: float
    gain: float = 1.0
    task_type: str = "search"


@dataclass
class EchoData:
    """
    回波数据

    Attributes:
        beam_info: 波束信息
        echo_signal: 回波复数信号
        clutter_signal: 杂波信号
        noise_signal: 噪声信号
        combined_signal: 合成信号
    """
    beam_info: BeamInfo
    echo_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    clutter_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    noise_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    combined_signal: np.ndarray = field(default_factory=lambda: np.array([]))


class EnvironmentSimulator:
    """
    环境模拟器

    模拟雷达环境中的所有目标、杂波和干扰。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化环境模拟器

        Args:
            config: 配置字典
        """
        self._logger = get_logger("environment")

        # 配置
        self._config = config or {}

        # 目标列表
        self._targets: Dict[int, TargetInfo] = {}

        # 目标ID计数器
        self._target_id_counter = 0

        # 杂波参数
        self._clutter_enabled = self._config.get('clutter_enabled', True)
        self._clutter_type = self._config.get('clutter_type', 'ground')

        # 干扰参数
        self._jamming_enabled = self._config.get('jamming_enabled', False)

        # 传播参数
        self._propagation_loss_enabled = self._config.get('propagation_loss', True)

        self._logger.info("环境模拟器初始化完成")

    async def initialize(self) -> None:
        """初始化环境模拟器"""
        self._logger.info("初始化环境...")

        # 加载默认场景（如果有）
        default_scenario = self._config.get('default_scenario')
        if default_scenario:
            await self.load_scenario(default_scenario)

    async def update(self, delta_time: float) -> None:
        """
        更新环境状态

        Args:
            delta_time: 时间步长 [秒]
        """
        # 更新所有目标位置
        for target in self._targets.values():
            if not target.is_active:
                continue

            # 更新位置: position = position + velocity * dt
            target.position.x += target.velocity.vx * delta_time
            target.position.y += target.velocity.vy * delta_time
            target.position.z += target.velocity.vz * delta_time

        # 移除超出范围的目标
        await self._remove_out_of_range_targets()

    async def add_target(self,
                       target_type: TargetType,
                       position: Position3D,
                       velocity: Velocity3D,
                       rcs: float) -> int:
        """
        添加目标

        Args:
            target_type: 目标类型
            position: 初始位置
            velocity: 速度
            rcs: RCS [m²]

        Returns:
            目标ID
        """
        self._target_id_counter += 1
        target_id = self._target_id_counter

        target = TargetInfo(
            id=target_id,
            type=target_type,
            position=position,
            velocity=velocity,
            rcs=rcs,
            is_active=True
        )

        self._targets[target_id] = target
        self._logger.info(f"添加目标: ID={target_id}, type={target_type.value}, rcs={rcs}m²")

        return target_id

    def remove_target(self, target_id: int) -> bool:
        """
        移除目标

        Args:
            target_id: 目标ID

        Returns:
            是否成功移除
        """
        if target_id in self._targets:
            del self._targets[target_id]
            self._logger.info(f"移除目标: ID={target_id}")
            return True
        return False

    def get_target(self, target_id: int) -> Optional[TargetInfo]:
        """获取目标信息"""
        return self._targets.get(target_id)

    def get_all_targets(self) -> List[TargetInfo]:
        """获取所有目标"""
        return list(self._targets.values())

    def get_active_targets(self) -> List[TargetInfo]:
        """获取所有活动目标"""
        return [t for t in self._targets.values() if t.is_active]

    async def _remove_out_of_range_targets(self) -> None:
        """移除超出范围的目标"""
        max_range = self._config.get('max_range', 500000)  # 500km

        to_remove = []
        for target in self._targets.values():
            range_val = np.sqrt(
                target.position.x**2 +
                target.position.y**2 +
                target.position.z**2
            )
            if range_val > max_range:
                to_remove.append(target.id)

        for target_id in to_remove:
            self.remove_target(target_id)

    async def generate_echo(self, beam_info: BeamInfo,
                          sample_rate: float = 20e6,
                          pulse_width: float = 10e-6) -> EchoData:
        """
        生成回波数据

        Args:
            beam_info: 波束信息
            sample_rate: 采样率 [Hz]
            pulse_width: 脉冲宽度 [秒]

        Returns:
            回波数据
        """
        # 计算采样点数
        n_samples = int(sample_rate * pulse_width)

        # 初始化信号
        echo_signal = np.zeros(n_samples, dtype=np.complex128)
        clutter_signal = np.zeros(n_samples, dtype=np.complex128)
        noise_signal = np.zeros(n_samples, dtype=np.complex128)

        # 计算每个目标的回波
        for target in self.get_active_targets():
            # 计算目标在波束坐标系中的位置
            target_range, target_az, target_el = self._position_to_radar(target.position)

            # 判断目标是否在波束覆盖范围内
            if self._is_target_in_beam(target_az, target_el, beam_info):
                # 计算接收功率
                power = self._calculate_received_power(
                    target_range, target.rcs, beam_info.gain
                )

                # 生成目标回波（单频信号）
                # 距离对应的延迟
                delay_samples = int(2 * target_range / PhysicsConstants.C * sample_rate)

                if 0 <= delay_samples < n_samples:
                    # 多普勒频移
                    # fd = -2 * v / lambda
                    wavelength = PhysicsConstants.C / 10e9  # X波段
                    radial_velocity = (
                        target.position.x * target.velocity.vx +
                        target.position.y * target.velocity.vy +
                        target.position.z * target.velocity.vz
                    ) / target_range
                    doppler_freq = -2 * radial_velocity / wavelength

                    # 生成回波
                    t = np.arange(n_samples) / sample_rate
                    phase = 2 * np.pi * doppler_freq * t
                    target_echo = np.sqrt(power) * np.exp(1j * phase)

                    # 添加延迟
                    echo_signal[delay_samples:] += target_echo[:n_samples - delay_samples]

        # 生成杂波
        if self._clutter_enabled:
            clutter_signal = await self._generate_clutter(n_samples, beam_info)

        # 生成噪声
        noise_signal = self._generate_noise(n_samples)

        # 合成信号
        combined_signal = echo_signal + clutter_signal + noise_signal

        return EchoData(
            beam_info=beam_info,
            echo_signal=echo_signal,
            clutter_signal=clutter_signal,
            noise_signal=noise_signal,
            combined_signal=combined_signal
        )

    def _position_to_radar(self, position: Position3D) -> tuple:
        """
        将位置转换为雷达坐标

        Args:
            position: 目标位置

        Returns:
            (range, azimuth, elevation)
        """
        range_val = np.sqrt(position.x**2 + position.y**2 + position.z**2)
        azimuth = np.arctan2(position.x, position.y)
        elevation = np.arcsin(position.z / range_val) if range_val > 0 else 0

        return range_val, azimuth, elevation

    def _is_target_in_beam(self, target_az: float, target_el: float,
                            beam_info: BeamInfo) -> bool:
        """
        判断目标是否在波束内

        Args:
            target_az: 目标方位角
            target_el: 目标俯仰角
            beam_info: 波束信息

        Returns:
            是否在波束内
        """
        # 波束宽度（假设3dB波束宽度为5度）
        beamwidth_az = np.deg2rad(5.0)
        beamwidth_el = np.deg2rad(5.0)

        az_diff = np.abs(target_az - beam_info.azimuth)
        el_diff = np.abs(target_el - beam_info.elevation)

        # 考虑角度周期性
        if az_diff > np.pi:
            az_diff = 2 * np.pi - az_diff

        return az_diff <= beamwidth_az / 2 and el_diff <= beamwidth_el / 2

    def _calculate_received_power(self, range_val: float, rcs: float,
                               beam_gain: float) -> float:
        """
        计算接收功率

        使用雷达方程：Pr = (Pt * G² * λ² * σ) / ((4π)³ * R⁴)

        Args:
            range_val: 距离 [米]
            rcs: RCS [m²]
            beam_gain: 波束增益（线性）

        Returns:
            接收功率 [W]
        """
        # 雷达参数
        Pt = 100e3  # 发射功率 100kW
        wavelength = PhysicsConstants.C / 10e9  # X波段波长
        G = beam_gain

        # 雷达方程
        power = (Pt * G**2 * wavelength**2 * rcs) / \
                 ((4 * np.pi)**3 * range_val**4)

        # 考虑系统损耗
        losses = 8.0  # 8dB
        power_lin = power * 10**(-losses / 10)

        return power_lin

    async def _generate_clutter(self, n_samples: int,
                               beam_info: BeamInfo) -> np.ndarray:
        """
        生成杂波信号

        Args:
            n_samples: 采样点数
            beam_info: 波束信息

        Returns:
            杂波信号
        """
        # 简化实现：生成相关高斯噪声
        # 实际应根据杂波类型（地/海/气象）使用对应模型

        # 杂波功率（假设比目标低20dB）
        clutter_power = 1e-6

        # 生成相关杂波
        # 使用指数相关模型
        rho = 0.9  # 相关系数
        white_noise = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        clutter = np.zeros(n_samples, dtype=np.complex128)
        clutter[0] = white_noise[0]

        for i in range(1, n_samples):
            clutter[i] = rho * clutter[i-1] + \
                         np.sqrt(1 - rho**2) * white_noise[i]

        return clutter * np.sqrt(clutter_power)

    def _generate_noise(self, n_samples: float) -> np.ndarray:
        """
        生成热噪声

        Args:
            n_samples: 采样点数

        Returns:
            噪声信号
        """
        # 热噪声功率
        k = PhysicsConstants.K
        T0 = 290  # 标准温度
        B = 10e6  # 带宽
        F = 3.0  # 噪声系数 3dB

        noise_power = k * T0 * B * 10**(F/10)

        # 生成复高斯白噪声
        noise = np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        noise = noise * np.sqrt(noise_power / 2)

        return noise

    async def load_scenario(self, scenario_data: Dict[str, Any]) -> None:
        """
        加载场景

        Args:
            scenario_data: 场景数据
        """
        self._logger.info("加载场景...")

        # 清空现有目标
        self._targets.clear()

        # 添加场景中的目标
        targets_data = scenario_data.get('targets', [])
        for target_data in targets_data:
            await self.add_target(
                target_type=TargetType(target_data.get('type', 'aircraft')),
                position=Position3D(
                    target_data['position']['x'],
                    target_data['position']['y'],
                    target_data['position']['z']
                ),
                velocity=Velocity3D(
                    target_data['velocity']['vx'],
                    target_data['velocity']['vy'],
                    target_data['velocity']['vz']
                ),
                rcs=target_data.get('rcs', 10.0)
            )

        self._logger.info(f"场景加载完成: {len(self._targets)}个目标")

    def get_scenario_summary(self) -> Dict[str, Any]:
        """获取场景摘要"""
        active = self.get_active_targets()

        return {
            'total_targets': len(self._targets),
            'active_targets': len(active),
            'targets_by_type': self._count_by_type(active),
        }

    def _count_by_type(self, targets: List[TargetInfo]) -> Dict[str, int]:
        """按类型统计目标"""
        count = {}
        for target in targets:
            type_str = target.type.value
            count[type_str] = count.get(type_str, 0) + 1
        return count


__all__ = [
    "TargetInfo",
    "BeamInfo",
    "EchoData",
    "EnvironmentSimulator",
]
