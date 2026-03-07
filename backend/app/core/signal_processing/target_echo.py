"""
目标回波模拟模块
完整实现雷达方程和目标回波模拟

参考文档 4.4.3 节
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TargetState:
    """目标状态"""
    x: float  # X位置 (m)
    y: float  # Y位置 (m)
    z: float  # Z位置 (m)
    vx: float  # X速度 (m/s)
    vy: float  # Y速度 (m/s)
    vz: float  # Z速度 (m/s)
    rcs: float  # RCS (m²)
    time: float  # 时间戳 (s)


def calculate_radar_equation(
    transmit_power: float,
    antenna_gain_tx: float,
    antenna_gain_rx: float,
    wavelength: float,
    rcs: float,
    range_distance: float,
    system_loss: float = 1.0,
) -> float:
    """
    完整雷达方程计算

    Pr = (Pt * Gt * Gr * λ² * σ) / ((4π)³ * R⁴ * L)

    Args:
        transmit_power: 发射功率 (W)
        antenna_gain_tx: 发射天线增益 (线性)
        antenna_gain_rx: 接收天线增益 (线性)
        wavelength: 波长 (m)
        rcs: 目标RCS (m²)
        range_distance: 目标距离 (m)
        system_loss: 系统损耗 (线性，1=无损耗)

    Returns:
        接收功率 (W)
    """
    c = 3e8  # 光速

    numerator = (
        transmit_power
        * antenna_gain_tx
        * antenna_gain_rx
        * (wavelength ** 2)
        * rcs
    )
    denominator = ((4 * np.pi) ** 3) * (range_distance ** 4) * system_loss

    return numerator / denominator


def calculate_two_way_range(
    target_pos: Tuple[float, float, float],
    radar_pos: Tuple[float, float, float] = (0, 0, 0),
) -> float:
    """
    计算雷达到目标的双向距离

    Args:
        target_pos: 目标位置 (x, y, z)
        radar_pos: 雷达位置 (x, y, z)

    Returns:
        双向距离 (m)
    """
    tx, ty, tz = target_pos
    rx, ry, rz = radar_pos

    # 单向距离
    one_way_range = np.sqrt((tx - rx)**2 + (ty - ry)**2 + (tz - rz)**2)

    # 双向距离
    return 2 * one_way_range


def calculate_doppler_shift(
    target_velocity: Tuple[float, float, float],
    target_position: Tuple[float, float, float],
    radar_position: Tuple[float, float, float] = (0, 0, 0),
    wavelength: float = 0.03,
) -> float:
    """
    计算多普勒频移

    fd = 2 * v_radial / λ

    Args:
        target_velocity: 目标速度 (vx, vy, vz) (m/s)
        target_position: 目标位置 (x, y, z) (m)
        radar_position: 雷达位置 (x, y, z) (m)
        wavelength: 波长 (m)

    Returns:
        多普勒频移 (Hz)
    """
    vx, vy, vz = target_velocity
    tx, ty, tz = target_position
    rx, ry, rz = radar_position

    # 计算径向速度
    dx = tx - rx
    dy = ty - ry
    dz = tz - rz
    range_distance = np.sqrt(dx**2 + dy**2 + dz**2)

    if range_distance == 0:
        return 0

    # 径向速度 = 速度向量在视线方向上的投影
    v_radial = (vx * dx + vy * dy + vz * dz) / range_distance

    # 多普勒频移
    fd = 2 * v_radial / wavelength

    return fd


def generate_swerling_rcs(
    model: Literal[1, 2, 3, 4],
    mean_rcs: float,
    num_pulses: int,
    num_scans: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    生成Swerling模型的RCS序列

    Args:
        model: Swerling模型类型 (1/2/3/4)
        mean_rcs: 平均RCS (m²)
        num_pulses: 每次扫描的脉冲数
        num_scans: 扫描次数
        rng: 随机数生成器

    Returns:
        RCS数组 (num_scans, num_pulses)
    """
    if rng is None:
        rng = np.random.default_rng()

    rcs_values = np.zeros((num_scans, num_pulses))

    if model == 1:
        # Swerling I: 慢起伏，脉冲间非相关
        # 指数分布（自由度=2的卡方分布）
        for scan in range(num_scans):
            # 每次扫描一个RCS值
            rcs = rng.exponential(mean_rcs)
            rcs_values[scan, :] = rcs

    elif model == 2:
        # Swerling II: 快起伏，脉冲间非相关
        # 每个脉冲独立同分布
        rcs_values = rng.exponential(mean_rcs, (num_scans, num_pulses))

    elif model == 3:
        # Swerling III: 慢起伏，脉冲间相关
        # 自由度=4的卡方分布
        for scan in range(num_scans):
            # 每次扫描一个RCS值
            chi2 = rng.chisquare(4, 1) / 2
            rcs = mean_rcs * chi2[0]
            rcs_values[scan, :] = rcs

    elif model == 4:
        # Swerling IV: 快起伏，脉冲间相关
        # 每个脉冲服从自由度=4的卡方分布
        chi2 = rng.chisquare(4, (num_scans, num_pulses)) / 2
        rcs_values = mean_rcs * chi2

    return rcs_values


def simulate_target_echo(
    transmitted_signal: np.ndarray,
    target_state: TargetState,
    radar_params: dict,
    sampling_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    模拟单个目标的回波信号

    Args:
        transmitted_signal: 发射信号（复数）
        target_state: 目标状态
        radar_params: 雷达参数字典
            - transmit_power: 发射功率 (W)
            - antenna_gain: 天线增益（可查询方向图）
            - wavelength: 波长 (m)
            - system_loss: 系统损耗
        sampling_rate: 采样率 (Hz)
        rng: 随机数生成器

    Returns:
        回波信号（与发射信号同长度）
    """
    c = 3e8

    if rng is None:
        rng = np.random.default_rng()

    # 计算目标距离
    target_pos = (target_state.x, target_state.y, target_state.z)
    range_distance = calculate_two_way_range(target_pos) / 2  # 单向距离

    # 计算时延
    time_delay = 2 * range_distance / c
    delay_samples = int(time_delay * sampling_rate)

    # 计算多普勒频移
    target_vel = (target_state.vx, target_state.vy, target_state.vz)
    doppler_shift = calculate_doppler_shift(
        target_vel, target_pos, wavelength=radar_params["wavelength"]
    )

    # 计算接收功率（雷达方程）
    # 假设发射和接收增益相同
    received_power = calculate_radar_equation(
        transmit_power=radar_params["transmit_power"],
        antenna_gain_tx=radar_params["antenna_gain"],
        antenna_gain_rx=radar_params["antenna_gain"],
        wavelength=radar_params["wavelength"],
        rcs=target_state.rcs,
        range_distance=range_distance,
        system_loss=radar_params.get("system_loss", 1.0),
    )

    # 计算幅度因子
    # Pr = |A|²，所以 A = sqrt(Pr)
    amplitude = np.sqrt(received_power)

    # 生成回波信号
    num_samples = len(transmitted_signal)
    echo = np.zeros(num_samples, dtype=complex)

    if delay_samples < num_samples:
        # 延迟信号
        delayed_signal = transmitted_signal[:num_samples - delay_samples]

        # 应用多普勒频移
        time = np.arange(len(delayed_signal)) / sampling_rate
        doppler_phase = 2 * np.pi * doppler_shift * time
        doppler_signal = delayed_signal * np.exp(1j * doppler_phase)

        # 应用幅度
        echo[delay_samples:delay_samples + len(doppler_signal)] = amplitude * doppler_signal

    return echo


def simulate_multi_target_echo(
    transmitted_signal: np.ndarray,
    target_states: List[TargetState],
    radar_params: dict,
    sampling_rate: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    模拟多个目标的回波信号

    Args:
        transmitted_signal: 发射信号
        target_states: 目标状态列表
        radar_params: 雷达参数
        sampling_rate: 采样率
        rng: 随机数生成器

    Returns:
        叠加后的回波信号
    """
    num_samples = len(transmitted_signal)
    total_echo = np.zeros(num_samples, dtype=complex)

    for target_state in target_states:
        echo = simulate_target_echo(
            transmitted_signal,
            target_state,
            radar_params,
            sampling_rate,
            rng,
        )
        total_echo += echo

    return total_echo


def calculate_range_folded(
    true_range: float,
    max_unambiguous_range: float,
) -> float:
    """
    计算距离折叠后的显示距离

    Args:
        true_range: 真实距离 (m)
        max_unambiguous_range: 最大无模糊距离 (m)

    Returns:
        折叠后的显示距离 (m)
    """
    return true_range % max_unambiguous_range


def calculate_doppler_folded(
    true_doppler: float,
    prf: float,
) -> float:
    """
    计算多普勒折叠后的显示频率

    Args:
        true_doppler: 真实多普勒频移 (Hz)
        prf: 脉冲重复频率 (Hz)

    Returns:
        折叠后的多普勒频率 (Hz)
    """
    # 折叠到 [-PRF/2, PRF/2)
    nyquist = prf / 2
    folded = ((true_doppler + nyquist) % prf) - nyquist
    return folded
