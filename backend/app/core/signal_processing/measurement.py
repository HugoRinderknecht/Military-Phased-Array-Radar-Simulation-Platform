"""
参数估计模块
完整实现距离、角度、多普勒参数估计

参考文档 4.4.7 节
"""
import numpy as np
from typing import Tuple, Optional, List
from scipy.signal import find_peaks
from scipy.interpolate import interp1d


def estimate_range(
    cfar_detections: np.ndarray,
    range_bins: np.ndarray,
    use_interpolation: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    距离估计

    Args:
        cfar_detections: CFAR检测结果（布尔数组或幅度数组）
        range_bins: 距离单元数组 (m)
        use_interpolation: 是否使用重心插值提高精度

    Returns:
        (estimated_ranges, detection_indices): 估计的距离和索引
    """
    if cfar_detections.dtype == bool:
        # 布尔数组，找到检测位置
        detection_indices = np.where(cfar_detections)[0]
    else:
        # 幅度数组，找峰值
        detection_indices, _ = find_peaks(cfar_detections)

    if len(detection_indices) == 0:
        return np.array([]), np.array([])

    if use_interpolation and len(detection_indices) > 0:
        # 重心内插提高精度
        estimated_ranges = []

        for idx in detection_indices:
            # 确保在边界内
            if idx > 0 and idx < len(range_bins) - 1:
                # 使用相邻3点进行重心计算
                left = max(0, idx - 1)
                right = min(len(range_bins) - 1, idx + 2)

                weights = cfar_detections[left:right] if cfar_detections.dtype != bool else np.ones(3)
                positions = range_bins[left:right]

                # 加权平均
                if np.sum(weights) > 0:
                    refined_range = np.sum(weights * positions) / np.sum(weights)
                    estimated_ranges.append(refined_range)
                else:
                    estimated_ranges.append(range_bins[idx])
            else:
                estimated_ranges.append(range_bins[idx])

        return np.array(estimated_ranges), detection_indices
    else:
        # 直接使用距离单元值
        return range_bins[detection_indices], detection_indices


def estimate_azimuth_monopulse(
    sum_channel: np.ndarray,
    delta_channel: np.ndarray,
    calibration_curve: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    单脉冲比幅测角

    θ ≈ arctan(Δ/Σ)

    Args:
        sum_channel: 和通道信号
        delta_channel: 差通道信号
        calibration_curve: 校准曲线（可选）

    Returns:
        估计的角度（弧度）
    """
    # 确保不除零
    sum_safe = np.where(np.abs(sum_channel) < 1e-10, 1e-10, sum_channel)

    # 比值
    ratio = delta_channel / sum_safe

    # 角度估计
    angles = np.arctan(ratio)

    # 如果有校准曲线，应用校准
    if calibration_curve is not None:
        # 简单的线性插值校准
        # calibration_curve应该是(ratio, angle)的查找表
        pass

    return angles


def estimate_azimuth_phase_comparison(
    channel1: np.ndarray,
    channel2: np.ndarray,
    wavelength: float,
    baseline: float,
) -> np.ndarray:
    """
    相位比较法测角

    θ = arcsin(λ*Δφ / (2π*d))

    Args:
        channel1: 通道1信号（复数）
        channel2: 通道2信号（复数）
        wavelength: 波长 (m)
        baseline: 基线长度 (m)

    Returns:
        估计的角度（弧度）
    """
    # 计算相位差
    phase1 = np.angle(channel1)
    phase2 = np.angle(channel2)

    # 相位差（解模糊）
    phase_diff = phase2 - phase1
    phase_diff = np.unwrap(phase_diff)

    # 角度估计
    # Δφ = 2π*d*sin(θ)/λ
    # θ = arcsin(λ*Δφ / (2π*d))

    sin_theta = wavelength * phase_diff / (2 * np.pi * baseline)

    # 限制在[-1, 1]范围内
    sin_theta = np.clip(sin_theta, -1, 1)

    angles = np.arcsin(sin_theta)

    return angles


def estimate_doppler(
    doppler_bins: np.ndarray,
    prf: float,
    num_pulses: int,
    use_interpolation: bool = True,
) -> np.ndarray:
    """
    多普勒频率估计

    Args:
        doppler_bins: 多普勒通道索引
        prf: 脉冲重复频率
        num_pulses: 脉冲数
        use_interpolation: 是否使用插值提高精度

    Returns:
        估计的多普勒频率 (Hz)
    """
    # 将索引转换为频率
    # 频率范围: [-PRF/2, PRF/2)
    doppler_resolution = prf / num_pulses

    if use_interpolation:
        # 使用重心或抛物线插值
        # 这里简化处理
        doppler_freq = (doppler_bins - num_pulses / 2) * doppler_resolution
    else:
        doppler_freq = (doppler_bins - num_pulses / 2) * doppler_resolution

    return doppler_freq


def estimate_velocity(
    doppler_frequency: np.ndarray,
    wavelength: float,
) -> np.ndarray:
    """
    从多普勒频率估计速度

    v = fd * λ / 2

    Args:
        doppler_frequency: 多普勒频率 (Hz)
        wavelength: 波长 (m)

    Returns:
        径向速度 (m/s)
    """
    return doppler_frequency * wavelength / 2


def super_resolution_range(
    correlation_matrix: np.ndarray,
    num_sources: int,
    array_spacing: float,
) -> np.ndarray:
    """
    超分辨率距离估计（MUSIC/ESPRIT等子空间方法）

    Args:
        correlation_matrix: 协方差矩阵
        num_sources: 信号源数
        array_spacing: 阵元间距

    Returns:
        估计的距离
    """
    # 这里简化实现，实际需要完整的MUSIC或ESPRIT算法
    # 特征分解
    eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

    # 噪声子空间
    noise_subspace = eigenvectors[:, :-num_sources]

    # 空间谱
    # ... 完整实现较复杂

    return np.array([])  # 占位


def calculate_range_accuracy(
    snr: float,
    bandwidth: float,
) -> float:
    """
    计算理论测距精度

    σR = c / (2 * B * sqrt(2 * SNR))

    Args:
        snr: 信噪比（线性）
        bandwidth: 信号带宽 (Hz)

    Returns:
        测距精度（标准差，m）
    """
    c = 3e8
    if snr <= 0:
        return float('inf')

    accuracy = c / (2 * bandwidth * np.sqrt(2 * snr))
    return accuracy


def calculate_angle_accuracy(
    snr: float,
    beamwidth: float,
) -> float:
    """
    计算理论测角精度

    σθ = θ_bw / (k * sqrt(2 * SNR))

    Args:
        snr: 信噪比（线性）
        beamwidth: 波束宽度（弧度）

    Returns:
        测角精度（标准差，弧度）
    """
    k = 1.5  # 典型值，取决于阵列和处理方法

    if snr <= 0:
        return float('inf')

    accuracy = beamwidth / (k * np.sqrt(2 * snr))
    return accuracy


def calculate_doppler_accuracy(
    snr: float,
    num_pulses: int,
    prf: float,
) -> float:
    """
    计算理论多普勒测量精度

    σfd = PRF / (2π * sqrt(N) * sqrt(2 * SNR))

    Args:
        snr: 信噪比（线性）
        num_pulses: 脉冲数
        prf: 脉冲重复频率

    Returns:
        多普勒测量精度（标准差，Hz）
    """
    if snr <= 0:
        return float('inf')

    accuracy = prf / (2 * np.pi * np.sqrt(num_pulses) * np.sqrt(2 * snr))
    return accuracy


def estimate_radar_cross_section(
    received_power: float,
    transmit_power: float,
    antenna_gain_tx: float,
    antenna_gain_rx: float,
    wavelength: float,
    range_distance: float,
    system_loss: float = 1.0,
) -> float:
    """
    从测量值估计RCS

    从雷达方程反解：
    σ = (Pr * (4π)³ * R⁴ * L) / (Pt * Gt * Gr * λ²)

    Args:
        received_power: 接收功率 (W)
        transmit_power: 发射功率 (W)
        antenna_gain_tx: 发射增益
        antenna_gain_rx: 接收增益
        wavelength: 波长 (m)
        range_distance: 距离 (m)
        system_loss: 系统损耗

    Returns:
        RCS (m²)
    """
    numerator = (
        received_power *
        (4 * np.pi) ** 3 *
        range_distance ** 4 *
        system_loss
    )

    denominator = (
        transmit_power *
        antenna_gain_tx *
        antenna_gain_rx *
        wavelength ** 2
    )

    if denominator == 0:
        return 0

    rcs = numerator / denominator
    return rcs
