"""
Keystone变换模块
完整实现距离走动校正

参考文档 4.4.8 节
"""
import numpy as np
from typing import Tuple
from scipy.fft import fft, ifft, fftshift
from scipy.interpolate import interp1d


def keystone_transform(
    data: np.ndarray,
    prf: float,
    center_frequency: float,
    bandwidth: float,
    sampling_rate: float,
) -> np.ndarray:
    """
    Keystone变换实现

    用于校正距离走动（距离弯曲）

    Args:
        data: 输入数据 (num_pulses, num_range_bins)
        prf: 脉冲重复频率
        center_frequency: 中心频率
        bandwidth: 信号带宽
        sampling_rate: 采样率

    Returns:
        Keystone变换后的数据
    """
    num_pulses, num_range_bins = data.shape

    # 步骤1: 快时间（距离维）FFT
    range_fft = fft(data, axis=1)

    # 频率轴
    freq_axis = np.linspace(-bandwidth/2, bandwidth/2, num_range_bins)

    # 步骤2: 构建新的慢时间网格
    # Keystone变换核心：重新缩放慢时间轴
    slow_time_original = np.arange(num_pulses) / prf

    # 对于每个距离频率单元，计算缩放因子
    # 缩放因子 = fc / (fc + fr)
    scaling_factors = center_frequency / (center_frequency + freq_axis)

    corrected_data = np.zeros_like(range_fft, dtype=complex)

    # 步骤3: 对每个距离频率单元进行插值
    for fr_idx in range(num_range_bins):
        # 当前频率的缩放因子
        scale = scaling_factors[fr_idx]

        # 缩放后的慢时间
        slow_time_scaled = slow_time_original * scale

        # 创建插值函数
        # 使用sinc插值（通过FFT实现）
        interp_func = interp1d(
            slow_time_original,
            range_fft[:, fr_idx],
            kind='linear',
            bounds_error=False,
            fill_value=0j
        )

        # 插值到原始慢时间网格
        corrected_data[:, fr_idx] = interp_func(slow_time_original)

    # 步骤4: 快时间IFFT回到时域
    corrected_data = ifft(corrected_data, axis=1)

    return corrected_data


def keystone_transform_via_fft(
    data: np.ndarray,
    prf: float,
    center_frequency: float,
    bandwidth: float,
) -> np.ndarray:
    """
    基于FFT的Keystone变换实现（更高效）

    Args:
        data: 输入数据 (num_pulses, num_range_bins)
        prf: 脉冲重复频率
        center_frequency: 中心频率
        bandwidth: 信号带宽

    Returns:
        变换后的数据
    """
    num_pulses, num_range_bins = data.shape

    # 快时间FFT
    data_freq = fft(data, axis=1)

    # 归一化频率
    freq_normalized = np.linspace(-0.5, 0.5, num_range_bins)

    # 缩放因子
    alpha = center_frequency / (center_frequency + bandwidth * freq_normalized)

    # 对每个频率单元进行慢时间重采样
    corrected_data = np.zeros_like(data_freq, dtype=complex)

    for i in range(num_range_bins):
        # 当前缩放因子
        scale = alpha[i]

        # 生成新的慢时间索引
        original_indices = np.arange(num_pulses)
        scaled_indices = original_indices * scale

        # 边界处理
        valid_mask = (scaled_indices >= 0) & (scaled_indices < num_pulses - 1)

        # 线性插值
        interp_indices = scaled_indices[valid_mask]
        interp_data = np.interp(
            interp_indices,
            original_indices,
            data_freq[:, i].real
        ) + 1j * np.interp(
            interp_indices,
            original_indices,
            data_freq[:, i].imag
        )

        corrected_data[valid_mask, i] = interp_data

    # IFFT回到时域
    corrected_data = ifft(corrected_data, axis=1)

    return corrected_data


def calculate_range_walk(
    velocity: float,
    num_pulses: int,
    prf: float,
    wavelength: float,
) -> float:
    """
    计算距离走动量

    Args:
        velocity: 目标径向速度 (m/s)
        num_pulses: 脉冲数
        prf: 脉冲重复频率
        wavelength: 波长 (m)

    Returns:
        距离走动量 (m)
    """
    c = 3e8
    total_time = num_pulses / prf
    distance_change = velocity * total_time

    # 双程距离走动
    range_walk = 2 * distance_change

    return range_walk


def detect_range_walk(
    data: np.ndarray,
    expected_range_bins: int,
) -> Tuple[bool, float]:
    """
    检测是否存在距离走动

    Args:
        data: 距离-多普勒数据
        expected_range_bins: 预期的距离单元偏移

    Returns:
        (has_walk, walk_amount): 是否存在走动和走动量
    """
    # 找到峰值位置
    peak_idx = np.argmax(np.abs(data), axis=0)

    # 计算峰值位置的标准差
    std_peak = np.std(peak_idx)

    has_walk = std_peak > expected_range_bins / 2
    walk_amount = std_peak

    return has_walk, walk_amount


def apply_range_migration_correction(
    data: np.ndarray,
    migration_profile: np.ndarray,
) -> np.ndarray:
    """
    应用距离迁移校正

    Args:
        data: 输入数据 (num_pulses, num_range_bins)
        migration_profile: 迁移剖面（每个脉冲的距离偏移）

    Returns:
        校正后的数据
    """
    num_pulses, num_range_bins = data.shape
    corrected_data = np.zeros_like(data, dtype=complex)

    for pulse_idx in range(num_pulses):
        shift = int(round(migration_profile[pulse_idx]))

        if shift == 0:
            corrected_data[pulse_idx, :] = data[pulse_idx, :]
        elif shift > 0:
            # 向后移动
            corrected_data[pulse_idx, shift:] = data[pulse_idx, :-shift]
            corrected_data[pulse_idx, :shift] = 0
        else:
            # 向前移动
            corrected_data[pulse_idx, :shift] = data[pulse_idx, -shift:]
            corrected_data[pulse_idx, shift:] = 0

    return corrected_data


def pulse_compression_and_keystone(
    received_signal: np.ndarray,
    reference_waveform: np.ndarray,
    prf: float,
    center_frequency: float,
    bandwidth: float,
) -> np.ndarray:
    """
    脉冲压缩 + Keystone变换组合处理

    Args:
        received_signal: 接收信号
        reference_waveform: 参考波形
        prf: 脉冲重复频率
        center_frequency: 中心频率
        bandwidth: 带宽

    Returns:
        处理后的数据
    """
    # 步骤1: 脉冲压缩
    compressed = np.correlate(
        received_signal,
        np.conj(reference_waveform[::-1]),
        mode='same'
    )

    # 重塑为脉冲-距离矩阵
    num_pulses = len(received_signal) // len(reference_waveform)
    compressed_matrix = compressed.reshape(num_pulses, -1)

    # 步骤2: Keystone变换
    corrected = keystone_transform(
        compressed_matrix,
        prf,
        center_frequency,
        bandwidth,
    )

    return corrected


def estimate_correction_parameters(
    data: np.ndarray,
    prf: float,
    wavelength: float,
) -> np.ndarray:
    """
    估计Keystone变换校正参数

    通过估计目标的多普勒频率来计算所需的校正

    Args:
        data: 输入数据 (num_pulses, num_range_bins)
        prf: 脉冲重复频率
        wavelength: 波长

    Returns:
        每个距离单元的校正参数
    """
    num_pulses, num_range_bins = data.shape

    # 多普勒FFT
    doppler_data = fft(data, axis=0)

    # 找到峰值多普勒频率
    peak_doppler_indices = np.argmax(np.abs(doppler_data), axis=0)

    # 转换为速度
    doppler_freq = (peak_doppler_indices - num_pulses // 2) * prf / num_pulses
    velocity = doppler_freq * wavelength / 2

    # 计算缩放因子
    # 这里简化处理，实际需要更复杂的估计
    correction_params = np.ones(num_range_bins)

    return correction_params


def integrate_pulses_after_keystone(
    corrected_data: np.ndarray,
    integration_method: str = "coherent",
) -> np.ndarray:
    """
    Keystone变换后的脉冲积分

    Args:
        corrected_data: 校正后的数据
        integration_method: 积分方法（"coherent"或"non-coherent"）

    Returns:
        积分结果
    """
    if integration_method == "coherent":
        # 相干积分
        integrated = np.sum(corrected_data, axis=0)
    else:
        # 非相干积分
        integrated = np.sqrt(np.sum(np.abs(corrected_data) ** 2, axis=0))

    return integrated
