"""
MTI/MTD杂波抑制模块
完整实现一次、二次、三次MTI对消和MTD多普勒滤波

参考文档 4.4.5 节
"""
import numpy as np
from typing import Literal, Optional, Tuple
from scipy import signal
from scipy.fft import fft, fftshift


def mti_canceler_1st(
    pulses: np.ndarray,
) -> np.ndarray:
    """
    一次MTI对消器

    y[n] = x[n] - x[n-1]

    Args:
        pulses: 脉冲序列 (num_pulses, num_range_bins)

    Returns:
        对消后的脉冲序列
    """
    # 沿脉冲维差分
    cancelled = np.diff(pulses, axis=0, prepend=0)

    return cancelled


def mti_canceler_2nd(
    pulses: np.ndarray,
) -> np.ndarray:
    """
    二次MTI对消器

    y[n] = x[n] - 2*x[n-1] + x[n-2]

    Args:
        pulses: 脉冲序列 (num_pulses, num_range_bins)

    Returns:
        对消后的脉冲序列
    """
    # 二阶差分
    cancelled = np.zeros_like(pulses)

    # 补零处理边界
    padded = np.vstack([np.zeros((1, pulses.shape[1])), pulses])

    for i in range(2, padded.shape[0]):
        cancelled[i-2, :] = (
            padded[i, :] -
            2 * padded[i-1, :] +
            padded[i-2, :]
        )

    return cancelled


def mti_canceler_3rd(
    pulses: np.ndarray,
) -> np.ndarray:
    """
    三次MTI对消器

    y[n] = x[n] - 3*x[n-1] + 3*x[n-2] - x[n-3]

    Args:
        pulses: 脉冲序列 (num_pulses, num_range_bins)

    Returns:
        对消后的脉冲序列
    """
    cancelled = np.zeros_like(pulses)

    padded = np.vstack([np.zeros((2, pulses.shape[1])), pulses])

    for i in range(3, padded.shape[0]):
        cancelled[i-3, :] = (
            padded[i, :] -
            3 * padded[i-1, :] +
            3 * padded[i-2, :] -
            padded[i-3, :]
        )

    return cancelled


def calculate_mti_improvement_factor(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    clutter_range: Optional[Tuple[int, int]] = None,
) -> float:
    """
    计算MTI改善因子

    I = (S_clutter_in / N_clutter_in) / (S_clutter_out / N_clutter_out)

    Args:
        input_signal: 输入信号（含杂波）
        output_signal: 输出信号（对消后）
        clutter_range: 杂波存在的距离单元范围 (start, end)

    Returns:
        改善因子 (dB)
    """
    if clutter_range is None:
        clutter_range = (0, len(input_signal))

    # 提取杂波区域
    clutter_in = input_signal[clutter_range[0]:clutter_range[1]]
    clutter_out = output_signal[clutter_range[0]:clutter_range[1]]

    # 计算功率比
    clutter_power_in = np.mean(np.abs(clutter_in) ** 2)
    clutter_power_out = np.mean(np.abs(clutter_out) ** 2)

    if clutter_power_out == 0:
        return float('inf')

    improvement_factor = clutter_power_in / clutter_power_out
    improvement_factor_db = 10 * np.log10(improvement_factor)

    return improvement_factor_db


def calculate_blind_speeds(
    prf: float,
    wavelength: float,
    max_order: int = 5,
) -> np.ndarray:
    """
    计算MTI盲速

    盲速：当 fd = k * PRF 时，多普勒频移被完全抑制

    Args:
        prf: 脉冲重复频率 (Hz)
        wavelength: 波长 (m)
        max_order: 最大盲速阶数

    Returns:
        盲速数组 (m/s)
    """
    blind_speeds = []
    for k in range(1, max_order + 1):
        # fd = 2*v/λ = k*PRF
        # v = k*PRF*λ/2
        v = k * prf * wavelength / 2
        blind_speeds.append(v)

    return np.array(blind_speeds)


def design_mtd_filter_bank(
    num_pulses: int,
    window: Optional[Literal["rectangular", "hamming", "hanning", "blackman"]] = "hamming",
) -> np.ndarray:
    """
    设计MTD多普勒滤波器组

    Args:
        num_pulses: 脉冲数（决定滤波器数量）
        window: 窗函数类型

    Returns:
        滤波器系数矩阵 (num_filters, num_pulses)
    """
    num_filters = num_pulses

    # 生成窗函数
    if window == "rectangular":
        win = np.ones(num_pulses)
    elif window == "hamming":
        win = np.hamming(num_pulses)
    elif window == "hanning":
        win = np.hanning(num_pulses)
    elif window == "blackman":
        win = np.blackman(num_pulses)
    else:
        win = np.hamming(num_pulses)

    # 使用FFT设计滤波器组
    # 每个滤波器对应一个多普勒频率
    filter_bank = np.zeros((num_filters, num_pulses), dtype=complex)

    for k in range(num_filters):
        # 目标多普勒频率
        target_freq = k / num_filters  # 归一化频率

        # 生成复指数
        n = np.arange(num_pulses)
        filter_response = win * np.exp(1j * 2 * np.pi * target_freq * n)

        filter_bank[k, :] = filter_response

    return filter_bank


def mtd_processing(
    pulse_data: np.ndarray,
    filter_bank: Optional[np.ndarray] = None,
    window: Literal["rectangular", "hamming", "hanning", "blackman"] = "hamming",
) -> np.ndarray:
    """
    MTD多普勒滤波处理

    对每个距离单元的脉冲序列进行多普勒滤波

    Args:
        pulse_data: 脉冲数据 (num_pulses, num_range_bins)
        filter_bank: 预设计的滤波器组
        window: 窗函数类型（如果filter_bank为None）

    Returns:
        多普勒通道输出 (num_doppler_bins, num_range_bins)
    """
    num_pulses, num_range_bins = pulse_data.shape

    # 设计滤波器组
    if filter_bank is None:
        filter_bank = design_mtd_filter_bank(num_pulses, window)

    num_doppler_bins = filter_bank.shape[0]

    # 初始化输出
    doppler_output = np.zeros((num_doppler_bins, num_range_bins), dtype=complex)

    # 对每个距离单元应用多普勒滤波
    for range_idx in range(num_range_bins):
        range_profile = pulse_data[:, range_idx]

        # 应用每个滤波器
        for doppler_idx in range(num_doppler_bins):
            filter_coeffs = filter_bank[doppler_idx, :]
            # 卷积/相关
            output = np.sum(filter_coeffs * range_profile)
            doppler_output[doppler_idx, range_idx] = output

    return doppler_output


def estimate_doppler_frequency(
    doppler_output: np.ndarray,
    prf: float,
) -> np.ndarray:
    """
    从MTD输出估计多普勒频率

    Args:
        doppler_output: MTD输出 (num_doppler_bins, num_range_bins)
        prf: 脉冲重复频率

    Returns:
        多普勒频率数组 (Hz), 每个距离单元一个
    """
    num_doppler_bins, num_range_bins = doppler_output.shape

    # 找到每个距离单元的最大响应位置
    max_indices = np.argmax(np.abs(doppler_output), axis=0)

    # 计算对应的多普勒频率
    # 频率索引映射：[-PRF/2, PRF/2)
    doppler_frequencies = np.zeros(num_range_bins)

    for range_idx in range(num_range_bins):
        idx = max_indices[range_idx]
        # 将索引映射到频率
        if idx >= num_doppler_bins / 2:
            freq_idx = idx - num_doppler_bins
        else:
            freq_idx = idx

        doppler_frequencies[range_idx] = freq_idx * prf / num_doppler_bins

    return doppler_frequencies


def calculate_doppler_resolution(
    prf: float,
    num_pulses: int,
) -> float:
    """
    计算多普勒分辨率

    Args:
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数

    Returns:
        多普勒分辨率 (Hz)
    """
    return prf / num_pulses


def calculate_mtd_filter_response(
    filter_coeffs: np.ndarray,
    num_points: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算MTD滤波器的频率响应

    Args:
        filter_coeffs: 滤波器系数
        num_points: FFT点数

    Returns:
        (frequency, response_db): 频率数组和响应(dB)
    """
    # 计算频率响应
    response = np.fft.fft(filter_coeffs, n=num_points)
    response = np.fft.fftshift(response)

    # 归一化频率 [-0.5, 0.5]
    freq = np.linspace(-0.5, 0.5, num_points)

    # 转换为dB
    response_db = 20 * np.log10(np.abs(response) + 1e-10)
    response_db = response_db - np.max(response_db)

    return freq, response_db


def adaptive_mti(
    pulses: np.ndarray,
    clutter_estimate: Optional[np.ndarray] = None,
    alpha: float = 0.95,
) -> np.ndarray:
    """
    自适应MTI对消

    根据估计的杂波功率自适应调整对消器参数

    Args:
        pulses: 脉冲序列
        clutter_estimate: 杂波估计（可选）
        alpha: 自适应因子

    Returns:
        对消后的脉冲序列
    """
    num_pulses, num_range_bins = pulses.shape

    if clutter_estimate is None:
        # 简单杂波估计：取前几个脉冲的平均
        clutter_estimate = np.mean(pulses[:min(5, num_pulses), :], axis=0)

    # 自适应对消
    cancelled = np.zeros_like(pulses)

    for i in range(num_pulses):
        # 自适应预测杂波
        if i > 0:
            clutter_estimate = alpha * clutter_estimate + (1 - alpha) * pulses[i-1, :]

        # 对消
        cancelled[i, :] = pulses[i, :] - clutter_estimate

    return cancelled
