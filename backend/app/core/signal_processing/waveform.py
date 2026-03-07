"""
发射信号生成模块
完整实现常规脉冲、LFM、相位编码等多种波形

参考文档 4.4.2 节
"""
import numpy as np
from typing import Literal, Optional, Tuple, Union
from scipy import signal
from scipy.fft import fft, fftshift


def generate_pulse(
    pulse_width: float,
    sampling_rate: float,
    prf: float,
    num_pulses: int = 1,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成常规矩形脉冲序列

    Args:
        pulse_width: 脉冲宽度 (s)
        sampling_rate: 采样率 (Hz)
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数量
        amplitude: 脉冲幅度

    Returns:
        (signal_array, time_array): 信号数组和时间数组
    """
    pri = 1.0 / prf  # 脉冲重复间隔
    samples_per_pulse = int(pulse_width * sampling_rate)
    samples_per_pri = int(pri * sampling_rate)

    # 创建脉冲序列
    total_samples = samples_per_pri * num_pulses
    waveform = np.zeros(total_samples)
    time = np.arange(total_samples) / sampling_rate

    # 填充脉冲
    for i in range(num_pulses):
        start = i * samples_per_pri
        end = start + samples_per_pulse
        waveform[start:end] = amplitude

    return waveform, time


def generate_lfm_pulse(
    pulse_width: float,
    bandwidth: float,
    sampling_rate: float,
    prf: float,
    num_pulses: int = 1,
    amplitude: float = 1.0,
    up_chirp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成线性调频（LFM）脉冲信号

    s(t) = A * rect(t/τ) * exp(j*π*B/τ*t²)

    Args:
        pulse_width: 脉冲宽度 (s)
        bandwidth: 调频带宽 (Hz)
        sampling_rate: 采样率 (Hz)
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数量
        amplitude: 脉冲幅度
        up_chirp: True为上调频，False为下调频

    Returns:
        (signal_array, time_array): 复数信号数组和时间数组
    """
    pri = 1.0 / prf
    samples_per_pulse = int(pulse_width * sampling_rate)
    samples_per_pri = int(pri * sampling_rate)

    total_samples = samples_per_pri * num_pulses
    waveform = np.zeros(total_samples, dtype=complex)
    time = np.arange(total_samples) / sampling_rate

    # 调频斜率
    chirp_rate = bandwidth / pulse_width  # Hz/s
    if not up_chirp:
        chirp_rate = -chirp_rate

    # 生成LFM脉冲
    t_pulse = np.linspace(-pulse_width/2, pulse_width/2, samples_per_pulse)

    # LFM相位: φ(t) = π * (B/τ) * t²
    phase = np.pi * chirp_rate * t_pulse ** 2
    lfm_pulse = amplitude * np.exp(1j * phase)

    # 填充脉冲序列
    for i in range(num_pulses):
        start = i * samples_per_pri
        end = start + samples_per_pulse
        waveform[start:end] = lfm_pulse

    return waveform, time


def generate_barker_code(
    code_length: Literal[2, 3, 4, 5, 7, 11, 13],
    chip_width: float,
    sampling_rate: float,
    prf: float,
    num_pulses: int = 1,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成Barker码相位编码脉冲

    Args:
        code_length: Barker码长度 (2/3/4/5/7/11/13)
        chip_width: 码片宽度 (s)
        sampling_rate: 采样率 (Hz)
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数量
        amplitude: 脉冲幅度

    Returns:
        (signal_array, time_array): 复数信号数组和时间数组
    """
    # Barker码序列
    barker_codes = {
        2: np.array([1, -1]),
        3: np.array([1, 1, -1]),
        4: np.array([1, 1, -1, 1]),
        5: np.array([1, 1, 1, -1, 1]),
        7: np.array([1, 1, 1, -1, -1, 1, -1]),
        11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
        13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
    }

    if code_length not in barker_codes:
        raise ValueError(f"不支持的Barker码长度: {code_length}")

    code = barker_codes[code_length]
    pulse_width = code_length * chip_width

    pri = 1.0 / prf
    samples_per_pulse = int(pulse_width * sampling_rate)
    samples_per_pri = int(pri * sampling_rate)
    samples_per_chip = int(chip_width * sampling_rate)

    total_samples = samples_per_pri * num_pulses
    waveform = np.zeros(total_samples, dtype=complex)
    time = np.arange(total_samples) / sampling_rate

    # 生成Barker码脉冲
    barker_pulse = np.zeros(samples_per_pulse, dtype=complex)
    for i, bit in enumerate(code):
        start = i * samples_per_chip
        end = start + samples_per_chip
        # 相位调制: 0° for +1, 180° for -1
        phase = 0 if bit > 0 else np.pi
        barker_pulse[start:end] = amplitude * np.exp(1j * phase)

    # 填充脉冲序列
    for i in range(num_pulses):
        start = i * samples_per_pri
        end = start + samples_per_pulse
        waveform[start:end] = barker_pulse

    return waveform, time


def generate_mseq_code(
    n: int,
    chip_width: float,
    sampling_rate: float,
    prf: float,
    num_pulses: int = 1,
    amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成M序列（最大长度序列）相位编码

    Args:
        n: 移位寄存器级数 (序列长度 = 2^n - 1)
        chip_width: 码片宽度 (s)
        sampling_rate: 采样率 (Hz)
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数量
        amplitude: 脉冲幅度

    Returns:
        (signal_array, time_array): 复数信号数组和时间数组
    """
    # 生成M序列
    seq_len = 2 ** n - 1

    # 本原多项式系数（对于不同n值）
    taps = {
        3: [3, 2],
        4: [4, 3],
        5: [5, 3],
        6: [6, 5],
        7: [7, 6],
    }

    if n not in taps:
        raise ValueError(f"不支持的M序列级数: {n}")

    # LFSR生成M序列
    register = np.ones(n, dtype=int)
    mseq = np.zeros(seq_len, dtype=int)

    for i in range(seq_len):
        mseq[i] = register[-1]
        # 反馈
        tap_positions = [n - t for t in taps[n]]
        feedback = np.sum(register[tap_positions]) % 2
        register = np.roll(register, 1)
        register[0] = feedback

    # 将0/1转换为1/-1
    mseq = 2 * mseq - 1

    pulse_width = seq_len * chip_width
    pri = 1.0 / prf
    samples_per_pulse = int(pulse_width * sampling_rate)
    samples_per_pri = int(pri * sampling_rate)
    samples_per_chip = int(chip_width * sampling_rate)

    total_samples = samples_per_pri * num_pulses
    waveform = np.zeros(total_samples, dtype=complex)
    time = np.arange(total_samples) / sampling_rate

    # 生成M序列脉冲
    mseq_pulse = np.zeros(samples_per_pulse, dtype=complex)
    for i, bit in enumerate(mseq):
        start = i * samples_per_chip
        end = start + samples_per_chip
        phase = 0 if bit > 0 else np.pi
        mseq_pulse[start:end] = amplitude * np.exp(1j * phase)

    # 填充脉冲序列
    for i in range(num_pulses):
        start = i * samples_per_pri
        end = start + samples_per_pulse
        waveform[start:end] = mseq_pulse

    return waveform, time


def pulse_compression(
    received_signal: np.ndarray,
    reference_signal: np.ndarray,
) -> np.ndarray:
    """
    脉冲压缩处理（匹配滤波）

    Args:
        received_signal: 接收信号
        reference_signal: 参考信号（发射信号副本）

    Returns:
        压缩后的信号
    """
    # 时域相关（匹配滤波）
    compressed = np.correlate(
        received_signal,
        np.conj(reference_signal[::-1]),
        mode='same'
    )

    return compressed


def calculate_ambiguity_function(
    waveform: np.ndarray,
    sampling_rate: float,
    max_doppler: float = 1000,
    doppler_resolution: float = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算模糊函数

    Args:
        waveform: 波形信号（复数）
        sampling_rate: 采样率
        max_doppler: 最大多普勒频移 (Hz)
        doppler_resolution: 多普勒分辨率 (Hz)

    Returns:
        (ambiguity_surface, doppler_axis): 模糊函数表面和多普勒轴
    """
    # 时间延迟轴
    n_samples = len(waveform)
    delay_axis = np.arange(-n_samples//2, n_samples//2) / sampling_rate

    # 多普勒轴
    doppler_axis = np.arange(-max_doppler, max_doppler, doppler_resolution)

    # 计算模糊函数
    ambiguity = np.zeros((len(doppler_axis), len(delay_axis)), dtype=complex)

    for i, fd in enumerate(doppler_axis):
        # 多普勒频移
        doppler_signal = waveform * np.exp(1j * 2 * np.pi * fd * np.arange(n_samples) / sampling_rate)
        # 与原信号相关
        correlation = np.correlate(doppler_signal, np.conj(waveform), mode='same')
        ambiguity[i, :] = correlation

    # 归一化
    ambiguity = np.abs(ambiguity) ** 2
    ambiguity = ambiguity / np.max(ambiguity)

    # 转换为dB
    ambiguity_db = 10 * np.log10(ambiguity + 1e-10)

    return ambiguity_db, doppler_axis


def calculate_range_resolution(
    bandwidth: float,
) -> float:
    """
    计算距离分辨率

    Args:
        bandwidth: 信号带宽 (Hz)

    Returns:
        距离分辨率 (m)
    """
    c = 3e8  # 光速
    return c / (2 * bandwidth)


def calculate_doppler_resolution(
    prf: float,
    num_pulses: int,
) -> float:
    """
    计算多普勒分辨率

    Args:
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数量

    Returns:
        多普勒分辨率 (Hz)
    """
    return prf / num_pulses
