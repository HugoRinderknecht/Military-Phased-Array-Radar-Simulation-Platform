# 信号工具函数模块 (Signal Utilities Module)

import numpy as np
from scipy import signal
from typing import Tuple, Optional


def next_power_of_2(n: int) -> int:
    """计算大于等于n的最小2的幂次"""
    return 1 << (n - 1).bit_length()


def fft_shift_1d(data: np.ndarray) -> np.ndarray:
    """将FFT结果的零频移到中心"""
    return np.fft.fftshift(data)


def ifft_shift_1d(data: np.ndarray) -> np.ndarray:
    """反向FFT移位"""
    return np.fft.ifftshift(data)


def fft_frequency(n: int, sample_rate: float) -> np.ndarray:
    """计算FFT频率轴"""
    return np.fft.fftfreq(n, 1.0 / sample_rate)


def fft_shift_frequency(n: int, sample_rate: float) -> np.ndarray:
    """计算移位后的FFT频率轴（零频在中心）"""
    return np.fft.fftshift(np.fft.fftfreq(n, 1.0 / sample_rate))


def generate_window(window_type: str, n: int, **kwargs) -> np.ndarray:
    """生成窗函数"""
    window_type = window_type.lower()
    
    if window_type == 'rectangular':
        return np.ones(n)
    elif window_type == 'hamming':
        return np.hamming(n)
    elif window_type == 'hanning':
        return np.hanning(n)
    elif window_type == 'blackman':
        return np.blackman(n)
    elif window_type == 'bartlett':
        return np.bartlett(n)
    elif window_type == 'kaiser':
        beta = kwargs.get('beta', 6.0)
        return np.kaiser(n, beta)
    elif window_type == 'chebwin':
        attenuation = kwargs.get('attenuation', 50.0)
        return signal.windows.chebwin(n, attenuation)
    elif window_type == 'flattop':
        return signal.windows.flattop(n)
    elif window_type == 'gaussian':
        std = kwargs.get('std', n / 6.0)
        return signal.windows.gaussian(n, std)
    else:
        raise ValueError(f'未知的窗函数类型: {window_type}')


def window_correction(window: np.ndarray) -> float:
    """计算窗函数的功率校正因子"""
    return np.mean(window ** 2)


def coherent_gain(window: np.ndarray) -> float:
    """计算窗函数的相干增益"""
    return np.mean(window)


def enbw(window: np.ndarray, sample_rate: float) -> float:
    """计算等效噪声带宽"""
    n = len(window)
    sum_w = np.sum(window)
    sum_w2 = np.sum(window ** 2)
    return sample_rate * n * sum_w2 / (sum_w ** 2)


def decimate(signal_data: np.ndarray, factor: int, n_taps: int = 20) -> np.ndarray:
    """对信号进行整数倍抽取"""
    return signal.decimate(signal_data, factor, n_taps=n_taps)


def resample(signal_data: np.ndarray, original_rate: float, target_rate: float) -> np.ndarray:
    """对信号进行重采样"""
    n_original = len(signal_data)
    n_target = int(n_original * target_rate / original_rate)
    return signal.resample(signal_data, n_target)


def upsample(signal_data: np.ndarray, factor: int) -> np.ndarray:
    """对信号进行整数倍上采样"""
    n_original = len(signal_data)
    n_upsampled = n_original * factor
    upsampled = np.zeros(n_upsampled, dtype=signal_data.dtype)
    upsampled[::factor] = signal_data
    return upsampled


def design_fir_filter(cutoff: float, sample_rate: float, num_taps: int = 64, 
                      window_type: str = 'hamming', pass_zero: bool = True) -> np.ndarray:
    """设计FIR滤波器"""
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff / nyquist
    return signal.firwin(num_taps, normalized_cutoff, window=window_type, pass_zero=pass_zero)


def design_iir_lowpass(cutoff: float, sample_rate: float, order: int = 4, ftype: str = 'butter') -> Tuple[np.ndarray, np.ndarray]:
    """设计IIR低通滤波器"""
    nyquist = sample_rate / 2.0
    normalized_cutoff = cutoff / nyquist
    return signal.iirfilter(order, normalized_cutoff, btype='low', ftype=ftype)


def apply_filter(signal_data: np.ndarray, b: np.ndarray, a: np.ndarray = np.array([1.0])) -> np.ndarray:
    """应用滤波器"""
    return signal.lfilter(b, a, signal_data)


def compute_psd(signal_data: np.ndarray, sample_rate: float, nfft: Optional[int] = None, 
                window_type: str = 'hamming') -> Tuple[np.ndarray, np.ndarray]:
    """计算功率谱密度"""
    n = len(signal_data)
    nfft = nfft or next_power_of_2(n)
    
    window = generate_window(window_type, n)
    windowed_signal = signal_data * window
    
    fft_result = np.fft.fft(windowed_signal, nfft)
    
    psd = np.abs(fft_result[:nfft//2]) ** 2
    psd /= (sample_rate * np.sum(window ** 2))
    psd[1:-1] *= 2
    
    psd_db = 10.0 * np.log10(psd + 1e-20)
    freq = np.fft.fftfreq(nfft, 1.0 / sample_rate)[:nfft//2]
    
    return freq, psd_db


def compute_spectrogram(signal_data: np.ndarray, sample_rate: float, window_size: int = 256,
                        overlap: int = 128, nfft: Optional[int] = None, 
                        window_type: str = 'hamming') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算短时傅里叶变换（语谱图）"""
    nfft = nfft or next_power_of_2(window_size)
    
    freq, time, Zxx = signal.spectrogram(
        signal_data, fs=sample_rate, window=window_type,
        nperseg=window_size, noverlap=overlap, nfft=nfft, mode='magnitude'
    )
    
    spectrogram_db = 20.0 * np.log10(np.abs(Zxx) + 1e-20)
    
    return freq, time, spectrogram_db


def matched_filter_coefficients(waveform: np.ndarray) -> np.ndarray:
    """生成匹配滤波器系数（共轭翻转）"""
    return np.conj(waveform[::-1])


def pulse_compression(received: np.ndarray, reference: np.ndarray, nfft: Optional[int] = None) -> np.ndarray:
    """脉冲压缩（频域实现）"""
    n = len(received) + len(reference) - 1
    nfft = nfft or next_power_of_2(n)
    
    received_fft = np.fft.fft(received, nfft)
    reference_fft = np.fft.fft(reference, nfft)
    
    output_fft = received_fft * np.conj(reference_fft)
    
    compressed = np.fft.ifft(output_fft)
    
    return np.abs(compressed[:n])


def exponential_correlation(n: int, rho: float) -> np.ndarray:
    """生成指数相关函数"""
    return np.array([rho ** abs(k) for k in range(n)])


def generate_correlated_gaussian(n: int, correlation_coeff: float, power: float = 1.0) -> np.ndarray:
    """生成相关高斯噪声序列"""
    white = np.random.randn(n) * np.sqrt(power)
    if abs(correlation_coeff) < 1e-10:
        return white
    
    correlated = np.zeros(n)
    correlated[0] = white[0]
    for k in range(1, n):
        correlated[k] = correlation_coeff * correlated[k-1] + np.sqrt(1 - correlation_coeff**2) * white[k]
    
    return correlated


__all__ = [
    'next_power_of_2', 'fft_shift_1d', 'ifft_shift_1d',
    'fft_frequency', 'fft_shift_frequency',
    'generate_window', 'window_correction', 'coherent_gain', 'enbw',
    'decimate', 'resample', 'resample_signal', 'interpolate',  # 重采样
    'upsample',
    'design_fir_filter', 'design_iir_filter', 'apply_fir_filter', 'apply_iir_filter',  # 滤波器
    'compute_fft', 'compute_psd', 'compute_spectrogram',  # 频谱分析
    'correlate', 'convolve',  # 相关和卷积
    'db_to_linear', 'linear_to_db', 'complex_to_db',  # dB转换
    'envelope', 'instantaneous_phase', 'instantaneous_frequency',  # 瞬时特性
    'matched_filter_coefficients', 'pulse_compression',
    'exponential_correlation', 'generate_correlated_gaussian',
    'generate_lfm_pulse', 'generate_complex_pulse', 'generate_phase_code', 'generate_noise',  # 信号生成
]


# 别名
resample_signal = resample
design_iir_filter = design_iir_lowpass
apply_fir_filter = apply_filter
apply_iir_filter = apply_filter
interpolate = resample
compute_fft = np.fft.fft
correlate = np.correlate
convolve = np.convolve
db_to_linear = lambda x: 10**(x/10)
linear_to_db = lambda x: 10*np.log10(x)
complex_to_db = lambda x: 20*np.log10(np.abs(x))
envelope = np.abs
instantaneous_phase = np.angle
instantaneous_frequency = lambda x: np.diff(np.unwrap(np.angle(x)))


def generate_noise(
    n_samples: int,
    noise_type: str = 'gaussian',
    power: float = 1.0
) -> np.ndarray:
    """生成噪声"""
    if noise_type == 'gaussian':
        return np.random.randn(n_samples) * np.sqrt(power)
    elif noise_type == 'uniform':
        return (np.random.rand(n_samples) - 0.5) * 2 * np.sqrt(3 * power)
    else:
        return np.random.randn(n_samples) * np.sqrt(power)


def generate_phase_code(
    code_length: int,
    code_type: str = 'barker'
) -> np.ndarray:
    """生成相位编码序列"""
    if code_type == 'barker':
        # Barker码（已知的最佳短码）
        barker_codes = {
            2: np.array([1, -1]),
            3: np.array([1, 1, -1]),
            4: np.array([1, 1, -1, 1]),
            5: np.array([1, 1, 1, -1, 1]),
            7: np.array([1, 1, 1, -1, -1, 1, -1]),
            11: np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1]),
            13: np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]),
        }
        return barker_codes.get(code_length, np.ones(code_length))
    elif code_type == 'm_sequence':
        # 简单的伪随机序列
        return np.random.choice([1, -1], size=code_length)
    else:
        return np.ones(code_length)


def generate_lfm_pulse(
    sample_rate: float,
    pulse_width: float,
    bandwidth: float,
    initial_phase: float = 0.0
) -> np.ndarray:
    """生成线性调频（LFM）脉冲"""
    n_samples = int(sample_rate * pulse_width)
    t = np.arange(n_samples) / sample_rate

    # 瞬时频率
    chirp_rate = bandwidth / pulse_width
    instantaneous_phase = 2 * np.pi * (0.5 * chirp_rate * t**2) + initial_phase

    # 复信号
    signal = np.exp(1j * instantaneous_phase)

    return signal


def generate_complex_pulse(
    sample_rate: float,
    pulse_width: float,
    frequency: float = 0.0,
    initial_phase: float = 0.0
) -> np.ndarray:
    """生成简单复脉冲（单频）"""
    n_samples = int(sample_rate * pulse_width)
    t = np.arange(n_samples) / sample_rate

    phase = 2 * np.pi * frequency * t + initial_phase
    signal = np.exp(1j * phase)

    return signal
