import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from typing import Tuple, List, Optional, Union
import math


class WaveformGenerator:
    """波形生成器"""

    @staticmethod
    def lfm_pulse(duration: float, bandwidth: float, fs: float,
                  f0: float = 0.0, phase: float = 0.0) -> np.ndarray:
        """线性调频脉冲生成"""
        t = np.arange(0, duration, 1 / fs)
        chirp_rate = bandwidth / duration

        # LFM信号: s(t) = exp(j*(2π*f0*t + π*K*t^2 + φ))
        phase_func = 2 * np.pi * f0 * t + np.pi * chirp_rate * t ** 2 + phase
        return np.exp(1j * phase_func)

    @staticmethod
    def nlfm_pulse(duration: float, bandwidth: float, fs: float,
                   window_type: str = 'hamming') -> np.ndarray:
        """非线性调频脉冲生成"""
        t = np.arange(0, duration, 1 / fs)

        # 窗函数映射
        if window_type == 'hamming':
            window = 0.54 - 0.46 * np.cos(2 * np.pi * t / duration)
        elif window_type == 'hanning':
            window = 0.5 - 0.5 * np.cos(2 * np.pi * t / duration)
        elif window_type == 'blackman':
            window = 0.42 - 0.5 * np.cos(2 * np.pi * t / duration) + 0.08 * np.cos(4 * np.pi * t / duration)
        else:
            window = np.ones_like(t)

        # 通过窗函数调制实现非线性频率变化
        normalized_window = (window - window.min()) / (window.max() - window.min())
        freq_modulation = -bandwidth / 2 + bandwidth * normalized_window

        phase_func = 2 * np.pi * np.cumsum(freq_modulation) / fs
        return np.exp(1j * phase_func)

    @staticmethod
    def barker_code(code_length: int, chip_duration: float, fs: float) -> np.ndarray:
        """Barker码生成"""
        barker_sequences = {
            2: [1, -1],
            3: [1, 1, -1],
            4: [1, 1, -1, 1],
            5: [1, 1, 1, -1, 1],
            7: [1, 1, 1, -1, -1, 1, -1],
            11: [1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1],
            13: [1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1]
        }

        if code_length not in barker_sequences:
            raise ValueError(f"Barker code length {code_length} not supported")

        code = barker_sequences[code_length]
        samples_per_chip = int(chip_duration * fs)

        signal_out = []
        for chip in code:
            signal_out.extend([chip] * samples_per_chip)

        return np.array(signal_out, dtype=complex)

    @staticmethod
    def m_sequence(order: int, chip_duration: float, fs: float) -> np.ndarray:
        """M序列生成"""
        # 本原多项式表 (简化实现)
        primitive_polynomials = {
            3: [3, 1],
            4: [4, 1],
            5: [5, 2],
            6: [6, 1],
            7: [7, 1],
            8: [8, 4, 3, 2],
            9: [9, 4],
            10: [10, 3]
        }

        if order not in primitive_polynomials:
            raise ValueError(f"M-sequence order {order} not supported")

        # 生成M序列
        poly = primitive_polynomials[order]
        register = np.ones(order, dtype=int)
        sequence = []

        for _ in range(2 ** order - 1):
            output = register[0]
            sequence.append(2 * output - 1)  # 转换为±1

            # 反馈计算
            feedback = 0
            for tap in poly[1:]:
                feedback ^= register[tap - 1]

            # 移位寄存器
            register[1:] = register[:-1]
            register[0] = feedback

        # 转换为时域信号
        samples_per_chip = int(chip_duration * fs)
        signal_out = []
        for chip in sequence:
            signal_out.extend([chip] * samples_per_chip)

        return np.array(signal_out, dtype=complex)

    @staticmethod
    def stepped_frequency_waveform(num_steps: int, step_size: float,
                                   pulse_duration: float, fs: float) -> np.ndarray:
        """步进频率波形"""
        t_pulse = np.arange(0, pulse_duration, 1 / fs)
        full_signal = []

        for step in range(num_steps):
            freq = step * step_size
            pulse = np.exp(1j * 2 * np.pi * freq * t_pulse)
            full_signal.extend(pulse)

        return np.array(full_signal)


class WindowFunctions:
    """窗函数库"""

    @staticmethod
    def rectangular(length: int) -> np.ndarray:
        """矩形窗"""
        return np.ones(length)

    @staticmethod
    def hamming(length: int) -> np.ndarray:
        """汉明窗"""
        return np.hamming(length)

    @staticmethod
    def hanning(length: int) -> np.ndarray:
        """汉宁窗"""
        return np.hanning(length)

    @staticmethod
    def blackman(length: int) -> np.ndarray:
        """布莱克曼窗"""
        return np.blackman(length)

    @staticmethod
    def kaiser(length: int, beta: float = 8.6) -> np.ndarray:
        """凯撒窗"""
        return np.kaiser(length, beta)

    @staticmethod
    def chebyshev(length: int, ripple: float = 60) -> np.ndarray:
        """切比雪夫窗"""
        return signal.chebwin(length, ripple)

    @staticmethod
    def taylor(length: int, nbar: int = 4, sll: float = -30) -> np.ndarray:
        """泰勒窗"""
        return signal.windows.taylor(length, nbar, sll)

    @staticmethod
    def tukey(length: int, alpha: float = 0.5) -> np.ndarray:
        """图基窗"""
        return signal.tukey(length, alpha)


class FilterDesign:
    """滤波器设计"""

    @staticmethod
    def butterworth_lowpass(cutoff_freq: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """巴特沃斯低通滤波器"""
        nyquist = fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return b, a

    @staticmethod
    def butterworth_highpass(cutoff_freq: float, fs: float, order: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """巴特沃斯高通滤波器"""
        nyquist = fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.butter(order, normalized_cutoff, btype='high')
        return b, a

    @staticmethod
    def butterworth_bandpass(low_freq: float, high_freq: float, fs: float, order: int = 5) -> Tuple[
        np.ndarray, np.ndarray]:
        """巴特沃斯带通滤波器"""
        nyquist = fs / 2
        low_norm = low_freq / nyquist
        high_norm = high_freq / nyquist
        b, a = signal.butter(order, [low_norm, high_norm], btype='band')
        return b, a

    @staticmethod
    def chebyshev1_filter(cutoff_freq: float, fs: float, order: int = 5,
                          ripple: float = 1.0, btype: str = 'low') -> Tuple[np.ndarray, np.ndarray]:
        """切比雪夫I型滤波器"""
        nyquist = fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.cheby1(order, ripple, normalized_cutoff, btype=btype)
        return b, a

    @staticmethod
    def elliptic_filter(cutoff_freq: float, fs: float, order: int = 5,
                        ripple: float = 1.0, attenuation: float = 40.0,
                        btype: str = 'low') -> Tuple[np.ndarray, np.ndarray]:
        """椭圆滤波器"""
        nyquist = fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = signal.ellip(order, ripple, attenuation, normalized_cutoff, btype=btype)
        return b, a

    @staticmethod
    def fir_filter_design(cutoff_freq: float, fs: float, num_taps: int,
                          window: str = 'hamming') -> np.ndarray:
        """FIR滤波器设计"""
        nyquist = fs / 2
        normalized_cutoff = cutoff_freq / nyquist
        return signal.firwin(num_taps, normalized_cutoff, window=window)


class SpectralAnalysis:
    """频谱分析工具"""

    @staticmethod
    def power_spectral_density(x: np.ndarray, fs: float, window: str = 'hanning',
                               nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """功率谱密度估计"""
        freqs, psd = signal.welch(x, fs, window=window, nperseg=nperseg)
        return freqs, psd

    @staticmethod
    def spectrogram(x: np.ndarray, fs: float, window: str = 'hanning',
                    nperseg: Optional[int] = None, noverlap: Optional[int] = None) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """时频分析（语谱图）"""
        freqs, times, Sxx = signal.spectrogram(x, fs, window=window,
                                               nperseg=nperseg, noverlap=noverlap)
        return freqs, times, Sxx

    @staticmethod
    def stft(x: np.ndarray, fs: float, window: str = 'hanning',
             nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """短时傅里叶变换"""
        freqs, times, Zxx = signal.stft(x, fs, window=window, nperseg=nperseg)
        return freqs, times, Zxx

    @staticmethod
    def instantaneous_frequency(x: np.ndarray, fs: float) -> np.ndarray:
        """瞬时频率估计"""
        analytic_signal = signal.hilbert(x)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * fs
        return instantaneous_frequency

    @staticmethod
    def cross_correlation(x: np.ndarray, y: np.ndarray, mode: str = 'full') -> np.ndarray:
        """互相关计算"""
        return signal.correlate(x, y, mode=mode)

    @staticmethod
    def auto_correlation(x: np.ndarray, maxlags: Optional[int] = None) -> np.ndarray:
        """自相关计算"""
        if maxlags is None:
            maxlags = len(x) - 1

        # 使用FFT加速计算
        correlation = np.correlate(x, x, mode='full')
        mid = len(correlation) // 2

        start = max(0, mid - maxlags)
        end = min(len(correlation), mid + maxlags + 1)

        return correlation[start:end]


class DigitalFiltering:
    """数字滤波工具"""

    @staticmethod
    def apply_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray,
                     zi: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """应用数字滤波器"""
        if zi is None:
            y, zf = signal.lfilter(b, a, x)
        else:
            y, zf = signal.lfilter(b, a, x, zi=zi)
        return y, zf

    @staticmethod
    def zero_phase_filter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
        """零相位滤波"""
        return signal.filtfilt(b, a, x)

    @staticmethod
    def median_filter(x: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """中值滤波"""
        return signal.medfilt(x, kernel_size=kernel_size)

    @staticmethod
    def savgol_filter(x: np.ndarray, window_length: int, polyorder: int,
                      deriv: int = 0, delta: float = 1.0) -> np.ndarray:
        """Savitzky-Golay滤波"""
        return signal.savgol_filter(x, window_length, polyorder, deriv=deriv, delta=delta)

    @staticmethod
    def wiener_filter(x: np.ndarray, noise_variance: Optional[float] = None) -> np.ndarray:
        """维纳滤波"""
        # 简化的维纳滤波实现
        if noise_variance is None:
            # 估计噪声方差
            noise_variance = np.var(x) * 0.1

        # 计算功率谱
        X = fft(x)
        power_spectrum = np.abs(X) ** 2

        # 维纳滤波器频率响应
        H = power_spectrum / (power_spectrum + noise_variance)

        # 应用滤波器
        Y = X * H
        y = np.real(ifft(Y))

        return y


class ModulationDemodulation:
    """调制解调工具"""

    @staticmethod
    def amplitude_modulation(message: np.ndarray, carrier_freq: float,
                             fs: float, modulation_index: float = 1.0) -> np.ndarray:
        """幅度调制"""
        t = np.arange(len(message)) / fs
        carrier = np.cos(2 * np.pi * carrier_freq * t)
        return (1 + modulation_index * message) * carrier

    @staticmethod
    def frequency_modulation(message: np.ndarray, carrier_freq: float,
                             fs: float, freq_deviation: float = 1000) -> np.ndarray:
        """频率调制"""
        t = np.arange(len(message)) / fs

        # 积分消息信号得到相位
        phase_deviation = 2 * np.pi * freq_deviation * np.cumsum(message) / fs

        return np.cos(2 * np.pi * carrier_freq * t + phase_deviation)

    @staticmethod
    def phase_modulation(message: np.ndarray, carrier_freq: float,
                         fs: float, phase_deviation: float = np.pi / 2) -> np.ndarray:
        """相位调制"""
        t = np.arange(len(message)) / fs
        carrier = np.cos(2 * np.pi * carrier_freq * t + phase_deviation * message)
        return carrier

    @staticmethod
    def qam_modulation(i_, q_data: np.ndarray,
                       carrier_freq: float, fs: float) -> np.ndarray:
        """正交幅度调制"""
        t = np.arange(len(i_data)) / fs
        i_carrier = np.cos(2 * np.pi * carrier_freq * t)
        q_carrier = -np.sin(2 * np.pi * carrier_freq * t)

        return i_data * i_carrier + q_data * q_carrier

    @staticmethod
    def envelope_detection(signal: np.ndarray) -> np.ndarray:
        """包络检测（AM解调）"""
        analytic_signal = signal.hilbert(signal)
        return np.abs(analytic_signal)

    @staticmethod
    def frequency_discrimination(signal: np.ndarray, fs: float) -> np.ndarray:
        """鉴频器（FM解调）"""
        analytic_signal = signal.hilbert(signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_freq = np.diff(instantaneous_phase) * fs / (2 * np.pi)
        return instantaneous_freq


class RadarSignalProcessing:
    """雷达信号处理专用工具"""

    @staticmethod
    def pulse_compression(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """脉冲压缩（匹配滤波）"""
        # 时域匹配滤波
        compressed = signal.correlate(signal, reference[::-1].conj(), mode='full')
        return compressed

    @staticmethod
    def pulse_compression_fft(signal: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """基于FFT的脉冲压缩"""
        # 频域匹配滤波
        signal_fft = fft(signal, n=len(signal) + len(reference) - 1)
        ref_fft = fft(reference[::-1].conj(), n=len(signal) + len(reference) - 1)

        compressed_fft = signal_fft * ref_fft
        compressed = ifft(compressed_fft)

        return compressed

    @staticmethod
    def doppler_processing(pulses: np.ndarray, axis: int = -1) -> np.ndarray:
        """多普勒处理（跨脉冲FFT）"""
        return fft(pulses, axis=axis)

    @staticmethod
    def cfar_detector(signal: np.ndarray, num_training: int = 32,
                      num_guard: int = 4, pfa: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """CFAR检测器"""
        signal_power = np.abs(signal) ** 2
        threshold = np.zeros_like(signal_power)
        detections = np.zeros_like(signal_power, dtype=bool)

        # 计算阈值因子
        alpha = num_training * (pfa ** (-1 / num_training) - 1)

        half_window = num_training // 2 + num_guard

        for i in range(half_window, len(signal_power) - half_window):
            # 前窗训练单元
            front_training = signal_power[i - half_window:i - num_guard]
            # 后窗训练单元
            back_training = signal_power[i + num_guard + 1:i + half_window + 1]

            # 组合训练数据
            training_data = np.concatenate([front_training, back_training])
            noise_level = np.mean(training_data)

            threshold[i] = alpha * noise_level
            detections[i] = signal_power[i] > threshold[i]

        return detections, threshold

    @staticmethod
    def range_migration_correction(data: np.ndarray, range_bins: np.ndarray,
                                   velocity: float, wavelength: float) -> np.ndarray:
        """距离徙动校正"""
        corrected_data = np.zeros_like(data)

        for pulse_idx in range(data.shape[1]):
            # 计算距离徙动量
            time_offset = pulse_idx  # 脉冲间隔时间
            range_migration = velocity * time_offset

            # 计算需要校正的距离门数
            range_bin_shift = int(round(2 * range_migration / (3e8 / range_bins.shape[0])))

            # 进行插值校正
            if abs(range_bin_shift) < len(range_bins):
                if range_bin_shift > 0:
                    corrected_data[range_bin_shift:, pulse_idx] = data[:-range_bin_shift, pulse_idx]
                elif range_bin_shift < 0:
                    corrected_data[:range_bin_shift, pulse_idx] = data[-range_bin_shift:, pulse_idx]
                else:
                    corrected_data[:, pulse_idx] = data[:, pulse_idx]

        return corrected_data

    @staticmethod
    def sidelobe_suppression(signal: np.ndarray, window_func: str = 'hamming') -> np.ndarray:
        """旁瓣抑制"""
        if window_func == 'hamming':
            window = WindowFunctions.hamming(len(signal))
        elif window_func == 'blackman':
            window = WindowFunctions.blackman(len(signal))
        elif window_func == 'kaiser':
            window = WindowFunctions.kaiser(len(signal))
        else:
            window = np.ones(len(signal))

        return signal * window


# 便捷函数
def next_power_of_2(n: int) -> int:
    """找到大于等于n的最小2的幂次"""
    return 2 ** math.ceil(math.log2(n))


def zero_pad(signal: np.ndarray, target_length: int, mode: str = 'center') -> np.ndarray:
    """信号零填充"""
    if len(signal) >= target_length:
        return signal[:target_length]

    pad_length = target_length - len(signal)

    if mode == 'center':
        pad_before = pad_length // 2
        pad_after = pad_length - pad_before
        return np.pad(signal, (pad_before, pad_after), 'constant')
    elif mode == 'end':
        return np.pad(signal, (0, pad_length), 'constant')
    elif mode == 'start':
        return np.pad(signal, (pad_length, 0), 'constant')
    else:
        return np.pad(signal, (0, pad_length), 'constant')


def circular_shift(signal: np.ndarray, shift: int) -> np.ndarray:
    """循环移位"""
    return np.roll(signal, shift)


def time_reverse(signal: np.ndarray) -> np.ndarray:
    """时间反转"""
    return signal[::-1]


def complex_conjugate(signal: np.ndarray) -> np.ndarray:
    """复共轭"""
    return np.conj(signal)


def signal_energy(signal: np.ndarray) -> float:
    """信号能量"""
    return np.sum(np.abs(signal) ** 2)


def signal_power(signal: np.ndarray) -> float:
    """信号平均功率"""
    return signal_energy(signal) / len(signal)


def snr_estimate(signal: np.ndarray, noise_samples: Optional[np.ndarray] = None) -> float:
    """信噪比估计"""
    if noise_samples is not None:
        signal_power_val = signal_power(signal)
        noise_power_val = signal_power(noise_samples)
    else:
        # 简单的信噪比估计：假设前10%为噪声
        noise_end = len(signal) // 10
        signal_power_val = signal_power(signal[noise_end:])
        noise_power_val = signal_power(signal[:noise_end])

    return 10 * np.log10(signal_power_val / max(noise_power_val, 1e-10))
