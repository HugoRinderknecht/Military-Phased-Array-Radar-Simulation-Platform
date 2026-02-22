# 数学工具函数模块 (Math Utilities Module)
# 本模块提供雷达仿真中常用的数学计算函数

import numpy as np
from typing import Tuple, Union
from radar.common.constants import PhysicsConstants


# ==================== dB转换函数 ====================

def db_to_linear(db_value: float) -> float:
    """将dB值转换为线性值
    
    Args:
        db_value: dB值
    
    Returns:
        线性值
    
    公式: linear = 10^(dB/10)
    """
    return 10.0 ** (db_value / 10.0)


def linear_to_db(linear_value: float) -> float:
    """将线性值转换为dB值
    
    Args:
        linear_value: 线性值（必须为正数）
    
    Returns:
        dB值
    """
    if linear_value <= 0:
        raise ValueError(f'线性值必须为正数, 得到: {linear_value}')
    return 10.0 * np.log10(linear_value)


# ==================== 距离分辨率 ====================

def range_resolution(bandwidth: float) -> float:
    """计算雷达的距离分辨率
    
    Args:
        bandwidth: 信号带宽 (Hz)
    
    Returns:
        距离分辨率 (m)
    """
    return PhysicsConstants.C / (2.0 * bandwidth)


# ==================== 多普勒频率 ====================

def doppler_frequency(velocity: float, wavelength: float) -> float:
    """计算多普勒频率
    
    Args:
        velocity: 目标径向速度 (m/s)
        wavelength: 雷达波长 (m)
    
    Returns:
        多普勒频率 (Hz)
    """
    return -2.0 * velocity / wavelength


def doppler_velocity(fd: float, wavelength: float) -> float:
    """从多普勒频率计算径向速度"""
    return -fd * wavelength / 2.0


# ==================== 雷达距离方程 ====================

def radar_equation(
    pt: float,
    g: float,
    rcs: float,
    wavelength: float,
    distance: float,
    losses: float = 0.0
) -> float:
    """计算雷达接收功率（基本雷达方程）"""
    l_linear = db_to_linear(losses)
    numerator = pt * (g ** 2) * (wavelength ** 2) * rcs
    denominator = ((4 * np.pi) ** 3) * (distance ** 4) * l_linear
    return numerator / denominator


def two_wayPropagation_loss(distance: float, wavelength: float) -> float:
    """计算双程传播损耗 (dB)"""
    return 20.0 * np.log10(4.0 * np.pi * distance / wavelength)


# ==================== SNR计算 ====================

def snr_linear(pr: float, noise_power: float) -> float:
    """计算信噪比（线性值）"""
    if noise_power <= 0:
        raise ValueError(f'噪声功率必须为正数')
    return pr / noise_power


def snr_db(pr: float, noise_power: float) -> float:
    """计算信噪比（dB值）"""
    return linear_to_db(snr_linear(pr, noise_power))


def noise_power(bandwidth: float, temperature: float = 290.0, noise_figure: float = 3.0) -> float:
    """计算热噪声功率 (W)"""
    k = PhysicsConstants.K
    f_linear = db_to_linear(noise_figure)
    return k * temperature * bandwidth * f_linear


# ==================== 脉冲压缩增益 ====================

def pulse_compression_gain(pulse_width: float, bandwidth: float) -> float:
    """计算脉冲压缩增益（线性值）"""
    return pulse_width * bandwidth


def pulse_compression_gain_db(pulse_width: float, bandwidth: float) -> float:
    """计算脉冲压缩增益 (dB)"""
    return linear_to_db(pulse_compression_gain(pulse_width, bandwidth))


# ==================== 相干积累增益 ====================

def coherent_integration_gain(n_pulses: int) -> float:
    """计算相干积累增益（线性值）"""
    return float(n_pulses)


def coherent_integration_gain_db(n_pulses: int) -> float:
    """计算相干积累增益 (dB)"""
    return linear_to_db(coherent_integration_gain(n_pulses))


# ==================== 非相干积累增益 ====================

def noncoherent_integration_gain(n_pulses: int, efficiency: float = 0.8) -> float:
    """计算非相干积累增益（线性值）"""
    return float(n_pulses ** efficiency)


def noncoherent_integration_gain_db(n_pulses: int, efficiency: float = 0.8) -> float:
    """计算非相干积累增益 (dB)"""
    return linear_to_db(noncoherent_integration_gain(n_pulses, efficiency))


# ==================== 波束宽度计算 ====================

def beam_width(aperture: float, wavelength: float) -> float:
    """计算天线波束宽度（弧度）"""
    k = 1.0
    return k * wavelength / aperture


def beam_width_degrees(aperture: float, wavelength: float) -> float:
    """计算天线波束宽度（度）"""
    return np.degrees(beam_width(aperture, wavelength))


# ==================== 天线增益计算 ====================

def antenna_gain(aperture_area: float, wavelength: float, efficiency: float = 0.55) -> float:
    """计算天线增益（线性值）"""
    effective_area = aperture_area * efficiency
    return 4.0 * np.pi * effective_area / (wavelength ** 2)


def antenna_gain_db(aperture_area: float, wavelength: float, efficiency: float = 0.55) -> float:
    """计算天线增益 (dB)"""
    return linear_to_db(antenna_gain(aperture_area, wavelength, efficiency))


# ==================== 最大探测距离 ====================

def max_detection_range(
    pt: float, g: float, wavelength: float, rcs: float,
    snr_min: float, noise_power: float, losses: float = 0.0
) -> float:
    """计算雷达最大探测距离 (m)"""
    l_linear = db_to_linear(losses)
    numerator = pt * (g ** 2) * (wavelength ** 2) * rcs
    denominator = ((4 * np.pi) ** 3) * snr_min * noise_power * l_linear
    return (numerator / denominator) ** 0.25


# ==================== 角度转换 ====================

def deg_to_rad(degrees: float) -> float:
    """度转弧度"""
    return degrees * np.pi / 180.0


def rad_to_deg(radians: float) -> float:
    """弧度转度"""
    return radians * 180.0 / np.pi


def wrap_angle(angle: float, max_angle: float = 360.0) -> float:
    """角度回绕到指定范围"""
    half = max_angle / 2.0
    return ((angle + half) % max_angle) - half


# ==================== 矢量运算 ====================

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """归一化向量"""
    norm = np.linalg.norm(v)
    if norm == 0:
        raise ValueError('无法归一化零向量')
    return v / norm


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的夹角 (弧度)"""
    v1_norm = normalize_vector(v1)
    v2_norm = normalize_vector(v2)
    return np.arccos(np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0))


# ==================== 角度归一化 ====================

def normalize_angle(angle: float) -> float:
    """将角度归一化到 [-pi, pi] 范围"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def angle_difference(angle1: float, angle2: float) -> float:
    """计算两个角度的最小差值"""
    diff = angle1 - angle2
    return normalize_angle(diff)


# ==================== 向量运算辅助函数 ====================

def vector_magnitude(v: np.ndarray) -> float:
    """计算向量模长"""
    return np.linalg.norm(v)


def vector_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算两个向量之间的距离"""
    return np.linalg.norm(v1 - v2)


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """计算向量点积"""
    return np.dot(v1, v2)


def cross_product(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """计算向量叉积"""
    return np.cross(v1, v2)


# ==================== 统计函数 ====================

def mean(data: np.ndarray) -> float:
    """计算均值"""
    return np.mean(data)


def std(data: np.ndarray) -> float:
    """计算标准差"""
    return np.std(data)


def variance(data: np.ndarray) -> float:
    """计算方差"""
    return np.var(data)


def median(data: np.ndarray) -> float:
    """计算中位数"""
    return np.median(data)


# ==================== 插值函数 ====================

def linear_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """线性插值"""
    return np.interp(x_new, x, y)


def spline_interpolate(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """样条插值"""
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    return cs(x_new)


def nearest_neighbor(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """最近邻插值"""
    indices = np.searchsorted(x, x_new)
    indices = np.clip(indices, 0, len(y) - 1)
    return y[indices]


# ==================== 窗函数 ====================

def get_window(window_type: str, n: int, **kwargs) -> np.ndarray:
    """获取窗函数"""
    return np.window(n, window_type, **kwargs)


def apply_window(signal: np.ndarray, window: np.ndarray) -> np.ndarray:
    """应用窗函数"""
    return signal * window


# ==================== FFT相关 ====================

def fft_shift(x: np.ndarray) -> np.ndarray:
    """FFT频移"""
    return np.fft.fftshift(x)


def fft_frequency(n: int, d: float = 1.0) -> np.ndarray:
    """计算FFT频率轴"""
    return np.fft.fftfreq(n, d)


def next_power_of_2(n: int) -> int:
    """计算下一个2的幂次"""
    return 1 << (n - 1).bit_length()


__all__ = [
    'db_to_linear', 'linear_to_db', 'range_resolution',
    'doppler_frequency', 'doppler_velocity', 'two_wayPropagation_loss',
    'snr_linear', 'snr_db', 'noise_power', 'radar_equation',
    'max_detection_range', 'pulse_compression_gain', 'pulse_compression_gain_db',
    'coherent_integration_gain', 'coherent_integration_gain_db',
    'noncoherent_integration_gain', 'noncoherent_integration_gain_db',
    'beam_width', 'beam_width_degrees', 'antenna_gain', 'antenna_gain_db',
    'deg_to_rad', 'rad_to_deg', 'wrap_angle',
    'normalize_vector', 'angle_between_vectors',
    'normalize_angle', 'angle_difference',
    'vector_magnitude', 'vector_distance', 'dot_product', 'cross_product',
    'mean', 'std', 'variance', 'median',
    'linear_interpolate', 'spline_interpolate', 'nearest_neighbor',
    'get_window', 'apply_window',
    'fft_shift', 'fft_frequency', 'next_power_of_2',
]
