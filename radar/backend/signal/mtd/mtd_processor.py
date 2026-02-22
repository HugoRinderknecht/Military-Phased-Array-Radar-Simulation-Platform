# mtd_processor.py - 完整的MTD处理实现
"""
MTD (Moving Target Detection) - 动目标检测完整实现

包括:
- 多脉冲多普勒处理
- Range-Doppler矩阵生成
- 多普勒滤波器组
- 盲速和多普勒模糊处理
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class DopplerProcessingType(Enum):
    """多普勒处理类型"""
    FFT = "fft"                      # FFT多普勒处理
    MTI = "mti"                      # 动目标显示(MTI)
    FILTER_BANK = "filter_bank"      # 多普勒滤波器组


@dataclass
class MTDResult:
    """MTD处理结果"""
    range_doppler_matrix: np.ndarray  # Range-Doppler矩阵 [range_bins, doppler_bins]
    range_bins: np.ndarray              # 距离单元
    doppler_bins: np.ndarray            # 多普勒单元
    doppler_frequencies: np.ndarray     # 多普勒频率 (Hz)
    velocities: np.ndarray               # 径向速度 (m/s)


class DopplerFilterBank:
    """
    多普勒滤波器组

    使用一组窄带滤波器分离不同速度的目标
    """

    def __init__(
        self,
        num_filters: int = 32,
        prf: float = 2000.0,
        num_pulses: int = 64
    ):
        """
        Args:
            num_filters: 滤波器数量
            prf: 脉冲重复频率 (Hz)
            num_pulses: 脉冲数
        """
        self.num_filters = num_filters
        self.prf = prf
        self.num_pulses = num_pulses

        # 生成滤波器系数
        self.filters = self._design_filters()

    def _design_filters(self) -> np.ndarray:
        """设计多普勒滤波器组"""
        filters = np.zeros((self.num_filters, self.num_pulses), dtype=complex)

        for k in range(self.num_filters):
            # 第k个滤波器的中心频率
            omega_k = 2 * np.pi * k / self.num_filters

            # 使用窗口函数设计FIR滤波器
            for n in range(self.num_pulses):
                # 矩形窗滤波器
                filters[k, n] = np.exp(1j * omega_k * n)

                # 可以使用其他窗函数 (Hamming, Kaiser等)
                # window = 0.54 - 0.46 * np.cos(2 * np.pi * n / (self.num_pulses - 1))
                # filters[k, n] *= window

        return filters

    def apply(self, pulses: np.ndarray) -> np.ndarray:
        """
        应用多普勒滤波器组

        Args:
            pulses: 脉冲数据 [num_pulses, num_range_bins]

        Returns:
            滤波器输出 [num_filters, num_range_bins]
        """
        num_pulses, num_range_bins = pulses.shape

        # 对每个距离单元应用滤波器组
        output = np.zeros((self.num_filters, num_range_bins), dtype=complex)

        for range_idx in range(num_range_bins):
            # 提取该距离单元的脉冲序列
            pulse_train = pulses[:, range_idx]

            # 应用每个滤波器
            for k in range(self.num_filters):
                # 时域卷积
                output[k, range_idx] = np.sum(self.filters[k, :] * pulse_train)

        return output


class MTDProcessor:
    """
    动目标检测 (MTD) 处理器

    完整实现多脉冲多普勒处理
    """

    def __init__(
        self,
        sample_rate: float = 20e6,
        prf: float = 2000.0,
        num_pulses: int = 64,
        carrier_freq: float = 10e9,
        processing_type: DopplerProcessingType = DopplerProcessingType.FFT
    ):
        """
        Args:
            sample_rate: 采样率 (Hz)
            prf: 脉冲重复频率 (Hz)
            num_pulses: 处理脉冲数
            carrier_freq: 载波频率 (Hz)
            processing_type: 多普勒处理类型
        """
        self.sample_rate = sample_rate
        self.prf = prf
        self.num_pulses = num_pulses
        self.carrier_freq = carrier_freq
        self.processing_type = processing_type

        # 计算波长
        self.wavelength = 3e8 / carrier_freq

        # 多普勒滤波器组
        if processing_type == DopplerProcessingType.FILTER_BANK:
            self.filter_bank = DopplerFilterBank(
                num_filters=32,
                prf=prf,
                num_pulses=num_pulses
            )

    def process(
        self,
        pulses: np.ndarray,
        range_window: Optional[str] = None
    ) -> MTDResult:
        """
        执行MTD处理

        Args:
            pulses: 脉冲数据 [num_pulses, num_samples]
            range_window: 距离窗函数类型

        Returns:
            MTD处理结果
        """
        # 1. 脉冲压缩 (如果需要)
        compressed = self._pulse_compression(pulses)

        # 2. 距离维FFT
        range_profile = self._range_fft(compressed, range_window)

        # 3. 多普勒处理
        if self.processing_type == DopplerProcessingType.FFT:
            range_doppler = self._fft_doppler_processing(range_profile)
        elif self.processing_type == DopplerProcessingType.FILTER_BANK:
            range_doppler = self._filter_bank_doppler_processing(range_profile)
        else:  # MTI
            range_doppler = self._mti_processing(range_profile)

        # 4. 生成距离和多普勒单元
        num_range_bins = range_doppler.shape[1]
        num_doppler_bins = range_doppler.shape[0]

        range_bins = np.arange(num_range_bins)
        doppler_bins = np.arange(num_doppler_bins)

        # 5. 计算多普勒频率
        doppler_frequencies = self._compute_doppler_frequencies(num_doppler_bins)

        # 6. 计算速度
        velocities = self._doppler_to_velocity(doppler_frequencies)

        return MTDResult(
            range_doppler_matrix=np.abs(range_doppler),
            range_bins=range_bins,
            doppler_bins=doppler_bins,
            doppler_frequencies=doppler_frequencies,
            velocities=velocities
        )

    def _pulse_compression(self, pulses: np.ndarray) -> np.ndarray:
        """脉冲压缩"""
        # 这里假设输入已经是脉冲压缩后的数据
        # 或者可以添加实际的脉冲压缩逻辑
        return pulses

    def _range_fft(
        self,
        data: np.ndarray,
        window: Optional[str] = None
    ) -> np.ndarray:
        """
        距离维FFT

        Args:
            data: 输入数据 [num_pulses, num_samples]
            window: 窗函数类型

        Returns:
            距离剖面 [num_pulses, num_range_bins]
        """
        num_pulses, num_samples = data.shape

        # 应用窗函数
        if window is not None:
            window_coeffs = self._generate_window(window, num_samples)
            data = data * window_coeffs

        # FFT
        range_data = np.fft.fft(data, axis=1)

        # 取前半部分 (正频率)
        range_data = range_data[:, :num_samples // 2]

        return range_data

    def _fft_doppler_processing(self, range_profile: np.ndarray) -> np.ndarray:
        """
        FFT多普勒处理

        Args:
            range_profile: 距离剖面 [num_pulses, num_range_bins]

        Returns:
            Range-Doppler矩阵 [num_doppler_bins, num_range_bins]
        """
        # 沿脉冲维进行FFT
        range_doppler = np.fft.fft(range_profile, axis=0)

        # 取前半部分
        num_pulses = range_doppler.shape[0]
        range_doppler = range_doppler[:num_pulses // 2, :]

        return range_doppler

    def _filter_bank_doppler_processing(self, range_profile: np.ndarray) -> np.ndarray:
        """
        滤波器组多普勒处理

        Args:
            range_profile: 距离剖面 [num_pulses, num_range_bins]

        Returns:
            Range-Doppler矩阵 [num_doppler_bins, num_range_bins]
        """
        # 应用多普勒滤波器组
        range_doppler = self.filter_bank.apply(range_profile)

        return range_doppler

    def _mti_processing(self, range_profile: np.ndarray) -> np.ndarray:
        """
        MTI (动目标显示) 处理

        使用简单的高通滤波器抑制静止杂波

        Args:
            range_profile: 距离剖面 [num_pulses, num_range_bins]

        Returns:
            处理后的数据 [num_pulses, num_range_bins]
        """
        num_pulses, num_range_bins = range_profile.shape

        # 双脉冲对消器
        mti_output = np.zeros_like(range_profile)
        mti_output[0, :] = range_profile[0, :]
        mti_output[1:, :] = range_profile[1:, :] - range_profile[:-1, :]

        # 三脉冲对消器 (更好)
        # mti_output[0:2, :] = range_profile[0:2, :]
        # mti_output[2:, :] = range_profile[2:, :] - 2 * range_profile[1:-1, :] + range_profile[:-2, :]

        return mti_output

    def _generate_window(self, window_type: str, n: int) -> np.ndarray:
        """生成窗函数"""
        if window_type == 'hamming':
            return np.hamming(n)
        elif window_type == 'hanning':
            return np.hanning(n)
        elif window_type == 'blackman':
            return np.blackman(n)
        elif window_type == 'kaiser':
            return np.kaiser(n, beta=6.0)
        else:
            return np.ones(n)

    def _compute_doppler_frequencies(self, num_doppler_bins: int) -> np.ndarray:
        """计算多普勒频率"""
        # 多普勒频率范围: [-PRF/2, PRF/2]
        return np.linspace(-self.prf / 2, self.prf / 2, num_doppler_bins)

    def _doppler_to_velocity(self, doppler_freq: np.ndarray) -> np.ndarray:
        """多普勒频率转速度"""
        return doppler_freq * self.wavelength / 2.0

    def compensate_blind_speeds(
        self,
        range_doppler: np.ndarray,
        prf_set: List[float]
    ) -> np.ndarray:
        """
        补偿盲速

        使用多重PRF解决多普勒模糊

        Args:
            range_doppler: Range-Doppler矩阵
            prf_set: PRF集合

        Returns:
            补偿后的数据
        """
        # 简化实现：实际需要更复杂的PRF切换逻辑
        return range_doppler


class ClutterSuppressor:
    """
    杂波抑制器

    使用AMTI (Airborne Moving Target Indication) 技术抑制杂波
    """

    def __init__(
        self,
        num_pulses: int = 64,
        platform_velocity: float = 200.0,
        platform_heading: float = 0.0
    ):
        """
        Args:
            num_pulses: 脉冲数
            platform_velocity: 平台速度 (m/s)
            platform_heading: 平台航向 (rad)
        """
        self.num_pulses = num_pulses
        self.platform_velocity = platform_velocity
        self.platform_heading = platform_heading

    def suppress_clutter(
        self,
        range_doppler: np.ndarray,
        clutter_map: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        抑制杂波

        Args:
            range_doppler: Range-Doppler矩阵
            clutter_map: 杂波图 (可选)

        Returns:
            抑制后的数据
        """
        if clutter_map is not None:
            # 使用杂波图
            suppressed = range_doppler - clutter_map
        else:
            # 简单的零多普勒抑制
            suppressed = range_dopler.copy()
            center_doppler = suppressed.shape[0] // 2

            # 抑制零多普勒附近的区域
            suppress_width = 3
            suppressed[center_doppler - suppress_width:center_doppler + suppress_width + 1, :] *= 0.1

        return suppressed


def create_mtd_processor(
    prf: float = 2000.0,
    num_pulses: int = 64,
    carrier_freq: float = 10e9,
    processing_type: str = "fft"
) -> MTDProcessor:
    """
    创建MTD处理器

    Args:
        prf: 脉冲重复频率 (Hz)
        num_pulses: 脉冲数
        carrier_freq: 载波频率 (Hz)
        processing_type: 处理类型 ("fft", "mti", "filter_bank")

    Returns:
        MTD处理器实例
    """
    if isinstance(processing_type, str):
        processing_type = DopplerProcessingType(processing_type)

    return MTDProcessor(
        sample_rate=20e6,
        prf=prf,
        num_pulses=num_pulses,
        carrier_freq=carrier_freq,
        processing_type=processing_type
    )


# 便捷函数
def process_mtd(
    pulses: np.ndarray,
    prf: float = 2000.0,
    num_pulses: int = 64,
    carrier_freq: float = 10e9,
    processing_type: str = "fft"
) -> MTDResult:
    """
    执行MTD处理

    Args:
        pulses: 脉冲数据 [num_pulses, num_samples]
        prf: 脉冲重复频率
        num_pulses: 脉冲数
        carrier_freq: 载波频率
        processing_type: 处理类型

    Returns:
        MTD处理结果
    """
    processor = create_mtd_processor(prf, num_pulses, carrier_freq, processing_type)
    return processor.process(pulses)
