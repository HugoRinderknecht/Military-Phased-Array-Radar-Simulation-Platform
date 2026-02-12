# signal_processor.py - 信号处理器
"""
本模块实现雷达信号处理链。

信号处理流程：
1. 波形产生
2. 脉冲压缩
3. MTI/MTD
4. CFAR检测
5. 测角
6. 点迹提取
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from radar.common.logger import get_logger
from radar.common.types import Plot, Position3D, Velocity3D, AzimuthElevation
from radar.common.constants import PhysicsConstants, MathConstants
from radar.common.utils import (
    generate_lfm_pulse, pulse_compression, generate_window,
    compute_fft, next_power_of_2, db_to_linear, linear_to_db
)


class ProcessingStage(Enum):
    """处理阶段"""
    WAVEFORM_GEN = "waveform_gen"       # 波形产生
    PULSE_COMPRESSION = "pulse_compression" # 脉冲压缩
    MTI = "mti"                       # MTI滤波
    MTD = "mtd"                       # MTD处理
    CFAR = "cfar"                      # CFAR检测
    ANGLE_EST = "angle_est"              # 测角
    PLOT_EXTRACT = "plot_extract"          # 点迹提取


@dataclass
class Detection:
    """
    检测结果

    Attributes:
        range_bin: 距离单元索引
        doppler_bin: 多普勒单元索引
        snr: 信噪比 [dB]
        amplitude: 幅度（线性）
        phase: 相位 [弧度]
    """
    range_bin: int
    doppler_bin: int
    snr: float
    amplitude: float
    phase: float


@dataclass
class SignalProcessorConfig:
    """
    信号处理器配置

    Attributes:
        # 波形参数
        pulse_width: 脉冲宽度 [秒]
        bandwidth: 带宽 [Hz]
        sample_rate: 采样率 [Hz]

        # 脉冲压缩
        compression_enabled: 是否脉冲压缩
        window_type: 加窗类型

        # MTD参数
        mtd_enabled: 是否启用MTD
        fft_size: FFT大小
        doppler_bins: 多普勒通道数

        # CFAR参数
        cfar_enabled: 是否启用CFAR
        cfar_type: CFAR类型 (CA, GO, SO, OS)
        training_cells: 训练单元数
        guard_cells: 保护单元数
        false_alarm_rate: 虚警率

        # 测角参数
        angle_est_enabled: 是否测角
        angle_method: 测角方法

        # 点迹提取
        min_plots: 最小点迹数
        max_plots: 最大点迹数
    """
    # 波形
    pulse_width: float = 10e-6
    bandwidth: float = 10e6
    sample_rate: float = 20e6

    # 脉冲压缩
    compression_enabled: bool = True
    window_type: str = 'hamming'

    # MTD
    mtd_enabled: bool = True
    fft_size: int = 1024
    doppler_bins: int = 64

    # CFAR
    cfar_enabled: bool = True
    cfar_type: str = 'ca'
    training_cells: int = 20
    guard_cells: int = 2
    false_alarm_rate: float = 1e-6

    # 测角
    angle_est_enabled: bool = True
    angle_method: str = 'monopulse'

    # 点迹
    min_plots: int = 1
    max_plots: int = 1000


class SignalProcessor:
    """
    信号处理器

    实现完整的雷达信号处理链。
    """

    def __init__(self, config: Optional[SignalProcessorConfig] = None):
        """
        初始化信号处理器

        Args:
            config: 处理器配置
        """
        self._logger = get_logger("signal_proc")
        self._config = config or SignalProcessorConfig()

        # 计算参数
        self._n_samples = int(
            self._config.sample_rate * self._config.pulse_width
        )
        self._range_resolution = PhysicsConstants.C / (2 * self._config.bandwidth)
        self._max_range = self._n_samples * self._range_resolution

        # 生成参考波形（用于脉冲压缩）
        self._reference_waveform = generate_lfm_pulse(
            self._config.sample_rate,
            self._config.pulse_width,
            self._config.bandwidth
        )

        # 预计算FFT长度
        self._fft_size = next_power_of_2(self._n_samples * 2)

        # 统计信息
        self._stats = {
            'processed_pulses': 0,
            'total_detections': 0,
            'average_snr': 0.0,
        }

        self._logger.info(
            f"信号处理器初始化: "
            f"BW={self._config.bandwidth/1e6}MHz, "
            f"PW={self._config.pulse_width*1e6}us, "
            f"Fs={self._config.sample_rate/1e6}MHz"
        )

    def process_pulse(self, received_signal: np.ndarray,
                    beam_az: float, beam_el: float) -> List[Plot]:
        """
        处理单个脉冲

        Args:
            received_signal: 接收信号 [复数]
            beam_az: 波束方位角 [弧度]
            beam_el: 波束俯仰角 [弧度]

        Returns:
            检测到的点迹列表
        """
        plots = []

        # Step 1: 脉冲压缩
        if self._config.compression_enabled:
            compressed = self._pulse_compression(received_signal)
        else:
            compressed = np.abs(received_signal)

        # Step 2: MTD处理（多普勒处理）
        if self._config.mtd_enabled:
            range_doppler_map = self._mtd_processing(compressed)
        else:
            range_doppler_map = compressed.reshape(-1, 1)

        # Step 3: CFAR检测
        if self._config.cfar_enabled:
            detections = self._cfar_detection(range_doppler_map)
        else:
            detections = self._threshold_detection(range_doppler_map)

        # Step 4: 测角和点迹提取
        for detection in detections:
            # 计算目标参数
            range_val = detection.range_bin * self._range_resolution

            # 多普勒速度
            if detection.doppler_bin < self._config.doppler_bins:
                doppler_freq = (detection.doppler_bin -
                               self._config.doppler_bins / 2)
                wavelength = PhysicsConstants.C / 10e9  # X波段
                velocity = -doppler_freq * wavelength / 2
            else:
                velocity = 0.0

            # 位置
            sin_el = np.sin(beam_el)
            sin_az = np.sin(beam_az)
            cos_el = np.cos(beam_el)
            cos_az = np.cos(beam_az)

            x = range_val * cos_el * sin_az
            y = range_val * cos_el * cos_az
            z = range_val * sin_el

            plot = Plot(
                id=self._stats['total_detections'],
                timestamp=0,  # TODO: 使用实际时间戳
                position=Position3D(x, y, z),
                velocity=Velocity3D(0, 0, velocity),
                range_val=range_val,
                azimuth=beam_az,
                elevation=beam_el,
                doppler_vel=velocity,
                snr=detection.snr
            )

            plots.append(plot)
            self._stats['total_detections'] += 1

        self._stats['processed_pulses'] += 1

        # 限制点迹数量
        if len(plots) > self._config.max_plots:
            plots = plots[:self._config.max_plots]

        return plots

    def _pulse_compression(self, signal: np.ndarray) -> np.ndarray:
        """
        脉冲压缩

        使用匹配滤波实现。

        Args:
            signal: 接收信号

        Returns:
            压缩后的信号（幅度）
        """
        # 应用窗函数
        window = generate_window(
            self._config.window_type,
            len(self._reference_waveform)
        )
        windowed_ref = self._reference_waveform * window

        # 脉冲压缩（频域实现）
        compressed = pulse_compression(signal, windowed_ref, self._fft_size)

        return np.abs(compressed[:len(signal)])

    def _mtd_processing(self, signal: np.ndarray) -> np.ndarray:
        """
        MTD处理（动目标显示）

        Args:
            signal: 输入信号（距离向）

        Returns:
            距离-多普勒图 [range_bins, doppler_bins]
        """
        # 多脉冲多普勒处理需要多个脉冲
        # 这里简化为单脉冲FFT

        # 计算每个距离单元的多普勒FFT
        # 实际应该积累多个脉冲

        # 简化：对信号进行FFT
        # 实际MTD需要多脉冲处理

        # 这里返回距离-幅度
        # 完整MTD会返回range-doppler矩阵
        return signal.reshape(-1, 1)

    def _cfar_detection(self, range_doppler_map: np.ndarray) -> List[Detection]:
        """
        CFAR检测

        Args:
            range_doppler_map: 距离-多普勒图

        Returns:
            检测列表
        """
        detections = []

        n_train = self._config.training_cells
        n_guard = self._config.guard_cells
        p_fa = self._config.false_alarm_rate

        # 计算CA-CFAR门限因子
        alpha = n_train * (p_fa ** (-1 / n_train) - 1)

        # 对每个距离单元进行处理
        n_range = len(range_doppler_map)
        half_train = n_train // 2
        half_guard = n_guard // 2

        for i in range(half_train + half_guard,
                       n_range - half_train - half_guard):

            # 计算杂波功率估计
            lead_cells = range_doppler_map[i - half_train - half_guard:i - half_guard]
            lag_cells = range_doppler_map[i + half_guard + 1:
                                              i + half_guard + half_train + 1]

            noise_level = (np.mean(lead_cells) + np.mean(lag_cells)) / 2

            # 门限
            threshold = alpha * noise_level

            # 检测
            signal_level = range_doppler_map[i]
            snr = linear_to_db(signal_level / (noise_level + 1e-10))

            if signal_level > threshold:
                detection = Detection(
                    range_bin=i,
                    doppler_bin=0,  # 简化
                    snr=snr,
                    amplitude=signal_level,
                    phase=np.angle(signal_level)
                )
                detections.append(detection)

        return detections

    def _threshold_detection(self, range_doppler_map: np.ndarray) -> List[Detection]:
        """
        固定门限检测

        Args:
            range_doppler_map: 距离-多普勒图

        Returns:
            检测列表
        """
        # 固定门限
        threshold_db = 10.0  # 10dB
        threshold = db_to_linear(threshold_db)

        detections = []

        for i, level in enumerate(range_doppler_map):
            if level > threshold:
                snr = linear_to_db(level)
                detection = Detection(
                    range_bin=i,
                    doppler_bin=0,
                    snr=snr,
                    amplitude=level,
                    phase=np.angle(level)
                )
                detections.append(detection)

        return detections

    def get_statistics(self) -> dict:
        """获取处理统计信息"""
        return self._stats.copy()


class MonopulseAngleEstimator:
    """
    单脉冲测角器

    使用和差波束进行角度估计。
    """

    def __init__(self):
        self._logger = get_logger("monopulse")

    def estimate_angle(self, sum_beam: complex,
                      delta_az: complex,
                      delta_el: complex) -> Tuple[float, float]:
        """
        估计角度

        Args:
            sum_beam: 和波束输出
            delta_az: 方位差波束输出
            delta_el: 俯仰差波束输出

        Returns:
            (azimuth_error, elevation_error) [弧度]
        """
        # 计算比值
        if np.abs(sum_beam) < 1e-10:
            return 0.0, 0.0

        ratio_az = delta_az / sum_beam
        ratio_el = delta_el / sum_beam

        # 比值到角度（需要标定）
        # 简化：角度 ≈ Re(ratio) / k
        k = 1.0  # 斜率因子，需要校准

        az_error = k * np.real(ratio_az)
        el_error = k * np.real(ratio_el)

        return az_error, el_error


__all__ = [
    "ProcessingStage",
    "Detection",
    "SignalProcessorConfig",
    "SignalProcessor",
    "MonopulseAngleEstimator",
]
