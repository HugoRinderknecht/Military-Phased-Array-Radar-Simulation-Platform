import numpy as np
from typing import Dict, List, Tuple, Deque, Optional
from collections import deque
from numba import njit, float64
from models.tracking import Track, Detection
from models.radar_system import RadarSystem
from models.environment import Environment

# 预编译常量
FOUR_PI = 4 * np.pi
FOUR_PI_CUBED = FOUR_PI ** 3


class RCSProcessor:
    def __init__(self, radar_system: RadarSystem, environment: Environment):
        self.radar_system = radar_system
        self.environment = environment
        self.k_factor = 1.0
        self.calibration_targets = {}

        # 预计算雷达相关常量
        self.base_k = self._precompute_base_k()
        self.pulse_width = 20e-6
        self.pulse_count = 64

        # 特征分析窗口大小
        self.feature_window = 50

    def _precompute_base_k(self) -> float:
        """预计算基础K因子"""
        return (self.radar_system.radar_power *
                (self.radar_system.wavelength ** 2) *
                self.radar_system.gain / FOUR_PI_CUBED)

    def process_rcs(self, track: Track) -> Dict[str, float]:
        """处理RCS的主流程"""
        norm = max(np.linalg.norm(track.state[:3]), 1e-6)
        k_factor = self._calibrate_k_factor(track, norm)

        self._apply_compensation_factors(track)
        rcs_value = self._calculate_rcs(track, k_factor)

        if self._is_calibration_target(track):
            self._dynamic_calibration(track, rcs_value)

        fused_rcs = self._fuse_multisensor_rcs(track)
        features = self._analyze_rcs_features(track)

        return {
            'rcs_value': fused_rcs,
            'k_factor': k_factor,
            'features': features
        }

    def _calibrate_k_factor(self, track: Track, norm: float) -> float:
        """校准K因子"""
        range_km = norm / 1000.0
        atmospheric_loss = self._calculate_atmospheric_loss(range_km)
        scan_loss = self._calculate_scan_loss(track.state[:3], norm)

        return self.base_k / (atmospheric_loss * scan_loss)

    def _calculate_atmospheric_loss(self, range_km: float) -> float:
        """计算大气损耗"""
        loss_db = self.environment.weather.atmospheric_loss(
            self.radar_system.frequency, range_km
        )
        return 10 ** (loss_db / 10.0)

    @staticmethod
    @njit(float64(float64[:], float64), cache=True)
    def _calculate_scan_loss(pos: np.ndarray, norm: float) -> float:
        """计算扫描损耗 (NUMBA加速)"""
        azimuth = np.arctan2(pos[1], pos[0])
        elevation = np.arcsin(pos[2] / norm)

        # 使用三角恒等式优化计算
        cos_az = np.cos(azimuth)
        cos_el = np.cos(elevation)
        scan_loss = (cos_az * cos_el) ** 2

        return max(scan_loss, 0.1)

    def _apply_compensation_factors(self, track: Track):
        """应用补偿因子"""
        if track.detections_history:
            latest_detection = track.detections_history[-1]
            range_val = latest_detection.range

            # 使用平方代替乘方
            range_correction = (range_val / 10000.0) ** 4
            freq_correction = (self.radar_system.frequency / 1e10) ** 2

            if track.rcs_history:
                track.rcs_history[-1] *= range_correction * freq_correction

    def _calculate_rcs(self, track: Track, k_factor: float) -> float:
        """计算RCS值"""
        if not track.detections_history:
            return 0.0

        detection = track.detections_history[-1]
        range_val = detection.range
        snr_linear = 10 ** (detection.snr / 10.0)

        return self._rcs_core(
            range_val,
            self.radar_system.frequency,
            snr_linear,
            k_factor,
            self.pulse_width,
            self.pulse_count
        )

    @staticmethod
    @njit(float64(float64, float64, float64, float64, float64, float64), cache=True)
    def _rcs_core(range_val, frequency, snr_linear, k_factor, pulse_width, pulse_count) -> float:
        """RCS核心计算 (NUMBA加速)"""
        range_sq = range_val * range_val
        rcs_linear = (range_sq * range_sq * frequency * frequency * snr_linear) / \
                     (k_factor * pulse_count * pulse_width)

        # 避免对数计算中的负值
        if rcs_linear < 1e-10:
            return -100.0  # 极小的dB值

        return 10.0 * np.log10(rcs_linear)

    def _is_calibration_target(self, track: Track) -> bool:
        """判断是否为校准目标"""
        return (
                len(track.rcs_history) >= 5 and
                np.std(track.rcs_history[-5:]) < 1.0 and
                track.confirmed
        )

    def _dynamic_calibration(self, track: Track, measured_rcs: float):
        """动态校准"""
        if track.track_id not in self.calibration_targets:
            self.calibration_targets[track.track_id] = {
                'expected_rcs': 10.0,  # dBsm
                'measurements': deque(maxlen=10)
            }

        calib_data = self.calibration_targets[track.track_id]
        calib_data['measurements'].append(measured_rcs)

        if len(calib_data['measurements']) >= 10:
            avg_measured = np.mean(calib_data['measurements'])
            calibration_error = calib_data['expected_rcs'] - avg_measured
            self.k_factor *= 10 ** (calibration_error / 20.0)

    def _fuse_multisensor_rcs(self, track: Track) -> float:
        """多传感器RCS融合"""
        if not track.rcs_history:
            return 0.0

        # 使用滑动窗口平均值
        window = min(5, len(track.rcs_history))
        radar_rcs = np.mean(track.rcs_history[-window:])

        # 模拟其他传感器读数
        weights = np.array([0.6, 0.25, 0.15])
        estimates = np.array([
            radar_rcs,
            radar_rcs + np.random.normal(0, 2),
            radar_rcs + np.random.normal(0, 3)
        ])

        return np.dot(weights, estimates)

    def _analyze_rcs_features(self, track: Track) -> Dict[str, float]:
        """分析RCS特征"""
        if len(track.rcs_history) < 3:
            return {'mean': 0, 'std': 0, 'trend': 0, 'max': 0, 'min': 0, 'range': 0}

        # 使用有限窗口分析
        window_size = min(self.feature_window, len(track.rcs_history))
        rcs_array = np.array(track.rcs_history[-window_size:])

        return self._compute_features(rcs_array)

    @staticmethod
    @njit(float64[:](float64[:]), cache=True)
    def _compute_features(rcs_array: np.ndarray) -> Dict[str, float]:
        """计算特征 (NUMBA加速)"""
        mean = np.mean(rcs_array)
        std = np.std(rcs_array)
        rcs_min = np.min(rcs_array)
        rcs_max = np.max(rcs_array)
        rcs_range = rcs_max - rcs_min

        # 仅当有足够数据时计算趋势
        trend = 0.0
        n = len(rcs_array)
        if n > 1:
            # 使用高效线性回归计算趋势
            x = np.arange(n)
            sum_x = np.sum(x)
            sum_y = np.sum(rcs_array)
            sum_xy = np.sum(x * rcs_array)
            sum_x2 = np.sum(x * x)

            numerator = n * sum_xy - sum_x * sum_y
            denominator = n * sum_x2 - sum_x * sum_x
            trend = numerator / denominator if denominator != 0 else 0.0

        return {
            'mean': mean,
            'std': std,
            'trend': trend,
            'max': rcs_max,
            'min': rcs_min,
            'range': rcs_range
        }
