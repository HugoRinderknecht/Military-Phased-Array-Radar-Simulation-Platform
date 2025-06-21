import numpy as np
from typing import List, Tuple, Optional
from models.tracking import Detection
from models.radar_system import RadarSystem
from models.environment import Environment
from config.settings import Config


class SignalProcessor:
    def __init__(self, radar_system: RadarSystem, environment: Environment):
        self.radar_system = radar_system
        self.environment = environment
        self.c = Config.SPEED_OF_LIGHT

    def process_radar_signal(self, raw_signal: np.ndarray, timestamp: float) -> List[Detection]:
        """处理雷达信号的主函数"""
        try:
            clutter_suppressed = self._adaptive_clutter_suppression(raw_signal)
            sidelobe_suppressed = self._two_stage_sidelobe_suppression(clutter_suppressed)
            compressed = self._pulse_compression(sidelobe_suppressed)
            detections = self._cfar_detection(compressed, timestamp)
            doppler_processed = self._adaptive_doppler_processing(detections)
            return doppler_processed
        except Exception as e:
            print(f"Error in signal processing: {e}")
            return []

    def _adaptive_clutter_suppression(self, signal: np.ndarray) -> np.ndarray:
        """自适应杂波抑制"""
        n_channels = 16
        training_length = max(1, int(len(signal) * getattr(self.environment, 'clutter_density', 0.1)))

        # 修复：初始化为复数类型
        clutter_covariance = np.eye(n_channels, dtype=complex) * 0.1

        # 确保有足够的数据进行训练
        max_iterations = min(training_length, len(signal) - n_channels)
        if max_iterations <= 0:
            return signal.copy()

        for i in range(max_iterations):
            if i + n_channels <= len(signal):
                data_vector = signal[i:i + n_channels]
                # 计算外积并累加到协方差矩阵
                clutter_covariance += np.outer(data_vector, np.conj(data_vector))

        # 归一化协方差矩阵
        if training_length > 0:
            clutter_covariance /= training_length

        try:
            # 添加对角加载以确保矩阵可逆
            inv_cov = np.linalg.inv(clutter_covariance + np.eye(n_channels, dtype=complex) * 1e-6)
            steering_vector = np.ones(n_channels, dtype=complex)
            weights = inv_cov @ steering_vector

            # 归一化权重
            denominator = np.conj(steering_vector).T @ inv_cov @ steering_vector
            if abs(denominator) > 1e-10:
                weights /= denominator
            else:
                weights = np.ones(n_channels, dtype=complex) / n_channels
        except np.linalg.LinAlgError:
            # 如果矩阵求逆失败，使用均匀权重
            weights = np.ones(n_channels, dtype=complex) / n_channels

        # 应用权重进行杂波抑制
        output = np.zeros(len(signal), dtype=complex)
        for i in range(len(signal) - n_channels + 1):
            if i + n_channels <= len(signal):
                data_vector = signal[i:i + n_channels]
                output[i + n_channels // 2] = np.conj(weights).T @ data_vector

        # 处理边界情况
        for i in range(n_channels // 2):
            output[i] = signal[i]
        for i in range(len(signal) - n_channels // 2, len(signal)):
            output[i] = signal[i]

        return output

    def _two_stage_sidelobe_suppression(self, signal: np.ndarray) -> np.ndarray:
        """两级旁瓣抑制"""
        try:
            stage1 = self._improved_gsc_sidelobe_suppression(signal)
            stage2 = self._sum_diff_mainlobe_suppression(stage1)
            return stage2
        except Exception as e:
            print(f"Error in sidelobe suppression: {e}")
            return signal.copy()

    def _improved_gsc_sidelobe_suppression(self, signal: np.ndarray) -> np.ndarray:
        """改进的GSC旁瓣抑制"""
        n_aux = 8
        aux_weights = np.zeros(n_aux, dtype=complex)
        mu = 0.01

        output = signal.copy()

        for i in range(n_aux, len(signal)):
            aux_input = signal[i - n_aux:i]
            error = output[i]

            # 更新权重
            aux_weights += mu * np.conj(error) * aux_input
            aux_weights = np.clip(aux_weights.real, -1, 1) + 1j * np.clip(aux_weights.imag, -1, 1)

            # 计算干扰抵消
            interference = np.sum(aux_weights * aux_input)
            output[i] = signal[i] - interference

        return output

    def _sum_diff_mainlobe_suppression(self, signal: np.ndarray) -> np.ndarray:
        """和差主瓣抑制"""
        sum_weights = np.array([1, 1, 1, 1], dtype=complex)
        diff_az_weights = np.array([1, -1, 1, -1], dtype=complex)
        diff_el_weights = np.array([1, 1, -1, -1], dtype=complex)

        output = signal.copy()

        for i in range(0, len(signal) - 4, 4):
            data_segment = signal[i:i + 4]

            sum_beam = np.sum(sum_weights * data_segment)
            diff_az = np.sum(diff_az_weights * data_segment)
            diff_el = np.sum(diff_el_weights * data_segment)

            if abs(sum_beam) > 1e-10:
                null_angle_az = np.angle(diff_az / sum_beam)
                null_angle_el = np.angle(diff_el / sum_beam)
                null_factor = np.exp(-1j * (null_angle_az + null_angle_el))

                output[i:i + 4] = data_segment * null_factor

        return output

    def _pulse_compression(self, signal: np.ndarray) -> np.ndarray:
        """脉冲压缩"""
        try:
            bandwidth = getattr(Config, 'DEFAULT_BANDWIDTH', 10e6)
            pulse_width = getattr(Config, 'DEFAULT_PULSE_WIDTH', 1e-6)
            fs = bandwidth * 2

            # 生成LFM参考信号
            chirp_samples = int(pulse_width * fs)
            chirp_rate = bandwidth / pulse_width

            lfm_ref = np.zeros(chirp_samples, dtype=complex)
            for i in range(chirp_samples):
                t_chirp = i / fs
                lfm_ref[i] = np.exp(1j * np.pi * chirp_rate * t_chirp ** 2)

            # 执行匹配滤波
            compressed = np.correlate(signal, np.conj(lfm_ref[::-1]), mode='same')
            return compressed

        except Exception as e:
            print(f"Error in pulse compression: {e}")
            return signal.copy()

    def _cfar_detection(self, signal: np.ndarray, timestamp: float) -> List[Detection]:
        """CFAR检测"""
        training_cells = 32
        guard_cells = 4
        pfa = 1e-6

        # 计算CFAR阈值因子
        alpha = training_cells * (pfa ** (-1 / training_cells) - 1)

        # 环境自适应调整
        clutter_density = getattr(self.environment, 'clutter_density', 0.1)
        interference_level = getattr(self.environment, 'interference_level', 0.0)

        if clutter_density > 0.5:
            alpha *= 1.5
        if interference_level > 0.1:
            alpha *= 1.3

        detections = []
        half_window = training_cells // 2 + guard_cells

        for i in range(half_window, len(signal) - half_window):
            try:
                # 训练数据（排除保护单元）
                training_data = np.concatenate([
                    signal[i - half_window:i - guard_cells],
                    signal[i + guard_cells + 1:i + half_window + 1]
                ])

                # 计算噪声功率
                noise_power = np.mean(np.abs(training_data) ** 2)
                threshold = alpha * noise_power
                test_power = abs(signal[i]) ** 2

                if test_power > threshold:
                    # 计算距离
                    range_val = i * self.c / (2 * getattr(Config, 'DEFAULT_SAMPLING_RATE', 100e6))
                    snr = 10 * np.log10(test_power / (noise_power + 1e-10))

                    detection = Detection(
                        range=range_val,
                        azimuth=0.0,
                        elevation=0.0,
                        velocity=0.0,
                        snr=snr,
                        rcs_estimate=0.0,
                        timestamp=timestamp,
                        cell_index=i,
                        low_altitude=(range_val * np.sin(0.0) < 500)
                    )
                    detections.append(detection)
            except Exception as e:
                print(f"Error in CFAR detection at index {i}: {e}")
                continue

        return detections

    def _adaptive_doppler_processing(self, detections: List[Detection]) -> List[Detection]:
        """自适应多普勒处理"""
        try:
            prf = getattr(Config, 'DEFAULT_PRF', 1000)
            n_pulses = 64

            for detection in detections:
                # 模拟相位历史
                phase_history = np.random.random(n_pulses) * 2 * np.pi
                phase_history_complex = np.exp(1j * phase_history)

                # FFT多普勒处理
                fft_result = np.fft.fft(phase_history_complex)
                max_bin = np.argmax(np.abs(fft_result))
                doppler_freq = (max_bin - n_pulses // 2) * prf / n_pulses

                # 计算径向速度
                frequency = getattr(self.radar_system, 'frequency', 10e9)
                detection.velocity = doppler_freq * self.c / (2 * frequency)

        except Exception as e:
            print(f"Error in Doppler processing: {e}")

        return detections
