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
        clutter_suppressed = self._adaptive_clutter_suppression(raw_signal)
        sidelobe_suppressed = self._two_stage_sidelobe_suppression(clutter_suppressed)
        compressed = self._pulse_compression(sidelobe_suppressed)
        detections = self._cfar_detection(compressed, timestamp)
        doppler_processed = self._adaptive_doppler_processing(detections)

        return doppler_processed

    def _adaptive_clutter_suppression(self, signal: np.ndarray) -> np.ndarray:
        n_channels = 16
        training_length = int(len(signal) * self.environment.clutter_density)

        clutter_covariance = np.eye(n_channels) * 0.1

        for i in range(min(training_length, len(signal) - n_channels)):
            data_vector = signal[i:i + n_channels]
            clutter_covariance += np.outer(data_vector, np.conj(data_vector))

        clutter_covariance /= training_length

        try:
            inv_cov = np.linalg.inv(clutter_covariance + np.eye(n_channels) * 1e-6)
            steering_vector = np.ones(n_channels)
            weights = inv_cov @ steering_vector
            weights /= np.conj(steering_vector.T) @ inv_cov @ steering_vector
        except:
            weights = np.ones(n_channels) / n_channels

        output = np.zeros(len(signal), dtype=complex)
        for i in range(len(signal) - n_channels + 1):
            data_vector = signal[i:i + n_channels]
            output[i + n_channels // 2] = np.conj(weights.T) @ data_vector

        return output

    def _two_stage_sidelobe_suppression(self, signal: np.ndarray) -> np.ndarray:
        stage1 = self._improved_gsc_sidelobe_suppression(signal)
        stage2 = self._sum_diff_mainlobe_suppression(stage1)
        return stage2

    def _improved_gsc_sidelobe_suppression(self, signal: np.ndarray) -> np.ndarray:
        n_aux = 8
        aux_weights = np.zeros(n_aux, dtype=complex)
        mu = 0.01

        output = signal.copy()

        for i in range(n_aux, len(signal)):
            aux_input = signal[i - n_aux:i]
            error = output[i]

            aux_weights += mu * np.conj(error) * aux_input
            aux_weights = np.clip(aux_weights, -1, 1)

            interference = np.sum(aux_weights * aux_input)
            output[i] = signal[i] - interference

        return output

    def _sum_diff_mainlobe_suppression(self, signal: np.ndarray) -> np.ndarray:
        sum_weights = np.array([1, 1, 1, 1])
        diff_az_weights = np.array([1, -1, 1, -1])
        diff_el_weights = np.array([1, 1, -1, -1])

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
        bandwidth = Config.DEFAULT_BANDWIDTH
        pulse_width = Config.DEFAULT_PULSE_WIDTH
        fs = bandwidth * 2
        t = np.arange(len(signal)) / fs

        lfm_ref = np.zeros_like(signal)
        chirp_samples = int(pulse_width * fs)
        chirp_rate = bandwidth / pulse_width

        for i in range(min(chirp_samples, len(signal))):
            t_chirp = i / fs
            lfm_ref[i] = np.exp(1j * np.pi * chirp_rate * t_chirp ** 2)

        compressed = np.correlate(signal, np.conj(lfm_ref[::-1]), mode='full')
        return compressed[len(compressed) // 2 - len(signal) // 2:len(compressed) // 2 + len(signal) // 2]

    def _cfar_detection(self, signal: np.ndarray, timestamp: float) -> List[Detection]:
        training_cells = 32
        guard_cells = 4
        pfa = 1e-6

        alpha = training_cells * (pfa ** (-1 / training_cells) - 1)

        if self.environment.clutter_density > 0.5:
            alpha *= 1.5
        if self.environment.interference_level > 0.1:
            alpha *= 1.3

        detections = []
        half_window = training_cells // 2 + guard_cells

        for i in range(half_window, len(signal) - half_window):
            training_data = np.concatenate([
                signal[i - half_window:i - guard_cells],
                signal[i + guard_cells + 1:i + half_window + 1]
            ])

            noise_power = np.mean(np.abs(training_data) ** 2)
            threshold = alpha * noise_power
            test_power = abs(signal[i]) ** 2

            if test_power > threshold:
                range_val = i * self.c / (2 * 100e6)
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

        return detections

    def _adaptive_doppler_processing(self, detections: List[Detection]) -> List[Detection]:
        prf = Config.DEFAULT_PRF
        n_pulses = 64

        for detection in detections:
            phase_history = np.random.random(n_pulses) * 2 * np.pi
            phase_history_complex = np.exp(1j * phase_history)

            fft_result = np.fft.fft(phase_history_complex)
            max_bin = np.argmax(np.abs(fft_result))
            doppler_freq = (max_bin - n_pulses // 2) * prf / n_pulses

            detection.velocity = doppler_freq * self.c / (2 * self.radar_system.frequency)

        return detections
