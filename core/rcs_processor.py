import numpy as np
from typing import Dict, List, Tuple
from models.tracking import Track, Detection
from models.radar_system import RadarSystem
from models.environment import Environment


class RCSProcessor:
    def __init__(self, radar_system: RadarSystem, environment: Environment):
        self.radar_system = radar_system
        self.environment = environment
        self.k_factor = 1.0
        self.calibration_targets = {}

    def process_rcs(self, track: Track) -> Dict[str, float]:
        k_factor = self._calibrate_k_factor(track)
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

    def _calibrate_k_factor(self, track: Track) -> float:
        base_k = self.radar_system.radar_power * (self.radar_system.wavelength ** 2) * self.radar_system.gain
        base_k /= (4 * np.pi) ** 3

        atmospheric_loss = self._calculate_atmospheric_loss(track)
        scan_loss = self._calculate_scan_loss(track)

        k_factor = base_k / (atmospheric_loss * scan_loss)

        return k_factor

    def _calculate_atmospheric_loss(self, track: Track) -> float:
        range_km = np.sqrt(track.state[0] ** 2 + track.state[1] ** 2 + track.state[2] ** 2) / 1000

        loss_db = self.environment.weather.atmospheric_loss(
            self.radar_system.frequency, range_km
        )

        return 10 ** (loss_db / 10)

    def _calculate_scan_loss(self, track: Track) -> float:
        azimuth = np.arctan2(track.state[1], track.state[0])
        elevation = np.arcsin(track.state[2] / max(np.sqrt(sum(track.state[:3] ** 2)), 1e-6))

        scan_loss = (np.cos(azimuth) * np.cos(elevation)) ** 2

        return max(scan_loss, 0.1)

    def _apply_compensation_factors(self, track: Track):
        if len(track.detections_history) > 0:
            latest_detection = track.detections_history[-1]

            range_correction = (latest_detection.range / 10000) ** 4
            frequency_correction = (self.radar_system.frequency / 10e9) ** 2

            track.rcs_history[-1] *= range_correction * frequency_correction

    def _calculate_rcs(self, track: Track, k_factor: float) -> float:
        if not track.detections_history:
            return 0.0

        detection = track.detections_history[-1]
        range_val = detection.range
        snr_linear = 10 ** (detection.snr / 10)

        pulse_width = 20e-6
        pulse_count = 64

        rcs_linear = (range_val ** 4 * self.radar_system.frequency ** 2 * snr_linear) / \
                     (k_factor * pulse_count * pulse_width)

        rcs_db = 10 * np.log10(max(rcs_linear, 1e-10))

        return rcs_db

    def _is_calibration_target(self, track: Track) -> bool:
        if len(track.rcs_history) < 5:
            return False

        rcs_std = np.std(track.rcs_history)
        return rcs_std < 1.0 and track.confirmed

    def _dynamic_calibration(self, track: Track, measured_rcs: float):
        if track.track_id not in self.calibration_targets:
            self.calibration_targets[track.track_id] = {
                'expected_rcs': 10.0,  # dBsm
                'measurements': []
            }

        calib_data = self.calibration_targets[track.track_id]
        calib_data['measurements'].append(measured_rcs)

        if len(calib_data['measurements']) > 10:
            avg_measured = np.mean(calib_data['measurements'][-10:])
            calibration_error = calib_data['expected_rcs'] - avg_measured
            self.k_factor *= 10 ** (calibration_error / 20)

    def _fuse_multisensor_rcs(self, track: Track) -> float:
        if not track.rcs_history:
            return 0.0

        radar_rcs = np.mean(track.rcs_history[-5:]) if len(track.rcs_history) >= 5 else track.rcs_history[-1]

        ir_rcs_estimate = radar_rcs + np.random.normal(0, 2)
        esm_rcs_estimate = radar_rcs + np.random.normal(0, 3)

        weights = [0.6, 0.25, 0.15]
        estimates = [radar_rcs, ir_rcs_estimate, esm_rcs_estimate]

        fused_rcs = sum(w * est for w, est in zip(weights, estimates))

        return fused_rcs

    def _analyze_rcs_features(self, track: Track) -> Dict[str, float]:
        if len(track.rcs_history) < 3:
            return {'mean': 0, 'std': 0, 'trend': 0}

        rcs_array = np.array(track.rcs_history)

        features = {
            'mean': np.mean(rcs_array),
            'std': np.std(rcs_array),
            'trend': np.polyfit(range(len(rcs_array)), rcs_array, 1)[0] if len(rcs_array) > 1 else 0,
            'max': np.max(rcs_array),
            'min': np.min(rcs_array),
            'range': np.max(rcs_array) - np.min(rcs_array)
        }

        return features
