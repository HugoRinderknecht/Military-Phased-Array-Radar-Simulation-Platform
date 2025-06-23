import numpy as np
from typing import List, Dict, Optional, Tuple
from numba import njit, prange, float64, complex128, int32, objmode
from numba.experimental import jitclass
from models.tracking import Track
from models.environment import Environment
import math

# 定义NUMBA兼容的数据结构
spec_terrain_map = [
    ('width', int32),
    ('height', int32),
    ('resolution', float64),
    ('elevation', float64[:, :]),
    ('origin_x', float64),
    ('origin_y', float64),
    ('terrain_type', int32[:, :])
]


@jitclass(spec_terrain_map)
class TerrainMap:
    """地形图模型 (NUMBA兼容)"""

    def __init__(self, width: int = 1024, height: int = 1024, resolution: float = 10.0):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per pixel
        self.elevation = np.zeros((height, width))
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.terrain_type = np.ones((height, width), dtype=np.int32)  # 1=land, 0=water

    def get_elevation(self, x: float, y: float) -> float:
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.elevation[grid_y, grid_x]
        return 0.0

    def get_terrain_type(self, x: float, y: float) -> int:
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.terrain_type[grid_y, grid_x]
        return 1


class MultipathModel:
    """多径传播模型"""

    def __init__(self):
        self.direct_path_delay = 0.0
        self.reflected_path_delay = 0.0
        self.reflection_coefficient = 0.0
        self.phase_difference = 0.0
        self.multipath_components = []

    def add_reflection_path(self, delay: float, amplitude: float, phase: float):
        self.multipath_components.append({
            'delay': delay,
            'amplitude': amplitude,
            'phase': phase
        })


class OcclusionMap:
    """遮蔽图"""

    def __init__(self):
        self.severity = 0.0
        self.probability = 0.0
        self.azimuth_range = [0.0, 0.0]
        self.elevation_range = [0.0, 0.0]
        self.occlusion_points = []


@njit(float64(float64[:], float64[:]))
def _calculate_grazing_angle(reflection_point, target_pos):
    """计算掠射角 (NUMBA优化)"""
    incident_vector = target_pos - reflection_point
    return np.arcsin(abs(incident_vector[2]) / np.linalg.norm(incident_vector))


@njit(float64(float64))
def _calculate_reflection_coefficient(grazing_angle):
    """计算反射系数 (NUMBA优化)"""
    if grazing_angle < np.pi / 90:  # <2度
        return 0.7
    elif grazing_angle < np.pi / 18:  # <10度
        return 0.4
    else:
        return 0.1


@njit(complex128[:, :](int32, int32))
def _generate_steering_vector(stap_channels, stap_pulses):
    """生成期望信号导向矢量 (NUMBA优化)"""
    vector_length = stap_channels * stap_pulses
    steering_vector = np.ones(vector_length, dtype=np.complex128)
    target_spatial_freq = 0.0
    target_doppler_freq = np.pi / 8

    for channel in range(stap_channels):
        for pulse in range(stap_pulses):
            idx = channel * stap_pulses + pulse
            phase = (channel * target_spatial_freq + pulse * target_doppler_freq)
            steering_vector[idx] = np.exp(1j * phase)

    return steering_vector.reshape(-1, 1)


@njit(complex128[:, :](int32, int32, float64[:], float64[:], float64))
def _build_clutter_covariance_matrix(stap_channels, stap_pulses, track_state, multipath_params, wavelength):
    """构建杂波协方差矩阵 (NUMBA优化)"""
    matrix_size = stap_channels * stap_pulses
    covariance = np.eye(matrix_size, dtype=np.complex128) * 0.01

    # 多径参数解包
    reflection_coeff = multipath_params[0]
    phase_diff = multipath_params[1]

    for channel in prange(stap_channels):
        for pulse in prange(stap_pulses):
            row_idx = channel * stap_pulses + pulse
            spatial_freq = channel * 2 * np.pi / stap_channels
            temporal_freq = pulse * 2 * np.pi / stap_pulses

            # 杂波功率计算
            base_clutter = 1.0
            angle_factor = max(0.1, np.cos(spatial_freq))
            doppler_factor = 1.0 + 0.5 * np.cos(temporal_freq)
            clutter_power = base_clutter * angle_factor * doppler_factor

            # 多径因子
            multipath_factor = 1.0 + reflection_coeff * np.cos(phase_diff + spatial_freq)
            covariance[row_idx, row_idx] += clutter_power * multipath_factor

    # 添加相关性
    for i in prange(matrix_size):
        for j in prange(i + 1, matrix_size):
            distance = abs(i - j)
            correlation_coeff = np.exp(-distance / 10.0)
            phase = np.random.uniform(0, 2 * np.pi)
            covariance[i, j] = correlation_coeff * np.exp(1j * phase)
            covariance[j, i] = np.conj(covariance[i, j])

    return covariance


@njit(complex128[:, :](complex128[:, :], complex128[:, :]))
def _calculate_stap_weights(clutter_covariance, steering_vector):
    """计算STAP权重向量 (NUMBA优化)"""
    try:
        inv_covariance = np.linalg.inv(clutter_covariance + np.eye(clutter_covariance.shape[0]) * 1e-6)
        numerator = inv_covariance @ steering_vector
        denominator = np.conj(steering_vector.T) @ inv_covariance @ steering_vector
        weights = numerator / denominator
    except:
        regularized_cov = clutter_covariance + np.eye(clutter_covariance.shape[0]) * 0.1
        inv_covariance = np.linalg.inv(regularized_cov)
        numerator = inv_covariance @ steering_vector
        denominator = np.conj(steering_vector.T) @ inv_covariance @ steering_vector
        weights = numerator / denominator

    return weights


@njit(float64(complex128[:, :], complex128[:, :]))
def _calculate_sincr_improvement(weights, covariance):
    """计算信杂噪比改善 (NUMBA优化)"""
    numerator = abs(np.conj(weights.T) @ weights) ** 2
    denominator = np.real(np.conj(weights.T) @ covariance @ weights)
    return 10 * np.log10(numerator / max(denominator, 1e-10))


@njit(float64[:](float64[:], float64[:], float64))
def _generate_line_of_sight_points(start, end, resolution):
    """生成视线点集 (NUMBA优化)"""
    distance = np.linalg.norm(end - start)
    num_points = int(distance / resolution) + 1
    points = np.zeros((num_points, 3))

    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        points[i] = start + t * (end - start)

    return points


@njit(float64(float64[:], float64[:], float64[:], float64[:, :]))
def _analyze_terrain_occlusion(radar_pos, target_pos, terrain_elevation, terrain_info):
    """地形遮蔽分析 (NUMBA优化)"""
    resolution = terrain_info[0, 0]
    width = int(terrain_info[0, 1])
    height = int(terrain_info[0, 2])
    origin_x = terrain_info[0, 3]
    origin_y = terrain_info[0, 4]

    los_points = _generate_line_of_sight_points(radar_pos, target_pos, resolution)
    blocked_points = 0

    for i in range(len(los_points)):
        x, y, z = los_points[i]
        grid_x = int((x - origin_x) / resolution)
        grid_y = int((y - origin_y) / resolution)

        if 0 <= grid_x < width and 0 <= grid_y < height:
            terrain_height = terrain_elevation[grid_y, grid_x]
            if terrain_height > z:
                blocked_points += 1

    return blocked_points / len(los_points)


@njit(float64(float64[:], float64[:], float64[:, :], float64[:, :]))
def _find_reflection_point(radar_pos, target_pos, terrain_elevation, terrain_info):
    """查找镜面反射点 (NUMBA优化)"""
    resolution = terrain_info[0, 0]
    width = int(terrain_info[0, 1])
    height = int(terrain_info[0, 2])
    origin_x = terrain_info[0, 3]
    origin_y = terrain_info[0, 4]

    midpoint = (radar_pos + target_pos) / 2
    grid_x = int((midpoint[0] - origin_x) / resolution)
    grid_y = int((midpoint[1] - origin_y) / resolution)

    if 0 <= grid_x < width and 0 <= grid_y < height:
        midpoint[2] = terrain_elevation[grid_y, grid_x]
        return midpoint

    return np.array([np.nan, np.nan, np.nan])


@njit(float64(complex128[:], complex128[:], complex128[:], float64))
def _lms_adaptive_filter(reference, received, coefficients, step_size):
    """LMS自适应滤波 (NUMBA优化)"""
    filter_order = len(coefficients)
    signal_length = min(len(reference), len(received))
    initial_mse = 0.0

    for i in range(signal_length):
        error = reference[i] - received[i]
        initial_mse += (error.real ** 2 + error.imag ** 2)
    initial_mse /= signal_length

    for i in range(signal_length - filter_order):
        input_vector = received[i:i + filter_order]
        output = np.dot(coefficients.conj(), input_vector)
        error = reference[i] - output

        coefficients += step_size * error.conj() * input_vector

    final_mse = 0.0
    for i in range(signal_length - filter_order):
        input_vector = received[i:i + filter_order]
        output = np.dot(coefficients.conj(), input_vector)
        error = reference[i] - output
        final_mse += (error.real ** 2 + error.imag ** 2)
    final_mse /= (signal_length - filter_order)

    return 10 * np.log10(initial_mse / max(final_mse, 1e-10))


class LowAltitudeEnhancer:
    """低空目标增强处理器"""

    def __init__(self, radar_frequency: float = 10e9, environment: Optional[Environment] = None):
        self.radar_frequency = radar_frequency
        self.environment = environment
        self.wavelength = 3e8 / radar_frequency
        self.low_altitude_threshold = 500.0  # meters

        # STAP参数
        self.stap_channels = 8
        self.stap_pulses = 16
        self.stap_weights = np.ones((self.stap_channels, self.stap_pulses), dtype=complex)

        # 极化参数
        self.polarization_modes = ['HH', 'VV', 'HV', 'VH']
        self.polarization_weights = np.ones(4)

        # 多频段融合参数
        self.frequency_bands = [
            {'center': 3e9, 'bandwidth': 100e6, 'weight': 0.3},
            {'center': 10e9, 'bandwidth': 200e6, 'weight': 0.5},
            {'center': 35e9, 'bandwidth': 500e6, 'weight': 0.2}
        ]

    def enhance_low_altitude_targets(self, tracks: List[Track], terrain: Optional[TerrainMap] = None) -> List[Track]:
        enhanced_tracks = []

        for track in tracks:
            if self._is_low_altitude(track):
                enhanced_track = self._process_low_altitude_track(track, terrain)
                enhanced_tracks.append(enhanced_track)
            else:
                enhanced_tracks.append(track)

        return enhanced_tracks

    def _is_low_altitude(self, track: Track) -> bool:
        altitude = track.state[2]  # z坐标
        return altitude < self.low_altitude_threshold

    def _process_low_altitude_track(self, track: Track, terrain: Optional[TerrainMap]) -> Track:
        # 1. 多径效应建模与补偿
        multipath_model = self.model_multipath_effect(track, terrain)

        # 2. 空时自适应处理(STAP)
        stap_improvement = self.apply_stap_processing(track, multipath_model)

        # 3. 极化处理增强
        polarization_improvement = self.apply_polarimetric_processing(track)

        # 4. 地形遮蔽分析与补偿
        occlusion_map = self.analyze_terrain_occlusion(track, terrain)

        # 5. 多频段数据融合
        multiband_improvement = self.apply_multiband_fusion(track)

        # 6. 自适应滤波
        adaptive_filter_improvement = self.apply_adaptive_filtering(track, multipath_model)

        # 7. 机器学习增强
        ml_enhancement = self.apply_ml_enhancement(track)

        # 综合改善效果
        total_improvement = (stap_improvement +
                             polarization_improvement +
                             multiband_improvement +
                             adaptive_filter_improvement +
                             ml_enhancement) / 5.0

        # 更新航迹质量
        track.score *= (1.0 + total_improvement * 0.5)

        # 如果遮蔽严重，启用多传感器融合
        if occlusion_map.severity > 0.6:
            self.fuse_with_auxiliary_sensors(track)

        return track

    def model_multipath_effect(self, track: Track, terrain: Optional[TerrainMap]) -> MultipathModel:
        model = MultipathModel()
        target_pos = np.array(track.state[:3], dtype=np.float64)
        target_range = np.linalg.norm(target_pos)
        model.direct_path_delay = 2 * target_range / 3e8

        if terrain is not None:
            # 准备地形数据
            terrain_elevation = terrain.elevation
            terrain_info = np.array([
                [terrain.resolution, terrain.width, terrain.height, terrain.origin_x, terrain.origin_y]
            ])

            # 使用NUMBA优化函数查找反射点
            radar_pos = np.array([0, 0, 0], dtype=np.float64)
            reflection_point = _find_reflection_point(
                radar_pos, target_pos, terrain_elevation, terrain_info
            )

            if not np.isnan(reflection_point[0]):
                radar_to_reflection = np.linalg.norm(reflection_point)
                reflection_to_target = np.linalg.norm(target_pos - reflection_point)
                reflected_range = radar_to_reflection + reflection_to_target
                model.reflected_path_delay = 2 * reflected_range / 3e8

                # 计算掠射角和反射系数
                grazing_angle = _calculate_grazing_angle(reflection_point, target_pos)
                model.reflection_coefficient = _calculate_reflection_coefficient(grazing_angle)

                # 相位差
                path_difference = reflected_range - target_range
                model.phase_difference = 4 * np.pi * path_difference / self.wavelength

        return model

    def apply_stap_processing(self, track: Track, multipath_model: MultipathModel) -> float:
        # 准备参数
        multipath_params = np.array([
            multipath_model.reflection_coefficient,
            multipath_model.phase_difference
        ])

        # 构建协方差矩阵
        clutter_covariance = _build_clutter_covariance_matrix(
            self.stap_channels,
            self.stap_pulses,
            np.array(track.state[:3]),
            multipath_params,
            self.wavelength
        )

        # 生成导向矢量
        steering_vector = _generate_steering_vector(self.stap_channels, self.stap_pulses)

        # 计算最优权重
        optimal_weights = _calculate_stap_weights(clutter_covariance, steering_vector)

        # 更新权重
        self.stap_weights = optimal_weights.reshape(self.stap_channels, self.stap_pulses)

        # 计算改善因子
        return _calculate_sincr_improvement(optimal_weights, clutter_covariance)

    def apply_polarimetric_processing(self, track: Track) -> float:
        # 模拟极化散射矩阵
        scattering_matrix = self._generate_polarimetric_scattering_matrix(track)

        # 计算极化特征参数
        polarimetric_features = self._extract_polarimetric_features(scattering_matrix)

        # 目标/杂波分类
        target_probability = self._classify_target_clutter_polarimetric(polarimetric_features)

        # 极化自适应权重
        optimal_polarization = self._optimize_polarization_weights(scattering_matrix)

        # 更新极化权重
        self.polarization_weights = optimal_polarization

        # 计算极化增益
        return self._calculate_polarization_gain(target_probability)

    def analyze_terrain_occlusion(self, track: Track, terrain: Optional[TerrainMap]) -> OcclusionMap:
        occlusion_map = OcclusionMap()

        if terrain is None:
            occlusion_map.severity = 0.0
            return occlusion_map

        # 准备数据
        radar_pos = np.array([0, 0, 50], dtype=np.float64)  # 雷达位置
        target_pos = np.array(track.state[:3], dtype=np.float64)
        terrain_elevation = terrain.elevation
        terrain_info = np.array([
            [terrain.resolution, terrain.width, terrain.height, terrain.origin_x, terrain.origin_y]
        ])

        # 使用NUMBA优化函数
        severity = _analyze_terrain_occlusion(
            radar_pos, target_pos, terrain_elevation, terrain_info
        )

        occlusion_map.severity = severity
        occlusion_map.probability = min(severity * 1.5, 1.0)

        # 计算遮蔽角度范围
        target_azimuth = np.arctan2(target_pos[1], target_pos[0])
        occlusion_map.azimuth_range = [target_azimuth - np.pi / 18, target_azimuth + np.pi / 18]

        target_elevation = np.arcsin(target_pos[2] / np.linalg.norm(target_pos))
        occlusion_map.elevation_range = [target_elevation - np.pi / 36, target_elevation + np.pi / 36]

        return occlusion_map

    def apply_multiband_fusion(self, track: Track) -> float:
        band_improvements = []

        for band in self.frequency_bands:
            band_performance = self._simulate_band_performance(track, band)
            band_improvements.append(band_performance * band['weight'])

        total_improvement = sum(band_improvements)
        diversity_gain = self._calculate_frequency_diversity_gain(len(self.frequency_bands))
        return total_improvement * diversity_gain

    def apply_adaptive_filtering(self, track: Track, multipath_model: MultipathModel) -> float:
        # 生成参考信号
        reference_signal = self._generate_reference_signal(track)

        # 生成接收信号
        received_signal = self._generate_received_signal_with_multipath(track, multipath_model)

        # 使用NUMBA优化LMS滤波
        filter_order = 32
        coefficients = np.zeros(filter_order, dtype=np.complex128)
        adaptation_step = 0.01

        with objmode(mse_reduction=float64):
            mse_reduction = _lms_adaptive_filter(
                reference_signal,
                received_signal,
                coefficients,
                adaptation_step
            )

        return mse_reduction

    def apply_ml_enhancement(self, track: Track) -> float:
        features = self._extract_ml_features(track)
        target_confidence = self._simple_target_classifier(features)
        return max(0, (target_confidence - 0.5) * 2)

    def fuse_with_auxiliary_sensors(self, track: Track):
        ir_detection = self._simulate_ir_detection(track)
        acoustic_detection = self._simulate_acoustic_detection(track)
        visual_detection = self._simulate_visual_detection(track)

        fused_position = self._fuse_sensor_measurements(
            track, ir_detection, acoustic_detection, visual_detection
        )

        if fused_position is not None:
            track.state[:3] = fused_position
            track.score *= 1.3
            track.confirmed = True

    # 以下函数保持原样，但部分调用已替换为NUMBA优化版本
    def _generate_reference_signal(self, track: Track) -> np.ndarray:
        signal_length = 1024
        return np.random.random(signal_length) + 1j * np.random.random(signal_length)

    def _generate_received_signal_with_multipath(self, track: Track, model: MultipathModel) -> np.ndarray:
        direct_signal = self._generate_reference_signal(track)
        delayed_signal = np.roll(direct_signal, int(model.reflected_path_delay * 1e6))
        multipath_signal = direct_signal + model.reflection_coefficient * delayed_signal * np.exp(
            1j * model.phase_difference)
        return multipath_signal

    def _generate_polarimetric_scattering_matrix(self, track: Track) -> np.ndarray:
        S = np.array([[1.0 + 0.1j, 0.2 + 0.05j],
                      [0.2 + 0.05j, 0.8 + 0.15j]])
        if hasattr(track, 'target_type') and track.target_type == 'aircraft':
            S[0, 0] *= 1.5
        return S

    def _extract_polarimetric_features(self, S: np.ndarray) -> Dict[str, float]:
        features = {
            'polarization_ratio': abs(S[1, 1] / S[0, 0]),
            'cross_pol_ratio': abs(S[0, 1] / S[0, 0]),
            'phase_diff': np.angle(S[1, 1]) - np.angle(S[0, 0]),
            'degree_of_polarization': np.linalg.det(S @ S.conj().T) / (np.trace(S @ S.conj().T)) ** 2
        }
        return features

    def _classify_target_clutter_polarimetric(self, features: Dict[str, float]) -> float:
        score = 0.5
        if 0.1 < features['polarization_ratio'] < 0.8:
            score += 0.2
        if abs(features['phase_diff']) < np.pi / 6:
            score += 0.1
        return min(score, 1.0)

    def _optimize_polarization_weights(self, S: np.ndarray) -> np.ndarray:
        eigenvals, eigenvecs = np.linalg.eig(S @ S.conj().T)
        optimal_weights = eigenvecs[:, np.argmax(eigenvals)]
        return np.abs(optimal_weights)

    def _calculate_polarization_gain(self, target_probability: float) -> float:
        return target_probability * 3.0

    def _simulate_band_performance(self, track: Track, band: Dict) -> float:
        frequency = band['center']
        if frequency < 5e9:
            return 0.8
        elif frequency < 20e9:
            return 1.0
        else:
            return 1.2 if track.state[2] < 1000 else 0.6

    def _calculate_frequency_diversity_gain(self, num_bands: int) -> float:
        return min(1.0 + 0.1 * (num_bands - 1), 1.5)

    def _extract_ml_features(self, track: Track) -> np.ndarray:
        features = []
        velocity = np.linalg.norm(track.state[3:6])
        features.append(velocity)
        features.append(0.0)  # 加速度
        features.append(track.state[2])  # 高度
        features.append(np.linalg.norm(track.state[:3]))  # 距离

        if track.rcs_history:
            rcs_mean = np.mean(track.rcs_history)
            rcs_std = np.std(track.rcs_history) if len(track.rcs_history) > 1 else 0
        else:
            rcs_mean = 0.0
            rcs_std = 0.0

        features.extend([rcs_mean, rcs_std])
        features.append(track.age)
        features.append(1.0 if track.confirmed else 0.0)
        return np.array(features)

    def _simple_target_classifier(self, features: np.ndarray) -> float:
        weights = np.array([0.2, 0.1, 0.3, -0.1, 0.2, 0.05, 0.1, 0.05])
        score = np.dot(features, weights)
        return 1.0 / (1.0 + np.exp(-score))

    def _simulate_ir_detection(self, track: Track) -> Optional[np.ndarray]:
        if track.state[2] > 200:
            noise = np.random.normal(0, 50, 3)
            return track.state[:3] + noise
        return None

    def _simulate_acoustic_detection(self, track: Track) -> Optional[np.ndarray]:
        velocity = np.linalg.norm(track.state[3:6])
        if track.state[2] < 1000 and velocity > 50:
            noise = np.random.normal(0, 100, 3)
            return track.state[:3] + noise
        return None

    def _simulate_visual_detection(self, track: Track) -> Optional[np.ndarray]:
        range_val = np.linalg.norm(track.state[:3])
        if range_val < 20000 and self.environment and self.environment.weather.weather_type == "clear":
            noise = np.random.normal(0, 30, 3)
            return track.state[:3] + noise
        return None

    def _fuse_sensor_measurements(self, track: Track, ir_pos: Optional[np.ndarray],
                                  acoustic_pos: Optional[np.ndarray],
                                  visual_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        measurements = [track.state[:3]]
        weights = [0.5]

        if ir_pos is not None:
            measurements.append(ir_pos)
            weights.append(0.3)
        if acoustic_pos is not None:
            measurements.append(acoustic_pos)
            weights.append(0.1)
        if visual_pos is not None:
            measurements.append(visual_pos)
            weights.append(0.1)

        if len(measurements) == 1:
            return None

        weights = np.array(weights) / sum(weights)
        fused_position = np.zeros(3)
        for measurement, weight in zip(measurements, weights):
            fused_position += weight * measurement

        return fused_position
