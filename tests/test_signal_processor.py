import numpy as np
from typing import List, Dict, Optional, Tuple
from numba import njit, prange, float64, complex128, int32, objmode
from numba.experimental import jitclass
import math
import scipy.special as sp  # 用于菲涅尔衍射计算


# 假设的Track和Environment类
class Track:
    def __init__(self, state, score=1.0, confirmed=False, target_type=None, rcs_history=None, age=0):
        self.state = state  # [x, y, z, vx, vy, vz, ...]
        self.score = score
        self.confirmed = confirmed
        self.target_type = target_type
        self.rcs_history = rcs_history if rcs_history else []
        self.age = age


class Environment:
    def __init__(self, terrain=None, buildings=None, weather=None):
        self.terrain = terrain
        self.buildings = buildings
        self.weather = weather


class Weather:
    def __init__(self, weather_type="clear", rain_rate=0.0):
        self.weather_type = weather_type
        self.rain_rate = rain_rate  # mm/h


# 复杂多径模型
class ComplexMultipathModel:
    def __init__(self):
        self.paths = []  # 存储所有路径信息
        self.direct_path = None  # 直达路径

    def add_path(self, path_type, delay, amplitude, phase, direction=None, reflections=None,
                 diffraction_edge=None, scattering_points=None):
        """
        添加一条传播路径
        :param path_type: 'direct', 'reflection', 'diffraction', 'scattering'
        :param delay: 路径延迟（秒）
        :param amplitude: 相对幅度
        :param phase: 相位偏移（弧度）
        :param direction: 到达方向（可选）
        :param reflections: 反射点列表（对于反射路径）
        :param diffraction_edge: 衍射边缘点（对于衍射路径）
        :param scattering_points: 散射点列表（对于散射路径）
        """
        path = {
            'type': path_type,
            'delay': delay,
            'amplitude': amplitude,
            'phase': phase
        }
        if direction is not None:
            path['direction'] = direction
        if reflections is not None:
            path['reflections'] = reflections
        if diffraction_edge is not None:
            path['diffraction_edge'] = diffraction_edge
        if scattering_points is not None:
            path['scattering_points'] = scattering_points

        self.paths.append(path)

        if path_type == 'direct':
            self.direct_path = path


# 复杂大气模型
class ComplexAtmosphereModel:
    def __init__(self, temperature=15.0, humidity=60.0, pressure=1013.25):
        self.temperature = temperature  # 摄氏度
        self.humidity = humidity  # 百分比
        self.pressure = pressure  # 毫巴

    def calculate_refraction_index(self, altitude: float) -> float:
        """
        计算给定高度的大气折射率（N单位）
        ITU-R P.453模型
        """
        # 简化模型：随高度指数下降
        N0 = 315  # 海平面的折射率（N单位）
        H = 7.35  # 指数衰减高度（km）
        return N0 * np.exp(-altitude / H)

    def calculate_attenuation(self, frequency: float, distance: float,
                              elevation_angle: float, rain_rate: float) -> float:
        """
        计算总的大气衰减（dB）
        :param frequency: 频率（Hz）
        :param distance: 路径长度（m）
        :param elevation_angle: 仰角（弧度）
        :param rain_rate: 降雨率（mm/h）
        """
        # 气体衰减（ITU-R P.676）
        gamma_gas = self._calculate_gas_attenuation(frequency, elevation_angle)
        # 雨衰（ITU-R P.838）
        gamma_rain = self._calculate_rain_attenuation(frequency, rain_rate)
        # 总衰减
        total_attenuation = gamma_gas * distance + gamma_rain * distance
        return total_attenuation

    def _calculate_gas_attenuation(self, frequency: float, elevation_angle: float) -> float:
        """计算气体衰减系数（dB/km）"""
        # 简化模型：氧气和水汽衰减
        # 实际中应使用ITU-R P.676中的复杂模型
        if frequency < 10e9:
            return 0.001 * (frequency / 1e9)  # 低频衰减小
        else:
            return 0.01 * (frequency / 1e9)  # 高频衰减大

    def _calculate_rain_attenuation(self, frequency: float, rain_rate: float) -> float:
        """计算雨衰系数（dB/km）"""
        # 简化模型：ITU-R P.838
        k = 0.0001 * (frequency / 1e9) ** 2
        alpha = 1.2
        return k * (rain_rate ** alpha)

    def apply_refraction(self, position: np.ndarray) -> np.ndarray:
        """
        应用折射效应修正目标位置（仅考虑垂直方向）
        :param position: 目标原始位置（m）[x, y, z]
        :return: 修正后的位置
        """
        # 简化模型：使用等效地球半径法（4/3地球半径）
        R_e = 6371e3  # 地球半径
        k = 4 / 3  # 等效地球半径因子
        R_eff = k * R_e

        # 计算水平距离
        horizontal_distance = np.linalg.norm(position[:2])
        # 计算原始仰角
        sin_theta = position[2] / np.linalg.norm(position)
        # 修正仰角（使用等效地球半径模型）
        # 实际中更复杂的模型应考虑折射率剖面
        # 这里直接使用等效地球半径模型修正高度
        # 修正公式：z_corrected = z + (horizontal_distance**2) / (2 * R_eff)
        corrected_z = position[2] + (horizontal_distance ** 2) / (2 * R_eff)
        return np.array([position[0], position[1], corrected_z])


# 地形图模型（NUMBA兼容）
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
        self.resolution = resolution
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


class OcclusionMap:
    """遮蔽图"""

    def __init__(self):
        self.severity = 0.0
        self.probability = 0.0
        self.azimuth_range = [0.0, 0.0]
        self.elevation_range = [0.0, 0.0]
        self.occlusion_points = []


# 辅助函数：NUMBA优化
@njit(float64(float64[:], float64[:]))
def _calculate_grazing_angle(reflection_point, target_pos):
    incident_vector = target_pos - reflection_point
    return np.arcsin(abs(incident_vector[2]) / np.linalg.norm(incident_vector))


@njit(float64(float64))
def _calculate_reflection_coefficient(grazing_angle):
    if grazing_angle < np.pi / 90:
        return 0.7
    elif grazing_angle < np.pi / 18:
        return 0.4
    else:
        return 0.1


@njit(complex128[:, :](int32, int32))
def _generate_steering_vector(stap_channels, stap_pulses):
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
    matrix_size = stap_channels * stap_pulses
    covariance = np.eye(matrix_size, dtype=np.complex128) * 0.01
    reflection_coeff = multipath_params[0]
    phase_diff = multipath_params[1]
    for channel in prange(stap_channels):
        for pulse in prange(stap_pulses):
            row_idx = channel * stap_pulses + pulse
            spatial_freq = channel * 2 * np.pi / stap_channels
            temporal_freq = pulse * 2 * np.pi / stap_pulses
            base_clutter = 1.0
            angle_factor = max(0.1, np.cos(spatial_freq))
            doppler_factor = 1.0 + 0.5 * np.cos(temporal_freq)
            clutter_power = base_clutter * angle_factor * doppler_factor
            multipath_factor = 1.0 + reflection_coeff * np.cos(phase_diff + spatial_freq)
            covariance[row_idx, row_idx] += clutter_power * multipath_factor
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
    numerator = abs(np.conj(weights.T) @ weights) ** 2
    denominator = np.real(np.conj(weights.T) @ covariance @ weights)
    return 10 * np.log10(numerator / max(denominator, 1e-10))


@njit(float64[:](float64[:], float64[:], float64))
def _generate_line_of_sight_points(start, end, resolution):
    distance = np.linalg.norm(end - start)
    num_points = int(distance / resolution) + 1
    points = np.zeros((num_points, 3))
    for i in range(num_points):
        t = i / max(num_points - 1, 1)
        points[i] = start + t * (end - start)
    return points


@njit(float64(float64[:], float64[:], float64[:, :], float64[:, :]))
def _analyze_terrain_occlusion(radar_pos, target_pos, terrain_elevation, terrain_info):
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


@njit(float64[:](float64[:], float64[:], float64[:, :], float64[:, :]))
def _find_reflection_point(radar_pos, target_pos, terrain_elevation, terrain_info):
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
    """低空目标增强处理器（含复杂多径和大气模型）"""

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

        # 大气模型
        self.atmosphere_model = ComplexAtmosphereModel()

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
        altitude = track.state[2]
        return altitude < self.low_altitude_threshold

    def _process_low_altitude_track(self, track: Track, terrain: Optional[TerrainMap]) -> Track:
        # 大气折射修正
        if self.environment:
            # 应用大气折射修正目标位置
            track.state[:3] = self.atmosphere_model.apply_refraction(np.array(track.state[:3]))

        # 1. 复杂多径效应建模
        multipath_model = self.model_complex_multipath_effect(track, terrain)

        # 2. 空时自适应处理(STAP)
        stap_improvement = self.apply_stap_processing(track, multipath_model)

        # 3. 极化处理增强
        polarization_improvement = self.apply_polarimetric_processing(track)

        # 4. 地形遮蔽分析与补偿
        occlusion_map = self.analyze_terrain_occlusion(track, terrain)

        # 5. 多频段数据融合（考虑大气衰减）
        multiband_improvement = self.apply_multiband_fusion(track)

        # 6. 自适应滤波
        adaptive_filter_improvement = self.apply_adaptive_filtering(track, multipath_model)

        # 7. 机器学习增强
        ml_enhancement = self.apply_ml_enhancement(track)

        total_improvement = (stap_improvement +
                             polarization_improvement +
                             multiband_improvement +
                             adaptive_filter_improvement +
                             ml_enhancement) / 5.0
        track.score *= (1.0 + total_improvement * 0.5)

        if occlusion_map.severity > 0.6:
            self.fuse_with_auxiliary_sensors(track)

        return track

    def model_complex_multipath_effect(self, track: Track, terrain: Optional[TerrainMap]) -> ComplexMultipathModel:
        model = ComplexMultipathModel()
        target_pos = np.array(track.state[:3], dtype=np.float64)
        radar_pos = np.array([0, 0, 0], dtype=np.float64)
        target_range = np.linalg.norm(target_pos - radar_pos)

        # 添加直达路径
        model.add_path('direct', 2 * target_range / 3e8, 1.0, 0.0)

        if terrain is not None:
            # 添加地面反射路径
            self._add_ground_reflection(model, radar_pos, target_pos, terrain)

            # 添加建筑物反射（如果环境中有建筑物）
            if self.environment and self.environment.buildings:
                self._add_building_reflections(model, radar_pos, target_pos, self.environment.buildings)

            # 添加衍射路径
            self._add_diffraction_paths(model, radar_pos, target_pos, terrain)

            # 添加散射路径
            self._add_scattering_paths(model, radar_pos, target_pos, terrain)

        return model

    def _add_ground_reflection(self, model, radar_pos, target_pos, terrain):
        """添加地面反射路径（镜面反射和漫反射）"""
        # 使用NUMBA优化函数查找反射点
        terrain_elevation = terrain.elevation
        terrain_info = np.array([
            [terrain.resolution, terrain.width, terrain.height, terrain.origin_x, terrain.origin_y]
        ])
        reflection_point = _find_reflection_point(radar_pos, target_pos, terrain_elevation, terrain_info)

        if not np.isnan(reflection_point[0]):
            radar_to_reflection = np.linalg.norm(reflection_point - radar_pos)
            reflection_to_target = np.linalg.norm(target_pos - reflection_point)
            reflected_range = radar_to_reflection + reflection_to_target
            delay = 2 * reflected_range / 3e8

            grazing_angle = _calculate_grazing_angle(reflection_point, target_pos)
            reflection_coeff = _calculate_reflection_coefficient(grazing_angle)

            # 相位差
            path_difference = reflected_range - np.linalg.norm(target_pos - radar_pos)
            phase_difference = 4 * np.pi * path_difference / self.wavelength

            # 添加镜面反射
            model.add_path('reflection', delay, reflection_coeff, phase_difference,
                           reflections=[reflection_point])

            # 添加漫反射分量（简化：作为独立路径）
            # 实际中漫反射可能有多条，这里只加一条作为示例
            diffuse_coeff = reflection_coeff * 0.3  # 漫反射系数较小
            model.add_path('diffuse', delay * 1.01, diffuse_coeff, phase_difference + np.pi / 4)

    def _add_building_reflections(self, model, radar_pos, target_pos, buildings):
        """添加建筑物反射路径"""
        # 简化：只考虑单个建筑物
        for building in buildings:
            # 建筑物用立方体表示：中心位置和尺寸
            center = building['center']
            size = building['size']

            # 计算雷达和目标相对于建筑物的位置
            # 简化：只考虑一次反射，反射点为建筑物墙面中心
            # 实际中应计算反射点
            # 这里仅示意
            reflection_point = np.array([
                center[0],
                center[1],
                center[2] + size[2] / 2  # 墙面中心
            ])

            # 检查反射点是否在墙面上（简化）
            # 计算反射路径
            radar_to_reflection = np.linalg.norm(reflection_point - radar_pos)
            reflection_to_target = np.linalg.norm(target_pos - reflection_point)
            reflected_range = radar_to_reflection + reflection_to_target
            total_delay = 2 * reflected_range / 3e8

            # 反射系数（建筑物墙面）
            # 根据材料，这里假设为0.8
            reflection_coeff = 0.8

            # 相位差（相对于直达路径）
            direct_range = 2 * np.linalg.norm(target_pos - radar_pos)
            path_difference = 2 * reflected_range - direct_range
            phase_difference = 4 * np.pi * path_difference / self.wavelength

            model.add_path('reflection', total_delay, reflection_coeff, phase_difference,
                           reflections=[reflection_point])

    def _add_diffraction_paths(self, model, radar_pos, target_pos, terrain):
        """添加衍射路径（如山峰边缘衍射）"""
        # 简化：考虑一个衍射边缘
        # 假设在雷达和目标连线上方有一个边缘点
        midpoint = (radar_pos + target_pos) / 2
        # 假设边缘点高度比连线高100米
        edge_point = midpoint + np.array([0, 0, 100.0])

        # 计算衍射路径
        radar_to_edge = np.linalg.norm(edge_point - radar_pos)
        edge_to_target = np.linalg.norm(target_pos - edge_point)
        diffracted_path_length = radar_to_edge + edge_to_target
        delay = 2 * diffracted_path_length / 3e8  # 双程延迟

        # 菲涅尔衍射系数
        # 计算菲涅尔参数
        d1 = radar_to_edge
        d2 = edge_to_target
        d = d1 + d2
        # 计算边缘高度相对于直连线的垂直距离（这里设为100米）
        h = 100.0
        # 波长
        wavelength = self.wavelength
        # 菲涅尔参数
        v = h * np.sqrt(2 * d / (wavelength * d1 * d2))
        # 使用菲涅尔积分
        # 注意：scipy的fresnel函数不能在numba中用，这里简化处理
        # 实际中应计算菲涅尔积分，这里用近似
        # 幅度衰减因子
        if v < 0:
            diff_coeff = 0.5  # 完全在阴影区
        elif v < 1.0:
            diff_coeff = 0.6
        else:
            diff_coeff = 0.3 / v

        # 相位差（相对于直达路径）
        direct_path = 2 * np.linalg.norm(target_pos - radar_pos)
        path_difference = 2 * diffracted_path_length - direct_path
        phase_difference = 4 * np.pi * path_difference / wavelength

        model.add_path('diffraction', delay, diff_coeff, phase_difference, diffraction_edge=edge_point)

    def _add_scattering_paths(self, model, radar_pos, target_pos, terrain):
        """添加散射路径（如树木、城市等）"""
        # 简化：在路径中随机添加几个散射点
        num_scatterers = 3
        scatter_points = []

        # 生成随机散射点
        for _ in range(num_scatterers):
            # 在直达路径附近随机位置
            t = np.random.uniform(0.2, 0.8)
            scatter_point = radar_pos + t * (target_pos - radar_pos)
            # 添加随机偏移
            scatter_point += np.random.uniform(-50, 50, 3)
            scatter_points.append(scatter_point)

        # 计算散射路径
        total_path_length = 0
        last_point = radar_pos
        for point in scatter_points:
            total_path_length += np.linalg.norm(point - last_point)
            last_point = point
        total_path_length += np.linalg.norm(target_pos - last_point)

        delay = total_path_length / 3e8  # 单程延迟，散射路径是双向吗？
        # 散射路径也是双向的，所以总延迟为2 * total_path_length / c
        delay *= 2

        # 散射系数（通常较小）
        scatter_coeff = 0.2 / num_scatterers

        # 相位差（随机）
        phase_difference = np.random.uniform(0, 2 * np.pi)

        model.add_path('scattering', delay, scatter_coeff, phase_difference, scattering_points=scatter_points)

    def apply_stap_processing(self, track: Track, multipath_model: ComplexMultipathModel) -> float:
        # 注意：STAP处理中，我们只考虑主要反射路径（第一条反射）来构建杂波
        if len(multipath_model.paths) > 1:
            # 取第一条反射路径的参数
            reflection_path = multipath_model.paths[1]  # 索引0是直达
            reflection_coeff = reflection_path['amplitude']
            phase_difference = reflection_path['phase']
        else:
            reflection_coeff = 0.0
            phase_difference = 0.0

        multipath_params = np.array([reflection_coeff, phase_difference])
        clutter_covariance = _build_clutter_covariance_matrix(
            self.stap_channels, self.stap_pulses,
            np.array(track.state[:3]), multipath_params, self.wavelength
        )
        steering_vector = _generate_steering_vector(self.stap_channels, self.stap_pulses)
        optimal_weights = _calculate_stap_weights(clutter_covariance, steering_vector)
        self.stap_weights = optimal_weights.reshape(self.stap_channels, self.stap_pulses)
        return _calculate_sincr_improvement(optimal_weights, clutter_covariance)

    def apply_polarimetric_processing(self, track: Track) -> float:
        # 与之前相同
        scattering_matrix = self._generate_polarimetric_scattering_matrix(track)
        polarimetric_features = self._extract_polarimetric_features(scattering_matrix)
        target_probability = self._classify_target_clutter_polarimetric(polarimetric_features)
        optimal_polarization = self._optimize_polarization_weights(scattering_matrix)
        self.polarization_weights = optimal_polarization
        return self._calculate_polarization_gain(target_probability)

    def analyze_terrain_occlusion(self, track: Track, terrain: Optional[TerrainMap]) -> OcclusionMap:
        occlusion_map = OcclusionMap()
        if terrain is None:
            occlusion_map.severity = 0.0
            return occlusion_map
        radar_pos = np.array([0, 0, 50], dtype=np.float64)
        target_pos = np.array(track.state[:3], dtype=np.float64)
        terrain_elevation = terrain.elevation
        terrain_info = np.array([
            [terrain.resolution, terrain.width, terrain.height, terrain.origin_x, terrain.origin_y]
        ])
        severity = _analyze_terrain_occlusion(radar_pos, target_pos, terrain_elevation, terrain_info)
        occlusion_map.severity = severity
        occlusion_map.probability = min(severity * 1.5, 1.0)
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

    def _simulate_band_performance(self, track: Track, band: Dict) -> float:
        # 考虑大气衰减
        frequency = band['center']
        distance = np.linalg.norm(track.state[:3])
        elevation_angle = np.arcsin(track.state[2] / distance)
        rain_rate = self.environment.weather.rain_rate if self.environment and self.environment.weather else 0.0
        attenuation = self.atmosphere_model.calculate_attenuation(frequency, distance, elevation_angle, rain_rate)

        # 衰减越大，性能越差
        attenuation_factor = 10 ** (-attenuation / 20)  # 转换为幅度因子

        if frequency < 5e9:
            base_performance = 0.8
        elif frequency < 20e9:
            base_performance = 1.0
        else:
            base_performance = 1.2 if track.state[2] < 1000 else 0.6

        return base_performance * attenuation_factor

    def apply_adaptive_filtering(self, track: Track, multipath_model: ComplexMultipathModel) -> float:
        # 生成参考信号（直达路径）
        reference_signal = self._generate_reference_signal(track)

        # 生成接收信号（含所有多径）
        received_signal = self._generate_received_signal_with_multipath(track, multipath_model)

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

    def _generate_received_signal_with_multipath(self, track: Track,
                                                 multipath_model: ComplexMultipathModel) -> np.ndarray:
        direct_signal = self._generate_reference_signal(track)
        received_signal = np.zeros_like(direct_signal)

        for path in multipath_model.paths:
            delay_samples = int(path['delay'] * 1e6)  # 假设采样率为1MHz
            if delay_samples < len(direct_signal):
                delayed_signal = np.roll(direct_signal, delay_samples)
                # 将延迟超出部分的信号置零
                if delay_samples > 0:
                    delayed_signal[:delay_samples] = 0
                else:
                    delayed_signal[delay_samples:] = 0
                # 添加该路径信号
                received_signal += path['amplitude'] * delayed_signal * np.exp(1j * path['phase'])

        return received_signal

    # 以下函数与之前相同，为保持完整性保留
    def apply_ml_enhancement(self, track: Track) -> float:
        features = self._extract_ml_features(track)
        target_confidence = self._simple_target_classifier(features)
        return max(0, (target_confidence - 0.5) * 2)

    def fuse_with_auxiliary_sensors(self, track: Track):
        ir_detection = self._simulate_ir_detection(track)
        acoustic_detection = self._simulate_acoustic_detection(track)
        visual_detection = self._simulate_visual_detection(track)
        fused_position = self._fuse_sensor_measurements(track, ir_detection, acoustic_detection, visual_detection)
        if fused_position is not None:
            track.state[:3] = fused_position
            track.score *= 1.3
            track.confirmed = True

    def _generate_reference_signal(self, track: Track) -> np.ndarray:
        signal_length = 1024
        return np.random.random(signal_length) + 1j * np.random.random(signal_length)

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
