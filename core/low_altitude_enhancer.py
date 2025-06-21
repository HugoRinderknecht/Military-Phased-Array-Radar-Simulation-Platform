import numpy as np
from typing import List, Dict, Optional, Tuple
from models.tracking import Track
from models.environment import Environment
import math


class MultipathModel:
    """多径传播模型"""

    def __init__(self):
        self.direct_path_delay = 0.0
        self.reflected_path_delay = 0.0
        self.reflection_coefficient = 0.0
        self.phase_difference = 0.0
        self.multipath_components = []

    def add_reflection_path(self, delay: float, amplitude: float, phase: float):
        """添加反射路径"""
        self.multipath_components.append({
            'delay': delay,
            'amplitude': amplitude,
            'phase': phase
        })


class TerrainMap:
    """地形图模型"""

    def __init__(self, width: int = 1024, height: int = 1024, resolution: float = 10.0):
        self.width = width
        self.height = height
        self.resolution = resolution  # meters per pixel
        self.elevation = np.zeros((height, width))
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.terrain_type = np.ones((height, width), dtype=int)  # 1=land, 0=water

    def get_elevation(self, x: float, y: float) -> float:
        """获取指定坐标的地形高度"""
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)

        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            return self.elevation[grid_y, grid_x]
        return 0.0

    def get_terrain_type(self, x: float, y: float) -> int:
        """获取指定坐标的地形类型"""
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
        """低空目标增强处理主函数"""
        enhanced_tracks = []

        for track in tracks:
            if self._is_low_altitude(track):
                enhanced_track = self._process_low_altitude_track(track, terrain)
                enhanced_tracks.append(enhanced_track)
            else:
                enhanced_tracks.append(track)

        return enhanced_tracks

    def _is_low_altitude(self, track: Track) -> bool:
        """判断是否为低空目标"""
        altitude = track.state[2]  # z坐标
        return altitude < self.low_altitude_threshold

    def _process_low_altitude_track(self, track: Track, terrain: Optional[TerrainMap]) -> Track:
        """处理单个低空目标"""
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
        """多径效应建模"""
        model = MultipathModel()

        target_x, target_y, target_z = track.state[:3]
        target_range = np.sqrt(target_x ** 2 + target_y ** 2 + target_z ** 2)

        # 直达路径
        model.direct_path_delay = 2 * target_range / 3e8

        if terrain is not None:
            # 地面反射模型
            ground_reflection_point = self._find_specular_reflection_point(
                np.array([0, 0, 0]),  # 雷达位置
                np.array([target_x, target_y, target_z]),
                terrain
            )

            if ground_reflection_point is not None:
                # 计算反射路径
                radar_to_reflection = np.linalg.norm(ground_reflection_point)
                reflection_to_target = np.linalg.norm(
                    np.array([target_x, target_y, target_z]) - ground_reflection_point
                )
                reflected_range = radar_to_reflection + reflection_to_target

                model.reflected_path_delay = 2 * reflected_range / 3e8

                # 反射系数（取决于地面类型和掠射角）
                grazing_angle = self._calculate_grazing_angle(
                    ground_reflection_point,
                    np.array([target_x, target_y, target_z])
                )

                model.reflection_coefficient = self._calculate_reflection_coefficient(
                    grazing_angle, terrain
                )

                # 相位差
                path_difference = reflected_range - target_range
                model.phase_difference = 4 * np.pi * path_difference / self.wavelength

            # 多重反射和散射
            self._add_complex_propagation_effects(model, track, terrain)
        else:
            # 简化的平面地面模型
            ground_distance = np.sqrt(target_x ** 2 + target_y ** 2)
            reflected_range = np.sqrt(ground_distance ** 2 + (2 * target_z) ** 2)

            model.reflected_path_delay = 2 * reflected_range / 3e8
            model.reflection_coefficient = 0.3  # 典型地面反射系数

            path_difference = reflected_range - target_range
            model.phase_difference = 4 * np.pi * path_difference / self.wavelength

        return model

    def apply_stap_processing(self, track: Track, multipath_model: MultipathModel) -> float:
        """空时自适应处理"""
        # 构造杂波加干扰协方差矩阵
        clutter_covariance = self._build_clutter_covariance_matrix(track, multipath_model)

        # 计算最优权重向量
        optimal_weights = self._calculate_stap_weights(clutter_covariance)

        # 更新STAP权重
        self.stap_weights = optimal_weights.reshape(self.stap_channels, self.stap_pulses)

        # 计算信杂噪比改善
        sincr_improvement = self._calculate_sincr_improvement(optimal_weights, clutter_covariance)

        return sincr_improvement

    def _build_clutter_covariance_matrix(self, track: Track, multipath_model: MultipathModel) -> np.ndarray:
        """构建杂波协方差矩阵"""
        matrix_size = self.stap_channels * self.stap_pulses
        covariance = np.eye(matrix_size, dtype=complex) * 0.01  # 噪声本底

        # 添加地面杂波分量
        for channel in range(self.stap_channels):
            for pulse in range(self.stap_pulses):
                row_idx = channel * self.stap_pulses + pulse

                # 空间频率
                spatial_freq = channel * 2 * np.pi / self.stap_channels

                # 时间频率（多普勒）
                temporal_freq = pulse * 2 * np.pi / self.stap_pulses

                # 杂波功率
                clutter_power = self._calculate_clutter_power(track, spatial_freq, temporal_freq)

                # 多径效应
                multipath_factor = self._calculate_multipath_factor(multipath_model, spatial_freq)

                covariance[row_idx, row_idx] += clutter_power * multipath_factor

        # 添加相关性
        for i in range(matrix_size):
            for j in range(i + 1, matrix_size):
                correlation = self._calculate_clutter_correlation(i, j, track)
                covariance[i, j] = correlation
                covariance[j, i] = np.conj(correlation)

        return covariance

    def _calculate_stap_weights(self, clutter_covariance: np.ndarray) -> np.ndarray:
        """计算STAP权重向量"""
        # 期望信号导向矢量
        steering_vector = self._generate_steering_vector()

        try:
            # MVDR权重计算
            inv_covariance = np.linalg.inv(clutter_covariance + np.eye(clutter_covariance.shape[0]) * 1e-6)
            weights = inv_covariance @ steering_vector
            weights /= (np.conj(steering_vector.T) @ inv_covariance @ steering_vector)
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用对角线加载
            regularized_cov = clutter_covariance + np.eye(clutter_covariance.shape[0]) * 0.1
            inv_covariance = np.linalg.inv(regularized_cov)
            weights = inv_covariance @ steering_vector
            weights /= (np.conj(steering_vector.T) @ inv_covariance @ steering_vector)

        return weights

    def apply_polarimetric_processing(self, track: Track) -> float:
        """极化处理增强"""
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
        polarization_gain = self._calculate_polarization_gain(target_probability)

        return polarization_gain

    def analyze_terrain_occlusion(self, track: Track, terrain: Optional[TerrainMap]) -> OcclusionMap:
        """地形遮蔽分析"""
        occlusion_map = OcclusionMap()

        if terrain is None:
            occlusion_map.severity = 0.0
            return occlusion_map

        radar_pos = np.array([0, 0, 50])  # 假设雷达高度50m
        target_pos = track.state[:3]

        # 视线分析
        los_points = self._generate_line_of_sight_points(radar_pos, target_pos, terrain.resolution)
        blocked_points = 0

        for point in los_points:
            terrain_height = terrain.get_elevation(point[0], point[1])
            if terrain_height > point[2]:
                blocked_points += 1
                occlusion_map.occlusion_points.append(point)

        occlusion_map.severity = blocked_points / len(los_points)
        occlusion_map.probability = min(occlusion_map.severity * 1.5, 1.0)

        # 计算遮蔽角度范围
        target_azimuth = np.arctan2(target_pos[1], target_pos[0])
        occlusion_map.azimuth_range = [target_azimuth - np.pi / 18, target_azimuth + np.pi / 18]

        target_elevation = np.arcsin(target_pos[2] / np.linalg.norm(target_pos))
        occlusion_map.elevation_range = [target_elevation - np.pi / 36, target_elevation + np.pi / 36]

        return occlusion_map

    def apply_multiband_fusion(self, track: Track) -> float:
        """多频段数据融合"""
        band_improvements = []

        for band in self.frequency_bands:
            # 模拟不同频段的检测性能
            band_performance = self._simulate_band_performance(track, band)
            band_improvements.append(band_performance * band['weight'])

        # 加权融合
        total_improvement = sum(band_improvements)

        # 多样性增益
        diversity_gain = self._calculate_frequency_diversity_gain(len(self.frequency_bands))

        return total_improvement * diversity_gain

    def apply_adaptive_filtering(self, track: Track, multipath_model: MultipathModel) -> float:
        """自适应滤波处理"""
        # 构建自适应滤波器
        filter_order = 32
        adaptation_step = 0.01

        # 参考信号（直达波）
        reference_signal = self._generate_reference_signal(track)

        # 接收信号（含多径）
        received_signal = self._generate_received_signal_with_multipath(track, multipath_model)

        # LMS自适应滤波
        filter_coefficients = np.zeros(filter_order, dtype=complex)
        mse_reduction = self._lms_adaptive_filter(
            reference_signal, received_signal, filter_coefficients, adaptation_step
        )

        return mse_reduction

    def apply_ml_enhancement(self, track: Track) -> float:
        """机器学习增强处理"""
        # 提取特征向量
        features = self._extract_ml_features(track)

        # 简化的目标分类器（实际中应使用训练好的模型）
        target_confidence = self._simple_target_classifier(features)

        # 基于置信度的增强
        enhancement_factor = max(0, (target_confidence - 0.5) * 2)

        return enhancement_factor

    def fuse_with_auxiliary_sensors(self, track: Track):
        """辅助传感器融合"""
        # 红外传感器数据融合
        ir_detection = self._simulate_ir_detection(track)

        # 声纳传感器数据融合(如果适用)
        acoustic_detection = self._simulate_acoustic_detection(track)

        # 视觉传感器数据融合
        visual_detection = self._simulate_visual_detection(track)

        # 多传感器数据融合
        fused_position = self._fuse_sensor_measurements(
            track, ir_detection, acoustic_detection, visual_detection
        )

        if fused_position is not None:
            # 更新航迹位置
            track.state[:3] = fused_position
            track.score *= 1.3  # 多传感器融合增强置信度
            track.confirmed = True

    # 辅助函数实现
    def _find_specular_reflection_point(self, radar_pos: np.ndarray, target_pos: np.ndarray,
                                        terrain: TerrainMap) -> Optional[np.ndarray]:
        """查找镜面反射点"""
        # 简化实现：在雷达和目标连线的中点附近搜索
        midpoint = (radar_pos + target_pos) / 2
        midpoint[2] = terrain.get_elevation(midpoint[0], midpoint[1])

        # 验证是否满足镜面反射条件
        if self._is_valid_reflection_point(radar_pos, target_pos, midpoint):
            return midpoint

        return None

    def _calculate_grazing_angle(self, reflection_point: np.ndarray, target_pos: np.ndarray) -> float:
        """计算掠射角"""
        incident_vector = target_pos - reflection_point
        return np.arcsin(abs(incident_vector[2]) / np.linalg.norm(incident_vector))

    def _calculate_reflection_coefficient(self, grazing_angle: float, terrain: TerrainMap) -> float:
        """计算反射系数"""
        # 简化的反射系数模型
        if grazing_angle < np.pi / 90:  # 小于2度
            return 0.7  # 高反射
        elif grazing_angle < np.pi / 18:  # 小于10度
            return 0.4
        else:
            return 0.1  # 低反射

    def _generate_steering_vector(self) -> np.ndarray:
        """生成期望信号导向矢量"""
        vector_length = self.stap_channels * self.stap_pulses
        steering_vector = np.ones(vector_length, dtype=complex)

        # 目标的空时频率
        target_spatial_freq = 0.0  # 假设目标在主瓣方向
        target_doppler_freq = np.pi / 8  # 假设目标多普勒频率

        for channel in range(self.stap_channels):
            for pulse in range(self.stap_pulses):
                idx = channel * self.stap_pulses + pulse

                phase = (channel * target_spatial_freq + pulse * target_doppler_freq)
                steering_vector[idx] = np.exp(1j * phase)

        return steering_vector

    def _calculate_clutter_power(self, track: Track, spatial_freq: float, temporal_freq: float) -> float:
        """计算杂波功率"""
        # 地面杂波功率模型
        base_clutter_power = 1.0

        # 角度依赖
        angle_factor = max(0.1, np.cos(spatial_freq))

        # 多普勒依赖
        doppler_factor = 1.0 + 0.5 * np.cos(temporal_freq)

        return base_clutter_power * angle_factor * doppler_factor

    def _extract_ml_features(self, track: Track) -> np.ndarray:
        """提取机器学习特征"""
        features = []

        # 运动学特征
        velocity = np.linalg.norm(track.state[3:6])
        features.append(velocity)

        acceleration = 0.0  # 简化：假设匀速
        features.append(acceleration)

        # 几何特征
        altitude = track.state[2]
        features.append(altitude)

        range_val = np.linalg.norm(track.state[:3])
        features.append(range_val)

        # 信号特征
        if track.rcs_history:
            rcs_mean = np.mean(track.rcs_history)
            rcs_std = np.std(track.rcs_history) if len(track.rcs_history) > 1 else 0
        else:
            rcs_mean = 0.0
            rcs_std = 0.0

        features.extend([rcs_mean, rcs_std])

        # 历史特征
        track_persistence = track.age
        features.append(track_persistence)

        confirmation_rate = 1.0 if track.confirmed else 0.0
        features.append(confirmation_rate)

        return np.array(features)

    def _simple_target_classifier(self, features: np.ndarray) -> float:
        """简单的目标分类器"""
        # 权重（简化模型）
        weights = np.array([0.2, 0.1, 0.3, -0.1, 0.2, 0.05, 0.1, 0.05])

        # 线性组合 + sigmoid
        score = np.dot(features, weights)
        confidence = 1.0 / (1.0 + np.exp(-score))

        return confidence

    def _simulate_ir_detection(self, track: Track) -> Optional[np.ndarray]:
        """模拟红外传感器检测"""
        # 简化的红外检测模拟
        if track.state[2] > 200:  # 高度大于200m时红外可检测
            noise = np.random.normal(0, 50, 3)  # 50m标准差
            return track.state[:3] + noise
        return None

    def _simulate_acoustic_detection(self, track: Track) -> Optional[np.ndarray]:
        """模拟声学传感器检测"""
        # 声学传感器主要在低空低速时有效
        velocity = np.linalg.norm(track.state[3:6])
        if track.state[2] < 1000 and velocity > 50:
            noise = np.random.normal(0, 100, 3)  # 100m标准差
            return track.state[:3] + noise
        return None

    def _simulate_visual_detection(self, track: Track) -> Optional[np.ndarray]:
        """模拟视觉传感器检测"""
        # 视觉传感器在晴天近距离时有效
        range_val = np.linalg.norm(track.state[:3])
        if range_val < 20000 and self.environment and self.environment.weather.weather_type == "clear":
            noise = np.random.normal(0, 30, 3)  # 30m标准差
            return track.state[:3] + noise
        return None

    def _fuse_sensor_measurements(self, track: Track, ir_pos: Optional[np.ndarray],
                                  acoustic_pos: Optional[np.ndarray],
                                  visual_pos: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """融合多传感器测量"""
        measurements = []
        weights = []

        # 雷达测量
        measurements.append(track.state[:3])
        weights.append(0.5)  # 雷达权重

        # 红外测量
        if ir_pos is not None:
            measurements.append(ir_pos)
            weights.append(0.3)

        # 声学测量
        if acoustic_pos is not None:
            measurements.append(acoustic_pos)
            weights.append(0.1)

        # 视觉测量
        if visual_pos is not None:
            measurements.append(visual_pos)
            weights.append(0.1)

        if len(measurements) == 1:
            return None  # 只有雷达数据，不需要融合

        # 归一化权重
        weights = np.array(weights)
        weights /= weights.sum()

        # 加权平均融合
        fused_position = np.zeros(3)
        for measurement, weight in zip(measurements, weights):
            fused_position += weight * measurement

        return fused_position

    # 其他辅助函数的简化实现
    def _add_complex_propagation_effects(self, model: MultipathModel, track: Track, terrain: TerrainMap):
        """添加复杂传播效应"""
        # 建筑物反射、大气折射等复杂效应的简化处理
        pass

    def _calculate_multipath_factor(self, model: MultipathModel, spatial_freq: float) -> float:
        """计算多径因子"""
        return 1.0 + model.reflection_coefficient * np.cos(model.phase_difference + spatial_freq)

    def _calculate_clutter_correlation(self, i: int, j: int, track: Track) -> complex:
        """计算杂波相关性"""
        # 简化的相关性模型
        distance = abs(i - j)
        correlation_coeff = np.exp(-distance / 10.0)
        return correlation_coeff * np.exp(1j * np.random.uniform(0, 2 * np.pi))

    def _calculate_sincr_improvement(self, weights: np.ndarray, covariance: np.ndarray) -> float:
        """计算信杂噪比改善"""
        # SINR改善计算
        numerator = abs(np.conj(weights.T) @ weights) ** 2
        denominator = np.real(np.conj(weights.T) @ covariance @ weights)

        return 10 * np.log10(numerator / max(denominator, 1e-10))

    def _generate_polarimetric_scattering_matrix(self, track: Track) -> np.ndarray:
        """生成极化散射矩阵"""
        # 2x2极化散射矩阵
        S = np.array([[1.0 + 0.1j, 0.2 + 0.05j],
                      [0.2 + 0.05j, 0.8 + 0.15j]])

        # 根据目标类型调整
        if hasattr(track, 'target_type'):
            if track.target_type == 'aircraft':
                S[0, 0] *= 1.5  # 飞机的HH分量较强

        return S

    def _extract_polarimetric_features(self, S: np.ndarray) -> Dict[str, float]:
        """提取极化特征"""
        features = {}

        # 极化比
        features['polarization_ratio'] = abs(S[1, 1] / S[0, 0])

        # 交叉极化比
        features['cross_pol_ratio'] = abs(S[0, 1] / S[0, 0])

        # 相位差
        features['phase_diff'] = np.angle(S[1, 1]) - np.angle(S[0, 0])

        # 极化度
        features['degree_of_polarization'] = np.linalg.det(S @ S.conj().T) / (np.trace(S @ S.conj().T)) ** 2

        return features

    def _classify_target_clutter_polarimetric(self, features: Dict[str, float]) -> float:
        """基于极化特征的目标/杂波分类"""
        score = 0.5  # 基础概率

        # 极化比判据
        if 0.1 < features['polarization_ratio'] < 0.8:
            score += 0.2  # 人工目标特征

        # 相位差判据
        if abs(features['phase_diff']) < np.pi / 6:
            score += 0.1  # 金属目标特征

        return min(score, 1.0)

    def _optimize_polarization_weights(self, S: np.ndarray) -> np.ndarray:
        """优化极化权重"""
        # 简化的极化权重优化
        eigenvals, eigenvecs = np.linalg.eig(S @ S.conj().T)
        optimal_weights = eigenvecs[:, np.argmax(eigenvals)]

        return np.abs(optimal_weights)

    def _calculate_polarization_gain(self, target_probability: float) -> float:
        """计算极化增益"""
        return target_probability * 3.0  # 最大3dB增益

    def _simulate_band_performance(self, track: Track, band: Dict) -> float:
        """模拟频段性能"""
        # 简化的频段性能模型
        frequency = band['center']

        # 不同频率的性能特点
        if frequency < 5e9:  # L/S波段
            return 0.8  # 对大目标有利
        elif frequency < 20e9:  # X波段
            return 1.0  # 平衡性能
        else:  # Ka波段
            return 1.2 if track.state[2] < 1000 else 0.6  # 对小目标有利但受天气影响

    def _calculate_frequency_diversity_gain(self, num_bands: int) -> float:
        """计算频率分集增益"""
        return min(1.0 + 0.1 * (num_bands - 1), 1.5)

    def _generate_reference_signal(self, track: Track) -> np.ndarray:
        """生成参考信号"""
        signal_length = 1024
        return np.random.random(signal_length) + 1j * np.random.random(signal_length)

    def _generate_received_signal_with_multipath(self, track: Track, model: MultipathModel) -> np.ndarray:
        """生成含多径的接收信号"""
        direct_signal = self._generate_reference_signal(track)

        # 添加多径分量
        delayed_signal = np.roll(direct_signal, int(model.reflected_path_delay * 1e6))
        multipath_signal = (direct_signal +
                            model.reflection_coefficient * delayed_signal *
                            np.exp(1j * model.phase_difference))

        return multipath_signal

    def _lms_adaptive_filter(self, reference: np.ndarray, received: np.ndarray,
                             coefficients: np.ndarray, step_size: float) -> float:
        """LMS自适应滤波"""
        initial_mse = np.mean(np.abs(received - reference) ** 2)

        # LMS迭代（简化实现）
        for i in range(min(len(reference), len(received)) - len(coefficients)):
            input_vector = received[i:i + len(coefficients)]
            output = np.dot(coefficients.conj(), input_vector)
            error = reference[i] - output

            coefficients += step_size * error.conj() * input_vector

        # 计算滤波后的MSE
        filtered_signal = np.convolve(received, coefficients.conj(), mode='same')
        final_mse = np.mean(np.abs(reference - filtered_signal[:len(reference)]) ** 2)

        return 10 * np.log10(initial_mse / max(final_mse, 1e-10))

    def _generate_line_of_sight_points(self, start: np.ndarray, end: np.ndarray,
                                       resolution: float) -> List[np.ndarray]:
        """生成视线点集"""
        distance = np.linalg.norm(end - start)
        num_points = int(distance / resolution)

        points = []
        for i in range(num_points + 1):
            t = i / max(num_points, 1)
            point = start + t * (end - start)
            points.append(point)

        return points

    def _is_valid_reflection_point(self, radar_pos: np.ndarray, target_pos: np.ndarray,
                                   reflection_point: np.ndarray) -> bool:
        """验证反射点有效性"""
        # 简化验证：检查反射角等于入射角
        incident = reflection_point - radar_pos
        reflected = target_pos - reflection_point

        # 归一化
        incident = incident / np.linalg.norm(incident)
        reflected = reflected / np.linalg.norm(reflected)

        # 检查角度（简化）
        cos_angle = np.dot(incident, reflected)
        return abs(cos_angle) > 0.7  # 允许一定误差
