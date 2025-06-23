import numpy as np
import numba
from numba import jit, vectorize, prange
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque

logger = logging.getLogger(__name__)


# Numba JIT优化的数学计算函数
@numba.jit(nopython=True, cache=True)
def polar_to_cartesian_batch(ranges, azimuths):
    """批量极坐标转换为直角坐标"""
    x = ranges * np.cos(np.radians(azimuths))
    y = ranges * np.sin(np.radians(azimuths))
    return x, y


@numba.jit(nopython=True, cache=True)
def cartesian_to_polar_batch(x_coords, y_coords):
    """批量直角坐标转换为极坐标"""
    ranges = np.sqrt(x_coords ** 2 + y_coords ** 2)
    azimuths = np.degrees(np.arctan2(y_coords, x_coords))
    # 确保方位角在0-360度范围内
    azimuths = np.where(azimuths < 0, azimuths + 360, azimuths)
    return ranges, azimuths


@numba.jit(nopython=True, cache=True)
def calculate_snr_batch(powers, gains, wavelengths, rcs_values, ranges,
                        loss_factors, pulse_widths, bandwidths, noise_temps):
    """批量计算SNR - 复杂雷达方程"""
    # 转换增益为线性值
    gain_linear = 10 ** (gains / 10)

    # 计算SNR (向量化)
    numerator = (powers * (gain_linear ** 2) * (wavelengths ** 2) * rcs_values *
                 pulse_widths * bandwidths)
    denominator = ((4 * np.pi) ** 3 * (ranges ** 4) * loss_factors * noise_temps)

    snr = numerator / denominator
    return snr


@numba.jit(nopython=True, cache=True)
def calculate_detection_probability_batch(snr_values, num_pulses):
    """批量计算检测概率"""
    # 多脉冲积累
    snr_accumulated = snr_values * num_pulses

    # Sigmoid检测概率模型
    detection_probs = 1.0 / (1.0 + np.exp(-snr_accumulated / 1000.0))

    # 限制最大检测概率为95%
    detection_probs = np.minimum(detection_probs, 0.95)

    return detection_probs


@numba.jit(nopython=True, cache=True)
def add_measurement_noise_batch(positions, noise_stds):
    """批量添加测量噪声"""
    noise_x = np.random.normal(0, noise_stds)
    noise_y = np.random.normal(0, noise_stds)

    noisy_positions = np.empty_like(positions)
    noisy_positions[:, 0] = positions[:, 0] + noise_x
    noisy_positions[:, 1] = positions[:, 1] + noise_y

    return noisy_positions


@numba.jit(nopython=True, cache=True, parallel=True)
def parallel_range_calculation(positions):
    """并行计算距离"""
    n = positions.shape[0]
    ranges = np.empty(n)

    for i in prange(n):
        ranges[i] = np.sqrt(positions[i, 0] ** 2 + positions[i, 1] ** 2)

    return ranges


# 预计算查找表
@numba.jit(nopython=True, cache=True)
def create_trig_lookup_tables(size=3600):
    """创建三角函数查找表（0.1度精度）"""
    angles = np.linspace(0, 360, size)
    cos_table = np.cos(np.radians(angles))
    sin_table = np.sin(np.radians(angles))
    return angles, cos_table, sin_table


# 全局查找表
ANGLES, COS_TABLE, SIN_TABLE = create_trig_lookup_tables()


@numba.jit(nopython=True, cache=True)
def fast_cos_sin(angle):
    """使用查找表的快速三角函数计算"""
    # 将角度规范化到0-360度
    angle = angle % 360
    # 找到最接近的索引
    idx = int(angle * 10)  # 0.1度精度
    if idx >= len(COS_TABLE):
        idx = len(COS_TABLE) - 1
    return COS_TABLE[idx], SIN_TABLE[idx]


class MemoryPool:
    """内存池管理器"""

    def __init__(self):
        self._pools = {}
        self._lock = threading.Lock()

    def get_array(self, shape, dtype=np.float32):
        """获取预分配的数组"""
        key = (shape, dtype)

        with self._lock:
            if key not in self._pools:
                self._pools[key] = deque()

            pool = self._pools[key]
            if pool:
                return pool.popleft()
            else:
                return np.empty(shape, dtype=dtype)

    def return_array(self, array):
        """归还数组到池中"""
        key = (array.shape, array.dtype)

        with self._lock:
            if key not in self._pools:
                self._pools[key] = deque()

            # 清零数组并归还
            array.fill(0)
            self._pools[key].append(array)


class VisualizationProcessor:
    """优化的可视化数据处理器"""

    def __init__(self, max_workers=4):
        self.data_processor = None
        self.memory_pool = MemoryPool()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 缓存管理
        self._grid_cache = None
        self._grid_cache_time = 0
        self._cache_ttl = 60.0  # 缓存60秒

        # 预分配常用数组
        self._temp_positions = np.empty((10000, 2), dtype=np.float32)
        self._temp_ranges = np.empty(10000, dtype=np.float32)
        self._temp_azimuths = np.empty(10000, dtype=np.float32)

    def process_radar_display_data(self, radar_data: Dict, targets: List[Dict],
                                   time_step: float) -> Dict[str, Any]:
        """优化的雷达显示数据处理"""
        try:
            current_time = datetime.now().timestamp()

            # 使用缓存的网格数据
            if (self._grid_cache is None or
                    (current_time - self._grid_cache_time) > self._cache_ttl):
                self._grid_cache = self._generate_radar_grid()
                self._grid_cache_time = current_time

            # 并行处理不同组件
            futures = []

            # 异步处理目标点
            futures.append(
                self.executor.submit(self._process_target_points_optimized, targets, time_step)
            )

            # 异步处理检测数据
            futures.append(
                self.executor.submit(self._process_detections_optimized, radar_data, targets)
            )

            # 异步处理航迹数据
            futures.append(
                self.executor.submit(self._process_tracks, radar_data.get('tracks', []))
            )

            # 收集结果
            target_points = futures[0].result()
            detections = futures[1].result()
            tracks = futures[2].result()

            # 生成扫描线（快速操作，不需要异步）
            sweep_line = self._generate_sweep_line(radar_data.get('current_angle', 0))

            return {
                'grid': self._grid_cache,
                'targets': target_points,
                'detections': detections,
                'tracks': tracks,
                'sweep_line': sweep_line,
                'timestamp': time_step,
                'radar_status': self._get_radar_status(radar_data)
            }

        except Exception as e:
            logger.error(f"Error processing radar display data: {str(e)}")
            return self._get_empty_radar_data()

    def _process_target_points_optimized(self, targets: List[Dict], time_step: float) -> List[Dict]:
        """优化的目标点处理"""
        if not targets:
            return []

        n_targets = len(targets)

        # 预分配或获取数组
        positions = self.memory_pool.get_array((n_targets, 2), np.float32)
        velocities = self.memory_pool.get_array((n_targets, 2), np.float32)
        rcs_values = self.memory_pool.get_array((n_targets,), np.float32)

        try:
            # 批量提取数据
            for i, target in enumerate(targets):
                pos = target.get('position', [0, 0])
                vel = target.get('velocity', [0, 0])
                positions[i] = [pos[0], pos[1]]
                velocities[i] = [vel[0], vel[1]]
                rcs_values[i] = target.get('rcs', 1.0)

            # 批量极坐标转换
            ranges, azimuths = cartesian_to_polar_batch(positions[:, 0], positions[:, 1])

            # 批量速度计算
            speeds = np.sqrt(velocities[:, 0] ** 2 + velocities[:, 1] ** 2)

            # 构建结果
            processed_targets = []
            for i in range(n_targets):
                target = targets[i]
                processed_targets.append({
                    'id': i,
                    'position': {
                        'x': float(positions[i, 0]),
                        'y': float(positions[i, 1]),
                        'range': float(ranges[i]),
                        'azimuth': float(azimuths[i])
                    },
                    'velocity': {
                        'x': float(velocities[i, 0]),
                        'y': float(velocities[i, 1]),
                        'speed': float(speeds[i])
                    },
                    'rcs': float(rcs_values[i]),
                    'altitude': target.get('altitude', 0),
                    'type': target.get('type', 'unknown'),
                    'formation_id': target.get('formation_id'),
                    'timestamp': time_step
                })

            return processed_targets

        finally:
            # 归还数组到内存池
            self.memory_pool.return_array(positions)
            self.memory_pool.return_array(velocities)
            self.memory_pool.return_array(rcs_values)

    def _process_detections_optimized(self, radar_data: Dict, targets: List[Dict]) -> List[Dict]:
        """优化的检测处理 - 使用复杂雷达方程"""
        if not targets:
            return []

        n_targets = len(targets)

        # 获取雷达参数
        power = radar_data.get('power', 1e6)
        frequency = radar_data.get('frequency', 3e9)
        gain = radar_data.get('gain', 30)
        loss_factor = radar_data.get('loss_factor', 1.2)
        pulse_width = radar_data.get('pulse_width', 1e-6)
        bandwidth = radar_data.get('bandwidth', 1e6)
        noise_temp = radar_data.get('noise_temp', 290)
        num_pulses = radar_data.get('num_pulses', 1)

        # 计算波长
        wavelength = 3e8 / frequency

        # 预分配数组
        positions = self.memory_pool.get_array((n_targets, 2), np.float32)
        rcs_values = self.memory_pool.get_array((n_targets,), np.float32)

        try:
            # 批量提取目标数据
            for i, target in enumerate(targets):
                pos = target.get('position', [0, 0])
                positions[i] = [pos[0], pos[1]]
                rcs_values[i] = target.get('rcs', 1.0)

            # 批量计算距离
            ranges = parallel_range_calculation(positions)

            # 创建雷达参数数组（广播）
            powers = np.full(n_targets, power, dtype=np.float32)
            gains = np.full(n_targets, gain, dtype=np.float32)
            wavelengths = np.full(n_targets, wavelength, dtype=np.float32)
            loss_factors = np.full(n_targets, loss_factor, dtype=np.float32)
            pulse_widths = np.full(n_targets, pulse_width, dtype=np.float32)
            bandwidths = np.full(n_targets, bandwidth, dtype=np.float32)
            noise_temps = np.full(n_targets, noise_temp, dtype=np.float32)

            # 批量计算SNR（复杂雷达方程）
            snr_values = calculate_snr_batch(
                powers, gains, wavelengths, rcs_values, ranges,
                loss_factors, pulse_widths, bandwidths, noise_temps
            )

            # 批量计算检测概率
            detection_probs = calculate_detection_probability_batch(snr_values, num_pulses)

            # 生成随机数用于检测决策
            random_values = np.random.random(n_targets)
            detected_mask = random_values < detection_probs
            detected_indices = np.where(detected_mask)[0]

            if len(detected_indices) == 0:
                return []

            # 为检测到的目标添加噪声
            detected_positions = positions[detected_indices]
            detected_ranges = ranges[detected_indices]

            # 计算测量噪声标准差
            noise_stds = self._get_measurement_noise_batch(detected_ranges)

            # 添加噪声
            noisy_positions = add_measurement_noise_batch(detected_positions, noise_stds)

            # 重新计算极坐标
            noisy_ranges, noisy_azimuths = cartesian_to_polar_batch(
                noisy_positions[:, 0], noisy_positions[:, 1]
            )

            # 构建检测结果
            detections = []
            for idx, det_idx in enumerate(detected_indices):
                detections.append({
                    'target_id': int(det_idx),
                    'position': {
                        'x': float(noisy_positions[idx, 0]),
                        'y': float(noisy_positions[idx, 1]),
                        'range': float(noisy_ranges[idx]),
                        'azimuth': float(noisy_azimuths[idx])
                    },
                    'snr': float(10 * np.log10(snr_values[det_idx] + 1e-12)),  # 转换为dB
                    'confidence': float(detection_probs[det_idx]),
                    'measurement_noise': float(noise_stds[idx])
                })

            return detections

        finally:
            # 归还数组
            self.memory_pool.return_array(positions)
            self.memory_pool.return_array(rcs_values)

    @numba.jit(nopython=True, cache=True)
    def _get_measurement_noise_batch(self, ranges):
        """批量计算测量噪声标准差"""
        base_noise = 10.0  # 基础噪声10米
        range_factors = ranges / 100000.0  # 100km标准化
        return base_noise * (1.0 + range_factors)

    def _process_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """优化的航迹处理"""
        if not tracks:
            return []

        processed_tracks = []

        for track in tracks:
            try:
                history = track.get('history', [])
                if len(history) < 2:
                    continue

                # 限制历史点数量
                recent_history = history[-10:]
                n_points = len(recent_history)

                # 批量处理航迹点
                positions = np.array([[p.get('position', [0, 0])[0],
                                       p.get('position', [0, 0])[1]]
                                      for p in recent_history], dtype=np.float32)

                ranges, azimuths = cartesian_to_polar_batch(positions[:, 0], positions[:, 1])

                track_points = []
                for i in range(n_points):
                    point = recent_history[i]
                    track_points.append({
                        'x': float(positions[i, 0]),
                        'y': float(positions[i, 1]),
                        'range': float(ranges[i]),
                        'azimuth': float(azimuths[i]),
                        'timestamp': point.get('timestamp', 0)
                    })

                processed_tracks.append({
                    'track_id': track.get('track_id'),
                    'points': track_points,
                    'predicted_position': track.get('predicted_position'),
                    'velocity': track.get('velocity'),
                    'quality': track.get('quality', 0.5)
                })

            except Exception as e:
                logger.warning(f"Error processing track: {str(e)}")
                continue

        return processed_tracks

    def process_chart_data(self, simulation_data: Dict, chart_type: str,
                           time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """优化的图表数据处理"""
        try:
            chart_processors = {
                'performance': self._process_performance_chart,
                'environment_impact': self._process_environment_chart,
                'target_analysis': self._process_target_chart,
                'detection_statistics': self._process_detection_stats
            }

            processor = chart_processors.get(chart_type)
            if not processor:
                raise ValueError(f"Unsupported chart type: {chart_type}")

            return processor(simulation_data, time_range)

        except Exception as e:
            logger.error(f"Error processing chart data for {chart_type}: {str(e)}")
            return {'error': str(e), 'data': []}

    def process_realtime_data(self, current_state: Dict) -> Dict[str, Any]:
        """优化的实时数据处理"""
        try:
            return {
                'targets': self._extract_target_positions_fast(current_state.get('targets', [])),
                'detections': self._extract_detections_fast(current_state.get('detections', [])),
                'radar_angle': current_state.get('radar_angle', 0),
                'performance_metrics': self._extract_key_metrics(current_state),
                'timestamp': current_state.get('timestamp', datetime.now().timestamp())
            }
        except Exception as e:
            logger.error(f"Error processing realtime data: {str(e)}")
            return {}

    def _extract_target_positions_fast(self, targets: List[Dict]) -> List[Dict]:
        """快速提取目标位置"""
        if not targets:
            return []

        # 使用列表推导式优化
        return [
            {
                'id': i,
                'x': target.get('position', [0, 0])[0],
                'y': target.get('position', [0, 0])[1],
                'type': target.get('type', 'unknown')
            }
            for i, target in enumerate(targets)
        ]

    def _extract_detections_fast(self, detections: List[Dict]) -> List[Dict]:
        """快速提取检测数据"""
        if not detections:
            return []

        return [
            {
                'id': det.get('target_id', i),
                'x': det.get('position', {}).get('x', 0),
                'y': det.get('position', {}).get('y', 0),
                'confidence': det.get('confidence', 0.5)
            }
            for i, det in enumerate(detections)
        ]

    # 保留原有的其他方法，进行性能优化
    def _generate_radar_grid(self) -> Dict[str, Any]:
        """生成雷达显示网格（缓存优化）"""
        range_rings = [10000, 20000, 50000, 100000, 200000]
        azimuth_lines = list(range(0, 360, 15))

        return {
            'range_rings': range_rings,
            'azimuth_lines': azimuth_lines,
            'center': [0, 0],
            'max_range': max(range_rings)
        }

    def _generate_sweep_line(self, current_angle: float) -> Dict[str, Any]:
        """生成雷达扫描线（使用快速三角函数）"""
        cos_val, sin_val = fast_cos_sin(current_angle)
        max_range = 200000

        return {
            'angle': current_angle,
            'start_point': [0, 0],
            'end_point': [
                max_range * cos_val,
                max_range * sin_val
            ]
        }

    def _get_radar_status(self, radar_data: Dict) -> Dict[str, Any]:
        """获取雷达状态信息"""
        return {
            'mode': radar_data.get('mode', 'search'),
            'power_level': radar_data.get('power_level', 1.0),
            'scan_rate': radar_data.get('scan_rate', 6.0),
            'beam_width': radar_data.get('beam_width', 3.0),
            'frequency': radar_data.get('frequency', 3e9),
            'status': 'operational'
        }

    def _process_performance_chart(self, data: Dict, time_range: Optional[Tuple[float, float]]) -> Dict:
        """处理性能图表数据"""
        time_points = np.array(data.get('time_points', []))
        detection_rates = np.array(data.get('detection_rates', []))
        false_alarm_rates = np.array(data.get('false_alarm_rates', []))

        # 使用numpy布尔索引优化时间范围过滤
        if time_range and len(time_points) > 0:
            start_time, end_time = time_range
            mask = (time_points >= start_time) & (time_points <= end_time)
            time_points = time_points[mask]
            detection_rates = detection_rates[mask]
            false_alarm_rates = false_alarm_rates[mask]

        return {
            'labels': time_points.tolist(),
            'datasets': [
                {
                    'label': 'Detection Rate',
                    'data': detection_rates.tolist(),
                    'borderColor': '#4CAF50',
                    'backgroundColor': 'rgba(76, 175, 80, 0.1)'
                },
                {
                    'label': 'False Alarm Rate',
                    'data': false_alarm_rates.tolist(),
                    'borderColor': '#F44336',
                    'backgroundColor': 'rgba(244, 67, 54, 0.1)'
                }
            ]
        }

    def _process_environment_chart(self, data: Dict, time_range: Optional[Tuple[float, float]]) -> Dict:
        """处理环境影响图表数据"""
        weather_impacts = data.get('weather_impacts', {})
        clutter_levels = data.get('clutter_levels', [])

        return {
            'weather_data': {
                'labels': list(weather_impacts.keys()),
                'data': list(weather_impacts.values())
            },
            'clutter_data': {
                'labels': [f'Range {i * 10}km' for i in range(len(clutter_levels))],
                'data': clutter_levels
            }
        }

    def _process_target_chart(self, data: Dict, time_range: Optional[Tuple[float, float]]) -> Dict:
        """处理目标分析图表数据"""
        target_types = data.get('target_types', {})
        altitude_distribution = data.get('altitude_distribution', {})

        return {
            'type_distribution': {
                'labels': list(target_types.keys()),
                'data': list(target_types.values())
            },
            'altitude_distribution': {
                'labels': list(altitude_distribution.keys()),
                'data': list(altitude_distribution.values())
            }
        }

    def _process_detection_stats(self, data: Dict, time_range: Optional[Tuple[float, float]]) -> Dict:
        """处理检测统计数据"""
        return {
            'total_detections': data.get('total_detections', 0),
            'confirmed_tracks': data.get('confirmed_tracks', 0),
            'false_alarms': data.get('false_alarms', 0),
            'detection_efficiency': data.get('detection_efficiency', 0.0)
        }

    def _extract_key_metrics(self, state: Dict) -> Dict[str, float]:
        """提取关键性能指标"""
        return {
            'detection_rate': state.get('detection_rate', 0.0),
            'false_alarm_rate': state.get('false_alarm_rate', 0.0),
            'track_count': len(state.get('tracks', [])),
            'target_count': len(state.get('targets', []))
        }

    def _get_empty_radar_data(self) -> Dict[str, Any]:
        """返回空的雷达数据结构"""
        return {
            'grid': self._generate_radar_grid(),
            'targets': [],
            'detections': [],
            'tracks': [],
            'sweep_line': self._generate_sweep_line(0),
            'timestamp': 0,
            'radar_status': {
                'mode': 'standby',
                'power_level': 0.0,
                'status': 'offline'
            }
        }

    def __del__(self):
        """清理资源"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


# 复杂雷达方程独立函数（保持向后兼容）
def calculate_detection_probability_complex(
        target: dict,
        radar_data: dict,
        range_val: float,
        velocity: float = 0.0,
        num_pulses: int = 1
):
    """
    使用复杂雷达方程计算检测概率
    这个函数现在内部使用了优化的批量计算
    """
    # 提取参数
    power = radar_data.get('power', 1e6)
    frequency = radar_data.get('frequency', 3e9)
    gain = radar_data.get('gain', 30)
    loss_factor = radar_data.get('loss_factor', 1.2)
    pulse_width = radar_data.get('pulse_width', 1e-6)
    bandwidth = radar_data.get('bandwidth', 1e6)
    noise_temp = radar_data.get('noise_temp', 290)
    rcs = target.get('rcs', 1.0)

    # 计算波长
    wavelength = 3e8 / frequency

    # 使用优化的批量计算函数（单个目标）
    powers = np.array([power], dtype=np.float32)
    gains = np.array([gain], dtype=np.float32)
    wavelengths = np.array([wavelength], dtype=np.float32)
    rcs_values = np.array([rcs], dtype=np.float32)
    ranges = np.array([range_val], dtype=np.float32)
    loss_factors = np.array([loss_factor], dtype=np.float32)
    pulse_widths = np.array([pulse_width], dtype=np.float32)
    bandwidths = np.array([bandwidth], dtype=np.float32)
    noise_temps = np.array([noise_temp], dtype=np.float32)

    # 计算SNR
    snr_values = calculate_snr_batch(
        powers, gains, wavelengths, rcs_values, ranges,
        loss_factors, pulse_widths, bandwidths, noise_temps
    )

    # 计算检测概率
    detection_probs = calculate_detection_probability_batch(snr_values, num_pulses)

    return float(detection_probs[0])
