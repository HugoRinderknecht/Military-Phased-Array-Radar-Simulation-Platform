import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from .data_processor import DataProcessor

logger = logging.getLogger(__name__)


class VisualizationProcessor:
    """可视化数据处理器 - 专门为前端提供优化的可视化数据"""

    def __init__(self):
        self.data_processor = DataProcessor()

    def process_radar_display_data(self, radar_data: Dict, targets: List[Dict],
                                   time_step: float) -> Dict[str, Any]:
        """
        处理雷达显示屏数据
        返回适合前端雷达显示的数据格式
        """
        try:
            # 生成极坐标网格数据
            grid_data = self._generate_radar_grid()

            # 处理目标点数据
            target_points = self._process_target_points(targets, time_step)

            # 处理检测数据
            detections = self._process_detections(radar_data, targets)

            # 处理航迹数据
            tracks = self._process_tracks(radar_data.get('tracks', []))

            # 生成扫描线数据
            sweep_line = self._generate_sweep_line(radar_data.get('current_angle', 0))

            return {
                'grid': grid_data,
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

    def process_chart_data(self, simulation_data: Dict, chart_type: str,
                           time_range: Optional[Tuple[float, float]] = None) -> Dict[str, Any]:
        """
        处理图表数据
        支持多种图表类型：performance, environment_impact, target_analysis
        """
        try:
            if chart_type == 'performance':
                return self._process_performance_chart(simulation_data, time_range)
            elif chart_type == 'environment_impact':
                return self._process_environment_chart(simulation_data, time_range)
            elif chart_type == 'target_analysis':
                return self._process_target_chart(simulation_data, time_range)
            elif chart_type == 'detection_statistics':
                return self._process_detection_stats(simulation_data, time_range)
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
        except Exception as e:
            logger.error(f"Error processing chart data for {chart_type}: {str(e)}")
            return {'error': str(e), 'data': []}

    def process_realtime_data(self, current_state: Dict) -> Dict[str, Any]:
        """
        处理实时数据更新
        返回适合WebSocket推送的轻量级数据
        """
        try:
            return {
                'targets': self._extract_target_positions(current_state.get('targets', [])),
                'detections': self._extract_detections(current_state.get('detections', [])),
                'radar_angle': current_state.get('radar_angle', 0),
                'performance_metrics': self._extract_key_metrics(current_state),
                'timestamp': current_state.get('timestamp', datetime.now().timestamp())
            }
        except Exception as e:
            logger.error(f"Error processing realtime data: {str(e)}")
            return {}

    def _generate_radar_grid(self) -> Dict[str, Any]:
        """生成雷达显示网格"""
        # 距离环
        range_rings = [10000, 20000, 50000, 100000, 200000]  # 米

        # 方位线 (每15度一条)
        azimuth_lines = list(range(0, 360, 15))

        return {
            'range_rings': range_rings,
            'azimuth_lines': azimuth_lines,
            'center': [0, 0],
            'max_range': max(range_rings)
        }

    def _process_target_points(self, targets: List[Dict], time_step: float) -> List[Dict]:
        """处理目标点数据"""
        processed_targets = []

        for i, target in enumerate(targets):
            try:
                position = target.get('position', [0, 0])
                velocity = target.get('velocity', [0, 0])

                # 计算极坐标
                range_val = np.sqrt(position[0] ** 2 + position[1] ** 2)
                azimuth = np.arctan2(position[1], position[0]) * 180 / np.pi
                if azimuth < 0:
                    azimuth += 360

                processed_targets.append({
                    'id': i,
                    'position': {
                        'x': position[0],
                        'y': position[1],
                        'range': range_val,
                        'azimuth': azimuth
                    },
                    'velocity': {
                        'x': velocity[0],
                        'y': velocity[1],
                        'speed': np.sqrt(velocity[0] ** 2 + velocity[1] ** 2)
                    },
                    'rcs': target.get('rcs', 1.0),
                    'altitude': target.get('altitude', 0),
                    'type': target.get('type', 'unknown'),
                    'formation_id': target.get('formation_id'),
                    'timestamp': time_step
                })
            except Exception as e:
                logger.warning(f"Error processing target {i}: {str(e)}")
                continue

        return processed_targets

    def _process_detections(self, radar_data: Dict, targets: List[Dict]) -> List[Dict]:
        """处理检测数据"""
        detections = []

        # 这里应该基于雷达性能和目标特性计算检测概率
        for i, target in enumerate(targets):
            position = target.get('position', [0, 0])
            range_val = np.sqrt(position[0] ** 2 + position[1] ** 2)

            # 简化的检测概率计算
            detection_prob = self._calculate_detection_probability(
                target, radar_data, range_val
            )

            if np.random.random() < detection_prob:
                # 添加测量噪声
                noise_std = self._get_measurement_noise(range_val)
                noisy_position = [
                    position[0] + np.random.normal(0, noise_std),
                    position[1] + np.random.normal(0, noise_std)
                ]

                azimuth = np.arctan2(noisy_position[1], noisy_position[0]) * 180 / np.pi
                if azimuth < 0:
                    azimuth += 360

                detections.append({
                    'target_id': i,
                    'position': {
                        'x': noisy_position[0],
                        'y': noisy_position[1],
                        'range': np.sqrt(noisy_position[0] ** 2 + noisy_position[1] ** 2),
                        'azimuth': azimuth
                    },
                    'snr': self._calculate_snr(target, radar_data, range_val),
                    'confidence': detection_prob,
                    'measurement_noise': noise_std
                })

        return detections

    def _process_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """处理航迹数据"""
        processed_tracks = []

        for track in tracks:
            try:
                history = track.get('history', [])
                if len(history) < 2:
                    continue

                # 计算航迹点的极坐标
                track_points = []
                for point in history[-10:]:  # 只保留最近10个点
                    pos = point.get('position', [0, 0])
                    range_val = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
                    azimuth = np.arctan2(pos[1], pos[0]) * 180 / np.pi
                    if azimuth < 0:
                        azimuth += 360

                    track_points.append({
                        'x': pos[0],
                        'y': pos[1],
                        'range': range_val,
                        'azimuth': azimuth,
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

    def _generate_sweep_line(self, current_angle: float) -> Dict[str, Any]:
        """生成雷达扫描线"""
        return {
            'angle': current_angle,
            'start_point': [0, 0],
            'end_point': [
                200000 * np.cos(np.radians(current_angle)),
                200000 * np.sin(np.radians(current_angle))
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
        time_points = data.get('time_points', [])
        detection_rates = data.get('detection_rates', [])
        false_alarm_rates = data.get('false_alarm_rates', [])

        # 应用时间范围过滤
        if time_range:
            start_time, end_time = time_range
            filtered_data = [(t, dr, far) for t, dr, far in zip(time_points, detection_rates, false_alarm_rates)
                             if start_time <= t <= end_time]
            if filtered_data:
                time_points, detection_rates, false_alarm_rates = zip(*filtered_data)

        return {
            'labels': list(time_points),
            'datasets': [
                {
                    'label': 'Detection Rate',
                    'data': list(detection_rates),
                    'borderColor': '#4CAF50',
                    'backgroundColor': 'rgba(76, 175, 80, 0.1)'
                },
                {
                    'label': 'False Alarm Rate',
                    'data': list(false_alarm_rates),
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

    def _calculate_detection_probability(self, target: Dict, radar_data: Dict, range_val: float) -> float:
        """计算检测概率（简化版）"""
        rcs = target.get('rcs', 1.0)
        power = radar_data.get('power', 1e6)
        frequency = radar_data.get('frequency', 3e9)

        # 简化的雷达方程
        # 实际应用中需要考虑更多因素
        snr = (power * rcs) / (range_val ** 4)
        detection_prob = 1 / (1 + np.exp(-snr / 1000))  # Sigmoid函数

        return min(detection_prob, 0.95)  # 最大95%检测概率

    def _calculate_snr(self, target: Dict, radar_data: Dict, range_val: float) -> float:
        """计算信噪比"""
        rcs = target.get('rcs', 1.0)
        power = radar_data.get('power', 1e6)

        # 简化的SNR计算
        snr_db = 10 * np.log10((power * rcs) / (range_val ** 4) / 1e-12)
        return max(snr_db, -20)  # 最小-20dB

    def _get_measurement_noise(self, range_val: float) -> float:
        """获取测量噪声标准差"""
        # 噪声随距离增加
        base_noise = 10  # 基础噪声10米
        range_factor = range_val / 100000  # 100km标准化
        return base_noise * (1 + range_factor)

    def _extract_target_positions(self, targets: List[Dict]) -> List[Dict]:
        """提取目标位置（轻量级）"""
        return [{
            'id': i,
            'x': target.get('position', [0, 0])[0],
            'y': target.get('position', [0, 0])[1],
            'type': target.get('type', 'unknown')
        } for i, target in enumerate(targets)]

    def _extract_detections(self, detections: List[Dict]) -> List[Dict]:
        """提取检测数据（轻量级）"""
        return [{
            'id': det.get('target_id'),
            'x': det.get('position', {}).get('x', 0),
            'y': det.get('position', {}).get('y', 0),
            'confidence': det.get('confidence', 0.5)
        } for det in detections]

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

    def __init__(self):
        self.data_processor = None  # 延迟初始化

    def _ensure_data_processor(self, radar_system=None, environment=None):
        """确保数据处理器已初始化"""
        if self.data_processor is None:
            if radar_system is None or environment is None:
                from models.radar_system import RadarSystem
                from models.environment import Environment

                radar_system = radar_system or RadarSystem()
                environment = environment or Environment()

            self.data_processor = DataProcessor(radar_system, environment)

        return self.data_processor

    # 然后在需要使用 data_processor 的方法中调用 _ensure_data_processor()
