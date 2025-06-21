import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from models.tracking import Track, Detection, SVMClutterFilter, TOMHTNode
from models.radar_system import RadarSystem
from models.environment import Environment
from sklearn.cluster import DBSCAN
import logging
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """数据处理性能指标"""
    detection_count: int = 0
    track_count: int = 0
    processing_time: float = 0.0
    cluster_count: int = 0
    false_alarm_rate: float = 0.0
    track_accuracy: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0


class DataProcessor:
    """增强的数据处理器 - 集成目标跟踪和可视化数据处理"""

    def __init__(self, radar_system: RadarSystem, environment: Environment):
        self.radar_system = radar_system
        self.environment = environment
        self.tracks = []
        self.svm_filter = SVMClutterFilter()
        self.track_id_counter = 1

        # 新增：性能监控和数据缓存
        self.processing_metrics = ProcessingMetrics()
        self.detection_history = deque(maxlen=1000)
        self.track_history = deque(maxlen=1000)
        self.metrics_history = deque(maxlen=100)

        # 配置参数
        self.max_tracks = 500
        self.gate_threshold = 1000.0
        self.track_max_age = 10
        self.track_confirmation_threshold = 3

        logger.info("DataProcessor initialized with enhanced capabilities")

    def process_tracking_data(self, detections: List[Detection], timestamp: float = None) -> Dict[str, Any]:
        """
        增强的跟踪数据处理 - 返回完整的处理结果
        """
        start_time = time.time()

        if timestamp is None:
            timestamp = time.time()

        try:
            # 原有的处理逻辑
            clusters = self._cluster_detections(detections)
            filtered_detections = self._tomht_svm_tracking(clusters)

            # 跟踪更新
            for track in self.tracks:
                track.predict(0.06)
                if self.environment.weather.weather_type in ["heavy_rain", "snow"]:
                    self._apply_imm_tracking([track])

            # 低空目标增强
            self._enhance_low_altitude_targets()

            # 更新历史记录
            self._update_history(detections, timestamp)

            # 计算性能指标
            processing_time = time.time() - start_time
            self._update_metrics(detections, processing_time)

            # 返回完整的处理结果
            return {
                'timestamp': timestamp,
                'detections': self._format_detections(filtered_detections),
                'tracks': self._format_tracks(),
                'clusters': self._format_clusters(clusters),
                'metrics': self._get_current_metrics(),
                'processing_time': processing_time,
                'status': 'success'
            }

        except Exception as e:
            logger.error(f"Error in tracking data processing: {str(e)}")
            return {
                'timestamp': timestamp,
                'error': str(e),
                'status': 'error'
            }

    def get_visualization_data(self, data_type: str = 'all') -> Dict[str, Any]:
        """
        获取可视化数据 - 为前端提供格式化的数据
        """
        try:
            if data_type == 'radar_display':
                return self._get_radar_display_data()
            elif data_type == 'performance':
                return self._get_performance_data()
            elif data_type == 'tracks':
                return self._get_tracks_data()
            elif data_type == 'detections':
                return self._get_detections_data()
            elif data_type == 'all':
                return {
                    'radar_display': self._get_radar_display_data(),
                    'performance': self._get_performance_data(),
                    'tracks': self._get_tracks_data(),
                    'detections': self._get_detections_data()
                }
            else:
                raise ValueError(f"Unknown data type: {data_type}")

        except Exception as e:
            logger.error(f"Error getting visualization data: {str(e)}")
            return {'error': str(e)}

    def _cluster_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        """聚类检测点"""
        if len(detections) < 2:
            return [[det] for det in detections]

        positions = np.array([
            [det.range * np.cos(det.azimuth), det.range * np.sin(det.azimuth)]
            for det in detections
        ])

        # 动态调整聚类参数
        eps = self._calculate_dynamic_eps(detections)
        clustering = DBSCAN(eps=eps, min_samples=1).fit(positions)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(detections[i])

        return list(clusters.values())

    def _calculate_dynamic_eps(self, detections: List[Detection]) -> float:
        """动态计算聚类参数"""
        if len(detections) < 5:
            return 100.0

        ranges = [det.range for det in detections]
        avg_range = np.mean(ranges)

        # 根据平均距离调整聚类阈值
        if avg_range < 5000:
            return 50.0
        elif avg_range < 20000:
            return 100.0
        else:
            return 200.0

    def _tomht_svm_tracking(self, clusters: List[List[Detection]]) -> List[Detection]:
        """TOMHT-SVM跟踪算法"""
        all_detections = [det for cluster in clusters for det in cluster]

        # SVM训练和预测
        if len(all_detections) > 10:
            labels = []
            for detection in all_detections:
                is_target = self._is_detection_associated_with_track(detection)
                labels.append(1 if is_target else 0)

            if len(set(labels)) > 1:
                self.svm_filter.train(all_detections, labels)

        # 过滤检测
        filtered_detections = []
        for detection in all_detections:
            prediction = self.svm_filter.predict(detection)
            if prediction == 1:
                filtered_detections.append(detection)

        # 更新跟踪
        self._update_tracks(filtered_detections)

        return filtered_detections

    def _is_detection_associated_with_track(self, detection: Detection) -> bool:
        """检查检测是否与现有航迹关联"""
        det_x = detection.range * np.cos(detection.azimuth)
        det_y = detection.range * np.sin(detection.azimuth)

        for track in self.tracks:
            pred_x, pred_y = track.state[0], track.state[1]
            distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)

            if distance < self.gate_threshold:
                return True

        return False

    def _update_tracks(self, detections: List[Detection]):
        """更新航迹"""
        if not detections:
            return

        association_matrix = self._build_association_matrix(detections)
        used_detections = set()

        # 更新现有航迹
        for i, track in enumerate(self.tracks):
            best_det_idx = self._find_best_detection_for_track(
                i, detections, association_matrix, used_detections
            )

            if best_det_idx >= 0:
                track.update(detections[best_det_idx])
                used_detections.add(best_det_idx)
                if track.age >= self.track_confirmation_threshold:
                    track.confirmed = True

        # 创建新航迹
        self._create_new_tracks(detections, used_detections)

        # 清理航迹
        self._cleanup_tracks()

    def _find_best_detection_for_track(self, track_idx: int, detections: List[Detection],
                                       association_matrix: np.ndarray,
                                       used_detections: set) -> int:
        """为航迹寻找最佳检测"""
        best_det_idx = -1
        min_distance = float('inf')
        track = self.tracks[track_idx]

        for j, detection in enumerate(detections):
            if j in used_detections or association_matrix[track_idx, j] == 0:
                continue

            det_x = detection.range * np.cos(detection.azimuth)
            det_y = detection.range * np.sin(detection.azimuth)
            pred_x, pred_y = track.state[0], track.state[1]
            distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)

            if distance < min_distance:
                min_distance = distance
                best_det_idx = j

        return best_det_idx

    def _create_new_tracks(self, detections: List[Detection], used_detections: set):
        """创建新航迹"""
        for j, detection in enumerate(detections):
            if j not in used_detections and len(self.tracks) < self.max_tracks:
                new_track = Track(
                    track_id=self.track_id_counter,
                    state=np.array([
                        detection.range * np.cos(detection.azimuth),
                        detection.range * np.sin(detection.azimuth),
                        detection.range * np.sin(detection.elevation),
                        detection.velocity * np.cos(detection.azimuth),
                        detection.velocity * np.sin(detection.azimuth),
                        0.0
                    ])
                )
                new_track.update(detection)
                self.tracks.append(new_track)
                self.track_id_counter += 1

    def _cleanup_tracks(self):
        """清理过期航迹"""
        self.tracks = [
            track for track in self.tracks
            if track.age < self.track_max_age or track.confirmed
        ]

    def _build_association_matrix(self, detections: List[Detection]) -> np.ndarray:
        """构建关联矩阵"""
        n_tracks = len(self.tracks)
        n_detections = len(detections)

        if n_tracks == 0 or n_detections == 0:
            return np.zeros((max(1, n_tracks), max(1, n_detections)))

        matrix = np.zeros((n_tracks, n_detections))

        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                if self._is_within_gate(track, detection):
                    matrix[i, j] = 1

        return matrix

    def _is_within_gate(self, track: Track, detection: Detection) -> bool:
        """检查检测是否在航迹门限内"""
        det_x = detection.range * np.cos(detection.azimuth)
        det_y = detection.range * np.sin(detection.azimuth)
        pred_x, pred_y = track.state[0], track.state[1]

        distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)
        return distance < self.gate_threshold

    def _apply_imm_tracking(self, tracks: List[Track]):
        """应用IMM跟踪"""
        for track in tracks:
            velocity = np.sqrt(track.state[3] ** 2 + track.state[4] ** 2)

            if velocity > 200.0:
                track.high_mobility = True
                track.covariance *= 2.0
            else:
                track.high_mobility = False

    def _enhance_low_altitude_targets(self):
        """增强低空目标处理"""
        for track in self.tracks:
            altitude = track.state[2]

            if altitude < 500.0:
                self._apply_stap_processing(track)
                self._apply_polarimetric_processing(track)

                if self.environment.terrain_type in ["hills", "urban"]:
                    self._fuse_with_ir_sensor(track)

    def _apply_stap_processing(self, track: Track):
        """应用STAP处理"""
        improvement_factor = 1.5 if track.state[2] < 200 else 1.2
        track.score *= improvement_factor

    def _apply_polarimetric_processing(self, track: Track):
        """应用极化处理"""
        if len(track.rcs_history) > 0:
            rcs_variation = np.std(track.rcs_history)
            if rcs_variation < 2.0:
                track.score *= 1.2
                track.confirmed = True

    def _fuse_with_ir_sensor(self, track: Track):
        """与红外传感器融合"""
        ir_confidence = 0.8
        current_range = np.sqrt(track.state[0] ** 2 + track.state[1] ** 2)

        noise_factor = np.random.normal(0, 10)
        ir_range = current_range + noise_factor

        fusion_weight = 0.6
        fused_range = fusion_weight * current_range + (1 - fusion_weight) * ir_range

        range_ratio = fused_range / current_range
        track.state[0] *= range_ratio
        track.state[1] *= range_ratio

        track.score *= (1.0 + ir_confidence * 0.5)
        track.confirmed = True

    def _update_history(self, detections: List[Detection], timestamp: float):
        """更新历史记录"""
        self.detection_history.append({
            'timestamp': timestamp,
            'count': len(detections),
            'detections': detections
        })

        self.track_history.append({
            'timestamp': timestamp,
            'count': len(self.tracks),
            'tracks': self.tracks.copy()
        })

    def _update_metrics(self, detections: List[Detection], processing_time: float):
        """更新性能指标"""
        self.processing_metrics.detection_count = len(detections)
        self.processing_metrics.track_count = len(self.tracks)
        self.processing_metrics.processing_time = processing_time

        # 计算虚警率
        confirmed_tracks = [t for t in self.tracks if t.confirmed]
        self.processing_metrics.false_alarm_rate = 1.0 - (
                len(confirmed_tracks) / max(len(self.tracks), 1)
        )

        # 计算跟踪精度
        self.processing_metrics.track_accuracy = self._calculate_track_accuracy()

        # 添加到历史记录
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': self.processing_metrics
        })

    def _calculate_track_accuracy(self) -> float:
        """计算跟踪精度"""
        if not self.tracks:
            return 0.0

        accurate_tracks = sum(1 for track in self.tracks if track.score > 0.8)
        return accurate_tracks / len(self.tracks)

    def _format_detections(self, detections: List[Detection]) -> List[Dict]:
        """格式化检测数据"""
        return [{
            'id': i,
            'range': det.range,
            'azimuth': det.azimuth,
            'elevation': det.elevation,
            'velocity': det.velocity,
            'snr': det.snr,
            'rcs': det.rcs,
            'position': {
                'x': det.range * np.cos(det.azimuth),
                'y': det.range * np.sin(det.azimuth),
                'z': det.range * np.sin(det.elevation)
            }
        } for i, det in enumerate(detections)]

    def _format_tracks(self) -> List[Dict]:
        """格式化航迹数据"""
        return [{
            'id': track.track_id,
            'position': {
                'x': track.state[0],
                'y': track.state[1],
                'z': track.state[2]
            },
            'velocity': {
                'x': track.state[3],
                'y': track.state[4],
                'z': track.state[5]
            },
            'confirmed': track.confirmed,
            'age': track.age,
            'score': track.score,
            'range': np.sqrt(track.state[0] ** 2 + track.state[1] ** 2),
            'azimuth': np.arctan2(track.state[1], track.state[0]),
            'high_mobility': getattr(track, 'high_mobility', False)
        } for track in self.tracks]

    def _format_clusters(self, clusters: List[List[Detection]]) -> List[Dict]:
        """格式化聚类数据"""
        return [{
            'id': i,
            'size': len(cluster),
            'center': self._calculate_cluster_center(cluster),
            'detections': len(cluster)
        } for i, cluster in enumerate(clusters)]

    def _calculate_cluster_center(self, cluster: List[Detection]) -> Dict:
        """计算聚类中心"""
        if not cluster:
            return {'x': 0, 'y': 0, 'z': 0}

        x_coords = [det.range * np.cos(det.azimuth) for det in cluster]
        y_coords = [det.range * np.sin(det.azimuth) for det in cluster]
        z_coords = [det.range * np.sin(det.elevation) for det in cluster]

        return {
            'x': np.mean(x_coords),
            'y': np.mean(y_coords),
            'z': np.mean(z_coords)
        }

    def _get_radar_display_data(self) -> Dict[str, Any]:
        """获取雷达显示数据"""
        return {
            'scan_angle': self.radar_system.antenna.current_angle,
            'max_range': self.radar_system.max_range,
            'targets': self._format_tracks(),
            'detections': self._format_detections([]),  # 当前帧检测
            'clutter_level': self.environment.clutter_density,
            'weather_impact': self._get_weather_impact()
        }

    def _get_performance_data(self) -> Dict[str, Any]:
        """获取性能数据"""
        return {
            'current_metrics': {
                'detection_count': self.processing_metrics.detection_count,
                'track_count': self.processing_metrics.track_count,
                'processing_time': self.processing_metrics.processing_time,
                'false_alarm_rate': self.processing_metrics.false_alarm_rate,
                'track_accuracy': self.processing_metrics.track_accuracy
            },
            'history': list(self.metrics_history)
        }

    def _get_tracks_data(self) -> Dict[str, Any]:
        """获取航迹数据"""
        return {
            'active_tracks': self._format_tracks(),
            'confirmed_tracks': [t for t in self._format_tracks() if t['confirmed']],
            'track_statistics': {
                'total': len(self.tracks),
                'confirmed': len([t for t in self.tracks if t.confirmed]),
                'high_mobility': len([t for t in self.tracks if getattr(t, 'high_mobility', False)])
            }
        }

    def _get_detections_data(self) -> Dict[str, Any]:
        """获取检测数据"""
        recent_detections = list(self.detection_history)[-10:] if self.detection_history else []

        return {
            'recent_detections': recent_detections,
            'detection_rate': self._calculate_detection_rate(),
            'average_snr': self._calculate_average_snr()
        }

    def _get_weather_impact(self) -> float:
        """获取天气影响"""
        weather_factors = {
            'clear': 1.0,
            'light_rain': 0.9,
            'heavy_rain': 0.7,
            'snow': 0.6,
            'fog': 0.8
        }
        return weather_factors.get(self.environment.weather.weather_type, 1.0)

    def _calculate_detection_rate(self) -> float:
        """计算检测率"""
        if len(self.detection_history) < 2:
            return 0.0

        recent_detections = list(self.detection_history)[-10:]
        total_detections = sum(entry['count'] for entry in recent_detections)
        return total_detections / len(recent_detections)

    def _calculate_average_snr(self) -> float:
        """计算平均信噪比"""
        if not self.detection_history:
            return 0.0

        recent_entry = self.detection_history[-1]
        detections = recent_entry.get('detections', [])

        if not detections:
            return 0.0

        snr_values = [det.snr for det in detections if hasattr(det, 'snr')]
        return np.mean(snr_values) if snr_values else 0.0

    def _get_current_metrics(self) -> Dict[str, Any]:
        """获取当前指标"""
        return {
            'detection_count': self.processing_metrics.detection_count,
            'track_count': self.processing_metrics.track_count,
            'processing_time': self.processing_metrics.processing_time,
            'cluster_count': self.processing_metrics.cluster_count,
            'false_alarm_rate': self.processing_metrics.false_alarm_rate,
            'track_accuracy': self.processing_metrics.track_accuracy
        }

    def get_historical_data(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """获取历史数据"""
        filtered_detections = [
            entry for entry in self.detection_history
            if start_time <= entry['timestamp'] <= end_time
        ]

        filtered_tracks = [
            entry for entry in self.track_history
            if start_time <= entry['timestamp'] <= end_time
        ]

        filtered_metrics = [
            entry for entry in self.metrics_history
            if start_time <= entry['timestamp'] <= end_time
        ]

        return {
            'detections': filtered_detections,
            'tracks': filtered_tracks,
            'metrics': filtered_metrics,
            'time_range': [start_time, end_time]
        }

    def reset(self):
        """重置处理器状态"""
        self.tracks.clear()
        self.track_id_counter = 1
        self.detection_history.clear()
        self.track_history.clear()
        self.metrics_history.clear()
        self.processing_metrics = ProcessingMetrics()
        logger.info("DataProcessor reset completed")
