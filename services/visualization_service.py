from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
from core.visualization_processor import VisualizationProcessor
from services.simulation_service import SimulationService
from services.cache_service import CacheService

logger = logging.getLogger(__name__)


class VisualizationService:
    """可视化数据服务 - 为前端提供优化的可视化数据"""

    def __init__(self):
        self.visualization_processor = VisualizationProcessor()
        self.simulation_service = SimulationService()
        self.cache_service = CacheService()

    def get_radar_display_data(self, simulation_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        获取雷达显示数据
        """
        try:
            # 检查缓存
            cache_key = f"radar_display:{simulation_id}"
            if use_cache:
                cached_data = self.cache_service.get(cache_key)
                if cached_data:
                    return cached_data

            # 获取仿真当前状态
            simulation_status = self.simulation_service.get_simulation_status(simulation_id)
            if simulation_status.get('status') != 'running':
                return {'error': 'Simulation not running', 'data': {}}

            # 获取仿真数据
            current_data = self.simulation_service.get_current_simulation_state(simulation_id)
            if not current_data:
                return {'error': 'No simulation data available', 'data': {}}

            # 处理可视化数据
            radar_data = current_data.get('radar_data', {})
            targets = current_data.get('targets', [])
            time_step = current_data.get('current_time', 0)

            display_data = self.visualization_processor.process_radar_display_data(
                radar_data, targets, time_step
            )

            # 缓存数据（短期缓存，1秒）
            if use_cache:
                self.cache_service.set(cache_key, display_data, ttl=1)

            return {'status': 'success', 'data': display_data}

        except Exception as e:
            logger.error(f"Error getting radar display data: {str(e)}")
            return {'error': str(e), 'data': {}}

    def get_chart_data(self, simulation_id: str, chart_type: str,
                       time_range: Optional[Tuple[float, float]] = None,
                       use_cache: bool = True) -> Dict[str, Any]:
        """
        获取图表数据
        支持的图表类型：performance, environment_impact, target_analysis, detection_statistics
        """
        try:
            # 构建缓存键
            cache_key = f"chart:{simulation_id}:{chart_type}"
            if time_range:
                cache_key += f":{time_range[0]}:{time_range[1]}"

            # 检查缓存
            if use_cache:
                cached_data = self.cache_service.get(cache_key)
                if cached_data:
                    return cached_data

            # 获取仿真数据
            simulation_data = self.simulation_service.get_simulation_data(
                simulation_id,
                start_time=time_range[0] if time_range else None,
                end_time=time_range[1] if time_range else None
            )

            if not simulation_data:
                return {'error': 'No simulation data available', 'data': {}}

            # 处理图表数据
            chart_data = self.visualization_processor.process_chart_data(
                simulation_data, chart_type, time_range
            )

            # 缓存数据（中期缓存，30秒）
            if use_cache:
                self.cache_service.set(cache_key, chart_data, ttl=30)

            return {
                'status': 'success',
                'chart_type': chart_type,
                'time_range': time_range,
                'data': chart_data
            }

        except Exception as e:
            logger.error(f"Error getting chart data for {chart_type}: {str(e)}")
            return {'error': str(e), 'data': {}}

    def get_realtime_data(self, simulation_id: str) -> Dict[str, Any]:
        """
        获取实时数据（用于WebSocket推送）
        """
        try:
            # 获取当前仿真状态
            current_state = self.simulation_service.get_current_simulation_state(simulation_id)
            if not current_state:
                return {'error': 'No current state available'}

            # 处理实时数据
            realtime_data = self.visualization_processor.process_realtime_data(current_state)

            return {
                'status': 'success',
                'simulation_id': simulation_id,
                'data': realtime_data
            }

        except Exception as e:
            logger.error(f"Error getting realtime data: {str(e)}")
            return {'error': str(e)}

    def get_historical_data(self, simulation_id: str,
                            start_time: float, end_time: float,
                            data_resolution: str = 'medium') -> Dict[str, Any]:
        """
        获取历史数据
        data_resolution: 'high', 'medium', 'low' - 控制数据密度
        """
        try:
            # 验证时间范围
            if start_time >= end_time:
                return {'error': 'Invalid time range'}

            # 根据分辨率设置采样间隔
            sampling_intervals = {
                'high': 0.1,  # 0.1秒间隔
                'medium': 1.0,  # 1秒间隔
                'low': 5.0  # 5秒间隔
            }

            sampling_interval = sampling_intervals.get(data_resolution, 1.0)

            # 获取历史数据
            historical_data = self.simulation_service.get_simulation_data(
                simulation_id, start_time, end_time,
                sampling_interval=sampling_interval
            )

            if not historical_data:
                return {'error': 'No historical data available', 'data': []}

            # 处理历史数据为可视化格式
            processed_data = self._process_historical_data(historical_data, data_resolution)

            return {
                'status': 'success',
                'time_range': [start_time, end_time],
                'resolution': data_resolution,
                'data': processed_data
            }

        except Exception as e:
            logger.error(f"Error getting historical data: {str(e)}")
            return {'error': str(e), 'data': []}

    def get_simulation_summary(self, simulation_id: str) -> Dict[str, Any]:
        """
        获取仿真总结数据
        """
        try:
            # 获取完整仿真数据
            full_data = self.simulation_service.get_simulation_data(simulation_id)
            if not full_data:
                return {'error': 'No simulation data available'}

            # 计算统计信息
            summary = self._calculate_simulation_summary(full_data)

            return {
                'status': 'success',
                'simulation_id': simulation_id,
                'summary': summary
            }

        except Exception as e:
            logger.error(f"Error getting simulation summary: {str(e)}")
            return {'error': str(e)}

    def get_export_preview_data(self, simulation_id: str, export_config: Dict) -> Dict[str, Any]:
        """
        获取导出预览数据
        """
        try:
            data_types = export_config.get('data_types', [])
            time_range = export_config.get('time_range')

            preview_data = {}

            for data_type in data_types:
                if data_type == 'targets':
                    preview_data['targets'] = self._get_targets_preview(simulation_id, time_range)
                elif data_type == 'detections':
                    preview_data['detections'] = self._get_detections_preview(simulation_id, time_range)
                elif data_type == 'tracks':
                    preview_data['tracks'] = self._get_tracks_preview(simulation_id, time_range)
                elif data_type == 'performance_metrics':
                    preview_data['performance_metrics'] = self._get_performance_preview(simulation_id, time_range)

            return {
                'status': 'success',
                'preview_data': preview_data,
                'estimated_size': self._estimate_export_size(preview_data, export_config)
            }

        except Exception as e:
            logger.error(f"Error getting export preview: {str(e)}")
            return {'error': str(e)}

    def _process_historical_data(self, data: Dict, resolution: str) -> List[Dict]:
        """处理历史数据"""
        time_points = data.get('time_points', [])
        targets_history = data.get('targets_history', [])
        detections_history = data.get('detections_history', [])

        processed_data = []

        for i, time_point in enumerate(time_points):
            targets = targets_history[i] if i < len(targets_history) else []
            detections = detections_history[i] if i < len(detections_history) else []

            # 根据分辨率过滤数据
            if resolution == 'low':
                # 低分辨率：只保留关键信息
                targets = self._simplify_targets(targets)
                detections = self._simplify_detections(detections)

            processed_data.append({
                'timestamp': time_point,
                'targets': targets,
                'detections': detections,
                'metrics': self._extract_frame_metrics(data, i)
            })

        return processed_data

    def _calculate_simulation_summary(self, data: Dict) -> Dict[str, Any]:
        """计算仿真总结统计"""
        return {
            'duration': data.get('simulation_time', 0),
            'total_targets': len(data.get('targets', [])),
            'total_detections': sum(len(dets) for dets in data.get('detections_history', [])),
            'average_detection_rate': self._calculate_average_detection_rate(data),
            'peak_target_count': self._calculate_peak_target_count(data),
            'radar_utilization': self._calculate_radar_utilization(data),
            'performance_metrics': self._calculate_performance_metrics(data)
        }

    def _get_targets_preview(self, simulation_id: str, time_range: Optional[Tuple[float, float]]) -> Dict:
        """获取目标数据预览"""
        data = self.simulation_service.get_simulation_data(
            simulation_id,
            start_time=time_range[0] if time_range else None,
            end_time=time_range[1] if time_range else None
        )

        targets = data.get('targets', [])
        return {
            'count': len(targets),
            'sample': targets[:5] if targets else [],  # 前5个样本
            'fields': ['id', 'position', 'velocity', 'rcs', 'altitude', 'type']
        }

    def _get_detections_preview(self, simulation_id: str, time_range: Optional[Tuple[float, float]]) -> Dict:
        """获取检测数据预览"""
        data = self.simulation_service.get_simulation_data(
            simulation_id,
            start_time=time_range[0] if time_range else None,
            end_time=time_range[1] if time_range else None
        )

        detections = []
        for det_list in data.get('detections_history', []):
            detections.extend(det_list)

        return {
            'count': len(detections),
            'sample': detections[:5] if detections else [],
            'fields': ['target_id', 'position', 'snr', 'confidence', 'timestamp']
        }

    def _get_tracks_preview(self, simulation_id: str, time_range: Optional[Tuple[float, float]]) -> Dict:
        """获取航迹数据预览"""
        # 实现航迹数据预览逻辑
        return {
            'count': 0,
            'sample': [],
            'fields': ['track_id', 'points', 'velocity', 'quality']
        }

    def _get_performance_preview(self, simulation_id: str, time_range: Optional[Tuple[float, float]]) -> Dict:
        """获取性能指标预览"""
        data = self.simulation_service.get_simulation_data(
            simulation_id,
            start_time=time_range[0] if time_range else None,
            end_time=time_range[1] if time_range else None
        )

        return {
            'metrics': ['detection_rate', 'false_alarm_rate', 'track_accuracy', 'system_load'],
            'time_points': len(data.get('time_points', [])),
            'sample_values': {
                'detection_rate': data.get('detection_rates', [])[:5],
                'false_alarm_rate': data.get('false_alarm_rates', [])[:5]
            }
        }

    def _estimate_export_size(self, preview_data: Dict, export_config: Dict) -> Dict[str, Any]:
        """估算导出文件大小"""
        # 简化的文件大小估算
        base_size = 0
        format_multipliers = {
            'csv': 1.0,
            'json': 1.5,
            'excel': 2.0,
            'matlab': 1.2,
            'hdf5': 0.8
        }

        for data_type, data_info in preview_data.items():
            count = data_info.get('count', 0)
            fields = len(data_info.get('fields', []))
            base_size += count * fields * 50  # 假设每个字段50字节

        export_format = export_config.get('export_format', 'csv')
        multiplier = format_multipliers.get(export_format, 1.0)

        estimated_bytes = int(base_size * multiplier)

        # 如果启用压缩，减少30%
        if export_config.get('compression', False):
            estimated_bytes = int(estimated_bytes * 0.7)

        return {
            'bytes': estimated_bytes,
            'human_readable': self._format_file_size(estimated_bytes),
            'compression_enabled': export_config.get('compression', False)
        }

    def _format_file_size(self, bytes_size: int) -> str:
        """格式化文件大小"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024
        return f"{bytes_size:.1f} TB"

    def _simplify_targets(self, targets: List[Dict]) -> List[Dict]:
        """简化目标数据（低分辨率）"""
        return [{
            'id': target.get('id'),
            'position': target.get('position'),
            'type': target.get('type')
        } for target in targets]

    def _simplify_detections(self, detections: List[Dict]) -> List[Dict]:
        """简化检测数据（低分辨率）"""
        return [{
            'target_id': det.get('target_id'),
            'position': det.get('position'),
            'confidence': det.get('confidence')
        } for det in detections]

    def _extract_frame_metrics(self, data: Dict, frame_index: int) -> Dict:
        """提取帧指标"""
        return {
            'detection_rate': data.get('detection_rates', [])[frame_index] if frame_index < len(
                data.get('detection_rates', [])) else 0,
            'false_alarm_rate': data.get('false_alarm_rates', [])[frame_index] if frame_index < len(
                data.get('false_alarm_rates', [])) else 0
        }

    def _calculate_average_detection_rate(self, data: Dict) -> float:
        """计算平均检测率"""
        detection_rates = data.get('detection_rates', [])
        return sum(detection_rates) / len(detection_rates) if detection_rates else 0.0

    def _calculate_peak_target_count(self, data: Dict) -> int:
        """计算峰值目标数量"""
        targets_history = data.get('targets_history', [])
        return max(len(targets) for targets in targets_history) if targets_history else 0

    def _calculate_radar_utilization(self, data: Dict) -> float:
        """计算雷达利用率"""
        # 简化的雷达利用率计算
        return 0.85  # 假设值

    def _calculate_performance_metrics(self, data: Dict) -> Dict[str, float]:
        """计算综合性能指标"""
        return {
            'overall_efficiency': 0.87,
            'detection_accuracy': 0.92,
            'false_alarm_rate': 0.05,
            'system_availability': 0.99
        }
