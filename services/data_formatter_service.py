from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from dataclasses import asdict
from models.tracking import Track, Detection
from models.radar_system import RadarSystem
from models.environment import Environment

logger = logging.getLogger(__name__)


class DataFormatterService:
    """数据格式化服务 - 将内部数据格式转换为前端友好的格式"""

    def __init__(self):
        self.supported_formats = ['json', 'csv', 'xml', 'binary']
        logger.info("DataFormatterService initialized")

    def format_for_frontend(self, data: Dict[str, Any], data_type: str = 'general') -> Dict[str, Any]:
        """将数据格式化为前端友好的格式"""
        try:
            if data_type == 'radar_display':
                return self._format_radar_display_data(data)
            elif data_type == 'chart':
                return self._format_chart_data(data)
            elif data_type == 'table':
                return self._format_table_data(data)
            elif data_type == 'export':
                return self._format_export_data(data)
            else:
                return self._format_general_data(data)
        except Exception as e:
            logger.error(f"Error formatting data for frontend: {str(e)}")
            return {'error': str(e), 'data': {}}

    def _format_radar_display_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化雷达显示数据"""
        return {
            'scan_data': {
                'current_angle': data.get('scan_angle', 0),
                'max_range': data.get('max_range', 50000),
                'sweep_time': data.get('sweep_time', 2.0),
                'resolution': data.get('resolution', 100)
            },
            'targets': self._format_targets_for_radar(data.get('targets', [])),
            'detections': self._format_detections_for_radar(data.get('detections', [])),
            'environment': {
                'clutter_level': data.get('clutter_level', 0.1),
                'weather_impact': data.get('weather_impact', 1.0),
                'noise_floor': data.get('noise_floor', -120)
            },
            'display_settings': {
                'range_rings': self._generate_range_rings(data.get('max_range', 50000)),
                'azimuth_lines': self._generate_azimuth_lines(),
                'color_scheme': 'default'
            }
        }

    def _format_targets_for_radar(self, targets: List[Dict]) -> List[Dict]:
        """格式化雷达显示的目标数据"""
        formatted_targets = []

        for target in targets:
            formatted_target = {
                'id': target.get('id'),
                'polar': {
                    'range': target.get('range', 0),
                    'azimuth': np.degrees(target.get('azimuth', 0)),
                    'elevation': np.degrees(target.get('elevation', 0))
                },
                'cartesian': target.get('position', {'x': 0, 'y': 0, 'z': 0}),
                'velocity': target.get('velocity', {'x': 0, 'y': 0, 'z': 0}),
                'properties': {
                    'confirmed': target.get('confirmed', False),
                    'age': target.get('age', 0),
                    'score': target.get('score', 0.0),
                    'high_mobility': target.get('high_mobility', False),
                    'rcs': target.get('rcs', 1.0)
                },
                'display': {
                    'color': self._get_target_color(target),
                    'size': self._get_target_size(target),
                    'symbol': self._get_target_symbol(target),
                    'trail_length': 10 if target.get('confirmed') else 3
                }
            }
            formatted_targets.append(formatted_target)

        return formatted_targets

    def _format_detections_for_radar(self, detections: List[Dict]) -> List[Dict]:
        """格式化雷达显示的检测数据"""
        formatted_detections = []

        for detection in detections:
            formatted_detection = {
                'id': detection.get('id'),
                'polar': {
                    'range': detection.get('range', 0),
                    'azimuth': np.degrees(detection.get('azimuth', 0)),
                    'elevation': np.degrees(detection.get('elevation', 0))
                },
                'cartesian': detection.get('position', {'x': 0, 'y': 0, 'z': 0}),
                'properties': {
                    'snr': detection.get('snr', 0.0),
                    'velocity': detection.get('velocity', 0.0),
                    'rcs': detection.get('rcs', 1.0)
                },
                'display': {
                    'color': self._get_detection_color(detection),
                    'size': 2,
                    'symbol': 'dot',
                    'fade_time': 1.0
                }
            }
            formatted_detections.append(formatted_detection)

        return formatted_detections

    def _format_chart_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化图表数据"""
        chart_type = data.get('chart_type', 'line')

        if chart_type == 'performance':
            return self._format_performance_chart(data)
        elif chart_type == 'detection_statistics':
            return self._format_detection_statistics_chart(data)
        elif chart_type == 'environment_impact':
            return self._format_environment_impact_chart(data)
        elif chart_type == 'track_distribution':
            return self._format_track_distribution_chart(data)
        else:
            return self._format_generic_chart(data)

    def _format_performance_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化性能图表数据"""
        metrics = data.get('metrics', [])

        return {
            'chart_type': 'line',
            'title': 'System Performance',
            'datasets': [
                {
                    'label': 'Processing Time (ms)',
                    'data': [m.get('processing_time', 0) * 1000 for m in metrics],
                    'color': '#3498db',
                    'yAxis': 'left'
                },
                {
                    'label': 'Track Count',
                    'data': [m.get('track_count', 0) for m in metrics],
                    'color': '#e74c3c',
                    'yAxis': 'right'
                },
                {
                    'label': 'Detection Count',
                    'data': [m.get('detection_count', 0) for m in metrics],
                    'color': '#f39c12',
                    'yAxis': 'right'
                }
            ],
            'x_axis': {
                'label': 'Time',
                'data': [m.get('timestamp', 0) for m in metrics]
            },
            'y_axes': {
                'left': {'label': 'Time (ms)', 'min': 0},
                'right': {'label': 'Count', 'min': 0}
            }
        }

    def _format_detection_statistics_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化检测统计图表数据"""
        stats = data.get('statistics', {})

        return {
            'chart_type': 'bar',
            'title': 'Detection Statistics',
            'datasets': [
                {
                    'label': 'Detection Rate',
                    'data': list(stats.get('detection_rates', {}).values()),
                    'color': '#2ecc71'
                }
            ],
            'x_axis': {
                'label': 'Range Bins',
                'data': list(stats.get('detection_rates', {}).keys())
            },
            'y_axis': {
                'label': 'Rate',
                'min': 0,
                'max': 1
            }
        }

    def _format_environment_impact_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化环境影响图表数据"""
        impact_data = data.get('impact_data', {})

        return {
            'chart_type': 'radar',
            'title': 'Environment Impact Analysis',
            'datasets': [
                {
                    'label': 'Current Conditions',
                    'data': [
                        impact_data.get('weather_impact', 0),
                        impact_data.get('clutter_impact', 0),
                        impact_data.get('interference_impact', 0),
                        impact_data.get('terrain_impact', 0),
                        impact_data.get('atmospheric_impact', 0)
                    ],
                    'color': '#9b59b6'
                }
            ],
            'labels': ['Weather', 'Clutter', 'Interference', 'Terrain', 'Atmospheric'],
            'scale': {'min': 0, 'max': 1}
        }

    def _format_track_distribution_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化航迹分布图表数据"""
        tracks = data.get('tracks', [])

        # 按距离分布
        range_bins = [0, 5000, 10000, 20000, 50000]
        range_counts = [0] * (len(range_bins) - 1)

        for track in tracks:
            track_range = track.get('range', 0)
            for i in range(len(range_bins) - 1):
                if range_bins[i] <= track_range < range_bins[i + 1]:
                    range_counts[i] += 1
                    break

        return {
            'chart_type': 'pie',
            'title': 'Track Distribution by Range',
            'datasets': [
                {
                    'data': range_counts,
                    'colors': ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
                }
            ],
            'labels': [f'{range_bins[i]}-{range_bins[i + 1]}m' for i in range(len(range_bins) - 1)]
        }

    def _format_table_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化表格数据"""
        if 'tracks' in data:
            return self._format_tracks_table(data['tracks'])
        elif 'detections' in data:
            return self._format_detections_table(data['detections'])
        elif 'performance' in data:
            return self._format_performance_table(data['performance'])
        else:
            return {'headers': [], 'rows': [], 'total': 0}

    def _format_tracks_table(self, tracks: List[Dict]) -> Dict[str, Any]:
        """格式化航迹表格数据"""
        headers = [
            {'key': 'id', 'label': 'Track ID', 'sortable': True},
            {'key': 'range', 'label': 'Range (m)', 'sortable': True, 'format': 'number'},
            {'key': 'azimuth', 'label': 'Azimuth (°)', 'sortable': True, 'format': 'angle'},
            {'key': 'velocity', 'label': 'Speed (m/s)', 'sortable': True, 'format': 'number'},
            {'key': 'confirmed', 'label': 'Status', 'sortable': True, 'format': 'status'},
            {'key': 'age', 'label': 'Age', 'sortable': True},
            {'key': 'score', 'label': 'Score', 'sortable': True, 'format': 'percentage'}
        ]

        rows = []
        for track in tracks:
            velocity_magnitude = np.sqrt(
                track.get('velocity', {}).get('x', 0) ** 2 +
                track.get('velocity', {}).get('y', 0) ** 2
            )

            row = {
                'id': track.get('id', 'N/A'),
                'range': round(track.get('range', 0), 1),
                'azimuth': round(np.degrees(track.get('azimuth', 0)), 1),
                'velocity': round(velocity_magnitude, 1),
                'confirmed': 'Confirmed' if track.get('confirmed') else 'Tentative',
                'age': track.get('age', 0),
                'score': round(track.get('score', 0) * 100, 1)
            }
            rows.append(row)

        return {
            'headers': headers,
            'rows': rows,
            'total': len(rows),
            'pagination': {
                'page_size': 20,
                'total_pages': (len(rows) + 19) // 20
            }
        }

    def _format_export_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化导出数据"""
        export_format = data.get('format', 'json')

        if export_format == 'csv':
            return self._format_for_csv_export(data)
        elif export_format == 'xml':
            return self._format_for_xml_export(data)
        elif export_format == 'binary':
            return self._format_for_binary_export(data)
        else:
            return self._format_for_json_export(data)

    def _format_for_csv_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化CSV导出数据"""
        csv_data = []

        if 'tracks' in data:
            df = pd.DataFrame(data['tracks'])
            csv_data.append({
                'filename': 'tracks.csv',
                'content': df.to_csv(index=False),
                'mime_type': 'text/csv'
            })

        if 'detections' in data:
            df = pd.DataFrame(data['detections'])
            csv_data.append({
                'filename': 'detections.csv',
                'content': df.to_csv(index=False),
                'mime_type': 'text/csv'
            })

        return {'files': csv_data, 'format': 'csv'}

    def _get_target_color(self, target: Dict) -> str:
        """获取目标显示颜色"""
        if target.get('confirmed'):
            return '#00ff00' if target.get('high_mobility') else '#ffff00'
        else:
            return '#ff8800'

    def _get_target_size(self, target: Dict) -> int:
        """获取目标显示大小"""
        base_size = 4
        if target.get('confirmed'):
            base_size += 2
        if target.get('high_mobility'):
            base_size += 1
        return base_size

    def _get_target_symbol(self, target: Dict) -> str:
        """获取目标显示符号"""
        if target.get('confirmed'):
            return 'triangle' if target.get('high_mobility') else 'diamond'
        else:
            return 'circle'

    def _get_detection_color(self, detection: Dict) -> str:
        """获取检测点显示颜色"""
        snr = detection.get('snr', 0)
        if snr > 20:
            return '#00ff00'
        elif snr > 10:
            return '#ffff00'
        else:
            return '#ff0000'

    def _generate_range_rings(self, max_range: float) -> List[Dict]:
        """生成距离环"""
        rings = []
        ring_interval = max_range / 10

        for i in range(1, 11):
            rings.append({
                'radius': i * ring_interval,
                'label': f'{int(i * ring_interval / 1000)}km',
                'style': 'solid' if i % 2 == 0 else 'dashed'
            })

        return rings

    def _generate_azimuth_lines(self) -> List[Dict]:
        """生成方位线"""
        lines = []
        for angle in range(0, 360, 30):
            lines.append({
                'angle': angle,
                'label': f'{angle}°',
                'style': 'solid' if angle % 90 == 0 else 'dashed'
            })

        return lines

    def _format_generic_chart(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化通用图表数据"""
        return {
            'chart_type': data.get('type', 'line'),
            'title': data.get('title', 'Chart'),
            'datasets': data.get('datasets', []),
            'x_axis': data.get('x_axis', {}),
            'y_axis': data.get('y_axis', {})
        }

    def _format_general_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化通用数据"""
        return {
            'formatted_data': data,
            'timestamp': datetime.now().isoformat(),
            'format_version': '1.0'
        }

    def _format_detections_table(self, detections: List[Dict]) -> Dict[str, Any]:
        """格式化检测表格数据"""
        headers = [
            {'key': 'id', 'label': 'Detection ID', 'sortable': True},
            {'key': 'range', 'label': 'Range (m)', 'sortable': True, 'format': 'number'},
            {'key': 'azimuth', 'label': 'Azimuth (°)', 'sortable': True, 'format': 'angle'},
            {'key': 'snr', 'label': 'SNR (dB)', 'sortable': True, 'format': 'number'},
            {'key': 'velocity', 'label': 'Velocity (m/s)', 'sortable': True, 'format': 'number'},
            {'key': 'rcs', 'label': 'RCS (m²)', 'sortable': True, 'format': 'number'}
        ]

        rows = []
        for detection in detections:
            row = {
                'id': detection.get('id', 'N/A'),
                'range': round(detection.get('range', 0), 1),
                'azimuth': round(np.degrees(detection.get('azimuth', 0)), 1),
                'snr': round(detection.get('snr', 0), 1),
                'velocity': round(detection.get('velocity', 0), 1),
                'rcs': round(detection.get('rcs', 0), 3)
            }
            rows.append(row)

        return {
            'headers': headers,
            'rows': rows,
            'total': len(rows)
        }

    def _format_performance_table(self, performance: Dict[str, Any]) -> Dict[str, Any]:
        """格式化性能表格数据"""
        headers = [
            {'key': 'metric', 'label': 'Metric', 'sortable': False},
            {'key': 'value', 'label': 'Value', 'sortable': True},
            {'key': 'unit', 'label': 'Unit', 'sortable': False}
        ]

        rows = [
            {'metric': 'Processing Time', 'value': round(performance.get('processing_time', 0) * 1000, 2),
             'unit': 'ms'},
            {'metric': 'Track Count', 'value': performance.get('track_count', 0), 'unit': 'tracks'},
            {'metric': 'Detection Count', 'value': performance.get('detection_count', 0), 'unit': 'detections'},
            {'metric': 'False Alarm Rate', 'value': round(performance.get('false_alarm_rate', 0) * 100, 1),
             'unit': '%'},
            {'metric': 'Track Accuracy', 'value': round(performance.get('track_accuracy', 0) * 100, 1), 'unit': '%'}
        ]

        return {
            'headers': headers,
            'rows': rows,
            'total': len(rows)
        }

    def _format_for_json_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化JSON导出数据"""
        return {
            'content': data,
            'format': 'json',
            'mime_type': 'application/json'
        }

    def _format_for_xml_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化XML导出数据"""
        # 简化的XML转换
        xml_content = self._dict_to_xml(data, 'root')
        return {
            'content': xml_content,
            'format': 'xml',
            'mime_type': 'application/xml'
        }

    def _format_for_binary_export(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """格式化二进制导出数据"""
        import pickle
        binary_content = pickle.dumps(data)
        return {
            'content': binary_content,
            'format': 'binary',
            'mime_type': 'application/octet-stream'
        }

    def _dict_to_xml(self, data: Dict, root_name: str) -> str:
        """将字典转换为XML"""
        import xml.etree.ElementTree as ET

        def build_xml(element, data):
            if isinstance(data, dict):
                for key, value in data.items():
                    child = ET.SubElement(element, str(key))
                    build_xml(child, value)
            elif isinstance(data, list):
                for item in data:
                    child = ET.SubElement(element, 'item')
                    build_xml(child, item)
            else:
                element.text = str(data)

        root = ET.Element(root_name)
        build_xml(root, data)

        return ET.tostring(root, encoding='unicode')
