import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import logging
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

logger = logging.getLogger(__name__)


class ChartUtils:
    """图表工具类 - 用于生成各种类型的图表"""

    def __init__(self):
        # 设置matplotlib样式
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')

        # 图表配置
        self.default_figsize = (12, 8)
        self.default_dpi = 100
        self.color_palette = [
            '#3498db', '#e74c3c', '#2ecc71', '#f39c12',
            '#9b59b6', '#1abc9c', '#34495e', '#e67e22',
            '#95a5a6', '#f1c40f', '#8e44ad', '#2c3e50'
        ]

        # Plotly配置
        self.plotly_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d']
        }

        logger.info("ChartUtils initialized with matplotlib and plotly support")

    def create_performance_chart(self, metrics_data: List[Dict],
                                 chart_type: str = 'line',
                                 output_format: str = 'plotly') -> Dict[str, Any]:
        """创建性能图表"""
        try:
            if output_format == 'plotly':
                return self._create_plotly_performance_chart(metrics_data, chart_type)
            else:
                return self._create_matplotlib_performance_chart(metrics_data, chart_type)
        except Exception as e:
            logger.error(f"Error creating performance chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def _create_plotly_performance_chart(self, metrics_data: List[Dict],
                                         chart_type: str) -> Dict[str, Any]:
        """使用Plotly创建性能图表"""
        if not metrics_data:
            return {'error': 'No metrics data provided', 'chart_data': None}

        # 提取数据
        timestamps = [datetime.fromtimestamp(m.get('timestamp', 0)) for m in metrics_data]
        processing_times = [m.get('processing_time', 0) * 1000 for m in metrics_data]  # 转换为毫秒
        track_counts = [m.get('track_count', 0) for m in metrics_data]
        detection_counts = [m.get('detection_count', 0) for m in metrics_data]
        false_alarm_rates = [m.get('false_alarm_rate', 0) * 100 for m in metrics_data]  # 转换为百分比

        # 创建子图
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Processing Time', 'Track & Detection Counts',
                            'False Alarm Rate', 'System Load'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # 处理时间
        fig.add_trace(
            go.Scatter(x=timestamps, y=processing_times,
                       mode='lines+markers', name='Processing Time (ms)',
                       line=dict(color=self.color_palette[0], width=2),
                       marker=dict(size=6)),
            row=1, col=1
        )

        # 航迹和检测计数
        fig.add_trace(
            go.Scatter(x=timestamps, y=track_counts,
                       mode='lines+markers', name='Track Count',
                       line=dict(color=self.color_palette[1], width=2),
                       marker=dict(size=6)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=timestamps, y=detection_counts,
                       mode='lines+markers', name='Detection Count',
                       line=dict(color=self.color_palette[2], width=2),
                       marker=dict(size=6)),
            row=1, col=2, secondary_y=True
        )

        # 虚警率
        fig.add_trace(
            go.Scatter(x=timestamps, y=false_alarm_rates,
                       mode='lines+markers', name='False Alarm Rate (%)',
                       line=dict(color=self.color_palette[3], width=2),
                       marker=dict(size=6)),
            row=2, col=1
        )

        # 系统负载（处理时间 vs 目标数量）
        system_load = [pt / max(tc, 1) for pt, tc in zip(processing_times, track_counts)]
        fig.add_trace(
            go.Scatter(x=timestamps, y=system_load,
                       mode='lines+markers', name='Load (ms/track)',
                       line=dict(color=self.color_palette[4], width=2),
                       marker=dict(size=6)),
            row=2, col=2
        )

        # 更新布局
        fig.update_layout(
            title='System Performance Dashboard',
            height=600,
            showlegend=True,
            template='plotly_white'
        )

        # 更新轴标签
        fig.update_yaxes(title_text="Time (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Track Count", row=1, col=2)
        fig.update_yaxes(title_text="Detection Count", row=1, col=2, secondary_y=True)
        fig.update_yaxes(title_text="Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Load (ms/track)", row=2, col=2)

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly',
            'title': 'System Performance Dashboard'
        }

    def create_radar_coverage_chart(self, tracks_data: List[Dict],
                                    max_range: float = 50000,
                                    output_format: str = 'plotly') -> Dict[str, Any]:
        """创建雷达覆盖图表"""
        try:
            if output_format == 'plotly':
                return self._create_plotly_radar_coverage(tracks_data, max_range)
            else:
                return self._create_matplotlib_radar_coverage(tracks_data, max_range)
        except Exception as e:
            logger.error(f"Error creating radar coverage chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def _create_plotly_radar_coverage(self, tracks_data: List[Dict],
                                      max_range: float) -> Dict[str, Any]:
        """使用Plotly创建雷达覆盖图表"""
        if not tracks_data:
            return {'error': 'No track data provided', 'chart_data': None}

        # 提取目标位置数据
        confirmed_tracks = {'r': [], 'theta': [], 'text': []}
        tentative_tracks = {'r': [], 'theta': [], 'text': []}
        high_mobility_tracks = {'r': [], 'theta': [], 'text': []}

        for track in tracks_data:
            range_val = track.get('range', 0)
            azimuth_val = np.degrees(track.get('azimuth', 0))
            track_id = track.get('id', 'Unknown')

            text = f"Track {track_id}<br>Range: {range_val:.0f}m<br>Azimuth: {azimuth_val:.1f}°"

            if track.get('high_mobility', False):
                high_mobility_tracks['r'].append(range_val)
                high_mobility_tracks['theta'].append(azimuth_val)
                high_mobility_tracks['text'].append(text)
            elif track.get('confirmed', False):
                confirmed_tracks['r'].append(range_val)
                confirmed_tracks['theta'].append(azimuth_val)
                confirmed_tracks['text'].append(text)
            else:
                tentative_tracks['r'].append(range_val)
                tentative_tracks['theta'].append(azimuth_val)
                tentative_tracks['text'].append(text)

        fig = go.Figure()

        # 添加不同类型的航迹
        if confirmed_tracks['r']:
            fig.add_trace(go.Scatterpolar(
                r=confirmed_tracks['r'],
                theta=confirmed_tracks['theta'],
                mode='markers',
                name='Confirmed Tracks',
                marker=dict(color='green', size=10, symbol='diamond'),
                text=confirmed_tracks['text'],
                hovertemplate='%{text}<extra></extra>'
            ))

        if tentative_tracks['r']:
            fig.add_trace(go.Scatterpolar(
                r=tentative_tracks['r'],
                theta=tentative_tracks['theta'],
                mode='markers',
                name='Tentative Tracks',
                marker=dict(color='orange', size=8, symbol='circle'),
                text=tentative_tracks['text'],
                hovertemplate='%{text}<extra></extra>'
            ))

        if high_mobility_tracks['r']:
            fig.add_trace(go.Scatterpolar(
                r=high_mobility_tracks['r'],
                theta=high_mobility_tracks['theta'],
                mode='markers',
                name='High Mobility Tracks',
                marker=dict(color='red', size=12, symbol='triangle-up'),
                text=high_mobility_tracks['text'],
                hovertemplate='%{text}<extra></extra>'
            ))

        # 更新布局
        fig.update_layout(
            title='Radar Coverage and Track Distribution',
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max_range],
                    tickmode='array',
                    tickvals=np.linspace(0, max_range, 6),
                    ticktext=[f'{int(r / 1000)}km' for r in np.linspace(0, max_range, 6)]
                ),
                angularaxis=dict(
                    tickmode='array',
                    tickvals=list(range(0, 360, 30)),
                    ticktext=[f'{i}°' for i in range(0, 360, 30)],
                    direction='clockwise',
                    theta0=90
                )
            ),
            showlegend=True,
            template='plotly_white'
        )

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly_polar',
            'title': 'Radar Coverage and Track Distribution'
        }

    def create_detection_statistics_chart(self, stats_data: Dict[str, Any],
                                          output_format: str = 'plotly') -> Dict[str, Any]:
        """创建检测统计图表"""
        try:
            if output_format == 'plotly':
                return self._create_plotly_detection_stats(stats_data)
            else:
                return self._create_matplotlib_detection_stats(stats_data)
        except Exception as e:
            logger.error(f"Error creating detection statistics chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def _create_plotly_detection_stats(self, stats_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用Plotly创建检测统计图表"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Range Distribution', 'SNR Distribution',
                            'Azimuth Distribution', 'Detection Rate by Time'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )

        # 1. 距离分布
        if 'range_distribution' in stats_data:
            ranges = stats_data['range_distribution']
            fig.add_trace(
                go.Histogram(x=ranges, nbinsx=20, name='Range',
                             marker_color=self.color_palette[0]),
                row=1, col=1
            )

        # 2. SNR分布
        if 'snr_distribution' in stats_data:
            snr_values = stats_data['snr_distribution']
            fig.add_trace(
                go.Histogram(x=snr_values, nbinsx=20, name='SNR',
                             marker_color=self.color_palette[1]),
                row=1, col=2
            )

        # 3. 方位角分布
        if 'azimuth_distribution' in stats_data:
            azimuths = np.degrees(stats_data['azimuth_distribution'])
            fig.add_trace(
                go.Histogram(x=azimuths, nbinsx=18, name='Azimuth',
                             marker_color=self.color_palette[2]),
                row=2, col=1
            )

        # 4. 检测率随时间变化
        if 'detection_rate_over_time' in stats_data:
            time_data = stats_data['detection_rate_over_time']
            timestamps = [datetime.fromtimestamp(t['timestamp']) for t in time_data]
            rates = [t['rate'] for t in time_data]

            fig.add_trace(
                go.Scatter(x=timestamps, y=rates, mode='lines+markers',
                           name='Detection Rate', line=dict(color=self.color_palette[3])),
                row=2, col=2
            )

        # 更新布局
        fig.update_layout(
            title='Detection Statistics Dashboard',
            height=600,
            showlegend=True,
            template='plotly_white'
        )

        # 更新轴标签
        fig.update_xaxes(title_text="Range (m)", row=1, col=1)
        fig.update_xaxes(title_text="SNR (dB)", row=1, col=2)
        fig.update_xaxes(title_text="Azimuth (degrees)", row=2, col=1)
        fig.update_xaxes(title_text="Time", row=2, col=2)

        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Detection Rate", row=2, col=2)

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly_subplots',
            'title': 'Detection Statistics Dashboard'
        }

    def create_environment_impact_chart(self, impact_data: Dict[str, Any],
                                        output_format: str = 'plotly') -> Dict[str, Any]:
        """创建环境影响图表"""
        try:
            if output_format == 'plotly':
                return self._create_plotly_environment_impact(impact_data)
            else:
                return self._create_matplotlib_environment_impact(impact_data)
        except Exception as e:
            logger.error(f"Error creating environment impact chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def _create_plotly_environment_impact(self, impact_data: Dict[str, Any]) -> Dict[str, Any]:
        """使用Plotly创建环境影响雷达图"""
        categories = ['Weather', 'Clutter', 'Interference', 'Terrain', 'Atmospheric']
        values = [
            impact_data.get('weather_impact', 0),
            impact_data.get('clutter_impact', 0),
            impact_data.get('interference_impact', 0),
            impact_data.get('terrain_impact', 0),
            impact_data.get('atmospheric_impact', 0)
        ]

        # 将影响值转换为性能百分比（值越小，性能越好）
        performance_values = [(1 - v) * 100 for v in values]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=performance_values,
            theta=categories,
            fill='toself',
            name='Current Performance',
            line_color=self.color_palette[0],
            fillcolor=self.color_palette[0],
            opacity=0.6
        ))

        # 添加理想性能基准线
        fig.add_trace(go.Scatterpolar(
            r=[100] * len(categories),
            theta=categories,
            fill='toself',
            name='Ideal Performance',
            line_color='gray',
            fillcolor='gray',
            opacity=0.2
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    ticksuffix='%'
                )
            ),
            title='Environment Impact on System Performance',
            showlegend=True,
            template='plotly_white'
        )

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly_radar',
            'title': 'Environment Impact Analysis'
        }

    def create_track_distribution_chart(self, tracks_data: List[Dict],
                                        distribution_type: str = 'range',
                                        output_format: str = 'plotly') -> Dict[str, Any]:
        """创建航迹分布图表"""
        try:
            if output_format == 'plotly':
                return self._create_plotly_track_distribution(tracks_data, distribution_type)
            else:
                return self._create_matplotlib_track_distribution(tracks_data, distribution_type)
        except Exception as e:
            logger.error(f"Error creating track distribution chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def _create_plotly_track_distribution(self, tracks_data: List[Dict],
                                          distribution_type: str) -> Dict[str, Any]:
        """使用Plotly创建航迹分布图表"""
        if not tracks_data:
            return {'error': 'No track data provided', 'chart_data': None}

        if distribution_type == 'range':
            # 按距离分布
            range_bins = [0, 5000, 10000, 20000, 35000, 50000]
            range_labels = [f'{range_bins[i] // 1000}-{range_bins[i + 1] // 1000}km'
                            for i in range(len(range_bins) - 1)]
            range_counts = [0] * (len(range_bins) - 1)

            for track in tracks_data:
                track_range = track.get('range', 0)
                for i in range(len(range_bins) - 1):
                    if range_bins[i] <= track_range < range_bins[i + 1]:
                        range_counts[i] += 1
                        break

            fig = go.Figure(data=[
                go.Pie(labels=range_labels, values=range_counts,
                       hole=0.3, marker_colors=self.color_palette[:len(range_labels)])
            ])

            fig.update_layout(
                title='Track Distribution by Range',
                template='plotly_white'
            )

        elif distribution_type == 'status':
            # 按状态分布
            confirmed_count = sum(1 for t in tracks_data if t.get('confirmed', False))
            tentative_count = len(tracks_data) - confirmed_count
            high_mobility_count = sum(1 for t in tracks_data if t.get('high_mobility', False))

            fig = go.Figure(data=[
                go.Pie(labels=['Confirmed', 'Tentative', 'High Mobility'],
                       values=[confirmed_count, tentative_count, high_mobility_count],
                       hole=0.3, marker_colors=['green', 'orange', 'red'])
            ])

            fig.update_layout(
                title='Track Distribution by Status',
                template='plotly_white'
            )

        elif distribution_type == 'velocity':
            # 按速度分布
            velocities = []
            for track in tracks_data:
                vel = track.get('velocity', {})
                velocity_magnitude = np.sqrt(vel.get('x', 0) ** 2 + vel.get('y', 0) ** 2)
                velocities.append(velocity_magnitude)

            fig = go.Figure(data=[
                go.Histogram(x=velocities, nbinsx=20,
                             marker_color=self.color_palette[2])
            ])

            fig.update_layout(
                title='Track Velocity Distribution',
                xaxis_title='Velocity (m/s)',
                yaxis_title='Number of Tracks',
                template='plotly_white'
            )

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly_distribution',
            'title': f'Track Distribution by {distribution_type.title()}'
        }

    def create_real_time_chart(self, data: Dict[str, Any],
                               chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建实时图表"""
        try:
            chart_type = chart_config.get('type', 'line')

            if chart_type == 'line':
                return self._create_real_time_line_chart(data, chart_config)
            elif chart_type == 'gauge':
                return self._create_real_time_gauge_chart(data, chart_config)
            elif chart_type == 'bar':
                return self._create_real_time_bar_chart(data, chart_config)
            else:
                return {'error': f'Unsupported chart type: {chart_type}'}

        except Exception as e:
            logger.error(f"Error creating real-time chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def _create_real_time_line_chart(self, data: Dict[str, Any],
                                     chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建实时线图"""
        fig = go.Figure()

        x_data = data.get('x', [])
        y_data = data.get('y', [])

        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines+markers',
            name=chart_config.get('title', 'Real-time Data'),
            line=dict(color=self.color_palette[0], width=2),
            marker=dict(size=4)
        ))

        fig.update_layout(
            title=chart_config.get('title', 'Real-time Chart'),
            xaxis_title=chart_config.get('x_label', 'Time'),
            yaxis_title=chart_config.get('y_label', 'Value'),
            template='plotly_white',
            height=400
        )

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly_line',
            'title': chart_config.get('title', 'Real-time Chart')
        }

    def _create_real_time_gauge_chart(self, data: Dict[str, Any],
                                      chart_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建实时仪表盘图"""
        value = data.get('value', 0)
        max_value = chart_config.get('max_value', 100)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': chart_config.get('title', 'Gauge')},
            gauge={
                'axis': {'range': [None, max_value]},
                'bar': {'color': self.color_palette[0]},
                'steps': [
                    {'range': [0, max_value * 0.5], 'color': "lightgray"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))

        fig.update_layout(
            template='plotly_white',
            height=400
        )

        return {
            'chart_data': fig.to_json(),
            'chart_type': 'plotly_gauge',
            'title': chart_config.get('title', 'Gauge Chart')
        }

    def create_heatmap_chart(self, data: np.ndarray,
                             labels: Dict[str, List[str]] = None,
                             title: str = 'Heatmap') -> Dict[str, Any]:
        """创建热力图"""
        try:
            fig = go.Figure(data=go.Heatmap(
                z=data,
                x=labels.get('x', []) if labels else None,
                y=labels.get('y', []) if labels else None,
                colorscale='Viridis'
            ))

            fig.update_layout(
                title=title,
                template='plotly_white'
            )

            return {
                'chart_data': fig.to_json(),
                'chart_type': 'plotly_heatmap',
                'title': title
            }

        except Exception as e:
            logger.error(f"Error creating heatmap chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}

    def export_chart(self, chart_data: str, format: str = 'png',
                     width: int = 800, height: int = 600) -> bytes:
        """导出图表为指定格式"""
        try:
            import plotly.io as pio

            fig = go.Figure(json.loads(chart_data))

            if format.lower() == 'png':
                return pio.to_image(fig, format='png', width=width, height=height)
            elif format.lower() == 'svg':
                return pio.to_image(fig, format='svg', width=width, height=height)
            elif format.lower() == 'pdf':
                return pio.to_image(fig, format='pdf', width=width, height=height)
            elif format.lower() == 'html':
                return fig.to_html().encode('utf-8')
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            logger.error(f"Error exporting chart: {str(e)}")
            raise

    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图形转换为base64字符串"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=self.default_dpi, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return image_base64

    def get_color_palette(self, n_colors: int = None) -> List[str]:
        """获取颜色调色板"""
        if n_colors is None:
            return self.color_palette
        else:
            return self.color_palette[:n_colors] if n_colors <= len(self.color_palette) else self.color_palette * (
                        (n_colors // len(self.color_palette)) + 1)

    def create_custom_chart(self, chart_spec: Dict[str, Any]) -> Dict[str, Any]:
        """根据规格创建自定义图表"""
        try:
            chart_type = chart_spec.get('type', 'scatter')
            data = chart_spec.get('data', {})
            layout = chart_spec.get('layout', {})

            if chart_type == 'scatter':
                fig = go.Figure(data=go.Scatter(**data))
            elif chart_type == 'bar':
                fig = go.Figure(data=go.Bar(**data))
            elif chart_type == 'line':
                fig = go.Figure(data=go.Scatter(mode='lines', **data))
            elif chart_type == 'pie':
                fig = go.Figure(data=go.Pie(**data))
            else:
                return {'error': f'Unsupported chart type: {chart_type}'}

            fig.update_layout(**layout)

            return {
                'chart_data': fig.to_json(),
                'chart_type': f'plotly_{chart_type}',
                'title': layout.get('title', 'Custom Chart')
            }

        except Exception as e:
            logger.error(f"Error creating custom chart: {str(e)}")
            return {'error': str(e), 'chart_data': None}
