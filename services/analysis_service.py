import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json
from numba import njit, float64, int64, prange


class AnalysisService:
    def analyze_simulation_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        performance_analysis = self._analyze_performance(results)
        detection_analysis = self._analyze_detection_performance(results)
        tracking_analysis = self._analyze_tracking_performance(results)
        resource_analysis = self._analyze_resource_utilization(results)

        return {
            'performance': performance_analysis,
            'detection': detection_analysis,
            'tracking': tracking_analysis,
            'resources': resource_analysis
        }

    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'summary' not in results:
            return {}

        summary = results['summary']

        return {
            'detection_rate': summary.get('avg_total_detections', 0) / max(summary.get('total_runs', 1), 1),
            'tracking_efficiency': summary.get('avg_final_tracks', 0),
            'overall_performance_score': self._calculate_performance_score(results)
        }

    def _analyze_detection_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'time_series' not in results:
            return {}

        time_series = results['time_series']
        detection_rates = np.array([step.get('avg_detections', 0) for step in time_series.values()])
        detection_variance = np.array([step.get('std_detections', 0) for step in time_series.values()])

        return self._analyze_detection_numba(detection_rates, detection_variance)

    def _analyze_tracking_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'time_series' not in results:
            return {}

        time_series = results['time_series']
        track_counts = np.array([step.get('avg_tracks', 0) for step in time_series.values()])
        confirmed_track_counts = np.array([step.get('avg_confirmed_tracks', 0) for step in time_series.values()])

        return self._analyze_tracking_numba(track_counts, confirmed_track_counts)

    def _analyze_resource_utilization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'time_series' not in results:
            return {}

        time_series = results['time_series']
        scheduling_efficiencies = np.array([step.get('avg_scheduling_efficiency', 0) for step in time_series.values()])

        return self._analyze_resources_numba(scheduling_efficiencies)

    def _calculate_performance_score(self, results: Dict[str, Any]) -> float:
        weights = {
            'detection': 0.3,
            'tracking': 0.4,
            'scheduling': 0.3
        }

        detection_analysis = self._analyze_detection_performance(results)
        tracking_analysis = self._analyze_tracking_performance(results)
        resource_analysis = self._analyze_resource_utilization(results)

        detection_score = (detection_analysis.get('mean_detection_rate', 0) / 10.0 +
                           detection_analysis.get('detection_stability', 0)) / 2

        tracking_score = (tracking_analysis.get('track_confirmation_rate', 0) +
                          tracking_analysis.get('tracking_stability', 0)) / 2

        scheduling_score = resource_analysis.get('average_scheduling_efficiency', 0)

        overall_score = (weights['detection'] * detection_score +
                         weights['tracking'] * tracking_score +
                         weights['scheduling'] * scheduling_score)

        return min(max(overall_score, 0), 1)

    def calculate_performance_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}

        if 'results' in data:
            results = data['results']

            if isinstance(results, dict) and 'time_series' in results:
                time_series = results['time_series']

                detections = np.array([step.get('avg_detections', 0) for step in time_series.values()])
                tracks = np.array([step.get('avg_tracks', 0) for step in time_series.values()])
                confirmed = np.array([step.get('avg_confirmed_tracks', 0) for step in time_series.values()])
                efficiencies = np.array([step.get('avg_scheduling_efficiency', 0) for step in time_series.values()])

                metrics['detection_metrics'] = self._calculate_detection_metrics_numba(detections)
                metrics['tracking_metrics'] = self._calculate_tracking_metrics_numba(tracks, confirmed)
                metrics['system_metrics'] = self._calculate_system_metrics_numba(efficiencies)

        return metrics

    @staticmethod
    @njit(float64(float64[:], float64[:]), cache=True)
    def _analyze_detection_numba(detection_rates, detection_variance):
        """Numba优化的检测性能分析"""
        n = len(detection_rates)
        if n == 0:
            return {
                'mean_detection_rate': 0.0,
                'detection_stability': 0.0,
                'peak_detection_rate': 0.0,
                'detection_trend': 0.0
            }

        mean_rate = np.mean(detection_rates)
        mean_var = np.mean(detection_variance)
        peak_rate = np.max(detection_rates)

        # 计算线性趋势 (最小二乘法)
        sum_x = 0.0
        sum_y = 0.0
        sum_xy = 0.0
        sum_x2 = 0.0

        for i in prange(n):
            sum_x += i
            sum_y += detection_rates[i]
            sum_xy += i * detection_rates[i]
            sum_x2 += i * i

        denominator = n * sum_x2 - sum_x * sum_x
        if abs(denominator) < 1e-10:
            slope = 0.0
        else:
            slope = (n * sum_xy - sum_x * sum_y) / denominator

        return {
            'mean_detection_rate': mean_rate,
            'detection_stability': 1 / (1 + mean_var) if mean_var > 0 else 0.0,
            'peak_detection_rate': peak_rate,
            'detection_trend': slope
        }

    @staticmethod
    @njit(float64(float64[:], float64[:]), cache=True)
    def _analyze_tracking_numba(track_counts, confirmed_track_counts):
        """Numba优化的跟踪性能分析"""
        n = len(track_counts)
        if n == 0:
            return {
                'track_initiation_rate': 0.0,
                'track_confirmation_rate': 0.0,
                'tracking_stability': 0.0,
                'average_confirmed_tracks': 0.0
            }

        mean_tracks = np.mean(track_counts)
        mean_confirmed = np.mean(confirmed_track_counts)

        # 计算稳定性
        std_tracks = np.std(track_counts)
        mean_tracks_nonzero = mean_tracks if mean_tracks > 1e-5 else 1.0
        stability = 1 - (std_tracks / mean_tracks_nonzero)

        # 计算确认率
        confirmation_rate = mean_confirmed / mean_tracks_nonzero if mean_tracks_nonzero > 0 else 0.0

        return {
            'track_initiation_rate': mean_tracks,
            'track_confirmation_rate': confirmation_rate,
            'tracking_stability': max(min(stability, 1.0), 0.0),
            'average_confirmed_tracks': mean_confirmed
        }

    @staticmethod
    @njit(float64(float64[:]), cache=True)
    def _analyze_resources_numba(scheduling_efficiencies):
        """Numba优化的资源利用分析"""
        n = len(scheduling_efficiencies)
        if n == 0:
            return {
                'average_scheduling_efficiency': 0.0,
                'resource_utilization_stability': 0.0,
                'peak_efficiency': 0.0,
                'minimum_efficiency': 0.0
            }

        mean_eff = np.mean(scheduling_efficiencies)
        std_eff = np.std(scheduling_efficiencies)
        peak_eff = np.max(scheduling_efficiencies)
        min_eff = np.min(scheduling_efficiencies)

        # 计算稳定性
        mean_eff_nonzero = mean_eff if mean_eff > 0.01 else 0.01
        stability = 1 - (std_eff / mean_eff_nonzero)

        return {
            'average_scheduling_efficiency': mean_eff,
            'resource_utilization_stability': max(min(stability, 1.0), 0.0),
            'peak_efficiency': peak_eff,
            'minimum_efficiency': min_eff
        }

    @staticmethod
    @njit(float64(float64[:]), cache=True)
    def _calculate_detection_metrics_numba(detections):
        """Numba优化的检测指标计算"""
        n = len(detections)
        if n == 0:
            return {
                'probability_of_detection': 0.0,
                'false_alarm_rate': 0.0,
                'detection_consistency': 0.0
            }

        # 概率检测
        pod_sum = 0.0
        for i in prange(n):
            pod_sum += min(detections[i] / 5.0, 1.0)
        pod = pod_sum / n

        # 误报率
        mean_det = np.mean(detections)
        mean_det_nonzero = mean_det if mean_det > 1e-5 else 1.0
        far = max(0, mean_det - 3) / mean_det_nonzero

        # 一致性
        std_det = np.std(detections)
        consistency = 1 - (std_det / mean_det_nonzero) if mean_det_nonzero > 0 else 0.0

        return {
            'probability_of_detection': pod,
            'false_alarm_rate': far,
            'detection_consistency': max(min(consistency, 1.0), 0.0)
        }

    @staticmethod
    @njit(float64(float64[:], float64[:]), cache=True)
    def _calculate_tracking_metrics_numba(tracks, confirmed):
        """Numba优化的跟踪指标计算"""
        n = len(tracks)
        if n == 0:
            return {
                'track_continuity': 0.0,
                'track_accuracy': 0.85,
                'track_purity': 0.90
            }

        # 跟踪连续性
        continuity_sum = 0.0
        for i in prange(n):
            t = tracks[i]
            c = confirmed[i]
            continuity_sum += c / max(t, 1.0)
        continuity = continuity_sum / n

        return {
            'track_continuity': continuity,
            'track_accuracy': 0.85,  # 模拟值
            'track_purity': 0.90  # 模拟值
        }

    @staticmethod
    @njit(float64(float64[:]), cache=True)
    def _calculate_system_metrics_numba(efficiencies):
        """Numba优化的系统指标计算"""
        n = len(efficiencies)
        if n == 0:
            return {
                'system_availability': 0.98,
                'response_time': 0.06,
                'throughput': 0.0,
                'reliability_score': 0.0
            }

        mean_eff = np.mean(efficiencies)
        reliability = min(mean_eff + 0.1, 1.0)

        return {
            'system_availability': 0.98,  # 模拟值
            'response_time': 0.06,  # 60ms
            'throughput': mean_eff,
            'reliability_score': reliability
        }

    def export_simulation_data(self, results: Dict[str, Any], format_type: str = 'json') -> Dict[str, Any]:
        try:
            if format_type.lower() == 'csv':
                return self._export_to_csv(results)
            elif format_type.lower() == 'excel':
                return self._export_to_excel(results)
            else:
                return self._export_to_json(results)

        except Exception as e:
            return {
                'error': str(e),
                'format': format_type
            }

    def _export_to_json(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'format': 'json',
            'data': json.dumps(results, indent=2),
            'filename': 'simulation_results.json'
        }

    def _export_to_csv(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'time_series' in results:
            df_data = []
            for step_idx, step_data in results['time_series'].items():
                row = {
                    'step': step_idx,
                    'time': step_data.get('time', 0),
                    'detections': step_data.get('avg_detections', 0),
                    'tracks': step_data.get('avg_tracks', 0),
                    'confirmed_tracks': step_data.get('avg_confirmed_tracks', 0),
                    'scheduling_efficiency': step_data.get('avg_scheduling_efficiency', 0)
                }
                df_data.append(row)

            df = pd.DataFrame(df_data)
            csv_data = df.to_csv(index=False)

            return {
                'format': 'csv',
                'data': csv_data,
                'filename': 'simulation_results.csv'
            }

        return {'error': 'No time series data available for CSV export'}

    def _export_to_excel(self, results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'format': 'excel',
            'message': 'Excel export not implemented yet',
            'filename': 'simulation_results.xlsx'
        }
