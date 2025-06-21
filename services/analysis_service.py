import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json


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

        detection_rates = []
        detection_variance = []

        for step_data in time_series.values():
            detection_rates.append(step_data.get('avg_detections', 0))
            detection_variance.append(step_data.get('std_detections', 0))

        return {
            'mean_detection_rate': np.mean(detection_rates) if detection_rates else 0,
            'detection_stability': 1 / (1 + np.mean(detection_variance)) if detection_variance else 0,
            'peak_detection_rate': max(detection_rates) if detection_rates else 0,
            'detection_trend': np.polyfit(range(len(detection_rates)), detection_rates, 1)[0] if len(
                detection_rates) > 1 else 0
        }

    def _analyze_tracking_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'time_series' not in results:
            return {}

        time_series = results['time_series']

        track_counts = []
        confirmed_track_counts = []

        for step_data in time_series.values():
            track_counts.append(step_data.get('avg_tracks', 0))
            confirmed_track_counts.append(step_data.get('avg_confirmed_tracks', 0))

        track_initiation_rate = np.mean(track_counts) if track_counts else 0
        track_confirmation_rate = (
                    np.mean(confirmed_track_counts) / max(track_initiation_rate, 1)) if confirmed_track_counts else 0

        return {
            'track_initiation_rate': track_initiation_rate,
            'track_confirmation_rate': track_confirmation_rate,
            'tracking_stability': 1 - (np.std(track_counts) / max(np.mean(track_counts), 1)) if track_counts else 0,
            'average_confirmed_tracks': np.mean(confirmed_track_counts) if confirmed_track_counts else 0
        }

    def _analyze_resource_utilization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        if 'time_series' not in results:
            return {}

        time_series = results['time_series']

        scheduling_efficiencies = []

        for step_data in time_series.values():
            scheduling_efficiencies.append(step_data.get('avg_scheduling_efficiency', 0))

        return {
            'average_scheduling_efficiency': np.mean(scheduling_efficiencies) if scheduling_efficiencies else 0,
            'resource_utilization_stability': 1 - (
                        np.std(scheduling_efficiencies) / max(np.mean(scheduling_efficiencies),
                                                              0.01)) if scheduling_efficiencies else 0,
            'peak_efficiency': max(scheduling_efficiencies) if scheduling_efficiencies else 0,
            'minimum_efficiency': min(scheduling_efficiencies) if scheduling_efficiencies else 0
        }

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

                metrics['detection_metrics'] = self._calculate_detection_metrics(time_series)
                metrics['tracking_metrics'] = self._calculate_tracking_metrics(time_series)
                metrics['system_metrics'] = self._calculate_system_metrics(time_series)

        return metrics

    def _calculate_detection_metrics(self, time_series: Dict) -> Dict[str, float]:
        detections = [step.get('avg_detections', 0) for step in time_series.values()]

        return {
            'probability_of_detection': np.mean([min(d / 5.0, 1.0) for d in detections]),
            'false_alarm_rate': max(0, np.mean(detections) - 3) / max(np.mean(detections), 1),
            'detection_consistency': 1 - (np.std(detections) / max(np.mean(detections), 1))
        }

    def _calculate_tracking_metrics(self, time_series: Dict) -> Dict[str, float]:
        tracks = [step.get('avg_tracks', 0) for step in time_series.values()]
        confirmed = [step.get('avg_confirmed_tracks', 0) for step in time_series.values()]

        return {
            'track_continuity': np.mean([c / max(t, 1) for t, c in zip(tracks, confirmed)]),
            'track_accuracy': 0.85,  # 模拟值，实际需要真值比较
            'track_purity': 0.90  # 模拟值，实际需要计算航迹纯净度
        }

    def _calculate_system_metrics(self, time_series: Dict) -> Dict[str, float]:
        efficiencies = [step.get('avg_scheduling_efficiency', 0) for step in time_series.values()]

        return {
            'system_availability': 0.98,  # 模拟值
            'response_time': 0.06,  # 60ms
            'throughput': np.mean(efficiencies),
            'reliability_score': min(np.mean(efficiencies) + 0.1, 1.0)
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
