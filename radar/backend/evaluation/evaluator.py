# evaluator.py - 效能评估器
"""
本模块实现系统效能评估功能。

效能评估包括：
- 跟踪精度统计
- 调度性能分析
- 检测概率评估
- 虚警率统计
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from radar.common.logger import get_logger
from radar.common.types import Track, Position3D, Velocity3D


@dataclass
class TrackingError:
    """
    跟踪误差统计

    Attributes:
        position_rmse: 位置均方根误差 [米]
        velocity_rmse: 速度均方根误差 [米/秒]
        position_bias: 位置偏差 [米]
        velocity_bias: 速度偏差 [米/秒]
        max_position_error: 最大位置误差 [米]
    """
    position_rmse: float = 0.0
    velocity_rmse: float = 0.0
    position_bias: float = 0.0
    velocity_bias: float = 0.0
    max_position_error: float = 0.0


@dataclass
class ScheduleAnalysis:
    """
    调度分析结果

    Attributes:
        utilization: 时间利用率 [0-1]
        success_rate: 调度成功率 [0-1]
        dropped_tasks: 丢弃任务数
        delayed_tasks: 延迟任务数
        avg_delay: 平均延迟 [秒]
    """
    utilization: float = 0.0
    success_rate: float = 0.0
    dropped_tasks: int = 0
    delayed_tasks: int = 0
    avg_delay: float = 0.0


@dataclass
class DetectionMetrics:
    """
    检测指标

    Attributes:
        detection_probability: 检测概率 [0-1]
        false_alarm_rate: 虚警率
        snr_mean: 平均SNR [dB]
        snr_std: SNR标准差 [dB]
    """
    detection_probability: float = 0.0
    false_alarm_rate: float = 0.0
    snr_mean: float = 0.0
    snr_std: float = 0.0


class Evaluator:
    """
    效能评估器

    收集和分析系统性能指标。
    """

    def __init__(self):
        """初始化评估器"""
        self._logger = get_logger("evaluator")

        # 跟踪误差历史
        self._position_errors: List[float] = []
        self._velocity_errors: List[float] = []

        # 检测统计
        self._detection_count = 0
        self._total_opportunities = 0
        self._false_alarm_count = 0
        self._snr_values: List[float] = []

        # 调度统计
        self._schedule_history: List[Dict] = []

        self._logger.info("效能评估器初始化")

    async def evaluate(self, tracks: List[Track],
                    true_targets: Optional[Dict[int, Position3D]] = None) -> Dict:
        """
        执行评估

        Args:
            tracks: 当前航迹列表
            true_targets: 真实目标位置 {target_id: position}

        Returns:
            评估结果字典
        """
        # 跟踪误差评估
        tracking_error = self._evaluate_tracking_error(
            tracks, true_targets
        )

        # 调度分析
        schedule_analysis = self._analyze_schedule()

        # 检测指标
        detection_metrics = self._calculate_detection_metrics()

        result = {
            'tracking_error': tracking_error,
            'schedule_analysis': schedule_analysis,
            'detection_metrics': detection_metrics,
            'timestamp': 0,  # TODO: 使用实际时间戳
        }

        return result

    def _evaluate_tracking_error(self,
                                  tracks: List[Track],
                                  true_targets: Optional[Dict[int, Position3D]]) -> TrackingError:
        """
        评估跟踪误差

        Args:
            tracks: 航迹列表
            true_targets: 真实目标

        Returns:
            跟踪误差统计
        """
        if not true_targets or not tracks:
            return TrackingError()

        position_errors = []
        velocity_errors = []

        for track in tracks:
            # 获取对应真实目标
            if track.id not in true_targets:
                continue

            true_pos = true_targets[track.id]

            # 位置误差
            pos_error = np.sqrt(
                (track.position.x - true_pos.x)**2 +
                (track.position.y - true_pos.y)**2 +
                (track.position.z - true_pos.z)**2
            )
            position_errors.append(pos_error)

            # 速度误差（假设真实目标有速度）
            # velocity_errors.append(...)

        if not position_errors:
            return TrackingError()

        # 计算统计量
        position_rmse = np.sqrt(np.mean(np.array(position_errors)**2))
        position_bias = np.mean(position_errors)
        max_error = np.max(position_errors)

        # 速度误差（简化）
        velocity_rmse = 0.0  # TODO: 完整实现
        velocity_bias = 0.0

        return TrackingError(
            position_rmse=position_rmse,
            velocity_rmse=velocity_rmse,
            position_bias=position_bias,
            velocity_bias=velocity_bias,
            max_position_error=max_error
        )

    def _analyze_schedule(self) -> ScheduleAnalysis:
        """
        分析调度性能

        Returns:
            调度分析结果
        """
        if not self._schedule_history:
            return ScheduleAnalysis()

        # 计算平均利用率
        utilizations = [s['utilization'] for s in self._schedule_history]
        avg_util = np.mean(utilizations) if utilizations else 0.0

        # 计算成功率
        total_tasks = sum(s['total_tasks'] for s in self._schedule_history)
        scheduled_tasks = sum(s['scheduled_tasks'] for s in self._schedule_history)
        success_rate = scheduled_tasks / total_tasks if total_tasks > 0 else 0.0

        # 延迟和丢弃统计
        delayed = sum(s['delayed'] for s in self._schedule_history)
        dropped = sum(s['dropped'] for s in self._schedule_history)

        # 平均延迟
        delays = [s['avg_delay'] for s in self._schedule_history if s['avg_delay'] > 0]
        avg_delay = np.mean(delays) if delays else 0.0

        return ScheduleAnalysis(
            utilization=avg_util,
            success_rate=success_rate,
            dropped_tasks=dropped,
            delayed_tasks=delayed,
            avg_delay=avg_delay
        )

    def _calculate_detection_metrics(self) -> DetectionMetrics:
        """
        计算检测指标

        Returns:
            检测指标
        """
        if self._total_opportunities == 0:
            return DetectionMetrics()

        # 检测概率
        pd = self._detection_count / self._total_opportunities

        # 虚警率
        far = self._false_alarm_count / max(self._detection_count, 1)

        # SNR统计
        if self._snr_values:
            snr_mean = np.mean(self._snr_values)
            snr_std = np.std(self._snr_values)
        else:
            snr_mean = 0.0
            snr_std = 0.0

        return DetectionMetrics(
            detection_probability=pd,
            false_alarm_rate=far,
            snr_mean=snr_mean,
            snr_std=snr_std
        )

    def record_detection(self, snr: float, is_target: bool) -> None:
        """
        记录检测事件

        Args:
            snr: 信噪比 [dB]
            is_target: 是否真实目标
        """
        self._snr_values.append(snr)
        self._total_opportunities += 1

        if is_target:
            self._detection_count += 1
        else:
            self._false_alarm_count += 1

    def record_schedule(self, schedule_result: Dict) -> None:
        """
        记录调度结果

        Args:
            schedule_result: 调度结果字典
        """
        self._schedule_history.append(schedule_result)

        # 限制历史长度
        if len(self._schedule_history) > 1000:
            self._schedule_history.pop(0)

    def get_summary(self) -> Dict:
        """获取评估摘要"""
        tracking_error = self._evaluate_tracking_error([], {})
        schedule_analysis = self._analyze_schedule()
        detection_metrics = self._calculate_detection_metrics()

        return {
            'tracking': {
                'position_rmse_m': tracking_error.position_rmse,
                'velocity_rmse_ms': tracking_error.velocity_rmse,
            },
            'schedule': {
                'utilization_percent': schedule_analysis.utilization * 100,
                'success_rate_percent': schedule_analysis.success_rate * 100,
            },
            'detection': {
                'probability': detection_metrics.detection_probability,
                'false_alarm_rate': detection_metrics.false_alarm_rate,
                'mean_snr_db': detection_metrics.snr_mean,
            },
        }


__all__ = [
    "TrackingError",
    "ScheduleAnalysis",
    "DetectionMetrics",
    "Evaluator",
]
