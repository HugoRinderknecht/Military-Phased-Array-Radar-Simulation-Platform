# scheduler.py - 资源调度器
"""
本模块实现雷达资源调度功能。

资源调度负责：
- 任务管理
- 优先级计算
- 调度策略
- 时间资源分配
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from heapq import heappush, heappop

from radar.common.logger import get_logger
from radar.common.types import TaskType, Position3D


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"        # 待调度
    SCHEDULED = "scheduled"    # 已调度
    EXECUTING = "executing"    # 执行中
    COMPLETED = "completed"    # 已完成
    DELAYED = "delayed"        # 延迟
    DELETED = "deleted"        # 删除


@dataclass
class ScheduleTask:
    """
    调度任务

    Attributes:
        id: 任务ID
        type: 任务类型
        priority: 优先级（数值越大越重要）
        dwell_time: 驻留时间 [微秒]
        deadline: 截止时间 [微秒]
        time_window: 时间窗 (front, back)
        direction_az: 波束方位角 [弧度]
        direction_el: 波束俯仰角 [弧度]
        target_id: 目标ID（如果适用）
        status: 任务状态
        can_delay: 是否可延迟
        delay_count: 延迟次数
        creation_time: 创建时间 [微秒]
    """
    id: int
    type: TaskType
    priority: float
    dwell_time: int
    deadline: int
    time_window: Tuple[int, int] = (0, int(1e9))
    direction_az: float = 0.0
    direction_el: float = 0.0
    target_id: int = 0
    status: TaskStatus = TaskStatus.PENDING
    can_delay: bool = True
    delay_count: int = 0
    creation_time: int = 0

    def __lt__(self, other):
        """优先级比较（用于堆排序）"""
        return self.priority < other.priority


@dataclass
class ScheduleResult:
    """
    调度结果

    Attributes:
        execute_queue: 执行队列
        delay_queue: 延迟队列
        delete_queue: 删除队列
        time_used: 使用的时间 [微秒]
        time_available: 剩余时间 [微秒]
        utilization: 时间利用率
    """
    execute_queue: List[ScheduleTask] = field(default_factory=list)
    delay_queue: List[ScheduleTask] = field(default_factory=list)
    delete_queue: List[ScheduleTask] = field(default_factory=list)
    time_used: int = 0
    time_available: int = 0
    utilization: float = 0.0


class Scheduler:
    """
    资源调度器

    实现自适应调度算法。
    """

    def __init__(self, schedule_period: int = 50000):  # 50ms
        """
        初始化调度器

        Args:
            schedule_period: 调度周期 [微秒]
        """
        self._logger = get_logger("scheduler")
        self._schedule_period = schedule_period

        # 任务管理
        self._tasks: Dict[int, ScheduleTask] = {}
        self._task_id_counter = 0
        self._task_heap: List[ScheduleTask] = []

        # 当前时间
        self._current_time = 0

        # 统计信息
        self._stats = {
            'total_scheduled': 0,
            'total_executed': 0,
            'total_delayed': 0,
            'total_deleted': 0,
            'average_utilization': 0.0,
        }

        self._logger.info(
            f"调度器初始化: 周期={schedule_period/1000}ms"
        )

    async def add_task(self, task_type: TaskType,
                     priority: float, dwell_time: int,
                     direction_az: float, direction_el: float,
                     deadline: Optional[int] = None,
                     target_id: int = 0) -> int:
        """
        添加任务

        Args:
            task_type: 任务类型
            priority: 优先级
            dwell_time: 驻留时间 [微秒]
            direction_az: 方位角 [弧度]
            direction_el: 俯仰角 [弧度]
            deadline: 截止时间 [微秒]
            target_id: 目标ID

        Returns:
            任务ID
        """
        self._task_id_counter += 1
        task_id = self._task_id_counter

        # 设置时间窗
        if deadline is None:
            deadline = self._current_time + int(1e6)  # 默认1秒

        # 创建任务
        task = ScheduleTask(
            id=task_id,
            type=task_type,
            priority=priority,
            dwell_time=dwell_time,
            deadline=deadline,
            time_window=(self._current_time, deadline),
            direction_az=direction_az,
            direction_el=direction_el,
            target_id=target_id,
            creation_time=self._current_time
        )

        # 添加到任务列表和优先队列
        self._tasks[task_id] = task
        heappush(self._task_heap, task)

        self._logger.info(
            f"添加任务: ID={task_id}, type={task_type.value}, "
            f"priority={priority}, dwell={dwell_time/1000}ms"
        )

        return task_id

    def remove_task(self, task_id: int) -> bool:
        """
        移除任务

        Args:
            task_id: 任务ID

        Returns:
            是否成功移除
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            self._logger.info(f"移除任务: ID={task_id}")
            return True
        return False

    async def schedule(self, current_time: int) -> ScheduleResult:
        """
        执行调度

        Args:
            current_time: 当前时间 [微秒]

        Returns:
            调度结果
        """
        self._current_time = current_time

        # 获取所有待调度任务
        pending_tasks = [
            task for task in self._tasks.values()
            if task.status == TaskStatus.PENDING
        ]

        if not pending_tasks:
            return ScheduleResult(time_available=self._schedule_period)

        # 按优先级排序
        pending_tasks.sort(key=lambda t: t.priority, reverse=True)

        # 调度算法：时间窗调度
        result = await self._adaptive_schedule(pending_tasks)

        # 更新任务状态
        for task in result.execute_queue:
            task.status = TaskStatus.SCHEDULED
            self._stats['total_scheduled'] += 1

        for task in result.delay_queue:
            task.status = TaskStatus.DELAYED
            task.delay_count += 1
            self._stats['total_delayed'] += 1

        for task in result.delete_queue:
            task.status = TaskStatus.DELETED
            self._stats['total_deleted'] += 1

        # 记录统计
        self._stats['average_utilization'] = (
            0.95 * self._stats['average_utilization'] +
            0.05 * result.utilization
        )

        self._logger.debug(
            f"调度结果: 执行={len(result.execute_queue)}, "
            f"延迟={len(result.delay_queue)}, "
            f"删除={len(result.delete_queue)}, "
            f"利用率={result.utilization:.1%}"
        )

        return result

    async def _adaptive_schedule(self, tasks: List[ScheduleTask]) -> ScheduleResult:
        """
        自适应调度算法

        Args:
            tasks: 待调度任务列表

        Returns:
            调度结果
        """
        execute_queue = []
        delay_queue = []
        delete_queue = []

        time_pointer = 0  # 已用时间 [微秒]
        period_end = self._schedule_period

        for task in tasks:
            # 计算可调度时间范围
            earliest = max(time_pointer, task.time_window[0])
            latest = min(period_end - task.dwell_time, task.time_window[1])

            if earliest <= latest:
                # 可以调度
                task.scheduled_time = earliest
                execute_queue.append(task)
                time_pointer = earliest + task.dwell_time

            elif task.can_delay:
                # 延迟到下一周期
                delay_queue.append(task)
            else:
                # 删除
                delete_queue.append(task)

        return ScheduleResult(
            execute_queue=execute_queue,
            delay_queue=delay_queue,
            delete_queue=delete_queue,
            time_used=time_pointer,
            time_available=period_end - time_pointer,
            utilization=time_pointer / period_end if period_end > 0 else 0
        )

    def update_task_priority(self, task_id: int, new_priority: float) -> bool:
        """
        更新任务优先级

        Args:
            task_id: 任务ID
            new_priority: 新优先级

        Returns:
            是否成功更新
        """
        if task_id in self._tasks:
            task = self._tasks[task_id]
            task.priority = new_priority

            # 重新加入堆
            heappush(self._task_heap, task)

            self._logger.info(f"更新优先级: ID={task_id}, priority={new_priority}")
            return True
        return False

    def get_task(self, task_id: int) -> Optional[ScheduleTask]:
        """获取任务"""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> List[ScheduleTask]:
        """获取所有任务"""
        return list(self._tasks.values())

    def get_statistics(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'total_tasks': len(self._tasks),
            'scheduled': self._stats['total_scheduled'],
            'delayed': self._stats['total_delayed'],
            'deleted': self._stats['total_deleted'],
            'utilization': self._stats['average_utilization'],
        }


class PriorityCalculator:
    """
    优先级计算器

    实现多种优先级计算方法。
    """

    @staticmethod
    def static_priority(task_type: TaskType) -> float:
        """
        静态优先级

        根据任务类型分配固定优先级。

        Args:
            task_type: 任务类型

        Returns:
            优先级值
        """
        priorities = {
            TaskType.GUIDANCE: 100.0,
            TaskType.MISSILE: 95.0,
            TaskType.TRACK: 80.0,
            TaskType.ACQUIRE: 70.0,
            TaskType.VERIFY: 60.0,
            TaskType.SEARCH: 50.0,
            TaskType.LOSS: 40.0,
            TaskType.CALIBRATE: 30.0,
        }
        return priorities.get(task_type, 50.0)

    @staticmethod
    def edf_priority(task: ScheduleTask, current_time: int) -> float:
        """
        最早截止时间优先 (EDF)

        Args:
            task: 任务
            current_time: 当前时间

        Returns:
            优先级值
        """
        slack = task.deadline - current_time
        if slack <= 0:
            return float('inf')  # 已超时，最高优先级
        return 1.0 / slack

    @staticmethod
    def dynamic_priority(task: ScheduleTask,
                      current_time: int,
                      age_weight: float = 0.1,
                      deadline_weight: float = 0.5,
                      type_weight: float = 0.4) -> float:
        """
        动态优先级

        综合考虑：
        - 任务年龄
        - 截止时间紧迫度
        - 任务类型重要性

        Args:
            task: 任务
            current_time: 当前时间
            age_weight: 年龄权重
            deadline_weight: 截止时间权重
            type_weight: 类型权重

        Returns:
            优先级值
        """
        # 任务年龄
        age = (current_time - task.creation_time) / 1e6  # 秒
        age_score = age * age_weight

        # 截止时间紧迫度
        time_to_deadline = (task.deadline - current_time) / 1e6  # 秒
        if time_to_deadline <= 0:
            deadline_score = 100.0
        else:
            deadline_score = 100.0 / time_to_deadline

        deadline_score *= deadline_weight

        # 任务类型重要性
        type_score = PriorityCalculator.static_priority(task.type)
        type_score *= type_weight

        # 总优先级
        total_priority = age_score + deadline_score + type_score

        return total_priority


__all__ = [
    "TaskStatus",
    "ScheduleTask",
    "ScheduleResult",
    "Scheduler",
    "PriorityCalculator",
]
