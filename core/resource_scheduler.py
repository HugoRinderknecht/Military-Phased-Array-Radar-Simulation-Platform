import heapq
import threading
from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import List, Dict, Any


class TaskType(Enum):
    TARGET_CONFIRMATION = 1
    HIGH_PRIORITY_TRACKING = 2
    LOST_TARGET_SEARCH = 3
    WEAK_TARGET_TRACKING = 4
    NORMAL_TRACKING = 5
    AREA_SEARCH = 6


@dataclass
class RadarTask:
    task_id: int
    task_type: TaskType
    duration: float
    release_time: float
    due_time: float
    priority: float
    target_id: int = None
    beam_position: Dict[str, float] = None
    hard_constraint: bool = False


@dataclass
class ScheduleResult:
    scheduled_tasks: List[RadarTask]
    delayed_tasks: List[RadarTask]
    cancelled_tasks: List[RadarTask]
    total_time: float
    efficiency: float


class ResourceScheduler:
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        return {
            'current_time': self.current_time,
            'schedule_interval': self.schedule_interval,
            'pending_tasks': len(self.task_queue),
            'active_threads': threading.active_count(),
            'scheduled_tasks_count': len(self.scheduled_tasks_history),
            'last_schedule_time': self.scheduled_tasks_history[-1][
                'timestamp'] if self.scheduled_tasks_history else None,
            'status': 'active' if self.task_queue else 'idle'
        }

    def schedule_resources(self, tasks: List[RadarTask], strategy: str = "priority") -> ScheduleResult:
        # 为每个任务计算动态优先级
        for task in tasks:
            task.priority = self._calculate_dynamic_priority(task)

        # 根据策略选择调度方法
        if strategy == "priority":
            result = self._priority_based_scheduling(tasks)
        else:
            result = self._time_pointer_scheduling(tasks)

        # 记录调度历史
        self.scheduled_tasks_history.append({
            'timestamp': time.time(),
            'result': result,
            'strategy': strategy
        })

        return result

    def _calculate_dynamic_priority(self, task: RadarTask) -> float:
        """计算动态优先级，考虑多种因素"""
        base_priority = task.task_type.value

        # 时间因子（考虑截止时间紧迫性）
        time_factor = 1.0
        if task.due_time > self.current_time:
            time_to_deadline = task.due_time - self.current_time
            if time_to_deadline <= task.duration:
                time_factor = 5.0  # 极紧急
            elif time_to_deadline <= task.duration * 2:
                time_factor = 3.0  # 紧急
            else:
                time_factor = 1.0 + (self.schedule_interval / time_to_deadline)
        else:
            time_factor = 10.0  # 已过期

        # 环境因子
        env_factor = 1.0
        if task.task_type in [TaskType.TARGET_CONFIRMATION, TaskType.HIGH_PRIORITY_TRACKING]:
            env_factor *= 1.5

        # 硬约束因子
        constraint_factor = 2.0 if task.hard_constraint else 1.0

        # 任务持续时间因子（短任务优先）
        duration_factor = 1.0 / (1.0 + task.duration / 10.0)

        # 综合优先级计算
        comprehensive_priority = (
                0.4 * base_priority * constraint_factor +
                0.3 * time_factor +
                0.2 * env_factor +
                0.1 * duration_factor
        )

        return comprehensive_priority

    def _priority_based_scheduling(self, tasks: List[RadarTask]) -> ScheduleResult:
        # 按优先级排序
        tasks_sorted = sorted(tasks, key=lambda t: t.priority, reverse=True)

        scheduled = []
        delayed = []
        cancelled = []
        current_time = self.current_time

        for task in tasks_sorted:
            # 检查任务是否能在截止时间内完成
            if current_time + task.duration > task.due_time:
                if task.hard_constraint:
                    # 硬约束任务必须执行
                    scheduled.append(task)
                    current_time += task.duration
                else:
                    # 非硬约束任务延迟或取消
                    if task.due_time - current_time > task.duration * 0.5:
                        delayed.append(task)
                    else:
                        cancelled.append(task)
            else:
                # 检查是否在调度时间窗口内
                if current_time < self.current_time + self.schedule_interval:
                    scheduled.append(task)
                    current_time += task.duration
                else:
                    delayed.append(task)

        efficiency = len(scheduled) / len(tasks) if tasks else 0.0

        return ScheduleResult(
            scheduled_tasks=scheduled,
            delayed_tasks=delayed,
            cancelled_tasks=cancelled,
            total_time=current_time - self.current_time,
            efficiency=efficiency
        )

    def _time_pointer_scheduling(self, tasks: List[RadarTask]) -> ScheduleResult:
        scheduled = []
        available_tasks = tasks.copy()
        cancelled = []
        delayed = []
        current_time = self.current_time

        # 按优先级和截止时间混合排序
        available_tasks.sort(key=lambda t: (-t.priority, t.due_time))

        while current_time < self.current_time + self.schedule_interval and available_tasks:
            # 查找当前时间点可执行的所有任务
            executable_tasks = []
            for i, task in enumerate(available_tasks):
                if (task.release_time <= current_time and
                        task.due_time >= current_time + task.duration):
                    executable_tasks.append((i, task))

            if not executable_tasks:
                # 如果没有可执行任务，前进到下一个任务的释放时间
                next_release_time = min(
                    (task.release_time for task in available_tasks if task.release_time > current_time),
                    default=current_time + 1.0
                )
                current_time = min(next_release_time, self.current_time + self.schedule_interval)
                continue

            # 处理优先级冲突：选择最佳任务
            best_task_info = self._resolve_priority_conflict(executable_tasks, current_time)

            if best_task_info:
                best_index, best_task = best_task_info

                # 检查是否会与已调度任务产生冲突
                if self._check_scheduling_conflict(best_task, scheduled, current_time):
                    # 如果有冲突，尝试重新调度或延迟
                    conflict_resolved = self._resolve_scheduling_conflict(
                        best_task, scheduled, current_time
                    )
                    if not conflict_resolved:
                        delayed.append(best_task)
                        available_tasks.remove(best_task)
                        continue

                scheduled.append(best_task)
                current_time += best_task.duration
                available_tasks.remove(best_task)
            else:
                # 没有找到合适的任务，前进时间
                current_time += 1.0

        # 处理剩余未调度的任务
        for task in available_tasks:
            if task.due_time > current_time:
                delayed.append(task)
            else:
                cancelled.append(task)

        efficiency = len(scheduled) / len(tasks) if tasks else 0.0

        return ScheduleResult(
            scheduled_tasks=scheduled,
            delayed_tasks=delayed,
            cancelled_tasks=cancelled,
            total_time=current_time - self.current_time,
            efficiency=efficiency
        )

    def _calculate_task_utility(self, task: RadarTask, current_time: float) -> float:
        base_utility = 10.0 - task.priority

        time_remaining = task.due_time - current_time
        time_utility = 1.0 / (1.0 + time_remaining / task.due_time)

        resource_utility = 1.0 / task.duration

        total_utility = base_utility * 0.5 + time_utility * 0.3 + resource_utility * 0.2

        return total_utility

    def add_task(self, task: RadarTask):
        heapq.heappush(self.task_queue, (-task.priority, task))  # 使用负值实现最大堆

    def update_time(self, new_time: float):
        self.current_time = new_time

    def _resolve_priority_conflict(self, executable_tasks: List[tuple], current_time: float) -> tuple:
        """解决优先级冲突，选择最佳任务"""
        if not executable_tasks:
            return None

        # 如果只有一个任务，直接返回
        if len(executable_tasks) == 1:
            return executable_tasks[0]

        # 计算每个任务的综合评分
        best_task_info = None
        best_score = -float('inf')

        for task_info in executable_tasks:
            index, task = task_info
            score = self._calculate_conflict_resolution_score(task, current_time)

            if score > best_score:
                best_score = score
                best_task_info = task_info

        return best_task_info

    def _calculate_conflict_resolution_score(self, task: RadarTask, current_time: float) -> float:
        """计算任务在冲突解决中的评分"""
        # 基础优先级权重
        priority_score = task.priority * 0.4

        # 紧急程度权重（基于截止时间）
        time_remaining = task.due_time - current_time
        if time_remaining <= task.duration:
            urgency_score = 10.0  # 非常紧急
        elif time_remaining <= task.duration * 2:
            urgency_score = 5.0  # 较紧急
        else:
            urgency_score = 1.0 / (time_remaining / task.duration)
        urgency_score *= 0.3

        # 硬约束任务权重
        constraint_score = 5.0 if task.hard_constraint else 0.0
        constraint_score *= 0.2

        # 任务类型权重
        type_score = self._get_task_type_weight(task.task_type) * 0.1

        total_score = priority_score + urgency_score + constraint_score + type_score

        return total_score

    def _get_task_type_weight(self, task_type: TaskType) -> float:
        """获取任务类型权重"""
        type_weights = {
            TaskType.TARGET_CONFIRMATION: 10.0,
            TaskType.HIGH_PRIORITY_TRACKING: 8.0,
            TaskType.LOST_TARGET_SEARCH: 6.0,
            TaskType.WEAK_TARGET_TRACKING: 4.0,
            TaskType.NORMAL_TRACKING: 3.0,
            TaskType.AREA_SEARCH: 2.0
        }
        return type_weights.get(task_type, 1.0)

    def _check_scheduling_conflict(self, new_task: RadarTask, scheduled_tasks: List[RadarTask],
                                   current_time: float) -> bool:
        """检查新任务是否与已调度任务产生冲突"""
        new_task_end = current_time + new_task.duration

        # 检查时间冲突
        task_start_time = self.current_time
        for scheduled_task in scheduled_tasks:
            task_end_time = task_start_time + scheduled_task.duration

            # 检查时间重叠
            if not (new_task_end <= task_start_time or current_time >= task_end_time):
                # 检查是否为资源冲突（如波束指向冲突）
                if self._has_resource_conflict(new_task, scheduled_task):
                    return True

            task_start_time = task_end_time

        return False

    def _has_resource_conflict(self, task1: RadarTask, task2: RadarTask) -> bool:
        """检查两个任务是否存在资源冲突"""
        # 如果两个任务都有波束位置信息，检查角度冲突
        if (task1.beam_position and task2.beam_position and
                'azimuth' in task1.beam_position and 'azimuth' in task2.beam_position):

            az_diff = abs(task1.beam_position['azimuth'] - task2.beam_position['azimuth'])
            el_diff = abs(task1.beam_position.get('elevation', 0) -
                          task2.beam_position.get('elevation', 0))

            # 如果角度差小于最小分离角度，则存在冲突
            min_separation = 5.0  # 度
            if az_diff < min_separation and el_diff < min_separation:
                return True

        # 检查目标冲突（同一目标不能同时执行多个任务）
        if (task1.target_id and task2.target_id and
                task1.target_id == task2.target_id):
            return True

        return False

    def _resolve_scheduling_conflict(self, new_task: RadarTask, scheduled_tasks: List[RadarTask],
                                     current_time: float) -> bool:
        """尝试解决调度冲突"""
        # 策略1：检查是否可以通过调整顺序解决冲突
        if self._can_reorder_tasks(new_task, scheduled_tasks, current_time):
            return True

        # 策略2：检查是否可以抢占低优先级任务
        if self._can_preempt_tasks(new_task, scheduled_tasks):
            return True

        # 策略3：检查是否可以分割任务
        if self._can_split_tasks(new_task, scheduled_tasks, current_time):
            return True

        return False

    def _can_reorder_tasks(self, new_task: RadarTask, scheduled_tasks: List[RadarTask],
                           current_time: float) -> bool:
        """检查是否可以通过重新排序解决冲突"""
        # 简化实现：检查是否可以将新任务插入到合适位置
        total_scheduled_time = sum(task.duration for task in scheduled_tasks)

        if (current_time + new_task.duration + total_scheduled_time <=
                self.current_time + self.schedule_interval):
            # 可以在时间窗口内完成所有任务，重新排序
            return True

        return False

    def _can_preempt_tasks(self, new_task: RadarTask, scheduled_tasks: List[RadarTask]) -> bool:
        """检查是否可以抢占低优先级任务"""
        if not new_task.hard_constraint:
            return False

        # 查找可以被抢占的任务
        for scheduled_task in scheduled_tasks:
            if (scheduled_task.priority < new_task.priority and
                    not scheduled_task.hard_constraint):
                return True

        return False

    def _can_split_tasks(self, new_task: RadarTask, scheduled_tasks: List[RadarTask],
                         current_time: float) -> bool:
        """检查是否可以通过分割任务解决冲突"""
        # 某些任务类型支持分割执行
        splittable_types = [TaskType.AREA_SEARCH, TaskType.NORMAL_TRACKING]

        if new_task.task_type in splittable_types:
            # 简化实现：检查是否有足够的时间片段
            available_time_slots = self._get_available_time_slots(scheduled_tasks, current_time)
            total_available_time = sum(slot[1] - slot[0] for slot in available_time_slots)

            if total_available_time >= new_task.duration:
                return True

        return False

    def _get_available_time_slots(self, scheduled_tasks: List[RadarTask],
                                  current_time: float) -> List[tuple]:
        """获取可用的时间槽"""
        slots = []
        task_start_time = current_time

        for scheduled_task in scheduled_tasks:
            task_end_time = task_start_time + scheduled_task.duration

            # 如果有间隙，记录为可用槽
            if task_start_time < current_time:
                slots.append((task_start_time, current_time))

            task_start_time = task_end_time

        # 添加最后的时间槽
        schedule_end = self.current_time + self.schedule_interval
        if task_start_time < schedule_end:
            slots.append((task_start_time, schedule_end))

        return slots
