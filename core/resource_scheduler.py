import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import heapq


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
    def __init__(self, schedule_interval: float = 60.0):
        self.schedule_interval = schedule_interval
        self.current_time = 0.0
        self.task_queue = []

    def schedule_resources(self, tasks: List[RadarTask], strategy: str = "priority") -> ScheduleResult:
        for task in tasks:
            task.priority = self._calculate_dynamic_priority(task)

        if strategy == "priority":
            return self._priority_based_scheduling(tasks)
        else:
            return self._time_pointer_scheduling(tasks)

    def _calculate_dynamic_priority(self, task: RadarTask) -> float:
        base_priority = task.task_type.value

        time_factor = 1.0
        if task.due_time > self.current_time:
            time_to_deadline = task.due_time - self.current_time
            time_factor = 1.0 + (self.schedule_interval / time_to_deadline)
        else:
            time_factor = 10.0

        env_factor = 1.0
        if task.task_type in [TaskType.TARGET_CONFIRMATION, TaskType.HIGH_PRIORITY_TRACKING]:
            env_factor *= 1.5

        comprehensive_priority = 0.5 * base_priority + 0.3 * time_factor + 0.2 * env_factor

        return comprehensive_priority

    def _priority_based_scheduling(self, tasks: List[RadarTask]) -> ScheduleResult:
        tasks_sorted = sorted(tasks, key=lambda t: t.priority)

        scheduled = []
        delayed = []
        cancelled = []
        current_time = self.current_time

        for task in tasks_sorted:
            if current_time + task.duration > task.due_time:
                if task.hard_constraint:
                    scheduled.append(task)
                    current_time += task.duration
                else:
                    if task.due_time - current_time > task.duration * 0.5:
                        delayed.append(task)
                    else:
                        cancelled.append(task)
            else:
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
        time_pointer = self.current_time
        available_tasks = tasks.copy()
        scheduled = []
        delayed = []
        cancelled = []

        while time_pointer < self.current_time + self.schedule_interval and available_tasks:
            best_task = None
            best_utility = -1.0
            best_index = -1

            for i, task in enumerate(available_tasks):
                if (task.release_time <= time_pointer and
                        task.due_time >= time_pointer + task.duration):

                    utility = self._calculate_task_utility(task, time_pointer)
                    if utility > best_utility:
                        best_utility = utility
                        best_task = task
                        best_index = i

            if best_task:
                scheduled.append(best_task)
                time_pointer += best_task.duration
                available_tasks.pop(best_index)
            else:
                time_pointer += 1.0

        for task in available_tasks:
            if task.due_time > time_pointer:
                delayed.append(task)
            else:
                cancelled.append(task)

        efficiency = len(scheduled) / len(tasks) if tasks else 0.0

        return ScheduleResult(
            scheduled_tasks=scheduled,
            delayed_tasks=delayed,
            cancelled_tasks=cancelled,
            total_time=time_pointer - self.current_time,
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
        heapq.heappush(self.task_queue, (task.priority, task))

    def update_time(self, new_time: float):
        self.current_time = new_time
