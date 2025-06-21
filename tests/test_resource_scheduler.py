import pytest
import numpy as np
from core.resource_scheduler import ResourceScheduler, RadarTask, TaskType, ScheduleResult
from models.environment import Environment, WeatherCondition


@pytest.fixture
def scheduler():
    return ResourceScheduler(schedule_interval=60.0)


@pytest.fixture
def sample_tasks():
    return [
        RadarTask(
            task_id=1,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=5.0,
            release_time=0.0,
            due_time=20.0,
            priority=1.0,
            target_id=101,
            hard_constraint=True
        ),
        RadarTask(
            task_id=2,
            task_type=TaskType.HIGH_PRIORITY_TRACKING,
            duration=3.0,
            release_time=2.0,
            due_time=15.0,
            priority=2.0,
            target_id=102,
            hard_constraint=True
        ),
        RadarTask(
            task_id=3,
            task_type=TaskType.NORMAL_TRACKING,
            duration=2.0,
            release_time=5.0,
            due_time=30.0,
            priority=5.0,
            target_id=103,
            hard_constraint=False
        ),
        RadarTask(
            task_id=4,
            task_type=TaskType.AREA_SEARCH,
            duration=10.0,
            release_time=0.0,
            due_time=50.0,
            priority=6.0,
            hard_constraint=False
        ),
        RadarTask(
            task_id=5,
            task_type=TaskType.WEAK_TARGET_TRACKING,
            duration=4.0,
            release_time=8.0,
            due_time=25.0,
            priority=4.0,
            target_id=104,
            hard_constraint=False
        )
    ]


@pytest.fixture
def overloaded_tasks():
    """创建超载的任务集合"""
    tasks = []
    for i in range(20):
        task = RadarTask(
            task_id=i + 1,
            task_type=TaskType.NORMAL_TRACKING,
            duration=5.0,
            release_time=0.0,
            due_time=30.0,
            priority=float(i),
            hard_constraint=(i < 5)  # 前5个是硬约束
        )
        tasks.append(task)
    return tasks


class TestResourceScheduler:

    def test_initialization(self, scheduler):
        """测试调度器初始化"""
        assert scheduler.schedule_interval == 60.0
        assert scheduler.current_time == 0.0
        assert scheduler.task_queue == []

    def test_priority_calculation(self, scheduler, sample_tasks):
        """测试优先级计算"""
        task = sample_tasks[0]  # TARGET_CONFIRMATION任务

        # 模拟环境
        scheduler.current_time = 5.0

        original_priority = task.priority
        calculated_priority = scheduler._calculate_dynamic_priority(task)

        assert calculated_priority > 0
        assert isinstance(calculated_priority, float)

        # 过期任务应该有很高的优先级
        overdue_task = RadarTask(
            task_id=99,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=5.0,
            release_time=0.0,
            due_time=2.0,  # 已过期
            priority=1.0
        )

        overdue_priority = scheduler._calculate_dynamic_priority(overdue_task)
        assert overdue_priority > calculated_priority

    def test_priority_based_scheduling_simple(self, scheduler, sample_tasks):
        """测试基于优先级的调度（简单情况）"""
        result = scheduler._priority_based_scheduling(sample_tasks)

        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) > 0
        assert result.efficiency > 0

        # 验证调度结果
        total_tasks = (len(result.scheduled_tasks) +
                       len(result.delayed_tasks) +
                       len(result.cancelled_tasks))
        assert total_tasks == len(sample_tasks)

        # 硬约束任务应该被优先调度
        scheduled_hard_tasks = [task for task in result.scheduled_tasks if task.hard_constraint]
        assert len(scheduled_hard_tasks) > 0

    def test_priority_based_scheduling_overload(self, scheduler, overloaded_tasks):
        """测试资源过载情况下的优先级调度"""
        result = scheduler._priority_based_scheduling(overloaded_tasks)

        # 在过载情况下，应该有任务被延迟或取消
        assert len(result.delayed_tasks) > 0 or len(result.cancelled_tasks) > 0
        assert result.efficiency < 1.0

        # 硬约束任务应该被优先保护
        scheduled_hard_tasks = [task for task in result.scheduled_tasks if task.hard_constraint]
        cancelled_hard_tasks = [task for task in result.cancelled_tasks if task.hard_constraint]

        # 硬约束任务的取消应该尽量避免
        assert len(cancelled_hard_tasks) <= len(scheduled_hard_tasks)

    def test_time_pointer_scheduling_simple(self, scheduler, sample_tasks):
        """测试基于时间指针的调度（简单情况）"""
        result = scheduler._time_pointer_scheduling(sample_tasks)

        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) > 0
        assert result.efficiency > 0

        # 验证时间约束
        current_time = 0.0
        for task in result.scheduled_tasks:
            assert task.release_time <= current_time + task.duration
            assert task.due_time >= current_time + task.duration
            current_time += task.duration

    def test_time_pointer_scheduling_with_timing(self, scheduler):
        """测试时间指针调度的时序性"""
        # 创建有明确时序要求的任务
        tasks = [
            RadarTask(
                task_id=1,
                task_type=TaskType.TARGET_CONFIRMATION,
                duration=5.0,
                release_time=10.0,  # 10秒后才能开始
                due_time=25.0,
                priority=1.0
            ),
            RadarTask(
                task_id=2,
                task_type=TaskType.NORMAL_TRACKING,
                duration=3.0,
                release_time=0.0,  # 立即可开始
                due_time=15.0,
                priority=3.0
            )
        ]

        result = scheduler._time_pointer_scheduling(tasks)

        # 第二个任务应该先被调度（可以立即开始）
        if len(result.scheduled_tasks) >= 2:
            assert result.scheduled_tasks[0].task_id == 2
            assert result.scheduled_tasks[1].task_id == 1

    def test_task_utility_calculation(self, scheduler):
        """测试任务效用函数计算"""
        task = RadarTask(
            task_id=1,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=5.0,
            release_time=0.0,
            due_time=20.0,
            priority=1.0
        )

        current_time = 5.0
        utility = scheduler._calculate_task_utility(task, current_time)

        assert utility > 0
        assert isinstance(utility, float)

        # 接近截止期的任务应该有更高的效用
        urgent_time = 18.0
        urgent_utility = scheduler._calculate_task_utility(task, urgent_time)
        assert urgent_utility > utility

    def test_schedule_resources_priority_strategy(self, scheduler, sample_tasks):
        """测试主调度函数（优先级策略）"""
        result = scheduler.schedule_resources(sample_tasks, strategy="priority")

        assert isinstance(result, ScheduleResult)
        assert hasattr(result, 'scheduled_tasks')
        assert hasattr(result, 'delayed_tasks')
        assert hasattr(result, 'cancelled_tasks')

        # 验证优先级计算已执行
        for task in sample_tasks:
            assert hasattr(task, 'priority')
            assert task.priority > 0

    def test_schedule_resources_time_pointer_strategy(self, scheduler, sample_tasks):
        """测试主调度函数（时间指针策略）"""
        result = scheduler.schedule_resources(sample_tasks, strategy="time_pointer")

        assert isinstance(result, ScheduleResult)
        assert len(result.scheduled_tasks) >= 0

    def test_add_task_to_queue(self, scheduler):
        """测试任务队列管理"""
        task = RadarTask(
            task_id=1,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=5.0,
            release_time=0.0,
            due_time=20.0,
            priority=1.0
        )

        assert len(scheduler.task_queue) == 0
        scheduler.add_task(task)
        assert len(scheduler.task_queue) == 1

    def test_time_update(self, scheduler):
        """测试时间更新"""
        initial_time = scheduler.current_time
        new_time = 100.0

        scheduler.update_time(new_time)
        assert scheduler.current_time == new_time
        assert scheduler.current_time != initial_time

    def test_scheduling_with_different_task_types(self, scheduler):
        """测试不同任务类型的调度"""
        tasks = []
        for task_type in TaskType:
            task = RadarTask(
                task_id=task_type.value,
                task_type=task_type,
                duration=2.0,
                release_time=0.0,
                due_time=30.0,
                priority=float(task_type.value)
            )
            tasks.append(task)

        result = scheduler.schedule_resources(tasks, strategy="priority")

        # 优先级高的任务应该被优先调度
        if len(result.scheduled_tasks) > 1:
            priorities = [task.priority for task in result.scheduled_tasks]
            # 优先级数字越小，优先级越高
            assert priorities == sorted(priorities)

    def test_hard_constraint_enforcement(self, scheduler):
        """测试硬约束任务的强制执行"""
        # 创建一个会导致超时的硬约束任务
        hard_task = RadarTask(
            task_id=1,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=70.0,  # 超过调度窗口
            release_time=0.0,
            due_time=10.0,  # 很紧的截止期
            priority=1.0,
            hard_constraint=True
        )

        soft_task = RadarTask(
            task_id=2,
            task_type=TaskType.AREA_SEARCH,
            duration=5.0,
            release_time=0.0,
            due_time=20.0,
            priority=5.0,
            hard_constraint=False
        )

        result = scheduler._priority_based_scheduling([hard_task, soft_task])

        # 硬约束任务应该被调度，即使超时
        hard_scheduled = any(task.task_id == 1 for task in result.scheduled_tasks)
        assert hard_scheduled

    def test_efficiency_calculation(self, scheduler, sample_tasks):
        """测试调度效率计算"""
        result = scheduler._priority_based_scheduling(sample_tasks)

        expected_efficiency = len(result.scheduled_tasks) / len(sample_tasks)
        assert abs(result.efficiency - expected_efficiency) < 1e-10

        assert 0 <= result.efficiency <= 1.0

    def test_empty_task_list(self, scheduler):
        """测试空任务列表处理"""
        empty_tasks = []

        result_priority = scheduler._priority_based_scheduling(empty_tasks)
        result_time_pointer = scheduler._time_pointer_scheduling(empty_tasks)

        assert len(result_priority.scheduled_tasks) == 0
        assert len(result_time_pointer.scheduled_tasks) == 0
        assert result_priority.efficiency == 0.0
        assert result_time_pointer.efficiency == 0.0


class TestTaskTypes:

    def test_task_type_enum(self):
        """测试任务类型枚举"""
        assert TaskType.TARGET_CONFIRMATION.value == 1
        assert TaskType.HIGH_PRIORITY_TRACKING.value == 2
        assert TaskType.LOST_TARGET_SEARCH.value == 3
        assert TaskType.WEAK_TARGET_TRACKING.value == 4
        assert TaskType.NORMAL_TRACKING.value == 5
        assert TaskType.AREA_SEARCH.value == 6

    def test_task_creation(self):
        """测试任务创建"""
        task = RadarTask(
            task_id=1,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=5.0,
            release_time=0.0,
            due_time=20.0,
            priority=1.0,
            target_id=101,
            hard_constraint=True
        )

        assert task.task_id == 1
        assert task.task_type == TaskType.TARGET_CONFIRMATION
        assert task.duration == 5.0
        assert task.hard_constraint == True
        assert task.target_id == 101


class TestScheduleResult:

    def test_schedule_result_creation(self):
        """测试调度结果创建"""
        task1 = RadarTask(
            task_id=1,
            task_type=TaskType.TARGET_CONFIRMATION,
            duration=5.0,
            release_time=0.0,
            due_time=20.0,
            priority=1.0
        )

        result = ScheduleResult(
            scheduled_tasks=[task1],
            delayed_tasks=[],
            cancelled_tasks=[],
            total_time=5.0,
            efficiency=1.0
        )

        assert len(result.scheduled_tasks) == 1
        assert len(result.delayed_tasks) == 0
        assert len(result.cancelled_tasks) == 0
        assert result.total_time == 5.0
        assert result.efficiency == 1.0


class TestIntegrationScenarios:

    def test_realistic_air_defense_scenario(self, scheduler):
        """测试现实的防空场景"""
        # 模拟一个现实的防空雷达调度场景
        tasks = [
            # 高优先级威胁确认
            RadarTask(1, TaskType.TARGET_CONFIRMATION, 3.0, 0.0, 10.0, 1.0, 201, True),
            RadarTask(2, TaskType.TARGET_CONFIRMATION, 3.0, 1.0, 12.0, 1.0, 202, True),

            # 高价值目标跟踪
            RadarTask(3, TaskType.HIGH_PRIORITY_TRACKING, 2.0, 0.0, 8.0, 2.0, 301, True),
            RadarTask(4, TaskType.HIGH_PRIORITY_TRACKING, 2.0, 3.0, 15.0, 2.0, 302, True),

            # 常规目标跟踪
            RadarTask(5, TaskType.NORMAL_TRACKING, 1.5, 0.0, 30.0, 5.0, 401, False),
            RadarTask(6, TaskType.NORMAL_TRACKING, 1.5, 2.0, 25.0, 5.0, 402, False),
            RadarTask(7, TaskType.NORMAL_TRACKING, 1.5, 4.0, 35.0, 5.0, 403, False),

            # 区域搜索
            RadarTask(8, TaskType.AREA_SEARCH, 8.0, 0.0, 60.0, 6.0, None, False),

            # 失跟搜索
            RadarTask(9, TaskType.LOST_TARGET_SEARCH, 5.0, 5.0, 20.0, 3.0, 501, True),

            # 弱目标跟踪
            RadarTask(10, TaskType.WEAK_TARGET_TRACKING, 4.0, 0.0, 40.0, 4.0, 601, False)
        ]

        # 测试优先级调度
        result_priority = scheduler.schedule_resources(tasks, strategy="priority")

        # 验证关键任务被调度
        scheduled_ids = [task.task_id for task in result_priority.scheduled_tasks]

        # 目标确认任务应该被优先调度
        confirmation_scheduled = any(tid in [1, 2] for tid in scheduled_ids)
        assert confirmation_scheduled

        # 高优先级跟踪任务应该被调度
        high_priority_scheduled = any(tid in [3, 4] for tid in scheduled_ids)
        assert high_priority_scheduled

        # 测试时间指针调度
        result_time_pointer = scheduler.schedule_resources(tasks, strategy="time_pointer")

        assert len(result_time_pointer.scheduled_tasks) >= 0
        assert result_time_pointer.efficiency >= 0

        # 比较两种策略的效果
        print(f"Priority strategy efficiency: {result_priority.efficiency:.3f}")
        print(f"Time pointer strategy efficiency: {result_time_pointer.efficiency:.3f}")
