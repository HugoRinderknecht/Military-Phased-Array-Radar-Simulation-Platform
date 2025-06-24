from core.resource_scheduler import ResourceScheduler, RadarTask, TaskType
import logging

logger = logging.getLogger(__name__)


class SchedulingService:
    def __init__(self):
        self.schedulers = {}

    def get_scheduler(self, config: dict) -> ResourceScheduler:
        """获取或创建资源调度器"""
        config_hash = hash(frozenset(config.items()))

        if config_hash not in self.schedulers:
            # 直接从配置字典创建调度器，不再需要SchedulingConfig
            schedule_interval = config.get('schedule_interval', 60.0)
            self.schedulers[config_hash] = ResourceScheduler(schedule_interval)
            logger.info(f"Created new ResourceScheduler for config {config_hash}")

        return self.schedulers[config_hash]

    def schedule_resources(self, tasks_data: list, scheduler_config: dict):
        """调度资源"""
        scheduler = self.get_scheduler(scheduler_config)

        # 转换任务数据为RadarTask对象
        tasks = []
        for task_data in tasks_data:
            task = RadarTask(
                task_id=task_data['task_id'],
                task_type=TaskType[task_data['task_type']],
                duration=task_data['duration'],
                release_time=task_data['release_time'],
                due_time=task_data['due_time'],
                priority=0.0,  # 初始优先级设为0，调度器会重新计算
                target_id=task_data.get('target_id'),
                beam_position=task_data.get('beam_position'),
                hard_constraint=task_data.get('hard_constraint', False)
            )
            tasks.append(task)

        return scheduler.schedule_resources(tasks)

    def get_scheduler_status(self, scheduler_config: dict):
        """获取调度器状态"""
        scheduler = self.get_scheduler(scheduler_config)
        return scheduler.get_scheduler_status()
