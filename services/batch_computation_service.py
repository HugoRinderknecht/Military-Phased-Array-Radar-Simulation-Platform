import uuid
import time
import threading
from services.signal_service import SignalService
from services.rcs_service import RCSService
from services.scheduling_service import SchedulingService
from services.enhancement_service import EnhancementService
import logging

logger = logging.getLogger(__name__)


class BatchComputationService:
    def __init__(self, websocket_service):
        self.tasks = {}
        self.lock = threading.Lock()
        self.signal_service = SignalService()
        self.rcs_service = RCSService()
        self.scheduling_service = SchedulingService()
        self.enhancement_service = EnhancementService()
        self.websocket_service = websocket_service

    def submit_async_task(self, task_type: str, task_data: dict, client_id: str = None) -> str:
        """提交异步计算任务"""
        task_id = str(uuid.uuid4())

        with self.lock:
            self.tasks[task_id] = {
                'task_id': task_id,
                'task_type': task_type,
                'client_id': client_id,
                'status': 'pending',
                'progress': 0,
                'start_time': time.time(),
                'result': None,
                'error': None
            }

        # 在后台线程中执行任务
        thread = threading.Thread(
            target=self._execute_task,
            args=(task_id, task_type, task_data, client_id)
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _execute_task(self, task_id: str, task_type: str, task_data: dict, client_id: str):
        """执行计算任务并发送进度更新"""
        try:
            # 更新任务状态为运行中
            self._update_task_status(task_id, 'running', 0, client_id)

            # 根据任务类型执行不同的处理
            result = None
            if task_type == 'signal_processing':
                result = self._process_signal_task(task_id, task_data, client_id)
            elif task_type == 'rcs_calculation':
                result = self._process_rcs_task(task_id, task_data, client_id)
            elif task_type == 'scheduling':
                result = self._process_scheduling_task(task_id, task_data, client_id)
            elif task_type == 'enhancement':
                result = self._process_enhancement_task(task_id, task_data, client_id)
            else:
                raise ValueError(f"Unsupported task type: {task_type}")

            # 更新任务状态为已完成
            self._update_task_status(task_id, 'completed', 100, client_id, result)

        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}")
            self._update_task_status(task_id, 'failed', 0, client_id, error=str(e))

    def _update_task_status(self, task_id: str, status: str, progress: float,
                            client_id: str, result: dict = None, error: str = None):
        """更新任务状态并通知客户端"""
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update({
                    'status': status,
                    'progress': progress,
                    'result': result,
                    'error': error,
                    'end_time': time.time() if status in ['completed', 'failed'] else None
                })

        # 通过WebSocket通知客户端
        if client_id:
            self.websocket_service.broadcast_task_update(
                task_id, status, progress, client_id, result
            )

    def _process_signal_task(self, task_id: str, task_data: dict, client_id: str) -> dict:
        """处理信号处理任务"""
        # 模拟处理过程
        steps = 10
        for i in range(1, steps + 1):
            time.sleep(0.5)  # 模拟处理时间
            progress = int((i / steps) * 100)
            self._update_task_status(task_id, 'running', progress, client_id)

        return {'status': 'success', 'processed_signals': len(task_data.get('signals', []))}

    def _process_rcs_task(self, task_id: str, task_data: dict, client_id: str) -> dict:
        """处理RCS计算任务"""
        # 模拟处理过程
        targets = task_data.get('targets', [])
        total = len(targets)
        results = []

        for i, target in enumerate(targets):
            time.sleep(0.3)  # 模拟处理时间
            progress = int(((i + 1) / total) * 100)
            self._update_task_status(task_id, 'running', progress, client_id)

            # 实际RCS计算逻辑
            result = {
                'target_id': target.get('id'),
                'rcs_value': 10.0,  # 模拟值
                'status': 'calculated'
            }
            results.append(result)

        return {'status': 'success', 'results': results}

    # 其他任务处理方法类似...

    def get_task_status(self, task_id: str) -> dict:
        """获取任务状态"""
        with self.lock:
            task = self.tasks.get(task_id)

        if not task:
            raise ValueError(f"Task {task_id} not found")

        return {
            'task_id': task_id,
            'status': task['status'],
            'progress': task['progress'],
            'client_id': task.get('client_id'),
            'start_time': task.get('start_time'),
            'end_time': task.get('end_time'),
            'error': task.get('error')
        }

    def get_task_result(self, task_id: str) -> dict:
        """获取任务结果"""
        with self.lock:
            task = self.tasks.get(task_id)

        if not task:
            raise ValueError(f"Task {task_id} not found")

        if task['status'] != 'completed':
            raise ValueError(f"Task {task_id} not completed")

        return {
            'task_id': task_id,
            'result': task['result'],
            'processing_time': task['end_time'] - task['start_time']
        }
