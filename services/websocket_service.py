from flask_socketio import emit, join_room, leave_room
from typing import Dict, List, Any, Optional, Set, Tuple
import uuid
import time
import threading
import logging
from collections import defaultdict
from datetime import datetime
from services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


class WebSocketService:
    """WebSocket服务 - 处理实时数据推送"""

    def __init__(self, socketio=None):
        self.socketio = socketio
        self.active_subscriptions = {}
        self.simulation_rooms = defaultdict(set)  # 使用defaultdict简化房间管理
        self.task_subscriptions = defaultdict(set)  # 任务订阅管理
        self.update_threads = {}
        self.visualization_service = VisualizationService()
        self.lock = threading.Lock()
        self.thread_stop_events = {}
        self.running = False
        self.main_thread = None

    def set_socketio(self, socketio):
        """设置SocketIO实例"""
        self.socketio = socketio

    def subscribe_simulation_updates(self, simulation_id: str, interval: float,
                                     data_types: List[str], client_id: str) -> str:
        """订阅仿真更新"""
        subscription_id = str(uuid.uuid4())

        with self.lock:
            self.active_subscriptions[subscription_id] = {
                'simulation_id': simulation_id,
                'interval': interval,
                'data_types': data_types,
                'client_id': client_id,
                'created_at': time.time(),
                'last_update': 0
            }

            # 如果这是第一个订阅该仿真的客户端，启动更新线程
            if simulation_id not in self.update_threads:
                self._start_update_thread(simulation_id)

        logger.info(f"New subscription {subscription_id} for simulation {simulation_id} (Client: {client_id})")
        return subscription_id

    def unsubscribe_simulation_updates(self, subscription_id: str, client_id: str) -> bool:
        """取消订阅仿真更新"""
        with self.lock:
            if subscription_id not in self.active_subscriptions:
                return False

            subscription = self.active_subscriptions.pop(subscription_id)
            simulation_id = subscription['simulation_id']

            # 检查是否还有其他订阅者
            remaining_subscriptions = [
                sub for sub in self.active_subscriptions.values()
                if sub['simulation_id'] == simulation_id
            ]

            # 如果没有其他订阅者，停止更新线程
            if not remaining_subscriptions and simulation_id in self.update_threads:
                self._stop_update_thread(simulation_id)

        logger.info(f"Unsubscribed {subscription_id} (Client: {client_id})")
        return True

    def broadcast_to_simulation(self, simulation_id: str, data: Dict[str, Any]) -> None:
        """向仿真房间广播数据"""
        if not self.socketio:
            logger.warning("SocketIO instance not set, cannot broadcast")
            return

        try:
            self.socketio.emit('simulation_update', data, room=simulation_id)
            logger.debug(f"Broadcasted data to simulation {simulation_id}")
        except Exception as e:
            logger.error(f"Error broadcasting to simulation {simulation_id}: {str(e)}")

    def emit_simulation_stopped(self, simulation_id: str) -> None:
        """发送仿真停止事件"""
        if not self.socketio:
            return

        try:
            self.socketio.emit('simulation_stopped', {
                'simulation_id': simulation_id,
                'timestamp': time.time()
            }, room=simulation_id)

            # 清理该仿真的所有订阅
            self._cleanup_simulation_subscriptions(simulation_id)
        except Exception as e:
            logger.error(f"Error emitting simulation stopped: {str(e)}")

    def emit_simulation_error(self, simulation_id: str, error_message: str) -> None:
        """发送仿真错误事件"""
        if not self.socketio:
            return

        try:
            self.socketio.emit('simulation_error', {
                'simulation_id': simulation_id,
                'error': error_message,
                'timestamp': time.time()
            }, room=simulation_id)
        except Exception as e:
            logger.error(f"Error emitting simulation error: {str(e)}")

    def join_simulation_room(self, simulation_id: str, client_id: str) -> None:
        """客户端加入仿真房间"""
        try:
            if self.socketio:
                join_room(simulation_id)

            with self.lock:
                self.simulation_rooms[simulation_id].add(client_id)

            logger.info(f"Client {client_id} joined simulation room {simulation_id}")
        except Exception as e:
            logger.error(f"Error joining simulation room: {str(e)}")

    def leave_simulation_room(self, simulation_id: str, client_id: str) -> None:
        """客户端离开仿真房间"""
        try:
            if self.socketio:
                leave_room(simulation_id)

            with self.lock:
                if simulation_id in self.simulation_rooms:
                    self.simulation_rooms[simulation_id].discard(client_id)

                    # 如果房间为空，清理房间
                    if not self.simulation_rooms[simulation_id]:
                        del self.simulation_rooms[simulation_id]

            logger.info(f"Client {client_id} left simulation room {simulation_id}")
        except Exception as e:
            logger.error(f"Error leaving simulation room: {str(e)}")

    def get_active_subscriptions(self) -> List[Dict[str, Any]]:
        """获取活跃订阅列表"""
        with self.lock:
            return [{
                'subscription_id': sub_id,
                'simulation_id': info['simulation_id'],
                'interval': info['interval'],
                'data_types': info['data_types'],
                'client_id': info['client_id'],
                'created_at': info['created_at'],
                'last_update': info['last_update']
            } for sub_id, info in self.active_subscriptions.items()]

    def subscribe_task_updates(self, task_id: str, client_id: str):
        """订阅任务更新"""
        with self.lock:
            self.task_subscriptions[task_id].add(client_id)
        logger.info(f"Client {client_id} subscribed to task updates for task {task_id}")

    def unsubscribe_task_updates(self, task_id: str, client_id: str):
        """取消订阅任务更新"""
        with self.lock:
            if task_id in self.task_subscriptions and client_id in self.task_subscriptions[task_id]:
                self.task_subscriptions[task_id].discard(client_id)
        logger.info(f"Client {client_id} unsubscribed from task updates for task {task_id}")

    def broadcast_task_update(self, task_id: str, status: str, progress: float,
                              client_id: str = None, result: dict = None):
        """广播任务更新到特定客户端或所有订阅者"""
        if not self.socketio:
            logger.warning("SocketIO instance not set, cannot broadcast task update")
            return

        update_data = {
            'task_id': task_id,
            'status': status,
            'progress': progress,
            'timestamp': time.time()
        }
        if result:
            update_data['result'] = result

        # 如果指定了客户端ID，只发送给该客户端
        if client_id:
            try:
                self.socketio.emit('task_update', update_data, room=client_id)
                logger.debug(f"Sent task update to client {client_id} for task {task_id}")
            except Exception as e:
                logger.error(f"Error sending task update to client {client_id}: {str(e)}")
            return

        # 如果没有指定客户端，发送给所有订阅者
        with self.lock:
            subscribers = self.task_subscriptions.get(task_id, set()).copy()

        for sub_client_id in subscribers:
            try:
                self.socketio.emit('task_update', update_data, room=sub_client_id)
                logger.debug(f"Sent task update to client {sub_client_id} for task {task_id}")
            except Exception as e:
                logger.error(f"Error sending task update to client {sub_client_id}: {str(e)}")

    def broadcast_computation_result(self, task_id: str, result: dict, client_id: str = None):
        """广播计算结果"""
        self.broadcast_task_update(task_id, 'completed', 100.0, client_id, result)

        # 任务完成后移除所有订阅
        with self.lock:
            if task_id in self.task_subscriptions:
                del self.task_subscriptions[task_id]

    def broadcast_simulation_data(self, simulation_id: str, data_type: str, data: dict):
        """广播仿真数据到房间"""
        if not self.socketio:
            return

        packet = {
            'simulation_id': simulation_id,
            'data_type': data_type,
            'data': data,
            'timestamp': time.time()
        }

        with self.lock:
            if simulation_id in self.simulation_rooms:
                for client_id in self.simulation_rooms[simulation_id]:
                    try:
                        self.socketio.emit('simulation_data', packet, room=client_id)
                        logger.debug(f"Sent simulation data to client {client_id}")
                    except Exception as e:
                        logger.error(f"Error sending data to {client_id}: {str(e)}")

    def notify_system_resources(self, resource_data: dict):
        """广播系统资源信息"""
        if not self.socketio:
            return

        with self.lock:
            for simulation_id, clients in self.simulation_rooms.items():
                for client_id in clients:
                    try:
                        self.socketio.emit('system_resources', resource_data, room=client_id)
                    except Exception as e:
                        logger.error(f"Error sending resources to {client_id}: {str(e)}")

    def start_update_thread(self):
        """启动主更新线程"""
        if self.running:
            return

        self.running = True
        self.main_thread = threading.Thread(target=self._main_update_loop, daemon=True)
        self.main_thread.start()
        logger.info("WebSocket main update thread started")

    def _main_update_loop(self):
        """主更新循环"""
        while self.running:
            try:
                # 检查所有活跃的仿真
                with self.lock:
                    simulation_ids = list(self.update_threads.keys())

                # 为每个仿真处理更新
                for sim_id in simulation_ids:
                    self._process_simulation_updates(sim_id)

                # 检查任务更新
                self._process_task_updates()

                time.sleep(0.5)  # 主循环间隔

            except Exception as e:
                logger.error(f"Error in main update loop: {str(e)}")
                time.sleep(5)

    def _process_simulation_updates(self, simulation_id: str):
        """处理特定仿真的更新"""
        try:
            # 获取该仿真的所有订阅
            with self.lock:
                relevant_subscriptions = [
                    (sub_id, sub_info) for sub_id, sub_info in self.active_subscriptions.items()
                    if sub_info['simulation_id'] == simulation_id
                ]

            if not relevant_subscriptions:
                return

            # 获取仿真数据（这里需要从您的服务中获取）
            # 实际实现中应调用仿真服务获取数据
            simulation_data = self.visualization_service.get_realtime_data(simulation_id)

            if 'error' in simulation_data:
                logger.error(f"Error getting realtime data for {simulation_id}: {simulation_data['error']}")
                return

            # 广播数据
            for sub_id, sub_info in relevant_subscriptions:
                current_time = time.time()

                # 检查是否到了更新时间
                if current_time - sub_info['last_update'] >= sub_info['interval']:
                    try:
                        # 根据数据类型过滤数据
                        filtered_data = self._filter_data_by_types(
                            simulation_data['data'], sub_info['data_types']
                        )

                        # 创建数据包
                        packet = {
                            'type': 'realtime_update',
                            'data': filtered_data,
                            'subscription_id': sub_id,
                            'timestamp': current_time
                        }

                        # 发送给客户端
                        if self.socketio:
                            self.socketio.emit('simulation_update', packet, room=simulation_id)

                        # 更新最后更新时间
                        with self.lock:
                            if sub_id in self.active_subscriptions:
                                self.active_subscriptions[sub_id]['last_update'] = current_time

                    except Exception as e:
                        logger.error(f"Error processing subscription {sub_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error processing updates for simulation {simulation_id}: {str(e)}")

    def _process_task_updates(self):
        """处理任务状态更新"""
        # 实际实现中应从批处理服务获取任务状态
        # 这里简化为空实现
        pass

    # 在 websocket_service.py 中添加以下方法

    def _start_update_thread(self, simulation_id: str) -> None:
        """启动仿真更新线程 - 优化版本"""
        if simulation_id in self.update_threads:
            logger.warning(f"Update thread for {simulation_id} already exists")
            return

        # 创建该仿真专用的停止事件
        stop_event = threading.Event()
        self.thread_stop_events[simulation_id] = stop_event

        def update_loop():
            logger.info(f"Started update thread for simulation {simulation_id}")
            last_update_time = time.time()

            while not stop_event.is_set():
                try:
                    current_time = time.time()

                    # 检查是否需要更新（基于最小间隔）
                    if current_time - last_update_time >= 0.1:  # 最小100ms间隔
                        self._process_simulation_updates(simulation_id)
                        last_update_time = current_time

                    # 使用stop_event.wait()替代time.sleep()
                    if stop_event.wait(timeout=0.05):  # 50ms检查间隔
                        break

                except Exception as e:
                    logger.error(f"Error in update thread for {simulation_id}: {str(e)}")
                    if stop_event.wait(timeout=1.0):  # 出错后等待更长时间
                        break

            logger.info(f"Update thread stopped for simulation {simulation_id}")

        thread = threading.Thread(target=update_loop, daemon=True, name=f"SimUpdate-{simulation_id}")
        with self.lock:
            self.update_threads[simulation_id] = thread
        thread.start()

    def _stop_update_thread(self, simulation_id: str) -> None:
        """停止仿真更新线程"""
        if simulation_id in self.update_threads:
            # 设置停止事件
            if simulation_id in self.thread_stop_events:
                self.thread_stop_events[simulation_id].set()

            # 等待线程结束
            thread = self.update_threads[simulation_id]
            try:
                thread.join(timeout=5.0)  # 最多等待5秒
                if thread.is_alive():
                    logger.warning(f"Thread for simulation {simulation_id} did not stop gracefully")
            except Exception as e:
                logger.error(f"Error stopping thread for simulation {simulation_id}: {str(e)}")

            # 清理线程引用和停止事件
            with self.lock:
                if simulation_id in self.update_threads:
                    del self.update_threads[simulation_id]
                if simulation_id in self.thread_stop_events:
                    del self.thread_stop_events[simulation_id]

            logger.info(f"Stopped update thread for simulation {simulation_id}")

    def _cleanup_simulation_subscriptions(self, simulation_id: str) -> None:
        """清理仿真相关的所有订阅"""
        with self.lock:
            # 删除所有相关订阅
            subscriptions_to_remove = [
                sub_id for sub_id, sub_info in self.active_subscriptions.items()
                if sub_info['simulation_id'] == simulation_id
            ]

            for sub_id in subscriptions_to_remove:
                del self.active_subscriptions[sub_id]

            # 清理房间
            if simulation_id in self.simulation_rooms:
                del self.simulation_rooms[simulation_id]

        # 停止更新线程
        self._stop_update_thread(simulation_id)

        logger.info(f"Cleaned up subscriptions for simulation {simulation_id}")

    def _filter_data_by_types(self, data: Dict[str, Any], data_types: List[str]) -> Dict[str, Any]:
        """根据数据类型过滤数据"""
        if 'all' in data_types:
            return data

        filtered_data = {}
        for data_type in data_types:
            if data_type in data:
                filtered_data[data_type] = data[data_type]

        return filtered_data

    def cleanup_all_subscriptions(self):
        """清理所有订阅和线程"""
        logger.info("Cleaning up all WebSocket subscriptions and threads")

        # 获取所有仿真ID
        simulation_ids = set()
        with self.lock:
            simulation_ids = {sub_info['simulation_id'] for sub_info in self.active_subscriptions.values()}
            simulation_ids.update(self.simulation_rooms.keys())

        # 清理每个仿真
        for simulation_id in simulation_ids:
            self._cleanup_simulation_subscriptions(simulation_id)

        # 确保所有线程都已停止
        with self.lock:
            remaining_threads = list(self.update_threads.keys())

        for simulation_id in remaining_threads:
            self._stop_update_thread(simulation_id)

        # 清理任务订阅
        with self.lock:
            self.task_subscriptions.clear()

        logger.info("All WebSocket subscriptions and threads cleaned up")

    def get_thread_status(self) -> Dict[str, Any]:
        """获取线程状态信息"""
        with self.lock:
            return {
                'active_threads': len(self.update_threads),
                'thread_details': {
                    sim_id: {
                        'is_alive': thread.is_alive(),
                        'name': thread.name
                    }
                    for sim_id, thread in self.update_threads.items()
                },
                'stop_events': len(self.thread_stop_events),
                'active_subscriptions': len(self.active_subscriptions),
                'simulation_rooms': len(self.simulation_rooms),
                'task_subscriptions': len(self.task_subscriptions)
            }
