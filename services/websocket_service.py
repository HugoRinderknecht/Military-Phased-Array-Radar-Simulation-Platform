from flask_socketio import emit, join_room, leave_room
from typing import Dict, List, Any, Optional
import uuid
import time
import threading
import logging
from datetime import datetime
from services.visualization_service import VisualizationService

logger = logging.getLogger(__name__)


class WebSocketService:
    """WebSocket服务 - 处理实时数据推送"""

    def __init__(self):
        self.active_subscriptions = {}
        self.simulation_rooms = {}
        self.update_threads = {}
        self.visualization_service = VisualizationService()
        self._lock = threading.Lock()
        # 添加线程控制变量
        self._running = False
        self._thread_stop_events = {}  # 每个仿真对应一个停止事件

    def subscribe_simulation_updates(self, simulation_id: str, interval: float,
                                     data_types: List[str]) -> str:
        """订阅仿真更新"""
        subscription_id = str(uuid.uuid4())

        with self._lock:
            self.active_subscriptions[subscription_id] = {
                'simulation_id': simulation_id,
                'interval': interval,
                'data_types': data_types,
                'created_at': time.time(),
                'last_update': 0
            }

            # 如果这是第一个订阅该仿真的客户端，启动更新线程
            if simulation_id not in self.update_threads:
                self._start_update_thread(simulation_id)

        logger.info(f"New subscription {subscription_id} for simulation {simulation_id}")
        return subscription_id

    def unsubscribe_simulation_updates(self, subscription_id: str) -> bool:
        """取消订阅仿真更新"""
        with self._lock:
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

        logger.info(f"Unsubscribed {subscription_id}")
        return True

    def broadcast_to_simulation(self, simulation_id: str, data: Dict[str, Any]) -> None:
        """向仿真房间广播数据"""
        try:
            emit('simulation_update', data, room=simulation_id)
            logger.debug(f"Broadcasted data to simulation {simulation_id}")
        except Exception as e:
            logger.error(f"Error broadcasting to simulation {simulation_id}: {str(e)}")

    def emit_simulation_stopped(self, simulation_id: str) -> None:
        """发送仿真停止事件"""
        try:
            emit('simulation_stopped', {
                'simulation_id': simulation_id,
                'timestamp': time.time()
            }, room=simulation_id)

            # 清理该仿真的所有订阅
            self._cleanup_simulation_subscriptions(simulation_id)
        except Exception as e:
            logger.error(f"Error emitting simulation stopped: {str(e)}")

    def emit_simulation_error(self, simulation_id: str, error_message: str) -> None:
        """发送仿真错误事件"""
        try:
            emit('simulation_error', {
                'simulation_id': simulation_id,
                'error': error_message,
                'timestamp': time.time()
            }, room=simulation_id)
        except Exception as e:
            logger.error(f"Error emitting simulation error: {str(e)}")

    def join_simulation_room(self, simulation_id: str, client_id: str) -> None:
        """客户端加入仿真房间"""
        try:
            join_room(simulation_id)

            with self._lock:
                if simulation_id not in self.simulation_rooms:
                    self.simulation_rooms[simulation_id] = set()
                self.simulation_rooms[simulation_id].add(client_id)

            logger.info(f"Client {client_id} joined simulation room {simulation_id}")
        except Exception as e:
            logger.error(f"Error joining simulation room: {str(e)}")

    def leave_simulation_room(self, simulation_id: str, client_id: str) -> None:
        """客户端离开仿真房间"""
        try:
            leave_room(simulation_id)

            with self._lock:
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
        with self._lock:
            subscriptions = []
            for sub_id, sub_info in self.active_subscriptions.items():
                subscriptions.append({
                    'subscription_id': sub_id,
                    'simulation_id': sub_info['simulation_id'],
                    'interval': sub_info['interval'],
                    'data_types': sub_info['data_types'],
                    'created_at': sub_info['created_at'],
                    'last_update': sub_info['last_update']
                })
            return subscriptions

    def _start_update_thread(self, simulation_id: str) -> None:
        """启动仿真更新线程"""

        # 创建该仿真专用的停止事件
        stop_event = threading.Event()
        self._thread_stop_events[simulation_id] = stop_event

        def update_loop():
            logger.info(f"Started update thread for simulation {simulation_id}")
            self._running = True

            while not stop_event.is_set():
                try:
                    # 获取该仿真的所有订阅（使用锁保护）
                    with self._lock:
                        relevant_subscriptions = [
                            (sub_id, sub_info.copy()) for sub_id, sub_info in self.active_subscriptions.items()
                            if sub_info['simulation_id'] == simulation_id
                        ]

                    if not relevant_subscriptions:
                        logger.info(f"No subscriptions left for simulation {simulation_id}, stopping thread")
                        break

                    # 获取实时数据
                    realtime_data = self.visualization_service.get_realtime_data(simulation_id)

                    if 'error' not in realtime_data:
                        # 向每个订阅者发送数据
                        for sub_id, sub_info in relevant_subscriptions:
                            current_time = time.time()

                            # 检查是否到了更新时间
                            if current_time - sub_info['last_update'] >= sub_info['interval']:
                                try:
                                    # 根据数据类型过滤数据
                                    filtered_data = self._filter_data_by_types(
                                        realtime_data['data'], sub_info['data_types']
                                    )

                                    # 广播数据
                                    self.broadcast_to_simulation(simulation_id, {
                                        'type': 'realtime_update',
                                        'data': filtered_data,
                                        'subscription_id': sub_id,
                                        'timestamp': current_time
                                    })

                                    # 更新最后更新时间（线程安全）
                                    with self._lock:
                                        if sub_id in self.active_subscriptions:
                                            self.active_subscriptions[sub_id]['last_update'] = current_time

                                except Exception as e:
                                    logger.error(f"Error processing subscription {sub_id}: {str(e)}")

                    # 等待最小间隔或停止事件
                    stop_event.wait(0.1)

                except Exception as e:
                    logger.error(f"Error in update thread for {simulation_id}: {str(e)}")
                    if not stop_event.wait(1.0):  # 错误时等待更长时间，但仍可被停止事件中断
                        continue

            self._running = False
            logger.info(f"Update thread stopped for simulation {simulation_id}")

        thread = threading.Thread(target=update_loop, daemon=True)
        self.update_threads[simulation_id] = thread
        thread.start()

    def _stop_update_thread(self, simulation_id: str) -> None:
        """停止仿真更新线程"""
        if simulation_id in self.update_threads:
            # 设置停止事件
            if simulation_id in self._thread_stop_events:
                self._thread_stop_events[simulation_id].set()

            # 等待线程结束
            thread = self.update_threads[simulation_id]
            try:
                thread.join(timeout=5.0)  # 最多等待5秒
                if thread.is_alive():
                    logger.warning(f"Thread for simulation {simulation_id} did not stop gracefully")
            except Exception as e:
                logger.error(f"Error stopping thread for simulation {simulation_id}: {str(e)}")

            # 清理线程引用和停止事件
            del self.update_threads[simulation_id]
            if simulation_id in self._thread_stop_events:
                del self._thread_stop_events[simulation_id]

            logger.info(f"Stopped update thread for simulation {simulation_id}")

    def _cleanup_simulation_subscriptions(self, simulation_id: str) -> None:
        """清理仿真相关的所有订阅"""
        with self._lock:
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

        # 停止更新线程（在锁外执行，避免死锁）
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
        with self._lock:
            simulation_ids = {sub_info['simulation_id'] for sub_info in self.active_subscriptions.values()}
            simulation_ids.update(self.simulation_rooms.keys())

        # 清理每个仿真
        for simulation_id in simulation_ids:
            self._cleanup_simulation_subscriptions(simulation_id)

        # 确保所有线程都已停止
        remaining_threads = list(self.update_threads.keys())
        for simulation_id in remaining_threads:
            self._stop_update_thread(simulation_id)

        logger.info("All WebSocket subscriptions and threads cleaned up")

    def get_thread_status(self) -> Dict[str, Any]:
        """获取线程状态信息"""
        with self._lock:
            return {
                'active_threads': len(self.update_threads),
                'thread_details': {
                    sim_id: {
                        'is_alive': thread.is_alive(),
                        'name': thread.name
                    }
                    for sim_id, thread in self.update_threads.items()
                },
                'stop_events': len(self._thread_stop_events),
                'active_subscriptions': len(self.active_subscriptions),
                'simulation_rooms': len(self.simulation_rooms)
            }
