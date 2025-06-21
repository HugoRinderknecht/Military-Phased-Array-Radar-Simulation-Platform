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

        def update_loop():
            logger.info(f"Started update thread for simulation {simulation_id}")

            while simulation_id in self.update_threads:
                try:
                    # 获取该仿真的所有订阅
                    relevant_subscriptions = [
                        (sub_id, sub_info) for sub_id, sub_info in self.active_subscriptions.items()
                        if sub_info['simulation_id'] == simulation_id
                    ]

                    if not relevant_subscriptions:
                        break

                    # 获取实时数据
                    realtime_data = self.visualization_service.get_realtime_data(simulation_id)

                    if 'error' not in realtime_data:
                        # 向每个订阅者发送数据
                        for sub_id, sub_info in relevant_subscriptions:
                            current_time = time.time()

                            # 检查是否到了更新时间
                            if current_time - sub_info['last_update'] >= sub_info['interval']:
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

                                # 更新最后更新时间
                                sub_info['last_update'] = current_time

                    # 等待最小间隔
                    time.sleep(0.1)

                except Exception as e:
                    logger.error(f"Error in update thread for {simulation_id}: {str(e)}")
                    time.sleep(1)  # 错误时等待更长时间

            logger.info(f"Update thread stopped for simulation {simulation_id}")

        thread = threading.Thread(target=update_loop, daemon=True)
        self.update_threads[simulation_id] = thread
        thread.start()

    def _stop_update_thread(self, simulation_id: str) -> None:
        """停止仿真更新线程"""
        if simulation_id in self.update_threads:
            # 线程会在下次循环时自动退出
            del self.update_threads[simulation_id]
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

            # 停止更新线程
            self._stop_update_thread(simulation_id)

            # 清理房间
            if simulation_id in self.simulation_rooms:
                del self.simulation_rooms[simulation_id]

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
