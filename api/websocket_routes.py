from flask_socketio import emit
from services.websocket_service import WebSocketService
import logging

logger = logging.getLogger(__name__)
websocket_service = WebSocketService()


def register_websocket_events(socketio):
    @socketio.on('connect')
    def handle_connect():
        logger.info('Client connected')
        websocket_service.handle_client_connect()

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info('Client disconnected')
        websocket_service.handle_client_disconnect()

    @socketio.on('join_simulation')
    def handle_join_simulation(data):
        simulation_id = data.get('simulation_id')
        client_id = data.get('client_id')
        if simulation_id and client_id:
            websocket_service.join_simulation_room(simulation_id, client_id)

    @socketio.on('leave_simulation')
    def handle_leave_simulation(data):
        simulation_id = data.get('simulation_id')
        client_id = data.get('client_id')
        if simulation_id and client_id:
            websocket_service.leave_simulation_room(simulation_id, client_id)

    @socketio.on('subscribe_task')
    def handle_subscribe_task(data):
        task_id = data.get('task_id')
        client_id = data.get('client_id')
        if task_id and client_id:
            websocket_service.subscribe_task_updates(task_id, client_id)

    @socketio.on('unsubscribe_task')
    def handle_unsubscribe_task(data):
        task_id = data.get('task_id')
        client_id = data.get('client_id')
        if task_id and client_id:
            websocket_service.unsubscribe_task_updates(task_id, client_id)

    @socketio.on('request_system_resources')
    def handle_request_system_resources():
        # 触发资源更新广播
        websocket_service.notify_system_resources()
