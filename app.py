from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room
from api.routes import api_bp
from api.visualization_routes import visualization_bp  # 新增
from api.realtime_routes import realtime_bp  # 新增
from config.settings import Config
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局socketio实例
socketio = None


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # 启用CORS
    CORS(app, resources={
        r"/api/*": {"origins": "*"},
        r"/socket.io/*": {"origins": "*"}
    })

    # 注册API蓝图
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(visualization_bp, url_prefix='/api/visualization')  # 新增
    app.register_blueprint(realtime_bp, url_prefix='/api/realtime')  # 新增

    return app


def create_socketio(app):
    """创建SocketIO实例"""
    global socketio
    socketio = SocketIO(
        app,
        cors_allowed_origins="*",
        async_mode='threading',
        logger=True,
        engineio_logger=True
    )
    return socketio


app = create_app()
socketio = create_socketio(app)

# 将socketio实例传递给其他模块使用
app.socketio = socketio


# WebSocket事件处理
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')


@socketio.on('join_simulation')
def handle_join_simulation(data):
    """客户端加入仿真会话"""
    simulation_id = data.get('simulation_id')
    client_id = data.get('client_id', 'unknown')

    if simulation_id:
        join_room(simulation_id)

        # 使用WebSocket服务管理房间
        from services.websocket_service import WebSocketService
        websocket_service = WebSocketService()
        websocket_service.join_simulation_room(simulation_id, client_id)

        logger.info(f'Client {client_id} joined simulation {simulation_id}')


@socketio.on('leave_simulation')
def handle_leave_simulation(data):
    """客户端离开仿真会话"""
    simulation_id = data.get('simulation_id')
    client_id = data.get('client_id', 'unknown')

    if simulation_id:
        leave_room(simulation_id)

        # 使用WebSocket服务管理房间
        from services.websocket_service import WebSocketService
        websocket_service = WebSocketService()
        websocket_service.leave_simulation_room(simulation_id, client_id)

        logger.info(f'Client {client_id} left simulation {simulation_id}')


# 新增：实时数据订阅事件
@socketio.on('subscribe_updates')
def handle_subscribe_updates(data):
    """订阅实时更新"""
    simulation_id = data.get('simulation_id')
    interval = data.get('interval', 1.0)
    data_types = data.get('data_types', ['all'])

    if simulation_id:
        from services.websocket_service import WebSocketService
        websocket_service = WebSocketService()
        subscription_id = websocket_service.subscribe_simulation_updates(
            simulation_id, interval, data_types
        )

        # 返回订阅ID给客户端
        socketio.emit('subscription_created', {
            'subscription_id': subscription_id,
            'simulation_id': simulation_id
        })


@socketio.on('unsubscribe_updates')
def handle_unsubscribe_updates(data):
    """取消订阅实时更新"""
    subscription_id = data.get('subscription_id')

    if subscription_id:
        from services.websocket_service import WebSocketService
        websocket_service = WebSocketService()
        success = websocket_service.unsubscribe_simulation_updates(subscription_id)

        socketio.emit('subscription_cancelled', {
            'subscription_id': subscription_id,
            'success': success
        })


@socketio.on('error')
def handle_error(e):
    """处理WebSocket错误"""
    logger.error(f'WebSocket error: {str(e)}')


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
