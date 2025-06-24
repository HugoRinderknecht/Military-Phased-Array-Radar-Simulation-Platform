# app.py
import os
import sys
import logging
import signal
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS
from api.routes import api_bp
from services.system_monitor import SystemMonitor
from services.websocket_service import WebSocketService
from config import get_config
import atexit
import time
import threading


# 获取配置
def get_app_config():
    return get_config()


# 配置日志
def setup_logging(config):
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


# 全局服务实例
system_monitor = None
websocket_service = None
services_initialized = False


def create_app(config_override=None):
    """应用工厂函数"""
    # 获取配置
    config = config_override or get_app_config()

    # 配置日志
    logger = setup_logging(config)

    # 创建Flask应用
    app = Flask(__name__)
    app.config.from_object(config)

    # 启用CORS
    CORS(app, origins=config.CORS_ORIGINS)

    # 创建SocketIO实例
    socketio = SocketIO(app, cors_allowed_origins=config.CORS_ORIGINS, logger=False, engineio_logger=False)

    # 注册API蓝图
    app.register_blueprint(api_bp, url_prefix='/api')

    # 添加健康检查路由
    @app.route('/health')
    def health_check():
        return {
            'status': 'healthy',
            'timestamp': time.time(),
            'services': {
                'system_monitor_running': system_monitor.is_running() if system_monitor else False,
                'websocket_service_active': websocket_service is not None,
                'services_initialized': services_initialized
            }
        }

    # 添加首页路由
    @app.route('/')
    def index():
        return {
            'message': 'Radar Simulation System API',
            'version': '1.0.0',
            'status': 'running',
            'endpoints': {
                'api': '/api',
                'health': '/health',
                'websocket': 'ws://localhost:5000'
            }
        }

    # 存储socketio实例到app配置中，供其他地方使用
    app.socketio = socketio

    return app


def is_main_process():
    """检查是否是主进程（避免Flask调试模式的重复初始化）"""
    if os.name == 'nt':  # Windows
        return os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    else:  # Unix/Linux
        return os.environ.get('WERKZEUG_RUN_MAIN') == 'true'


def initialize_services(socketio):
    """初始化服务"""
    global system_monitor, websocket_service, services_initialized

    if services_initialized:
        return

    config = get_app_config()
    logger = logging.getLogger(__name__)

    try:
        # 只在主进程或非调试模式下初始化服务
        if not config.DEBUG or is_main_process():
            logger.info("Initializing services in main process...")

            # 初始化WebSocket服务
            if websocket_service is None:
                websocket_service = WebSocketService(socketio)
                logger.info("WebSocket service initialized")
            else:
                websocket_service.set_socketio(socketio)

            # 初始化系统监控（如果启用）
            if config.SYSTEM_MONITOR_ENABLED and system_monitor is None:
                system_monitor = SystemMonitor(socketio, interval=config.SYSTEM_MONITOR_INTERVAL)

                # 延迟启动系统监控
                def delayed_start():
                    time.sleep(2)
                    if system_monitor and not system_monitor.is_running():
                        if system_monitor.start():
                            logger.info("System monitor started successfully")
                        else:
                            logger.warning("System monitor failed to start")

                threading.Thread(target=delayed_start, daemon=True).start()
                logger.info("System monitor initialization scheduled")

            # 启动WebSocket更新线程
            if websocket_service and hasattr(websocket_service, 'start_update_thread'):
                try:
                    websocket_service.start_update_thread()
                    logger.info("WebSocket service update thread started")
                except Exception as e:
                    logger.error(f"WebSocket service failed to start: {str(e)}")

            services_initialized = True
            logger.info("Services initialization completed")
        else:
            logger.info("Skipping service initialization in reloader process")

    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")


def cleanup_services():
    """清理服务"""
    global system_monitor, websocket_service, services_initialized

    logger = logging.getLogger(__name__)
    logger.info("Cleaning up services...")

    if system_monitor:
        try:
            system_monitor.stop()
            logger.info("System monitor stopped")
        except Exception as e:
            logger.warning(f"Error stopping system monitor: {str(e)}")

    if websocket_service:
        try:
            websocket_service.cleanup_all_subscriptions()
            logger.info("WebSocket subscriptions cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up websocket subscriptions: {str(e)}")

    services_initialized = False
    logger.info("Service cleanup completed")


# 注册清理函数
atexit.register(cleanup_services)


# 信号处理
def signal_handler(signum, frame):
    logger = logging.getLogger(__name__)
    logger.info(f"Received signal {signum}, cleaning up...")
    cleanup_services()
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
if os.name != 'nt':
    signal.signal(signal.SIGTERM, signal_handler)

# 为直接运行时创建应用实例
if __name__ == '__main__':
    config = get_app_config()
    app = create_app()
    socketio = app.socketio


    # SocketIO事件处理
    @socketio.on('connect')
    def handle_connect():
        logger = logging.getLogger(__name__)
        logger.debug('Client connected')
        # 确保服务已初始化
        if not services_initialized:
            initialize_services(socketio)

        if websocket_service:
            socketio.emit('welcome', {
                'message': 'Connected to radar simulation system',
                'timestamp': time.time(),
                'config': {
                    'monitor_interval': config.SYSTEM_MONITOR_INTERVAL,
                    'debug_mode': config.DEBUG
                }
            })


    @socketio.on('disconnect')
    def handle_disconnect():
        logger = logging.getLogger(__name__)
        logger.debug('Client disconnected')


    @socketio.on('subscribe_simulation')
    def handle_subscribe_simulation(data):
        if websocket_service:
            simulation_id = data.get('simulation_id')
            interval = data.get('interval', 1.0)
            data_types = data.get('data_types', ['all'])
            client_id = data.get('client_id', 'unknown')

            subscription_id = websocket_service.subscribe_simulation_updates(
                simulation_id, interval, data_types, client_id
            )

            socketio.emit('subscription_confirmed', {
                'subscription_id': subscription_id,
                'simulation_id': simulation_id
            })


    @socketio.on('unsubscribe_simulation')
    def handle_unsubscribe_simulation(data):
        if websocket_service:
            subscription_id = data.get('subscription_id')
            client_id = data.get('client_id', 'unknown')

            success = websocket_service.unsubscribe_simulation_updates(
                subscription_id, client_id
            )

            socketio.emit('unsubscription_confirmed', {
                'subscription_id': subscription_id,
                'success': success
            })


    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Starting application in {'DEBUG' if config.DEBUG else 'PRODUCTION'} mode")

        # 在主线程启动前初始化服务
        if not config.DEBUG:
            # 生产模式下立即初始化
            initialize_services(socketio)

        # 启动应用
        socketio.run(
            app,
            debug=config.DEBUG,
            host='0.0.0.0',
            port=5000,
            allow_unsafe_werkzeug=True,
            use_reloader=config.DEBUG
        )

    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Application interrupted by user")
    except Exception as e:
        logging.getLogger(__name__).error(f"Application error: {str(e)}")
    finally:
        cleanup_services()
