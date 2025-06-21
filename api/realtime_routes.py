from flask import Blueprint, request, jsonify
from flask_socketio import emit, join_room, leave_room
from services.visualization_service import VisualizationService
from services.websocket_service import WebSocketService
from api.schemas import validate_uuid_format
import logging
from functools import wraps

logger = logging.getLogger(__name__)

realtime_bp = Blueprint('realtime', __name__)
visualization_service = VisualizationService()
websocket_service = WebSocketService()


def handle_realtime_error(f):
    """实时数据API错误处理装饰器"""

    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {f.__name__}: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': str(e)
            }), 400
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Internal server error'
            }), 500

    return decorated_function


@realtime_bp.route('/subscribe/<simulation_id>', methods=['POST'])
@handle_realtime_error
def subscribe_realtime_data(simulation_id):
    """订阅实时数据推送"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    data = request.get_json() or {}
    update_interval = data.get('interval', 1.0)  # 默认1秒更新一次
    data_types = data.get('data_types', ['all'])  # 数据类型过滤

    # 验证更新间隔
    if update_interval < 0.1 or update_interval > 10.0:
        raise ValueError("Update interval must be between 0.1 and 10.0 seconds")

    # 注册订阅
    subscription_id = websocket_service.subscribe_simulation_updates(
        simulation_id, update_interval, data_types
    )

    return jsonify({
        'status': 'success',
        'subscription_id': subscription_id,
        'simulation_id': simulation_id,
        'update_interval': update_interval,
        'data_types': data_types
    })


@realtime_bp.route('/unsubscribe/<subscription_id>', methods=['POST'])
@handle_realtime_error
def unsubscribe_realtime_data(subscription_id):
    """取消订阅实时数据推送"""
    result = websocket_service.unsubscribe_simulation_updates(subscription_id)

    if not result:
        return jsonify({
            'status': 'error',
            'message': 'Subscription not found'
        }), 404

    return jsonify({
        'status': 'success',
        'message': 'Unsubscribed successfully'
    })


@realtime_bp.route('/broadcast/<simulation_id>', methods=['POST'])
@handle_realtime_error
def broadcast_data(simulation_id):
    """手动广播数据到订阅者"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    data = request.get_json() or {}
    message_type = data.get('type', 'update')
    message_data = data.get('data', {})

    # 广播数据
    websocket_service.broadcast_to_simulation(simulation_id, {
        'type': message_type,
        'data': message_data,
        'timestamp': data.get('timestamp')
    })

    return jsonify({
        'status': 'success',
        'message': 'Data broadcasted successfully'
    })


@realtime_bp.route('/subscriptions', methods=['GET'])
@handle_realtime_error
def get_active_subscriptions():
    """获取活跃订阅列表"""
    subscriptions = websocket_service.get_active_subscriptions()

    return jsonify({
        'status': 'success',
        'subscriptions': subscriptions,
        'total_count': len(subscriptions)
    })
