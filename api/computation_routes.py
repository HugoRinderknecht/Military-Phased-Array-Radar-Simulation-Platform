import json

from flask import Blueprint, request, jsonify
from services.batch_computation_service import BatchComputationService
from services.config_service import ConfigService
from services.websocket_service import WebSocketService
import logging

logger = logging.getLogger(__name__)
computation_bp = Blueprint('computation', __name__, url_prefix='/api/computation')

@computation_bp.route('/subscribe', methods=['POST'])
def subscribe_task_updates():
    data = request.json
    try:
        client_id = data['client_id']
        # 实现订阅逻辑
        return jsonify({
            'status': 'success',
            'message': f'Subscribed for task updates'
        })
    except Exception as e:
        logger.error(f"Subscription error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@computation_bp.route('/unsubscribe', methods=['POST'])
def unsubscribe_task_updates():
    data = request.json
    try:
        client_id = data['client_id']
        # 实现取消订阅逻辑
        return jsonify({
            'status': 'success',
            'message': f'Unsubscribed from task updates'
        })
    except Exception as e:
        logger.error(f"Unsubscription error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@computation_bp.route('/schedule', methods=['POST'])
def schedule_resources():
    data = request.json
    try:
        tasks = data['tasks']

        # 模拟资源调度结果
        scheduled_tasks = []
        delayed_tasks = []
        cancelled_tasks = []

        # 简单的调度算法：按截止时间排序并分配
        sorted_tasks = sorted(tasks, key=lambda x: x['due_time'])
        current_time = 0.0
        for task in sorted_tasks:
            if current_time + task['duration'] <= task['due_time']:
                scheduled_tasks.append({
                    'task_id': task['task_id'],
                    'start_time': current_time,
                    'end_time': current_time + task['duration']
                })
                current_time += task['duration']
            elif task.get('hard_constraint', False):
                # 硬约束任务必须执行
                scheduled_tasks.append({
                    'task_id': task['task_id'],
                    'start_time': current_time,
                    'end_time': current_time + task['duration']
                })
                current_time += task['duration']
            else:
                delayed_tasks.append(task['task_id'])

        efficiency = len(scheduled_tasks) / len(tasks) if tasks else 0.0

        return jsonify({
            'status': 'success',
            'scheduled_tasks': scheduled_tasks,
            'delayed_tasks': delayed_tasks,
            'cancelled_tasks': cancelled_tasks,
            'efficiency': efficiency
        })
    except Exception as e:
        logger.error(f"Resource scheduling error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@computation_bp.route('/scheduler/status', methods=['GET'], endpoint='scheduler_status')
def get_scheduler_status():
    try:
        # 获取详细的调度器状态信息
        scheduler_status = {
            'active_threads': 5,
            'pending_tasks': 3,
            'completed_tasks': 42,
            'last_update': '2023-08-15T10:00:00Z',
            'status': 'active',
            'resource_utilization': {
                'cpu': 75.4,
                'memory': 62.3
            },
            'active_subscriptions': 8,
            'simulation_rooms': 3  # 添加模拟的仿真房间数量
        }

        return jsonify({
            'status': 'success',
            'scheduler_status': scheduler_status
        })
    except Exception as e:
        logger.error(f"Error getting scheduler status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@computation_bp.route('/validate', methods=['POST'], endpoint='validate_config')
def validate_configuration():
    data = request.json
    try:
        # 记录接收到的配置数据
        logger.info(f"Received configuration for validation: {json.dumps(data, indent=2)}")

        # 调用配置服务进行验证
        config_service = ConfigService()
        validation_result = config_service.validate_configuration(data)

        # 记录验证结果
        logger.info(f"Validation result: {json.dumps(validation_result, indent=2)}")

        # 确保验证结果包含必要的字段
        valid = validation_result.get('valid', False)
        errors = validation_result.get('errors', [])
        warnings = validation_result.get('warnings', [])

        # 添加详细的验证结果消息
        if valid:
            message = '配置验证成功'
        else:
            message = f'配置验证失败，发现 {len(errors)} 个错误'

        return jsonify({
            'status': 'success',
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'message': message
        })
    except Exception as e:
        logger.error(f"Configuration validation error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


@computation_bp.route('/broadcast', methods=['POST'])
def simulation_data_broadcast():
    data = request.json
    try:
        # 实现数据广播逻辑
        return jsonify({
            'status': 'success',
            'message': 'Data broadcasted'
        })
    except Exception as e:
        logger.error(f"Data broadcast error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

@computation_bp.route('/resources/notifications', methods=['GET'])
def system_resources_notification():
    try:
        # 获取系统资源通知
        return jsonify({
            'status': 'success',
            'notifications': []  # 示例通知列表
        })
    except Exception as e:
        logger.error(f"Error getting resource notifications: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

batch_service = BatchComputationService(WebSocketService())
@computation_bp.route('/batch', methods=['POST'])
def submit_batch_task():
    data = request.json
    try:
        task_id = batch_service.submit_async_task(
            data['task_type'],
            data['task_data'],
            data.get('client_id')
        )
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': 'Computation task submitted successfully'
        })
    except Exception as e:
        logger.error(f"Error submitting computation task: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@computation_bp.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    try:
        status = batch_service.get_task_status(task_id)
        return jsonify({
            'status': 'success',
            'task_status': status
        })
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404

@computation_bp.route('/result/<task_id>', methods=['GET'])
def get_task_result(task_id):
    try:
        result = batch_service.get_task_result(task_id)
        return jsonify({
            'status': 'success',
            'task_result': result
        })
    except Exception as e:
        logger.error(f"Error getting task result: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 404

@computation_bp.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    try:
        # 实现取消逻辑
        return jsonify({
            'status': 'success',
            'message': f'Task {task_id} cancelled'
        })
    except Exception as e:
        logger.error(f"Error cancelling task: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
