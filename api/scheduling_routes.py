import logging

from flask import Blueprint, request, jsonify

from services.scheduling_service import SchedulingService

logger = logging.getLogger(__name__)
scheduling_bp = Blueprint('scheduling', __name__)
scheduling_service = SchedulingService()


@scheduling_bp.route('/schedule', methods=['POST'])
def schedule_tasks():
    data = request.json
    try:
        result = scheduling_service.schedule_resources(
            data['tasks'],
            data['scheduler_config']
        )

        # 直接返回序列化结果，不包装在'result'字段中
        return jsonify({
            'scheduled_tasks': [t.__dict__ for t in result.scheduled_tasks],
            'delayed_tasks': [t.__dict__ for t in result.delayed_tasks],
            'cancelled_tasks': [t.__dict__ for t in result.cancelled_tasks],
            'total_time': result.total_time,
            'efficiency': result.efficiency
        })
    except Exception as e:
        logger.error(f"Resource scheduling failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@scheduling_bp.route('/status', methods=['POST'])
def get_scheduler_status():
    data = request.json
    try:
        status = scheduling_service.get_scheduler_status(
            data['scheduler_config']
        )
        return jsonify({
            'status': 'success',
            'scheduler_status': status
        })
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
