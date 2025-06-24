import time
from flask import Blueprint, jsonify
from services.system_service import SystemService
import logging
import threading

logger = logging.getLogger(__name__)
system_bp = Blueprint('system', __name__)
system_service = SystemService()

# 添加应用启动时间记录
app_start_time = time.time()


@system_bp.route('/resources', methods=['GET'])
def get_system_resources():
    try:
        resources = system_service.get_resource_usage()
        return jsonify({
            'status': 'success',
            'resources': resources
        })
    except Exception as e:
        logger.error(f"Failed to get system resources: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@system_bp.route('/load', methods=['GET'])
def get_compute_load():
    try:
        load = system_service.get_compute_load()
        return jsonify({
            'status': 'success',
            'load': load
        })
    except Exception as e:
        logger.error(f"Failed to get compute load: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# 添加更详细的系统状态接口
@system_bp.route('/status', methods=['GET'])
def get_system_status():
    """获取系统整体状态信息"""
    try:
        # 获取系统资源使用情况
        resources = system_service.get_resource_usage()
        # 获取计算负载
        load = system_service.get_compute_load()

        return jsonify({
            'status': 'operational',
            'version': '1.0.0',
            'uptime': round(time.time() - app_start_time, 2),  # 系统运行时间（秒）
            'active_threads': threading.active_count(),  # 活跃线程数
            'resources': resources,
            'load': load
        })
    except Exception as e:
        logger.error(f"Failed to get system status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
