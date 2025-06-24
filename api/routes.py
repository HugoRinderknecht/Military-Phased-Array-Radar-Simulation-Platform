from flask import Blueprint, request, jsonify, send_file, current_app
from flask_socketio import emit
from services.simulation_service import SimulationService
from services.analysis_service import AnalysisService
from services.websocket_service import WebSocketService
from services.config_service import ConfigService
from services.export_service import ExportService
from api.schemas import (
    validate_simulation_request,
    validate_random_target_request,
    validate_config_request,
    validate_export_request
)
import traceback
import logging
import numpy as np
import uuid
import os
from datetime import datetime
from functools import wraps

# 设置日志
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

# 服务实例初始化
simulation_service = SimulationService()
analysis_service = AnalysisService()
websocket_service = WebSocketService()
config_service = ConfigService()
export_service = ExportService()


def handle_api_error(f):
    """API错误处理装饰器"""

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
        except FileNotFoundError as e:
            logger.error(f"File not found in {f.__name__}: {str(e)}")
            return jsonify({
                'status': 'error',
                'message': 'Resource not found'
            }), 404
        except Exception as e:
            logger.error(f"Unexpected error in {f.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'status': 'error',
                'message': 'Internal server error',
                'debug': str(e) if current_app.debug else None
            }), 500

    return decorated_function


# === 仿真管理接口 ===

@api_bp.route('/simulation/start', methods=['POST'])
@handle_api_error
def start_simulation():
    """启动仿真"""
    data = request.get_json()
    if not data:
        raise ValueError("Request body is required")

    # 验证请求
    is_valid, error_msg = validate_simulation_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 生成仿真ID
    simulation_id = str(uuid.uuid4())

    # 初始化仿真
    init_result = simulation_service.initialize_simulation(data, simulation_id)
    if init_result['status'] == 'error':
        raise ValueError(init_result.get('message', 'Failed to initialize simulation'))

    # 启动异步仿真任务
    result = simulation_service.start_async_simulation(simulation_id)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'message': 'Simulation started',
        'data': result
    })


@api_bp.route('/simulation/<simulation_id>/stop', methods=['POST'])
@handle_api_error
def stop_simulation(simulation_id):
    """停止仿真"""
    if not simulation_id:
        raise ValueError("Simulation ID is required")

    result = simulation_service.stop_simulation(simulation_id)

    # 通知前端仿真已停止
    try:
        websocket_service.emit_simulation_stopped(simulation_id)
    except Exception as e:
        logger.warning(f"Failed to emit websocket message: {str(e)}")

    return jsonify({
        'status': 'success',
        'message': 'Simulation stopped',
        'data': result
    })


@api_bp.route('/simulation/<simulation_id>/status', methods=['GET'])
@handle_api_error
def get_simulation_status(simulation_id):
    """获取仿真状态"""
    if not simulation_id:
        raise ValueError("Simulation ID is required")

    status = simulation_service.get_simulation_status(simulation_id)
    return jsonify({
        'status': 'success',
        'data': status
    })


@api_bp.route('/simulation/<simulation_id>/data', methods=['GET'])
@handle_api_error
def get_simulation_data(simulation_id):
    """获取仿真数据"""
    if not simulation_id:
        raise ValueError("Simulation ID is required")

    # 获取并验证查询参数
    start_time = request.args.get('start_time', type=float, default=0.0)
    end_time = request.args.get('end_time', type=float)
    data_types = request.args.getlist('data_types')

    # 验证时间范围
    if end_time is not None and start_time >= end_time:
        raise ValueError("start_time must be less than end_time")

    data = simulation_service.get_simulation_data(
        simulation_id, start_time, end_time, data_types
    )

    return jsonify({
        'status': 'success',
        'data': data
    })


@api_bp.route('/simulation/<simulation_id>/pause', methods=['POST'])
@handle_api_error
def pause_simulation(simulation_id):
    """暂停仿真"""
    if not simulation_id:
        raise ValueError("Simulation ID is required")

    result = simulation_service.pause_simulation(simulation_id)
    return jsonify({
        'status': 'success',
        'message': 'Simulation paused',
        'data': result
    })


@api_bp.route('/simulation/<simulation_id>/resume', methods=['POST'])
@handle_api_error
def resume_simulation(simulation_id):
    """恢复仿真"""
    if not simulation_id:
        raise ValueError("Simulation ID is required")

    result = simulation_service.resume_simulation(simulation_id)
    return jsonify({
        'status': 'success',
        'message': 'Simulation resumed',
        'data': result
    })


# === 配置管理接口 ===

@api_bp.route('/config/save', methods=['POST'])
@handle_api_error
def save_configuration():
    """保存配置"""
    data = request.get_json()
    if not data:
        raise ValueError("Configuration data is required")

    is_valid, error_msg = validate_config_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    config_id = config_service.save_configuration(data)

    return jsonify({
        'status': 'success',
        'config_id': config_id,
        'message': 'Configuration saved successfully'
    })


@api_bp.route('/config/<config_id>', methods=['GET'])
@handle_api_error
def load_configuration(config_id):
    """加载配置"""
    if not config_id:
        raise ValueError("Configuration ID is required")

    config = config_service.load_configuration(config_id)
    return jsonify({
        'status': 'success',
        'data': config
    })


@api_bp.route('/config/list', methods=['GET'])
@handle_api_error
def list_configurations():
    """获取配置列表"""
    # 获取分页参数
    page = request.args.get('page', type=int, default=1)
    per_page = request.args.get('per_page', type=int, default=20)

    if page < 1 or per_page < 1 or per_page > 100:
        raise ValueError("Invalid pagination parameters")

    configs = config_service.list_configurations(page=page, per_page=per_page)
    return jsonify({
        'status': 'success',
        'data': configs
    })


@api_bp.route('/analyze', methods=['POST'])
@handle_api_error
def analyze_results():
    """分析仿真结果"""
    data = request.get_json()
    if not data:
        raise ValueError("Analysis data is required")

    # 验证数据结构
    if 'results' not in data:
        raise ValueError("Results data is required")

    results = data['results']

    # 验证必要的数据字段
    if 'summary' not in results:
        raise ValueError("Summary data is required")

    # 调用分析服务
    analysis_result = analysis_service.analyze_simulation_results(results)

    return jsonify({
        'status': 'success',
        'analysis': analysis_result,
        'message': 'Analysis completed successfully'
    })


# === 数据导出接口 ===

@api_bp.route('/export/start', methods=['POST'])
@handle_api_error
def start_export():
    """开始导出"""
    data = request.get_json()
    if not data:
        raise ValueError("Export configuration is required")

    is_valid, error_msg = validate_export_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    export_id = export_service.start_export(data)

    return jsonify({
        'status': 'success',
        'export_id': export_id,
        'message': 'Export started'
    })


@api_bp.route('/export/<export_id>/download', methods=['GET'])
@handle_api_error
def download_export_file(export_id):
    """下载导出文件"""
    if not export_id:
        raise ValueError("Export ID is required")

    file_path = export_service.get_export_file_path(export_id)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Export file not found for ID: {export_id}")

    return send_file(file_path, as_attachment=True)


# === 系统信息接口 ===

@api_bp.route('/system/info', methods=['GET'])
@handle_api_error
def get_system_info():
    """获取系统信息"""
    try:
        info = {
            'version': '1.0.0',
            'api_version': '1.0',
            'server_time': datetime.now().isoformat(),
            'active_simulations': simulation_service.get_active_simulation_count(),
            'system_status': 'running'
        }
        return jsonify({'status': 'success', 'data': info})
    except Exception as e:
        logger.error(f"Failed to fetch system info: {str(e)}")
        raise  # 触发装饰器的500处理


@api_bp.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'success',
        'message': 'Radar simulation API is running',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    })


# === 兼容性接口 ===

@api_bp.route('/simulate', methods=['POST'])
@handle_api_error
def run_simulation():
    """运行仿真（单次，兼容性保留）"""
    data = request.get_json()
    if not data:
        raise ValueError("Simulation configuration is required")

    is_valid, error_msg = validate_simulation_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    init_result = simulation_service.initialize_simulation(data)
    if init_result['status'] == 'error':
        raise ValueError(init_result.get('message', 'Failed to initialize simulation'))

    simulation_result = simulation_service.run_simulation()
    return jsonify(simulation_result)


@api_bp.route('/status', methods=['GET'])
@handle_api_error
def get_status():
    """获取仿真状态（兼容性保留）"""
    try:
        status = simulation_service.get_legacy_status()
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        logger.error(f"Failed to fetch legacy status: {str(e)}")
        raise  # 触发装饰器的500处理


@api_bp.route('/reset', methods=['POST'])
@handle_api_error
def reset():
    """重置仿真"""
    result = simulation_service.reset_all_simulations()
    return jsonify({
        'status': 'success',
        'message': 'All simulations reset',
        'data': result
    })


@api_bp.route('/debug/monitor', methods=['GET'])
@handle_api_error
def get_monitor_debug_info():
    """获取监控调试信息"""
    try:
        from app import system_monitor, websocket_service

        info = {
            'system_monitor': system_monitor.get_monitor_info() if system_monitor else None,
            'websocket_service': websocket_service.get_thread_status() if websocket_service else None,
            'simulation_service': {
                'active_simulations': simulation_service.get_active_simulation_count(),
                'current_time': getattr(simulation_service, 'current_time', 0.0)
            }
        }

        return jsonify({
            'status': 'success',
            'data': info
        })
    except Exception as e:
        logger.error(f"Error getting debug info: {str(e)}")
        raise
