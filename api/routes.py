"""
API路由定义
包含所有雷达仿真系统的RESTful API接口
"""

from flask import Blueprint, request, jsonify, send_file
from werkzeug.exceptions import BadRequest
import uuid
import os
from datetime import datetime
import traceback

# 导入服务层
from services.simulation_service import SimulationService
from services.analysis_service import AnalysisService
from services.config_service import ConfigService
from services.export_service import ExportService
from services.target_service import TargetService
from services.signal_processing_service import SignalProcessingService
from services.batch_service import BatchService

# 导入验证模块
from api.schemas import (
    validate_simulation_request, validate_analysis_request,
    validate_config_request, validate_export_request,
    validate_signal_processing_request, validate_rcs_calculation_request,
    validate_batch_simulation_request, validate_random_target_request
)

# 创建蓝图
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 初始化服务实例
simulation_service = SimulationService()
analysis_service = AnalysisService()
config_service = ConfigService()
export_service = ExportService()
target_service = TargetService()
signal_processing_service = SignalProcessingService()
batch_service = BatchService()


# 工具函数
def validate_uuid_format(uuid_string):
    """验证UUID格式"""
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def handle_api_error(func):
    """API错误处理装饰器"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            return jsonify({
                'status': 'error',
                'error': 'validation_error',
                'message': str(e)
            }), 400
        except FileNotFoundError as e:
            return jsonify({
                'status': 'error',
                'error': 'not_found',
                'message': str(e)
            }), 404
        except PermissionError as e:
            return jsonify({
                'status': 'error',
                'error': 'permission_denied',
                'message': str(e)
            }), 403
        except Exception as e:
            # 记录详细错误信息
            error_trace = traceback.format_exc()
            print(f"API Error: {error_trace}")

            return jsonify({
                'status': 'error',
                'error': 'internal_error',
                'message': 'An internal server error occurred',
                'details': str(e) if os.getenv('DEBUG') == 'true' else None
            }), 500

    wrapper.__name__ = func.__name__
    return wrapper


# ===============================
# 核心仿真接口
# ===============================

@api_bp.route('/simulations', methods=['POST'])
@handle_api_error
def create_simulation():
    """创建新的仿真任务"""
    data = request.get_json()
    if not data:
        raise ValueError("Simulation configuration is required")

    # 验证输入数据
    is_valid, error_msg = validate_simulation_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 创建仿真
    simulation_id = simulation_service.create_simulation(data)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'message': 'Simulation created successfully'
    }), 201


@api_bp.route('/simulations/<simulation_id>/start', methods=['POST'])
@handle_api_error
def start_simulation(simulation_id):
    """启动仿真"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    # 检查仿真是否存在
    if not simulation_service.simulation_exists(simulation_id):
        raise FileNotFoundError(f"Simulation {simulation_id} not found")

    # 启动仿真
    result = simulation_service.start_simulation(simulation_id)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'start_time': result['start_time'],
        'message': 'Simulation started successfully'
    })


@api_bp.route('/simulations/<simulation_id>/stop', methods=['POST'])
@handle_api_error
def stop_simulation(simulation_id):
    """停止仿真"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    result = simulation_service.stop_simulation(simulation_id)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'stop_time': result['stop_time'],
        'message': 'Simulation stopped successfully'
    })


@api_bp.route('/simulations/<simulation_id>/pause', methods=['POST'])
@handle_api_error
def pause_simulation(simulation_id):
    """暂停仿真"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    result = simulation_service.pause_simulation(simulation_id)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'pause_time': result['pause_time'],
        'message': 'Simulation paused successfully'
    })


@api_bp.route('/simulations/<simulation_id>/resume', methods=['POST'])
@handle_api_error
def resume_simulation(simulation_id):
    """恢复仿真"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    result = simulation_service.resume_simulation(simulation_id)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'resume_time': result['resume_time'],
        'message': 'Simulation resumed successfully'
    })


@api_bp.route('/simulations/<simulation_id>/status', methods=['GET'])
@handle_api_error
def get_simulation_status(simulation_id):
    """获取仿真状态"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    status = simulation_service.get_simulation_status(simulation_id)

    return jsonify({
        'status': 'success',
        'simulation_status': status,
        'message': 'Simulation status retrieved successfully'
    })


@api_bp.route('/simulations', methods=['GET'])
@handle_api_error
def list_simulations():
    """获取仿真列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    status_filter = request.args.get('status')

    # 限制每页数量
    if per_page > 100:
        per_page = 100

    simulations = simulation_service.list_simulations(
        page=page,
        per_page=per_page,
        status_filter=status_filter
    )

    return jsonify({
        'status': 'success',
        'simulations': simulations['items'],
        'pagination': {
            'page': simulations['page'],
            'per_page': simulations['per_page'],
            'total': simulations['total'],
            'pages': simulations['pages']
        },
        'message': 'Simulations retrieved successfully'
    })


@api_bp.route('/simulations/<simulation_id>', methods=['DELETE'])
@handle_api_error
def delete_simulation(simulation_id):
    """删除仿真"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    simulation_service.delete_simulation(simulation_id)

    return jsonify({
        'status': 'success',
        'message': 'Simulation deleted successfully'
    })


# ===============================
# 结果分析接口
# ===============================

@api_bp.route('/simulations/<simulation_id>/results', methods=['GET'])
@handle_api_error
def get_simulation_results(simulation_id):
    """获取仿真结果"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    result_type = request.args.get('type', 'summary')
    time_range = request.args.get('time_range')

    results = analysis_service.get_simulation_results(
        simulation_id,
        result_type=result_type,
        time_range=time_range
    )

    return jsonify({
        'status': 'success',
        'results': results,
        'message': 'Simulation results retrieved successfully'
    })


@api_bp.route('/analysis/radar-equation', methods=['POST'])
@handle_api_error
def calculate_radar_equation():
    """雷达方程计算"""
    data = request.get_json()
    if not data:
        raise ValueError("Radar equation parameters are required")

    is_valid, error_msg = validate_analysis_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    result = analysis_service.calculate_radar_equation(data)

    return jsonify({
        'status': 'success',
        'calculation_result': result,
        'message': 'Radar equation calculated successfully'
    })


@api_bp.route('/analysis/range-doppler', methods=['POST'])
@handle_api_error
def generate_range_doppler_map():
    """生成距离-多普勒图"""
    data = request.get_json()
    if not data:
        raise ValueError("Range-Doppler parameters are required")

    map_data = analysis_service.generate_range_doppler_map(data)

    return jsonify({
        'status': 'success',
        'map_data': map_data,
        'message': 'Range-Doppler map generated successfully'
    })


# ===============================
# 配置管理接口
# ===============================

@api_bp.route('/configs', methods=['POST'])
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
    }), 201


@api_bp.route('/configs', methods=['GET'])
@handle_api_error
def list_configurations():
    """获取配置列表"""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    category = request.args.get('category')

    configs = config_service.list_configurations(
        page=page,
        per_page=per_page,
        category=category
    )

    return jsonify({
        'status': 'success',
        'configurations': configs['items'],
        'pagination': {
            'page': configs['page'],
            'per_page': configs['per_page'],
            'total': configs['total'],
            'pages': configs['pages']
        },
        'message': 'Configurations retrieved successfully'
    })


@api_bp.route('/configs/<config_id>', methods=['GET'])
@handle_api_error
def get_configuration(config_id):
    """获取特定配置"""
    if not validate_uuid_format(config_id):
        raise ValueError("Invalid configuration ID format")

    config = config_service.get_configuration(config_id)

    return jsonify({
        'status': 'success',
        'configuration': config,
        'message': 'Configuration retrieved successfully'
    })


@api_bp.route('/configs/<config_id>', methods=['PUT'])
@handle_api_error
def update_configuration(config_id):
    """更新配置"""
    if not validate_uuid_format(config_id):
        raise ValueError("Invalid configuration ID format")

    data = request.get_json()
    if not data:
        raise ValueError("Configuration data is required")

    is_valid, error_msg = validate_config_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    config_service.update_configuration(config_id, data)

    return jsonify({
        'status': 'success',
        'message': 'Configuration updated successfully'
    })


@api_bp.route('/configs/<config_id>', methods=['DELETE'])
@handle_api_error
def delete_configuration(config_id):
    """删除配置"""
    if not validate_uuid_format(config_id):
        raise ValueError("Invalid configuration ID format")

    config_service.delete_configuration(config_id)

    return jsonify({
        'status': 'success',
        'message': 'Configuration deleted successfully'
    })


# ===============================
# 数据导出接口
# ===============================

@api_bp.route('/export/simulation/<simulation_id>', methods=['POST'])
@handle_api_error
def export_simulation_data():
    """导出仿真数据"""
    simulation_id = request.view_args['simulation_id']
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    data = request.get_json()
    if not data:
        raise ValueError("Export parameters are required")

    is_valid, error_msg = validate_export_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 生成导出文件
    file_path = export_service.export_simulation_data(simulation_id, data)

    return jsonify({
        'status': 'success',
        'download_url': f'/api/download/{os.path.basename(file_path)}',
        'message': 'Export completed successfully'
    })


@api_bp.route('/export/report/<simulation_id>', methods=['POST'])
@handle_api_error
def generate_report():
    """生成报告"""
    simulation_id = request.view_args['simulation_id']
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    data = request.get_json() or {}

    report_path = export_service.generate_report(simulation_id, data)

    return jsonify({
        'status': 'success',
        'download_url': f'/api/download/{os.path.basename(report_path)}',
        'message': 'Report generated successfully'
    })


@api_bp.route('/download/<filename>', methods=['GET'])
@handle_api_error
def download_file(filename):
    """下载文件"""
    file_path = export_service.get_download_file_path(filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError("File not found")

    return send_file(file_path, as_attachment=True)


# ===============================
# 目标管理接口
# ===============================

@api_bp.route('/targets/generate', methods=['POST'])
@handle_api_error
def generate_random_targets():
    """生成随机目标"""
    data = request.get_json()
    if not data:
        raise ValueError("Target generation parameters are required")

    is_valid, error_msg = validate_random_target_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 调用目标生成服务
    targets = target_service.generate_random_targets(data)

    return jsonify({
        'status': 'success',
        'targets': targets,
        'count': len(targets),
        'message': 'Random targets generated successfully'
    })


@api_bp.route('/targets/validate', methods=['POST'])
@handle_api_error
def validate_targets():
    """验证目标配置"""
    data = request.get_json()
    if not data:
        raise ValueError("Target data is required")

    # 验证目标配置
    validation_result = target_service.validate_targets_configuration(data)

    return jsonify({
        'status': 'success',
        'validation': validation_result,
        'is_valid': validation_result['is_valid'],
        'warnings': validation_result.get('warnings', []),
        'errors': validation_result.get('errors', [])
    })


@api_bp.route('/simulation/<simulation_id>/targets/update', methods=['POST'])
@handle_api_error
def update_simulation_targets(simulation_id):
    """动态更新仿真中的目标"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    data = request.get_json()
    if not data:
        raise ValueError("Target update data is required")

    # 检查仿真状态
    status = simulation_service.get_simulation_status(simulation_id)
    if status.get('state') not in ['running', 'paused']:
        raise ValueError("Can only update targets for running or paused simulations")

    result = simulation_service.update_simulation_targets(simulation_id, data)

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'updated_targets': result['updated_count'],
        'message': 'Targets updated successfully'
    })


@api_bp.route('/targets/formation/create', methods=['POST'])
@handle_api_error
def create_target_formation():
    """创建目标编队"""
    data = request.get_json()
    if not data:
        raise ValueError("Formation data is required")

    # 验证编队参数
    required_fields = ['formation_type', 'leader_target', 'member_targets', 'spacing']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    formation = target_service.create_target_formation(data)

    return jsonify({
        'status': 'success',
        'formation': formation,
        'message': 'Target formation created successfully'
    })


# ===============================
# 信号处理接口
# ===============================

@api_bp.route('/signal/process', methods=['POST'])
@handle_api_error
def process_signal_data():
    """信号处理计算"""
    data = request.get_json()
    if not data:
        raise ValueError("Signal data is required")

    is_valid, error_msg = validate_signal_processing_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 调用信号处理服务
    result = signal_processing_service.process_signal_data(data)

    return jsonify({
        'status': 'success',
        'processing_type': data['processing_type'],
        'result': result,
        'processing_time': result.get('processing_time'),
        'message': 'Signal processing completed'
    })


@api_bp.route('/rcs/calculate', methods=['POST'])
@handle_api_error
def calculate_rcs():
    """RCS计算"""
    data = request.get_json()
    if not data:
        raise ValueError("RCS calculation parameters are required")

    is_valid, error_msg = validate_rcs_calculation_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    rcs_result = signal_processing_service.calculate_rcs(data)

    return jsonify({
        'status': 'success',
        'rcs_value': rcs_result['rcs'],
        'rcs_pattern': rcs_result.get('pattern'),
        'calculation_method': rcs_result['method'],
        'message': 'RCS calculation completed'
    })


@api_bp.route('/detection/probability', methods=['POST'])
@handle_api_error
def calculate_detection_probability():
    """检测概率计算"""
    data = request.get_json()
    if not data:
        raise ValueError("Detection probability parameters are required")

    # 验证参数
    required_fields = ['snr', 'pfa']  # Signal-to-Noise Ratio, Probability of False Alarm
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    detection_result = signal_processing_service.calculate_detection_probability(data)

    return jsonify({
        'status': 'success',
        'probability_of_detection': detection_result['pd'],
        'threshold': detection_result['threshold'],
        'roc_curve': detection_result.get('roc_curve'),
        'message': 'Detection probability calculated'
    })


@api_bp.route('/clutter/simulation', methods=['POST'])
@handle_api_error
def simulate_clutter():
    """杂波仿真"""
    data = request.get_json()
    if not data:
        raise ValueError("Clutter simulation parameters are required")

    clutter_result = signal_processing_service.simulate_clutter(data)

    return jsonify({
        'status': 'success',
        'clutter_data': clutter_result['data'],
        'statistics': clutter_result['statistics'],
        'message': 'Clutter simulation completed'
    })


# ===============================
# 性能评估接口
# ===============================

@api_bp.route('/performance/metrics', methods=['POST'])
@handle_api_error
def calculate_performance_metrics():
    """性能指标计算"""
    data = request.get_json()
    if not data:
        raise ValueError("Performance calculation data is required")

    if 'simulation_results' not in data:
        raise ValueError("Simulation results are required")

    metrics = analysis_service.calculate_performance_metrics(data['simulation_results'])

    return jsonify({
        'status': 'success',
        'metrics': {
            'detection_rate': metrics['detection_rate'],
            'false_alarm_rate': metrics['false_alarm_rate'],
            'tracking_accuracy': metrics['tracking_accuracy'],
            'range_accuracy': metrics['range_accuracy'],
            'angle_accuracy': metrics['angle_accuracy'],
            'overall_score': metrics['overall_score']
        },
        'detailed_analysis': metrics.get('detailed_analysis'),
        'message': 'Performance metrics calculated'
    })


@api_bp.route('/analysis/monte-carlo', methods=['POST'])
@handle_api_error
def run_monte_carlo_analysis():
    """蒙特卡洛分析"""
    data = request.get_json()
    if not data:
        raise ValueError("Monte Carlo analysis parameters are required")

    # 验证参数
    if 'runs' not in data or data['runs'] < 1 or data['runs'] > 1000:
        raise ValueError("Number of runs must be between 1 and 1000")

    if 'base_config' not in data:
        raise ValueError("Base configuration is required")

    # 启动异步蒙特卡洛分析
    analysis_id = analysis_service.start_monte_carlo_analysis(data)

    return jsonify({
        'status': 'success',
        'analysis_id': analysis_id,
        'estimated_time': data['runs'] * 2,  # 估算时间（秒）
        'message': 'Monte Carlo analysis started'
    })


@api_bp.route('/analysis/<analysis_id>/status', methods=['GET'])
@handle_api_error
def get_analysis_status(analysis_id):
    """获取分析状态"""
    if not analysis_id:
        raise ValueError("Analysis ID is required")

    status = analysis_service.get_analysis_status(analysis_id)

    return jsonify({
        'status': 'success',
        'analysis_status': status,
        'message': 'Analysis status retrieved'
    })


@api_bp.route('/sensitivity/analysis', methods=['POST'])
@handle_api_error
def run_sensitivity_analysis():
    """敏感性分析"""
    data = request.get_json()
    if not data:
        raise ValueError("Sensitivity analysis parameters are required")

    result = analysis_service.run_sensitivity_analysis(data)

    return jsonify({
        'status': 'success',
        'sensitivity_results': result,
        'message': 'Sensitivity analysis completed'
    })


# ===============================
# 批量操作接口
# ===============================

@api_bp.route('/batch/simulations', methods=['POST'])
@handle_api_error
def run_batch_simulations():
    """批量仿真"""
    data = request.get_json()
    if not data:
        raise ValueError("Batch simulation parameters are required")

    is_valid, error_msg = validate_batch_simulation_request(data)
    if not is_valid:
        raise ValueError(error_msg)

    # 启动批量仿真
    batch_id = batch_service.start_batch_simulations(data['configurations'])

    return jsonify({
        'status': 'success',
        'batch_id': batch_id,
        'total_simulations': len(data['configurations']),
        'estimated_time': len(data['configurations']) * 30,  # 估算时间
        'message': 'Batch simulations started'
    })


@api_bp.route('/batch/<batch_id>/status', methods=['GET'])
@handle_api_error
def get_batch_status(batch_id):
    """获取批量仿真状态"""
    if not batch_id:
        raise ValueError("Batch ID is required")

    status = batch_service.get_batch_status(batch_id)

    return jsonify({
        'status': 'success',
        'batch_status': status,
        'message': 'Batch status retrieved'
    })


@api_bp.route('/comparison/results', methods=['POST'])
@handle_api_error
def compare_simulation_results():
    """结果对比分析"""
    data = request.get_json()
    if not data:
        raise ValueError("Comparison parameters are required")

    if 'simulation_ids' not in data or len(data['simulation_ids']) < 2:
        raise ValueError("At least 2 simulation IDs are required for comparison")

    if len(data['simulation_ids']) > 10:
        raise ValueError("Maximum 10 simulations can be compared at once")

    comparison_result = analysis_service.compare_simulation_results(data)

    return jsonify({
        'status': 'success',
        'comparison': comparison_result,
        'message': 'Simulation comparison completed'
    })


@api_bp.route('/optimization/parameters', methods=['POST'])
@handle_api_error
def optimize_parameters():
    """参数优化"""
    data = request.get_json()
    if not data:
        raise ValueError("Optimization parameters are required")

    optimization_id = analysis_service.start_parameter_optimization(data)

    return jsonify({
        'status': 'success',
        'optimization_id': optimization_id,
        'message': 'Parameter optimization started'
    })


# ===============================
# 资源管理接口
# ===============================

@api_bp.route('/resources/status', methods=['GET'])
@handle_api_error
def get_resource_status():
    """获取计算资源状态"""
    resource_status = simulation_service.get_resource_status()

    return jsonify({
        'status': 'success',
        'resources': resource_status,
        'message': 'Resource status retrieved'
    })


@api_bp.route('/queue/status', methods=['GET'])
@handle_api_error
def get_queue_status():
    """获取任务队列状态"""
    queue_status = simulation_service.get_queue_status()

    return jsonify({
        'status': 'success',
        'queue': queue_status,
        'message': 'Queue status retrieved'
    })


# ===============================
# 系统健康检查接口
# ===============================

@api_bp.route('/health', methods=['GET'])
@handle_api_error
def health_check():
    """系统健康检查"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0',
        'services': {
            'simulation': simulation_service.health_check(),
            'analysis': analysis_service.health_check(),
            'config': config_service.health_check(),
            'export': export_service.health_check()
        }
    }

    return jsonify(health_status)


# ===============================
# 错误处理
# ===============================

@api_bp.errorhandler(400)
def bad_request(error):
    return jsonify({
        'status': 'error',
        'error': 'bad_request',
        'message': 'Bad request'
    }), 400


@api_bp.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error': 'not_found',
        'message': 'Resource not found'
    }), 404


@api_bp.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'error': 'internal_error',
        'message': 'Internal server error'
    }), 500
