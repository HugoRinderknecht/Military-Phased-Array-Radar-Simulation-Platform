from flask import Blueprint, request, jsonify
from services.visualization_service import VisualizationService
from api.schemas import validate_uuid_format, validate_time_range
from functools import wraps
import logging

logger = logging.getLogger(__name__)

visualization_bp = Blueprint('visualization', __name__)
visualization_service = VisualizationService()


def handle_visualization_error(f):
    """可视化API错误处理装饰器"""

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


@visualization_bp.route('/radar-display/<simulation_id>', methods=['GET'])
@handle_visualization_error
def get_radar_display_data(simulation_id):
    """获取雷达显示数据"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    use_cache = request.args.get('use_cache', 'true').lower() == 'true'

    result = visualization_service.get_radar_display_data(simulation_id, use_cache)

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': result['error']
        }), 404

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'data': result['data']
    })


@visualization_bp.route('/chart-data/<simulation_id>/<chart_type>', methods=['GET'])
@handle_visualization_error
def get_chart_data(simulation_id, chart_type):
    """获取图表数据"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    valid_chart_types = ['performance', 'environment_impact', 'target_analysis', 'detection_statistics']
    if chart_type not in valid_chart_types:
        raise ValueError(f"Invalid chart type. Must be one of: {', '.join(valid_chart_types)}")

    # 解析时间范围参数
    start_time = request.args.get('start_time', type=float)
    end_time = request.args.get('end_time', type=float)
    time_range = None

    if start_time is not None and end_time is not None:
        is_valid, error_msg = validate_time_range(start_time, end_time)
        if not is_valid:
            raise ValueError(error_msg)
        time_range = (start_time, end_time)

    use_cache = request.args.get('use_cache', 'true').lower() == 'true'

    result = visualization_service.get_chart_data(simulation_id, chart_type, time_range, use_cache)

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': result['error']
        }), 404

    return jsonify({
        'status': 'success',
        'simulation_id': simulation_id,
        'chart_type': chart_type,
        'data': result['data']
    })


@visualization_bp.route('/realtime-data/<simulation_id>', methods=['GET'])
@handle_visualization_error
def get_realtime_data(simulation_id):
    """获取实时数据"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    result = visualization_service.get_realtime_data(simulation_id)

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': result['error']
        }), 404

    return jsonify({
        'status': 'success',
        'data': result['data']
    })


@visualization_bp.route('/historical-data/<simulation_id>', methods=['GET'])
@handle_visualization_error
def get_historical_data(simulation_id):
    """获取历史数据"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    # 解析参数
    start_time = request.args.get('start_time', type=float)
    end_time = request.args.get('end_time', type=float)
    resolution = request.args.get('resolution', 'medium')

    if start_time is None or end_time is None:
        raise ValueError("start_time and end_time are required")

    is_valid, error_msg = validate_time_range(start_time, end_time)
    if not is_valid:
        raise ValueError(error_msg)

    if resolution not in ['high', 'medium', 'low']:
        raise ValueError("resolution must be 'high', 'medium', or 'low'")

    result = visualization_service.get_historical_data(simulation_id, start_time, end_time, resolution)

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': result['error']
        }), 404

    return jsonify({
        'status': 'success',
        'data': result['data']
    })


@visualization_bp.route('/summary/<simulation_id>', methods=['GET'])
@handle_visualization_error
def get_simulation_summary(simulation_id):
    """获取仿真总结"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    result = visualization_service.get_simulation_summary(simulation_id)

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': result['error']
        }), 404

    return jsonify({
        'status': 'success',
        'summary': result['summary']
    })


@visualization_bp.route('/export-preview/<simulation_id>', methods=['POST'])
@handle_visualization_error
def get_export_preview(simulation_id):
    """获取导出预览"""
    if not validate_uuid_format(simulation_id):
        raise ValueError("Invalid simulation ID format")

    export_config = request.get_json()
    if not export_config:
        raise ValueError("Export configuration is required")

    result = visualization_service.get_export_preview_data(simulation_id, export_config)

    if 'error' in result:
        return jsonify({
            'status': 'error',
            'message': result['error']
        }), 404

    return jsonify({
        'status': 'success',
        'preview': result['preview_data'],
        'estimated_size': result['estimated_size']
    })
