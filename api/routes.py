from flask import Blueprint, request, jsonify
from services.simulation_service import SimulationService
from services.analysis_service import AnalysisService
from api.schemas import validate_simulation_request, validate_random_target_request
import traceback
import logging
import numpy as np

# 设置日志
logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)
simulation_service = SimulationService()
analysis_service = AnalysisService()


@api_bp.route('/simulate', methods=['POST'])
def run_simulation():
    try:
        data = request.get_json()

        is_valid, error_msg = validate_simulation_request(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'msg': error_msg
            }), 400

        init_result = simulation_service.initialize_simulation(data)
        if init_result['status'] == 'error':
            return jsonify(init_result), 400

        simulation_result = simulation_service.run_simulation()

        return jsonify(simulation_result)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api_bp.route('/status', methods=['GET'])
def get_simulation_status():
    try:
        status = simulation_service.get_simulation_status()
        return jsonify({
            'status': 'success',
            'data': status
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/analyze', methods=['POST'])
def analyze_results():
    try:
        data = request.get_json()

        if 'results' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing results data'
            }), 400

        analysis = analysis_service.analyze_simulation_results(data['results'])

        return jsonify({
            'status': 'success',
            'analysis': analysis
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/performance', methods=['POST'])
def get_performance_metrics():
    try:
        data = request.get_json()

        metrics = analysis_service.calculate_performance_metrics(data)

        return jsonify({
            'status': 'success',
            'metrics': metrics
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/export', methods=['POST'])
def export_results():
    try:
        data = request.get_json()

        export_result = analysis_service.export_simulation_data(
            data['results'],
            data.get('format', 'json')
        )

        return jsonify({
            'status': 'success',
            'export': export_result
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@api_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'success',
        'message': 'Radar simulation API is running',
        'version': '1.0.0'
    })


@api_bp.route('/reset', methods=['POST'])
def reset_simulation():
    try:
        simulation_service.reset_simulation()
        return jsonify({
            'status': 'success',
            'message': 'Simulation reset successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# 新增的随机目标生成API
@api_bp.route('/generate-random-targets', methods=['POST'])
def generate_random_targets():
    """生成随机目标"""
    try:
        data = request.get_json()

        # 验证输入参数
        is_valid, errors = validate_random_target_request(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid input parameters',
                'details': errors
            }), 400

        # 从models.target导入生成器
        from models.target import RandomTargetGenerator

        generator = RandomTargetGenerator()

        # 提取参数
        num_targets = data.get('num_targets', 5)
        altitude_type = data.get('altitude_type', 'mixed')
        velocity_type = data.get('velocity_type', 'mixed')
        rcs_type = data.get('rcs_type', 'mixed')
        radar_range = data.get('radar_range', 100000)
        enable_formation = data.get('enable_formation', True)
        specific_types = data.get('specific_types', None)
        scenario_type = data.get('scenario_type', None)

        # 生成目标
        if scenario_type:
            targets = generator.generate_threat_scenario(scenario_type)
        else:
            targets = generator.generate_random_targets(
                num_targets=num_targets,
                altitude_type=altitude_type,
                velocity_type=velocity_type,
                rcs_type=rcs_type,
                radar_range=radar_range,
                enable_formation=enable_formation,
                specific_types=specific_types
            )

        return jsonify({
            'success': True,
            'targets': targets,
            'summary': {
                'total_targets': len(targets),
                'formations': len(set([t['formation_id'] for t in targets if t['formation_id']])),
                'target_types': list(set([t['type'] for t in targets]))
            }
        })

    except Exception as e:
        logger.error(f"Random target generation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Random target generation failed',
            'details': str(e)
        }), 500


@api_bp.route('/target-presets', methods=['GET'])
def get_target_presets():
    """获取目标预设配置"""
    try:
        from models.target import RandomTargetGenerator

        generator = RandomTargetGenerator()

        return jsonify({
            'success': True,
            'presets': {
                'altitude_types': list(generator.altitude_ranges.keys()),
                'velocity_types': list(generator.velocity_ranges.keys()),
                'rcs_types': list(generator.rcs_ranges.keys()),
                'target_types': list(generator.target_types.keys()),
                'formation_types': list(generator.formation_types.keys()),
                'scenario_types': ['air_raid', 'patrol', 'low_altitude_penetration', 'swarm_attack', 'mixed']
            },
            'ranges': {
                'altitude_ranges': generator.altitude_ranges,
                'velocity_ranges': generator.velocity_ranges,
                'rcs_ranges': generator.rcs_ranges,
                'target_specifications': generator.target_types
            }
        })

    except Exception as e:
        logger.error(f"Get target presets error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to get target presets',
            'details': str(e)
        }), 500


@api_bp.route('/random-scenario/<scenario_type>', methods=['POST'])
def generate_random_scenario(scenario_type):
    """生成预定义威胁场景"""
    try:
        from models.target import RandomTargetGenerator

        generator = RandomTargetGenerator()
        targets = generator.generate_threat_scenario(scenario_type)

        return jsonify({
            'success': True,
            'scenario_type': scenario_type,
            'targets': targets,
            'summary': {
                'total_targets': len(targets),
                'formations': len(set([t['formation_id'] for t in targets if t['formation_id']])),
                'target_types': list(set([t['type'] for t in targets])),
                'altitude_range': [
                    min([t['altitude'] for t in targets]),
                    max([t['altitude'] for t in targets])
                ] if targets else [0, 0],
                'velocity_range': [
                    min([np.linalg.norm(t['velocity']) for t in targets]),
                    max([np.linalg.norm(t['velocity']) for t in targets])
                ] if targets else [0, 0]
            }
        })

    except Exception as e:
        logger.error(f"Random scenario generation error: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Random scenario generation failed',
            'details': str(e)
        }), 500


@api_bp.route('/simulate-with-random-targets', methods=['POST'])
def simulate_with_random_targets():
    """运行带随机目标的仿真"""
    try:
        data = request.get_json()

        # 验证基本仿真参数
        is_valid, error_msg = validate_simulation_request(data)
        if not is_valid:
            return jsonify({
                'status': 'error',
                'message': error_msg
            }), 400

        # 如果包含随机目标配置，先验证随机目标参数
        if 'random_targets' in data:
            is_valid, errors = validate_random_target_request(data['random_targets'])
            if not is_valid:
                return jsonify({
                    'status': 'error',
                    'message': 'Invalid random target parameters',
                    'details': errors
                }), 400

        # 运行带随机目标的仿真
        result = simulation_service.run_simulation_with_random_targets(data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Random target simulation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Random target simulation failed',
            'details': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api_bp.route('/target-statistics', methods=['POST'])
def get_target_statistics():
    """获取目标统计信息"""
    try:
        data = request.get_json()

        if 'targets' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing targets data'
            }), 400

        targets = data['targets']

        # 计算统计信息
        stats = {
            'total_count': len(targets),
            'type_distribution': {},
            'altitude_stats': {},
            'velocity_stats': {},
            'rcs_stats': {},
            'formation_stats': {}
        }

        if not targets:
            return jsonify({
                'status': 'success',
                'statistics': stats
            })

        # 提取数据
        altitudes = [t.get('altitude', 0) for t in targets]
        velocities = [np.linalg.norm(t.get('velocity', [0, 0, 0])) for t in targets]
        rcs_values = [t.get('rcs', 0) for t in targets]
        target_types = [t.get('type', 'unknown') for t in targets]
        formation_ids = [t.get('formation_id') for t in targets]

        # 类型分布
        for ttype in target_types:
            stats['type_distribution'][ttype] = stats['type_distribution'].get(ttype, 0) + 1

        # 高度统计
        stats['altitude_stats'] = {
            'min': float(np.min(altitudes)),
            'max': float(np.max(altitudes)),
            'mean': float(np.mean(altitudes)),
            'std': float(np.std(altitudes))
        }

        # 速度统计
        stats['velocity_stats'] = {
            'min': float(np.min(velocities)),
            'max': float(np.max(velocities)),
            'mean': float(np.mean(velocities)),
            'std': float(np.std(velocities))
        }

        # RCS统计
        stats['rcs_stats'] = {
            'min': float(np.min(rcs_values)),
            'max': float(np.max(rcs_values)),
            'mean': float(np.mean(rcs_values)),
            'std': float(np.std(rcs_values))
        }

        # 编队统计
        formations = [fid for fid in formation_ids if fid is not None]
        unique_formations = set(formations)

        stats['formation_stats'] = {
            'formation_count': len(unique_formations),
            'formation_members': len(formations),
            'single_targets': len(targets) - len(formations),
            'avg_formation_size': len(formations) / len(unique_formations) if unique_formations else 0
        }

        return jsonify({
            'status': 'success',
            'statistics': stats
        })

    except Exception as e:
        logger.error(f"Target statistics error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Failed to calculate target statistics',
            'details': str(e)
        }), 500
