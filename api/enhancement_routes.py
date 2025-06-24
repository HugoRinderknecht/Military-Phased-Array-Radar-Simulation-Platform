from flask import Blueprint, request, jsonify
from services.enhancement_service import EnhancementService
import logging

logger = logging.getLogger(__name__)
enhancement_bp = Blueprint('enhancement', __name__)
enhancement_service = EnhancementService()


@enhancement_bp.route('/low-altitude', methods=['POST'])
def enhance_low_altitude():
    data = request.json
    try:
        # 参数验证
        required_fields = ['tracks', 'terrain_data', 'radar_config', 'environment_config']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        enhanced_tracks = enhancement_service.enhance_targets(
            data['tracks'],
            data['terrain_data'],
            data['radar_config'],
            data['environment_config']
        )

        return jsonify({
            'status': 'success',
            'enhanced_tracks': enhanced_tracks
        })
    except Exception as e:
        logger.error(f"Low altitude enhancement failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@enhancement_bp.route('/fuse-sensors', methods=['POST'])
def fuse_sensors():
    data = request.json
    try:
        # 参数验证
        required_fields = ['radar_data', 'ir_data', 'radar_config', 'environment_config']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'status': 'error',
                    'message': f'Missing required field: {field}'
                }), 400

        fused_data = enhancement_service.fuse_sensors(
            data['radar_data'],
            data['ir_data'],
            data['radar_config'],
            data['environment_config']
        )

        return jsonify({
            'status': 'success',
            'fused_data': fused_data
        })
    except Exception as e:
        logger.error(f"Sensor fusion failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
