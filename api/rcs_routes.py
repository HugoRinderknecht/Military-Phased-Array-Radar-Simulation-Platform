from flask import Blueprint, request, jsonify
from services.rcs_service import RCSService
import logging

logger = logging.getLogger(__name__)
rcs_bp = Blueprint('rcs', __name__)
rcs_service = RCSService()

@rcs_bp.route('/calculate', methods=['POST'])
def calculate_rcs():
    data = request.json
    try:
        result = rcs_service.calculate_rcs(
            data['target_data'],
            data['radar_config'],
            data['environment_config']
        )
        return jsonify({
            'status': 'success',
            'result': result
        })
    except Exception as e:
        logger.error(f"RCS calculation failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@rcs_bp.route('/batch-calculate', methods=['POST'])
def batch_calculate_rcs():
    data = request.json
    try:
        results = rcs_service.batch_calculate(
            data['targets'],
            data['radar_config'],
            data['environment_config']
        )
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        logger.error(f"Batch RCS calculation failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
