from flask import Blueprint, request, jsonify
from services.signal_service import SignalService
import logging

logger = logging.getLogger(__name__)
signal_bp = Blueprint('signal', __name__)
signal_service = SignalService()

@signal_bp.route('/process', methods=['POST'])
def process_signal():
    data = request.json
    try:
        result = signal_service.process_signal(
            data['signal_data'],
            data['radar_config'],
            data['environment_config'],
            data.get('processing_params', {})
        )
        return jsonify({
            'status': 'success',
            'detections': [d.__dict__ for d in result]
        })
    except Exception as e:
        logger.error(f"Signal processing failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@signal_bp.route('/batch-process', methods=['POST'])
def batch_process_signals():
    data = request.json
    try:
        results = signal_service.batch_process(
            data['signals'],
            data['radar_config'],
            data['environment_config']
        )
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        logger.error(f"Batch signal processing failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
