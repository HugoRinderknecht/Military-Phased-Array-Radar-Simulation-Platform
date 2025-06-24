import numpy as np
from core.signal_processor import SignalProcessor
from models.radar_system import RadarSystem
from models.environment import Environment
import logging

logger = logging.getLogger(__name__)


class SignalService:
    def __init__(self):
        self.processors = {}

    def get_processor(self, radar_config: dict, env_config: dict) -> SignalProcessor:
        """获取或创建信号处理器"""
        config_hash = hash(frozenset({**radar_config, **env_config}.items()))

        if config_hash not in self.processors:
            radar = RadarSystem(**radar_config)
            env = Environment(**env_config)
            self.processors[config_hash] = SignalProcessor(radar, env)
            logger.info(f"Created new SignalProcessor for config {config_hash}")

        return self.processors[config_hash]

    def process_signal(self, signal_data: list, radar_config: dict, env_config: dict, params: dict = None):
        """处理原始信号数据"""
        processor = self.get_processor(radar_config, env_config)
        signal_array = np.array(signal_data, dtype=complex)
        return processor.process_radar_signal(signal_array, params)

    def batch_process(self, signals: list, radar_config: dict, env_config: dict):
        """批量处理信号数据"""
        processor = self.get_processor(radar_config, env_config)
        results = []

        for i, signal in enumerate(signals):
            try:
                signal_array = np.array(signal, dtype=complex)
                result = processor.process_radar_signal(signal_array)
                results.append({
                    'index': i,
                    'detections': [d.__dict__ for d in result],
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'status': 'error'
                })
                logger.error(f"Error processing signal {i}: {str(e)}")

        return results
