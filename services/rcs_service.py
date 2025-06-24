from core.rcs_processor import RCSProcessor
from models.radar_system import RadarSystem
from models.environment import Environment
from models.target import Target
import logging

logger = logging.getLogger(__name__)


class RCSService:
    def __init__(self):
        self.processors = {}

    def get_processor(self, radar_config: dict, env_config: dict) -> RCSProcessor:
        """获取或创建RCS处理器"""
        config_hash = hash(frozenset({**radar_config, **env_config}.items()))

        if config_hash not in self.processors:
            radar = RadarSystem(**radar_config)
            env = Environment(**env_config)
            self.processors[config_hash] = RCSProcessor(radar, env)
            logger.info(f"Created new RCSProcessor for config {config_hash}")

        return self.processors[config_hash]

    def calculate_rcs(self, target_data: dict, radar_config: dict, env_config: dict):
        """计算目标RCS"""
        processor = self.get_processor(radar_config, env_config)
        target = Target(**target_data)
        return processor.process_rcs(target)

    def batch_calculate(self, targets: list, radar_config: dict, env_config: dict):
        """批量计算RCS"""
        processor = self.get_processor(radar_config, env_config)
        results = []

        for target_data in targets:
            try:
                target = Target(**target_data)
                result = processor.process_rcs(target)
                results.append({
                    'target_id': target.target_id,
                    'rcs_value': result['rcs_value'],
                    'features': result['features'],
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'target_id': target_data.get('target_id', 'unknown'),
                    'error': str(e),
                    'status': 'error'
                })
                logger.error(f"RCS calculation failed for target: {str(e)}")

        return results
