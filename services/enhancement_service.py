from core.low_altitude_enhancer import LowAltitudeEnhancer, Track
from models.radar_system import RadarSystem
from models.environment import Environment
import numpy as np
import logging

logger = logging.getLogger(__name__)


class EnhancementService:
    def __init__(self):
        self.enhancers = {}
        self.terrain_cache = {}

    def get_enhancer(self, radar_config: dict, env_config: dict) -> LowAltitudeEnhancer:
        config_hash = self._generate_config_hash(radar_config, env_config)

        if config_hash not in self.enhancers:
            radar = RadarSystem(**radar_config)
            env = Environment(**env_config)
            self.enhancers[config_hash] = LowAltitudeEnhancer(
                radar_frequency=radar.frequency,
                environment=env
            )
            logger.info(f"Created new LowAltitudeEnhancer for config {config_hash}")

        return self.enhancers[config_hash]

    def enhance_targets(self, tracks: list, terrain_data: dict,
                        radar_config: dict, env_config: dict) -> list:
        enhancer = self.get_enhancer(radar_config, env_config)

        # 转换轨道数据为Track对象
        track_objects = []
        for track in tracks:
            track_obj = Track(
                state=np.array(track['state']),
                score=track.get('score', 1.0),
                confirmed=track.get('confirmed', False),
                target_type=track.get('target_type'),
                rcs_history=track.get('rcs_history', []),
                age=track.get('age', 0)
            )
            track_objects.append(track_obj)

        # 创建地形图
        terrain_map = self._create_terrain_map(terrain_data)

        # 执行增强处理
        enhanced_tracks = enhancer.enhance_low_altitude_targets(track_objects, terrain_map)

        # 转换回字典格式
        result = []
        for track in enhanced_tracks:
            result.append({
                'state': track.state.tolist(),
                'score': track.score,
                'confirmed': track.confirmed,
                'enhancement_applied': True
            })

        return result

    def fuse_sensors(self, radar_data: dict, ir_data: dict,
                     radar_config: dict, env_config: dict) -> dict:
        enhancer = self.get_enhancer(radar_config, env_config)
        return enhancer.fuse_with_auxiliary_sensors(radar_data, ir_data)

    def _create_terrain_map(self, terrain_data: dict) -> 'TerrainMap':
        """创建地形图对象"""
        # 使用缓存避免重复创建
        cache_key = hash(frozenset(terrain_data.items()))
        if cache_key in self.terrain_cache:
            return self.terrain_cache[cache_key]

        # 实际实现中需要根据地形数据创建TerrainMap对象
        # 这里简化为返回None，实际应实现完整地形处理
        return None

    def _generate_config_hash(self, radar_config: dict, env_config: dict) -> int:
        """生成配置哈希值"""
        combined = {
            'radar': radar_config,
            'environment': env_config
        }
        return hash(frozenset(combined.items()))
