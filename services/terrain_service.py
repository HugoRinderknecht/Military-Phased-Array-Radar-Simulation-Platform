import numpy as np
from core.low_altitude_enhancer import TerrainMap, Track
import logging

logger = logging.getLogger(__name__)


class TerrainService:
    def __init__(self):
        self.terrain_cache = {}

    def create_terrain_map(self, elevation_data: np.ndarray, resolution: float,
                           origin: tuple, terrain_types: np.ndarray = None) -> TerrainMap:
        """创建地形图对象"""
        cache_key = hash((elevation_data.tobytes(), resolution, origin))
        if cache_key in self.terrain_cache:
            return self.terrain_cache[cache_key]

        height, width = elevation_data.shape
        terrain_map = TerrainMap(width, height, resolution)
        terrain_map.elevation = elevation_data
        terrain_map.origin_x, terrain_map.origin_y = origin

        if terrain_types is not None:
            terrain_map.terrain_type = terrain_types.astype(np.int32)

        self.terrain_cache[cache_key] = terrain_map
        return terrain_map

    def generate_terrain_from_gis(self, gis_data: dict) -> TerrainMap:
        """从GIS数据生成地形图"""
        # 实际实现中应解析GIS数据
        # 这里简化为创建基本地形
        width = gis_data.get('width', 1000)
        height = gis_data.get('height', 1000)
        resolution = gis_data.get('resolution', 10.0)
        origin = gis_data.get('origin', (0, 0))

        # 生成随机地形高程
        elevation = np.random.rand(height, width) * 500

        return self.create_terrain_map(elevation, resolution, origin)

    def calculate_occlusion_map(self, radar_pos: tuple, target_pos: tuple,
                                terrain_map: TerrainMap) -> dict:
        """计算地形遮蔽图"""
        # 使用低空增强器中的方法
        from core.low_altitude_enhancer import LowAltitudeEnhancer
        enhancer = LowAltitudeEnhancer()
        occlusion = enhancer.analyze_terrain_occlusion(
            Track(state=np.array(target_pos)),
            terrain_map
        )

        return {
            'severity': occlusion.severity,
            'probability': occlusion.probability,
            'azimuth_range': occlusion.azimuth_range,
            'elevation_range': occlusion.elevation_range
        }
