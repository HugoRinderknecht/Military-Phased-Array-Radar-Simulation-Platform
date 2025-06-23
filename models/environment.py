from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


@dataclass
class WeatherCondition:
    # 添加有效天气类型常量
    VALID_TYPES = ["clear", "rain", "snow", "fog", "storm"]

    weather_type: str  # "clear", "rain", "snow", "fog", "storm"
    precipitation_rate: float = 0.0  # mm/h
    visibility: float = 10000.0  # meters
    wind_speed: float = 0.0  # m/s
    wind_direction: float = 0.0  # radians
    temperature: float = 15.0  # celsius
    humidity: float = 50.0  # percentage

    def __post_init__(self):
        """验证参数有效性"""
        if self.weather_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid weather_type: '{self.weather_type}'. Must be one of: {self.VALID_TYPES}")

        # 验证其他参数范围
        if self.precipitation_rate < 0:
            raise ValueError("precipitation_rate must be non-negative")

        if self.visibility <= 0:
            raise ValueError("visibility must be positive")

        if self.wind_speed < 0:
            raise ValueError("wind_speed must be non-negative")

        if not 0 <= self.humidity <= 100:
            raise ValueError("humidity must be between 0 and 100")

    def atmospheric_loss(self, frequency: float, range_km: float) -> float:
        """计算大气损失，细化降水分类逻辑"""
        loss_map = {
            "clear": 0.1,
            "light_rain": 0.3,
            "heavy_rain": 1.2,
            "snow": 0.8,
            "fog": 0.5,
            "storm": 1.5  # 添加风暴类型
        }

        # 细化雨天的分类逻辑
        if self.weather_type == "rain":
            if self.precipitation_rate < 2.5:
                weather_key = "light_rain"
            elif self.precipitation_rate < 10.0:
                weather_key = "heavy_rain"
            else:
                weather_key = "storm"  # 极强降雨按风暴处理
        elif self.weather_type == "storm":
            weather_key = "storm"
        else:
            weather_key = self.weather_type

        base_loss = loss_map.get(weather_key, 0.1)
        freq_factor = (frequency / 10e9) ** 0.6

        # 根据能见度调整损失
        if self.visibility < 1000:  # 低能见度增加额外损失
            visibility_factor = 1.0 + (1000 - self.visibility) / 1000 * 0.5
        else:
            visibility_factor = 1.0

        return base_loss * freq_factor * range_km * visibility_factor


@dataclass
class Environment:
    # 添加有效地形类型常量
    VALID_TERRAIN_TYPES = ["flat", "hills", "urban", "forest", "sea", "mountain"]

    weather: WeatherCondition
    clutter_density: float = 0.3
    interference_level: float = 0.1
    multipath_factor: float = 0.2
    electronic_warfare: bool = False
    terrain_type: str = "flat"

    def __post_init__(self):
        """验证参数有效性"""
        if self.terrain_type not in self.VALID_TERRAIN_TYPES:
            raise ValueError(f"Invalid terrain_type: '{self.terrain_type}'. Must be one of: {self.VALID_TERRAIN_TYPES}")

        if not 0 <= self.clutter_density <= 1:
            raise ValueError("clutter_density must be between 0 and 1")

        if not 0 <= self.interference_level <= 1:
            raise ValueError("interference_level must be between 0 and 1")

        if not 0 <= self.multipath_factor <= 1:
            raise ValueError("multipath_factor must be between 0 and 1")

    def get_clutter_rcs(self, cell_area: float) -> float:
        """计算杂波RCS"""
        if cell_area <= 0:
            raise ValueError("cell_area must be positive")

        clutter_map = {
            "sea": -30,
            "flat": -20,
            "hills": -15,
            "urban": -10,
            "forest": -25,
            "mountain": -12  # 添加山地类型
        }
        base_clutter = clutter_map.get(self.terrain_type, -20)
        return base_clutter + 10 * np.log10(cell_area * self.clutter_density)

    def get_noise_figure(self) -> float:
        """计算噪声系数"""
        base_nf = 3.0
        if self.electronic_warfare:
            base_nf += 5.0
        if self.interference_level > 0.5:
            base_nf += 2.0

        # 根据天气条件调整噪声系数
        if self.weather.weather_type in ["storm", "heavy_rain"]:
            base_nf += 1.0

        return base_nf

    def validate_environment(self) -> Dict[str, Any]:
        """验证环境配置的完整性"""
        try:
            # 验证天气条件
            if not isinstance(self.weather, WeatherCondition):
                return {"status": "error", "message": "Invalid weather condition object"}

            # 验证参数组合的合理性
            warnings = []

            # 检查参数组合的一致性
            if (self.weather.weather_type == "clear" and
                    self.weather.precipitation_rate > 0):
                warnings.append("Clear weather with non-zero precipitation rate")

            if (self.terrain_type == "sea" and
                    self.clutter_density > 0.1):
                warnings.append("High clutter density for sea terrain")

            if (self.electronic_warfare and
                    self.interference_level < 0.3):
                warnings.append("Electronic warfare enabled but low interference level")

            return {
                "status": "success",
                "warnings": warnings,
                "message": "Environment validation completed"
            }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Environment validation failed: {str(e)}"
            }

    @classmethod
    def create_preset_environment(cls, preset_name: str) -> 'Environment':
        """创建预设环境"""
        presets = {
            "clear_day": cls(
                weather=WeatherCondition(
                    weather_type="clear",
                    visibility=15000.0,
                    temperature=20.0,
                    humidity=40.0
                ),
                clutter_density=0.2,
                interference_level=0.05,
                terrain_type="flat"
            ),
            "heavy_rain": cls(
                weather=WeatherCondition(
                    weather_type="rain",
                    precipitation_rate=15.0,
                    visibility=2000.0,
                    temperature=15.0,
                    humidity=90.0
                ),
                clutter_density=0.4,
                interference_level=0.2,
                terrain_type="flat"
            ),
            "urban_interference": cls(
                weather=WeatherCondition(
                    weather_type="clear",
                    temperature=25.0,
                    humidity=55.0
                ),
                clutter_density=0.6,
                interference_level=0.7,
                electronic_warfare=True,
                terrain_type="urban"
            ),
            "mountain_snow": cls(
                weather=WeatherCondition(
                    weather_type="snow",
                    precipitation_rate=5.0,
                    visibility=5000.0,
                    temperature=-5.0,
                    humidity=80.0
                ),
                clutter_density=0.3,
                interference_level=0.1,
                multipath_factor=0.4,
                terrain_type="mountain"
            )
        }

        if preset_name not in presets:
            raise ValueError(f"Unknown preset: '{preset_name}'. Available presets: {list(presets.keys())}")

        return presets[preset_name]
