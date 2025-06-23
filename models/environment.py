from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
from numba import njit, float64, int32, prange

# 定义天气类型常量
WEATHER_TYPES = ["clear", "rain", "snow", "fog", "storm"]
TERRAIN_TYPES = ["flat", "hills", "urban", "forest", "sea", "mountain"]


# Numba优化的大气损失计算函数
@njit(float64(int32, float64, float64, float64, float64), cache=True)
def calculate_atmospheric_loss_numba(
        weather_type_idx: int,
        precipitation_rate: float,
        visibility: float,
        frequency: float,
        range_km: float
) -> float:
    """
    使用Numba优化的计算大气损失函数
    参数:
        weather_type_idx: 天气类型索引 (0=clear, 1=rain, 2=snow, 3=fog, 4=storm)
        precipitation_rate: 降水量 (mm/h)
        visibility: 能见度 (meters)
        frequency: 频率 (Hz)
        range_km: 距离 (km)
    返回:
        大气损失 (dB)
    """
    # 天气损失基础值
    base_loss = 0.1

    # 雨天细化分类
    if weather_type_idx == 1:  # rain
        if precipitation_rate < 2.5:
            base_loss = 0.3
        elif precipitation_rate < 10.0:
            base_loss = 1.2
        else:
            base_loss = 1.5  # 极强降雨按风暴处理
    elif weather_type_idx == 2:  # snow
        base_loss = 0.8
    elif weather_type_idx == 3:  # fog
        base_loss = 0.5
    elif weather_type_idx == 4:  # storm
        base_loss = 1.5

    # 频率因子
    freq_factor = (frequency / 10e9) ** 0.6

    # 能见度因子
    visibility_factor = 1.0
    if visibility < 1000:
        visibility_factor = 1.0 + (1000 - visibility) / 1000 * 0.5

    return base_loss * freq_factor * range_km * visibility_factor


# Numba优化的杂波RCS计算函数
@njit(float64(int32, float64, float64), cache=True)
def calculate_clutter_rcs_numba(
        terrain_type_idx: int,
        clutter_density: float,
        cell_area: float
) -> float:
    """
    使用Numba优化的计算杂波RCS函数
    参数:
        terrain_type_idx: 地形类型索引 (0=flat, 1=hills, 2=urban, 3=forest, 4=sea, 5=mountain)
        clutter_density: 杂波密度 (0-1)
        cell_area: 单元面积 (m²)
    返回:
        杂波RCS (dB)
    """
    # 地形基础杂波值
    base_clutter = -20.0  # 默认值

    if terrain_type_idx == 0:  # flat
        base_clutter = -20.0
    elif terrain_type_idx == 1:  # hills
        base_clutter = -15.0
    elif terrain_type_idx == 2:  # urban
        base_clutter = -10.0
    elif terrain_type_idx == 3:  # forest
        base_clutter = -25.0
    elif terrain_type_idx == 4:  # sea
        base_clutter = -30.0
    elif terrain_type_idx == 5:  # mountain
        base_clutter = -12.0

    return base_clutter + 10.0 * np.log10(cell_area * clutter_density)


# Numba优化的噪声系数计算函数
@njit(float64(int32, int32, float64), cache=True)
def calculate_noise_figure_numba(
        weather_type_idx: int,
        electronic_warfare: int,
        interference_level: float
) -> float:
    """
    使用Numba优化的计算噪声系数函数
    参数:
        weather_type_idx: 天气类型索引 (0=clear, 1=rain, 2=snow, 3=fog, 4=storm)
        electronic_warfare: 电子战状态 (0=False, 1=True)
        interference_level: 干扰水平 (0-1)
    返回:
        噪声系数 (dB)
    """
    base_nf = 3.0

    # 电子战影响
    if electronic_warfare:
        base_nf += 5.0

    # 干扰水平影响
    if interference_level > 0.5:
        base_nf += 2.0

    # 天气影响
    if weather_type_idx in (1, 4):  # rain or storm
        if weather_type_idx == 1:  # rain
            # 根据降雨强度调整噪声系数
            base_nf += min(interference_level * 2.0, 1.5)
        else:  # storm
            base_nf += 1.0

    return base_nf


@dataclass
class WeatherCondition:
    # 使用全局常量代替类常量
    VALID_TYPES = WEATHER_TYPES

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

        # 预计算天气类型索引用于高效计算
        self.weather_type_idx = self.VALID_TYPES.index(self.weather_type)

    def atmospheric_loss(self, frequency: float, range_km: float) -> float:
        """计算大气损失 - 使用Numba优化版本"""
        return calculate_atmospheric_loss_numba(
            self.weather_type_idx,
            self.precipitation_rate,
            self.visibility,
            frequency,
            range_km
        )


@dataclass
class Environment:
    # 使用全局常量代替类常量
    VALID_TERRAIN_TYPES = TERRAIN_TYPES

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

        # 预计算地形类型索引用于高效计算
        self.terrain_type_idx = self.VALID_TERRAIN_TYPES.index(self.terrain_type)
        # 预转换电子战状态为整数 (0/1)
        self.electronic_warfare_int = 1 if self.electronic_warfare else 0

    def get_clutter_rcs(self, cell_area: float) -> float:
        """计算杂波RCS - 使用Numba优化版本"""
        if cell_area <= 0:
            raise ValueError("cell_area must be positive")

        return calculate_clutter_rcs_numba(
            self.terrain_type_idx,
            self.clutter_density,
            cell_area
        )

    def get_noise_figure(self) -> float:
        """计算噪声系数 - 使用Numba优化版本"""
        return calculate_noise_figure_numba(
            self.weather.weather_type_idx,
            self.electronic_warfare_int,
            self.interference_level
        )

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


# 预编译Numba函数以提高首次调用性能
_ = calculate_atmospheric_loss_numba(0, 0.0, 10000.0, 10e9, 100.0)
_ = calculate_clutter_rcs_numba(0, 0.3, 100.0)
_ = calculate_noise_figure_numba(0, 0, 0.1)
