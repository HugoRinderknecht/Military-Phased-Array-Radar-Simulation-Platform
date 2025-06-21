import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple, Union
from jsonschema import validate, ValidationError
import json


class DataValidator:
    """数据验证工具类"""

    @staticmethod
    def is_numeric(value: Any) -> bool:
        """检查是否为数值"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_positive(value: Union[int, float]) -> bool:
        """检查是否为正数"""
        return DataValidator.is_numeric(value) and float(value) > 0

    @staticmethod
    def is_non_negative(value: Union[int, float]) -> bool:
        """检查是否为非负数"""
        return DataValidator.is_numeric(value) and float(value) >= 0

    @staticmethod
    def is_in_range(value: Union[int, float], min_val: Union[int, float],
                    max_val: Union[int, float], inclusive: bool = True) -> bool:
        """检查数值是否在指定范围内"""
        if not DataValidator.is_numeric(value):
            return False

        val = float(value)
        if inclusive:
            return min_val <= val <= max_val
        else:
            return min_val < val < max_val

    @staticmethod
    def is_integer(value: Any) -> bool:
        """检查是否为整数"""
        try:
            return float(value).is_integer()
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_boolean(value: Any) -> bool:
        """检查是否为布尔值"""
        return isinstance(value, bool) or str(value).lower() in ['true', 'false', '1', '0']

    @staticmethod
    def is_list_of_type(value: Any, expected_type: type) -> bool:
        """检查是否为指定类型的列表"""
        return isinstance(value, list) and all(isinstance(item, expected_type) for item in value)

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """检查邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """检查URL格式"""
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
        return re.match(pattern, url) is not None

    @staticmethod
    def is_valid_json(json_string: str) -> bool:
        """检查JSON格式"""
        try:
            json.loads(json_string)
            return True
        except json.JSONDecodeError:
            return False


class RadarDataValidator:
    """雷达数据验证器"""

    @staticmethod
    def validate_frequency(freq: Union[int, float]) -> Tuple[bool, str]:
        """验证频率参数"""
        if not DataValidator.is_numeric(freq):
            return False, "频率必须是数值"

        freq_val = float(freq)
        if freq_val <= 0:
            return False, "频率必须为正数"

        # 雷达频段检查
        if not (1e6 <= freq_val <= 100e9):  # 1MHz to 100GHz
            return False, "频率超出合理范围 (1MHz - 100GHz)"

        return True, "频率参数有效"

    @staticmethod
    def validate_power(power: Union[int, float]) -> Tuple[bool, str]:
        """验证功率参数"""
        if not DataValidator.is_numeric(power):
            return False, "功率必须是数值"

        power_val = float(power)
        if power_val <= 0:
            return False, "功率必须为正数"

        # 雷达功率合理范围检查 (1W to 10MW)
        if not (1 <= power_val <= 10e6):
            return False, "功率超出合理范围 (1W - 10MW)"

        return True, "功率参数有效"

    @staticmethod
    def validate_range(range_val: Union[int, float]) -> Tuple[bool, str]:
        """验证距离参数"""
        if not DataValidator.is_numeric(range_val):
            return False, "距离必须是数值"

        range_float = float(range_val)
        if range_float < 0:
            return False, "距离不能为负数"

        # 雷达探测距离合理范围 (0 - 1000km)
        if range_float > 1000000:  # 1000 km in meters
            return False, "距离超出合理范围 (0 - 1000km)"

        return True, "距离参数有效"

    @staticmethod
    def validate_angle(angle: Union[int, float], angle_type: str = "azimuth") -> Tuple[bool, str]:
        """验证角度参数"""
        if not DataValidator.is_numeric(angle):
            return False, f"{angle_type}角度必须是数值"

        angle_val = float(angle)

        if angle_type == "azimuth":
            if not (-np.pi <= angle_val <= np.pi):
                return False, "方位角必须在 [-π, π] 范围内"
        elif angle_type == "elevation":
            if not (-np.pi / 2 <= angle_val <= np.pi / 2):
                return False, "俯仰角必须在 [-π/2, π/2] 范围内"

        return True, f"{angle_type}角度参数有效"

    @staticmethod
    def validate_velocity(velocity: Union[int, float, List]) -> Tuple[bool, str]:
        """验证速度参数"""
        if isinstance(velocity, (list, tuple, np.ndarray)):
            for v in velocity:
                if not DataValidator.is_numeric(v):
                    return False, "速度分量必须是数值"
                if abs(float(v)) > 3000:  # 马赫数约为3的速度
                    return False, f"速度分量超出合理范围 [-3000, 3000] m/s: {v}"
        else:
            if not DataValidator.is_numeric(velocity):
                return False, "速度必须是数值"
            if abs(float(velocity)) > 3000:
                return False, "速度超出合理范围 [-3000, 3000] m/s"

        return True, "速度参数有效"

    @staticmethod
    def validate_rcs(rcs: Union[int, float]) -> Tuple[bool, str]:
        """验证RCS参数"""
        if not DataValidator.is_numeric(rcs):
            return False, "RCS必须是数值"

        rcs_val = float(rcs)
        # RCS合理范围 (-60 to 60 dBsm)
        if not (-60 <= rcs_val <= 60):
            return False, "RCS超出合理范围 [-60, 60] dBsm"

        return True, "RCS参数有效"


class EnvironmentValidator:
    """环境参数验证器"""

    @staticmethod
    def validate_weather_type(weather_type: str) -> Tuple[bool, str]:
        """验证天气类型"""
        valid_types = ["clear", "rain", "snow", "fog", "storm"]
        if weather_type not in valid_types:
            return False, f"无效的天气类型。支持的类型: {valid_types}"
        return True, "天气类型有效"

    @staticmethod
    def validate_precipitation_rate(rate: Union[int, float]) -> Tuple[bool, str]:
        """验证降雨率"""
        if not DataValidator.is_numeric(rate):
            return False, "降雨率必须是数值"

        rate_val = float(rate)
        if rate_val < 0:
            return False, "降雨率不能为负数"

        if rate_val > 200:  # mm/h
            return False, "降雨率超出合理范围 (0-200 mm/h)"

        return True, "降雨率参数有效"

    @staticmethod
    def validate_visibility(visibility: Union[int, float]) -> Tuple[bool, str]:
        """验证能见度"""
        if not DataValidator.is_numeric(visibility):
            return False, "能见度必须是数值"

        vis_val = float(visibility)
        if vis_val <= 0:
            return False, "能见度必须为正数"

        if vis_val > 50000:  # 50km
            return False, "能见度超出合理范围 (0-50km)"

        return True, "能见度参数有效"

    @staticmethod
    def validate_terrain_type(terrain_type: str) -> Tuple[bool, str]:
        """验证地形类型"""
        valid_types = ["sea", "flat", "hills", "urban", "forest", "desert", "mountain"]
        if terrain_type not in valid_types:
            return False, f"无效的地形类型。支持的类型: {valid_types}"
        return True, "地形类型有效"


class SimulationValidator:
    """仿真参数验证器"""

    @staticmethod
    def validate_simulation_time(sim_time: Union[int, float]) -> Tuple[bool, str]:
        """验证仿真时间"""
        if not DataValidator.is_numeric(sim_time):
            return False, "仿真时间必须是数值"

        time_val = float(sim_time)
        if time_val <= 0:
            return False, "仿真时间必须为正数"

        if time_val > 3600:  # 1 hour
            return False, "仿真时间超出合理范围 (0-3600秒)"

        return True, "仿真时间参数有效"

    @staticmethod
    def validate_time_step(time_step: Union[int, float]) -> Tuple[bool, str]:
        """验证时间步长"""
        if not DataValidator.is_numeric(time_step):
            return False, "时间步长必须是数值"

        step_val = float(time_step)
        if step_val <= 0:
            return False, "时间步长必须为正数"

        if not (0.001 <= step_val <= 1.0):
            return False, "时间步长超出合理范围 (0.001-1.0秒)"

        return True, "时间步长参数有效"

    @staticmethod
    def validate_monte_carlo_runs(runs: int) -> Tuple[bool, str]:
        """验证蒙特卡洛运行次数"""
        if not DataValidator.is_integer(runs):
            return False, "蒙特卡洛运行次数必须是整数"

        runs_val = int(runs)
        if runs_val <= 0:
            return False, "蒙特卡洛运行次数必须为正整数"

        if runs_val > 1000:
            return False, "蒙特卡洛运行次数超出合理范围 (1-1000)"

        return True, "蒙特卡洛运行次数有效"


class ArrayValidator:
    """数组数据验证器"""

    @staticmethod
    def validate_array_dimension(array: np.ndarray, expected_dims: int) -> Tuple[bool, str]:
        """验证数组维度"""
        if array.ndim != expected_dims:
            return False, f"数组维度不匹配。期望: {expected_dims}, 实际: {array.ndim}"
        return True, "数组维度正确"

    @staticmethod
    def validate_array_shape(array: np.ndarray, expected_shape: Tuple[int, ...]) -> Tuple[bool, str]:
        """验证数组形状"""
        if array.shape != expected_shape:
            return False, f"数组形状不匹配。期望: {expected_shape}, 实际: {array.shape}"
        return True, "数组形状正确"

    @staticmethod
    def validate_array_dtype(array: np.ndarray, expected_dtype: np.dtype) -> Tuple[bool, str]:
        """验证数组数据类型"""
        if array.dtype != expected_dtype:
            return False, f"数组数据类型不匹配。期望: {expected_dtype}, 实际: {array.dtype}"
        return True, "数组数据类型正确"

    @staticmethod
    def validate_no_nan_inf(array: np.ndarray) -> Tuple[bool, str]:
        """验证数组中没有NaN或无穷大值"""
        if np.isnan(array).any():
            return False, "数组包含NaN值"
        if np.isinf(array).any():
            return False, "数组包含无穷大值"
        return True, "数组数据有效"

    @staticmethod
    def validate_array_range(array: np.ndarray, min_val: float, max_val: float) -> Tuple[bool, str]:
        """验证数组值范围"""
        if np.any(array < min_val) or np.any(array > max_val):
            return False, f"数组值超出范围 [{min_val}, {max_val}]"
        return True, "数组值在有效范围内"


class ComplexValidator:
    """复合数据验证器"""

    @staticmethod
    def validate_target_configuration(target_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证目标配置"""
        errors = []

        # 验证位置
        if 'position' in target_config:
            position = target_config['position']
            if not isinstance(position, (list, tuple)) or len(position) not in [2, 3]:
                errors.append("位置必须是长度为2或3的数组")
            else:
                for i, pos in enumerate(position):
                    if not DataValidator.is_numeric(pos):
                        errors.append(f"位置坐标{i}必须是数值")

        # 验证速度
        if 'velocity' in target_config:
            is_valid, msg = RadarDataValidator.validate_velocity(target_config['velocity'])
            if not is_valid:
                errors.append(msg)

        # 验证RCS
        if 'rcs' in target_config:
            is_valid, msg = RadarDataValidator.validate_rcs(target_config['rcs'])
            if not is_valid:
                errors.append(msg)

        # 验证高度
        if 'altitude' in target_config:
            is_valid, msg = RadarDataValidator.validate_range(target_config['altitude'])
            if not is_valid:
                errors.append(f"高度参数错误: {msg}")

        # 验证方位角
        if 'aspect_angle' in target_config:
            is_valid, msg = RadarDataValidator.validate_angle(target_config['aspect_angle'], "aspect")
            if not is_valid:
                errors.append(msg)

        return len(errors) == 0, errors

    @staticmethod
    def validate_radar_configuration(radar_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证雷达配置"""
        errors = []

        # 必需参数
        required_params = ['radar_area', 'tr_components', 'radar_power']
        for param in required_params:
            if param not in radar_config:
                errors.append(f"缺少必需参数: {param}")

        # 验证雷达面积
        if 'radar_area' in radar_config:
            if not DataValidator.is_positive(radar_config['radar_area']):
                errors.append("雷达面积必须为正数")

        # 验证TR组件数
        if 'tr_components' in radar_config:
            tr_components = radar_config['tr_components']
            if not DataValidator.is_integer(tr_components) or int(tr_components) <= 0:
                errors.append("TR组件数必须为正整数")
            elif int(tr_components) > 100000:
                errors.append("TR组件数超出合理范围 (1-100000)")

        # 验证雷达功率
        if 'radar_power' in radar_config:
            is_valid, msg = RadarDataValidator.validate_power(radar_config['radar_power'])
            if not is_valid:
                errors.append(msg)

        # 验证频率
        if 'frequency' in radar_config:
            is_valid, msg = RadarDataValidator.validate_frequency(radar_config['frequency'])
            if not is_valid:
                errors.append(msg)

        return len(errors) == 0, errors

    @staticmethod
    def validate_formation_configuration(formation_config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证编队配置"""
        errors = []

        required_params = ['formation_id', 'leader_id', 'member_ids']
        for param in required_params:
            if param not in formation_config:
                errors.append(f"缺少必需参数: {param}")

        # 验证编队ID
        if 'formation_id' in formation_config:
            if not DataValidator.is_integer(formation_config['formation_id']):
                errors.append("编队ID必须是整数")

        # 验证长机ID
        if 'leader_id' in formation_config:
            if not DataValidator.is_integer(formation_config['leader_id']):
                errors.append("长机ID必须是整数")

        # 验证僚机ID列表
        if 'member_ids' in formation_config:
            member_ids = formation_config['member_ids']
            if not isinstance(member_ids, list):
                errors.append("僚机ID列表必须是数组")
            else:
                for mid in member_ids:
                    if not DataValidator.is_integer(mid):
                        errors.append(f"僚机ID必须是整数: {mid}")

        # 验证编队类型
        if 'formation_type' in formation_config:
            valid_types = ['line', 'v', 'diamond', 'column']
            if formation_config['formation_type'] not in valid_types:
                errors.append(f"无效的编队类型。支持的类型: {valid_types}")

        # 验证间距
        if 'spacing' in formation_config:
            spacing = formation_config['spacing']
            if not DataValidator.is_positive(spacing):
                errors.append("编队间距必须为正数")
            elif float(spacing) > 10000:  # 10km
                errors.append("编队间距超出合理范围 (0-10km)")

        return len(errors) == 0, errors


class SecurityValidator:
    """安全验证器"""

    @staticmethod
    def validate_input_size(data: Any, max_size: int = 1024 * 1024) -> Tuple[bool, str]:
        """验证输入数据大小"""
        try:
            data_size = len(str(data))
            if data_size > max_size:
                return False, f"输入数据过大: {data_size} > {max_size}"
            return True, "输入数据大小合适"
        except Exception:
            return False, "无法计算数据大小"

    @staticmethod
    def validate_no_script_injection(data: str) -> Tuple[bool, str]:
        """验证没有脚本注入"""
        danger_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'on\w+\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
        ]

        for pattern in danger_patterns:
            if re.search(pattern, data, re.IGNORECASE):
                return False, "检测到潜在的脚本注入"

        return True, "输入安全"

    @staticmethod
    def sanitize_string(input_str: str) -> str:
        """字符串清理"""
        # 移除危险字符
        sanitized = re.sub(r'[<>"\']', '', input_str)
        # 限制长度
        sanitized = sanitized[:1000]
        return sanitized.strip()


# 便捷验证函数
def validate_simulation_request(request_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """验证完整的仿真请求"""
    all_errors = []
    
    # 验证雷达配置
    if 'radar' in request_data:
        is_valid, errors = ComplexValidator.validate_radar_configuration(request_data['radar'])
        if not is_valid:
            all_errors.extend([f"雷达配置: {err}" for err in errors])
    else:
        all_errors.append("缺少雷达配置")
    
    # 验证环境配置
    if 'environment' in request_data:
        env_config = request_data['environment']
        
        if 'weather_type' in env_config:
            is_valid, msg = EnvironmentValidator.validate_weather_type(env_config['weather_type'])
            if not is_valid:
                all_errors.append(f"环境配置: {msg}")
        
        if 'precipitation_rate' in env_config:
            is_valid, msg = EnvironmentValidator.validate_precipitation_rate(env_config['precipitation_rate'])
            if not is_valid:
                all_errors.append(f"环境配置: {msg}")

    # 验证目标配置
    if 'targets' in request_data:
        targets_config = request_data['targets']
        
        if 'num_targets' not in targets_config:
            all_errors.append("缺少目标数量配置")
        elif not DataValidator.is_positive(targets_config['num_targets']):
            all_errors.append("目标数量必须为正数")
        
        # 验证具体目标配置
        if 'specific_targets' in targets_config:
            for i, target in enumerate(targets_config['specific_targets']):
                is_valid, errors = ComplexValidator.validate_target_configuration(target)
                if not is_valid:
                    all_errors.extend([f"目标{i + 1}: {err}" for err in errors])
        
        # 验证编队配置
        if 'formations' in targets_config:
            for i, formation in enumerate(targets_config['formations']):
                is_valid, errors = ComplexValidator.validate_formation_configuration(formation)
                if not is_valid:
                    all_errors.extend([f"编队{i + 1}: {err}" for err in errors])
    
    # 验证仿真参数
    if 'simulation_time' in request_data:
        is_valid, msg = SimulationValidator.validate_simulation_time(request_data['simulation_time'])
        if not is_valid:
            all_errors.append(msg)
    
    if 'time_step' in request_data:
        is_valid, msg = SimulationValidator.validate_time_step(request_data['time_step'])
        if not is_valid:
            all_errors.append(msg)
    
    if 'monte_carlo_runs' in request_data:
        is_valid, msg = SimulationValidator.validate_monte_carlo_runs(request_data['monte_carlo_runs'])
        if not is_valid:
            all_errors.append(msg)
    
    return len(all_errors) == 0, all_errors


def quick_validate(value: Any, validator_type: str, **kwargs) -> bool:
    """快速验证函数"""
    validators = {
        'numeric': lambda x: DataValidator.is_numeric(x),
        'positive': lambda x: DataValidator.is_positive(x),
        'integer': lambda x: DataValidator.is_integer(x),
        'boolean': lambda x: DataValidator.is_boolean(x),
        'frequency': lambda x: RadarDataValidator.validate_frequency(x)[0],
        'power': lambda x: RadarDataValidator.validate_power(x)[0],
        'range': lambda x: RadarDataValidator.validate_range(x)[0],
        'rcs': lambda x: RadarDataValidator.validate_rcs(x)[0],
        'velocity': lambda x: RadarDataValidator.validate_velocity(x)[0],
    }

    if validator_type in validators:
        return validators[validator_type](value)
    else:
        return False
