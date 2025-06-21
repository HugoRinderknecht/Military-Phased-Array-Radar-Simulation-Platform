from jsonschema import validate, ValidationError

RADAR_SYSTEM_SCHEMA = {
    "type": "object",
    "properties": {
        "radar_area": {"type": "number", "minimum": 1},
        "tr_components": {"type": "integer", "minimum": 1},
        "radar_power": {"type": "number", "minimum": 1},
        "frequency": {"type": "number", "minimum": 1e9},
        "antenna_elements": {"type": "integer", "minimum": 4},
        "beam_width": {"type": "number", "minimum": 0.1},
        "scan_rate": {"type": "number", "minimum": 0.1}
    },
    "required": ["radar_area", "tr_components", "radar_power"]
}

ENVIRONMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "weather_type": {
            "type": "string",
            "enum": ["clear", "rain", "snow", "fog"]
        },
        "precipitation_rate": {"type": "number", "minimum": 0},
        "clutter_density": {"type": "number", "minimum": 0, "maximum": 1},
        "interference_level": {"type": "number", "minimum": 0, "maximum": 1},
        "electronic_warfare": {"type": "boolean"},
        "terrain_type": {
            "type": "string",
            "enum": ["sea", "flat", "hills", "urban", "forest"]
        }
    },
    "required": ["weather_type"]
}

TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "position": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 3
        },
        "velocity": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 3
        },
        "rcs": {"type": "number"},
        "altitude": {"type": "number", "minimum": 0},
        "aspect_angle": {"type": "number", "minimum": 0, "maximum": 6.28},
        "is_formation": {"type": "boolean"},
        "formation_id": {"type": ["integer", "null"]}
    },
    "required": ["position", "velocity", "rcs", "altitude", "aspect_angle"]
}

TARGETS_SCHEMA = {
    "type": "object",
    "properties": {
        "num_targets": {"type": "integer", "minimum": 1},
        "max_range": {"type": "number", "minimum": 1000},
        "specific_targets": {
            "type": "array",
            "items": TARGET_SCHEMA
        },
        "formations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "formation_id": {"type": "integer"},
                    "leader_id": {"type": "integer"},
                    "member_ids": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "formation_type": {
                        "type": "string",
                        "enum": ["line", "v", "diamond"]
                    },
                    "spacing": {"type": "number", "minimum": 10}
                },
                "required": ["formation_id", "leader_id", "member_ids"]
            }
        }
    },
    "required": ["num_targets"]
}

SIMULATION_SCHEMA = {
    "type": "object",
    "properties": {
        "radar": RADAR_SYSTEM_SCHEMA,
        "environment": ENVIRONMENT_SCHEMA,
        "targets": TARGETS_SCHEMA,
        "simulation_time": {"type": "number", "minimum": 1},
        "time_step": {"type": "number", "minimum": 0.01},
        "monte_carlo_runs": {"type": "integer", "minimum": 1, "maximum": 100}
    },
    "required": ["radar", "environment", "targets"]
}

def validate_simulation_request(data):
    try:
        validate(instance=data, schema=SIMULATION_SCHEMA)
        return True, None
    except ValidationError as e:
        return False, str(e)


# 在现有的 api/schemas.py 文件末尾添加以下内容

def validate_random_target_request(data):
    """验证随机目标生成请求"""
    if data is None:
        return False, ["Request data is empty"]

    errors = []

    # 验证目标数量
    if 'num_targets' in data:
        num_targets = data['num_targets']
        if not isinstance(num_targets, int) or num_targets <= 0:
            errors.append("num_targets must be a positive integer")
        elif num_targets > 100:
            errors.append("num_targets cannot exceed 100")

    # 验证类型参数
    valid_altitude_types = ['low', 'medium', 'high', 'mixed']
    if 'altitude_type' in data and data['altitude_type'] not in valid_altitude_types:
        errors.append(f"altitude_type must be one of {valid_altitude_types}")

    valid_velocity_types = ['slow', 'medium', 'fast', 'mixed']
    if 'velocity_type' in data and data['velocity_type'] not in valid_velocity_types:
        errors.append(f"velocity_type must be one of {valid_velocity_types}")

    valid_rcs_types = ['small', 'medium', 'large', 'mixed']
    if 'rcs_type' in data and data['rcs_type'] not in valid_rcs_types:
        errors.append(f"rcs_type must be one of {valid_rcs_types}")

    # 验证雷达范围
    if 'radar_range' in data:
        radar_range = data['radar_range']
        if not isinstance(radar_range, (int, float)) or radar_range <= 0:
            errors.append("radar_range must be a positive number")
        elif radar_range > 500000:  # 500km
            errors.append("radar_range cannot exceed 500km")

    # 验证编队设置
    if 'enable_formation' in data and not isinstance(data['enable_formation'], bool):
        errors.append("enable_formation must be a boolean")

    # 验证特定类型
    valid_target_types = ['fighter', 'bomber', 'transport', 'helicopter', 'drone', 'cruise_missile']
    if 'specific_types' in data:
        specific_types = data['specific_types']
        if specific_types is not None:
            if not isinstance(specific_types, list):
                errors.append("specific_types must be a list or null")
            else:
                for target_type in specific_types:
                    if target_type not in valid_target_types:
                        errors.append(f"Invalid target type: {target_type}. Valid types: {valid_target_types}")

    # 验证场景类型
    valid_scenarios = ['air_raid', 'patrol', 'low_altitude_penetration', 'swarm_attack', 'mixed']
    if 'scenario_type' in data:
        scenario_type = data['scenario_type']
        if scenario_type is not None and scenario_type not in valid_scenarios:
            errors.append(f"scenario_type must be one of {valid_scenarios}")

    return len(errors) == 0, errors


# 添加随机目标请求的JSON Schema
RANDOM_TARGET_SCHEMA = {
    "type": "object",
    "properties": {
        "num_targets": {
            "type": "integer",
            "minimum": 1,
            "maximum": 100
        },
        "altitude_type": {
            "type": "string",
            "enum": ["low", "medium", "high", "mixed"]
        },
        "velocity_type": {
            "type": "string",
            "enum": ["slow", "medium", "fast", "mixed"]
        },
        "rcs_type": {
            "type": "string",
            "enum": ["small", "medium", "large", "mixed"]
        },
        "radar_range": {
            "type": "number",
            "minimum": 1000,
            "maximum": 500000
        },
        "enable_formation": {
            "type": "boolean"
        },
        "specific_types": {
            "type": ["array", "null"],
            "items": {
                "type": "string",
                "enum": ["fighter", "bomber", "transport", "helicopter", "drone", "cruise_missile"]
            }
        },
        "scenario_type": {
            "type": ["string", "null"],
            "enum": ["air_raid", "patrol", "low_altitude_penetration", "swarm_attack", "mixed"]
        }
    },
    "additionalProperties": False
}
# 配置请求验证 Schema
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100
        },
        "description": {
            "type": ["string", "null"],
            "maxLength": 500
        },
        "config_type": {
            "type": "string",
            "enum": ["radar", "environment", "simulation", "full"]
        },
        "config_data": {
            "type": "object",
            "properties": {
                "radar": RADAR_SYSTEM_SCHEMA,
                "environment": ENVIRONMENT_SCHEMA,
                "targets": TARGETS_SCHEMA,
                "simulation_parameters": {
                    "type": "object",
                    "properties": {
                        "simulation_time": {"type": "number", "minimum": 1},
                        "time_step": {"type": "number", "minimum": 0.01},
                        "monte_carlo_runs": {"type": "integer", "minimum": 1, "maximum": 100}
                    }
                }
            }
        },
        "tags": {
            "type": ["array", "null"],
            "items": {
                "type": "string",
                "maxLength": 50
            },
            "maxItems": 10
        },
        "is_default": {
            "type": "boolean"
        },
        "created_by": {
            "type": ["string", "null"],
            "maxLength": 100
        }
    },
    "required": ["name", "config_type", "config_data"],
    "additionalProperties": False
}

# 导出请求验证 Schema
EXPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "simulation_id": {
            "type": "string",
            "pattern": "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
        },
        "export_format": {
            "type": "string",
            "enum": ["csv", "json", "excel", "matlab", "hdf5"]
        },
        "data_types": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["targets", "detections", "tracks", "radar_data", "environment", "performance_metrics"]
            },
            "minItems": 1,
            "uniqueItems": True
        },
        "time_range": {
            "type": ["object", "null"],
            "properties": {
                "start_time": {
                    "type": "number",
                    "minimum": 0
                },
                "end_time": {
                    "type": "number",
                    "minimum": 0
                }
            },
            "required": ["start_time", "end_time"]
        },
        "compression": {
            "type": "boolean"
        },
        "include_metadata": {
            "type": "boolean"
        },
        "file_name": {
            "type": ["string", "null"],
            "maxLength": 255,
            "pattern": "^[a-zA-Z0-9_\\-\\.]+$"
        },
        "export_options": {
            "type": ["object", "null"],
            "properties": {
                "decimal_places": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10
                },
                "coordinate_system": {
                    "type": "string",
                    "enum": ["cartesian", "polar", "geographic"]
                },
                "time_format": {
                    "type": "string",
                    "enum": ["timestamp", "relative", "absolute"]
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"]
                }
            }
        }
    },
    "required": ["simulation_id", "export_format", "data_types"],
    "additionalProperties": False
}


def validate_config_request(data):
    """验证配置请求"""
    try:
        validate(instance=data, schema=CONFIG_SCHEMA)

        # 额外的业务逻辑验证
        config_data = data.get('config_data', {})
        config_type = data.get('config_type')

        # 根据配置类型验证必要的数据
        if config_type == 'radar' and 'radar' not in config_data:
            return False, "Radar configuration is required for radar config type"
        elif config_type == 'environment' and 'environment' not in config_data:
            return False, "Environment configuration is required for environment config type"
        elif config_type == 'simulation' and 'simulation_parameters' not in config_data:
            return False, "Simulation parameters are required for simulation config type"
        elif config_type == 'full':
            # 完整配置需要所有组件
            required_components = ['radar', 'environment', 'targets']
            missing_components = [comp for comp in required_components if comp not in config_data]
            if missing_components:
                return False, f"Missing required components for full config: {', '.join(missing_components)}"

        # 验证时间范围合理性
        if 'time_range' in data and data['time_range']:
            time_range = data['time_range']
            if time_range['start_time'] >= time_range['end_time']:
                return False, "start_time must be less than end_time"

        return True, None

    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {str(e)}"


def validate_export_request(data):
    """验证导出请求"""
    try:
        validate(instance=data, schema=EXPORT_SCHEMA)

        # 额外的业务逻辑验证
        time_range = data.get('time_range')
        if time_range:
            start_time = time_range.get('start_time', 0)
            end_time = time_range.get('end_time', 0)

            if start_time >= end_time:
                return False, "start_time must be less than end_time"

            # 检查时间范围是否合理（不超过24小时的仿真）
            if end_time - start_time > 86400:  # 24小时
                return False, "Time range cannot exceed 24 hours"

        # 验证文件名格式
        file_name = data.get('file_name')
        if file_name:
            # 检查文件名是否包含不安全字符
            unsafe_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
            if any(char in file_name for char in unsafe_chars):
                return False, "File name contains unsafe characters"

        # 根据导出格式验证特定选项
        export_format = data.get('export_format')
        export_options = data.get('export_options', {})

        if export_format == 'matlab' and export_options.get('coordinate_system') == 'geographic':
            # MATLAB格式可能不支持某些坐标系统
            pass  # 这里可以添加格式特定的验证

        return True, None

    except ValidationError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Validation error: {str(e)}"


# 添加一个通用的验证辅助函数
def validate_uuid_format(uuid_string):
    """验证UUID格式"""
    import re
    uuid_pattern = re.compile(
        r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
        re.IGNORECASE
    )
    return bool(uuid_pattern.match(uuid_string))


def validate_time_range(start_time, end_time):
    """验证时间范围"""
    if start_time is None or end_time is None:
        return True, None

    if not isinstance(start_time, (int, float)) or not isinstance(end_time, (int, float)):
        return False, "Time values must be numbers"

    if start_time < 0 or end_time < 0:
        return False, "Time values must be non-negative"

    if start_time >= end_time:
        return False, "start_time must be less than end_time"

    return True, None


def get_validation_summary():
    """获取所有验证器的摘要信息"""
    return {
        "available_validators": [
            "validate_simulation_request",
            "validate_random_target_request",
            "validate_config_request",
            "validate_export_request"
        ],
        "schemas": {
            "simulation": "SIMULATION_SCHEMA",
            "random_target": "RANDOM_TARGET_SCHEMA",
            "config": "CONFIG_SCHEMA",
            "export": "EXPORT_SCHEMA"
        },
        "helper_functions": [
            "validate_uuid_format",
            "validate_time_range",
            "get_validation_summary"
        ]
    }




