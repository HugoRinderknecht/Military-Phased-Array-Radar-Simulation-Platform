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




