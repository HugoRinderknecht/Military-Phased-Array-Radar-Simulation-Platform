import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Target:
    target_id: int
    position: List[float]  # [x, y, z]
    velocity: List[float]  # [vx, vy, vz]
    rcs: float
    altitude: float
    aspect_angle: float
    is_formation: bool = False
    formation_id: Optional[int] = None
    formation_spacing: float = 100.0
    maneuver_type: str = "constant_velocity"

    def __post_init__(self):
        if len(self.position) == 2:
            self.position.append(self.altitude)
        if len(self.velocity) == 2:
            self.velocity.append(0.0)

        self.range = np.sqrt(sum(p ** 2 for p in self.position))
        self.azimuth = np.arctan2(self.position[1], self.position[0])
        self.elevation = np.arcsin(self.position[2] / max(self.range, 1e-6))


@dataclass
class Formation:
    formation_id: int
    leader_target: Target
    members: List[Target]
    formation_type: str = "line"
    spacing: float = 100.0

    def update_formation_positions(self, dt: float):
        self.leader_target.position[0] += self.leader_target.velocity[0] * dt
        self.leader_target.position[1] += self.leader_target.velocity[1] * dt
        self.leader_target.position[2] += self.leader_target.velocity[2] * dt

        for i, member in enumerate(self.members):
            if self.formation_type == "line":
                offset_x = (i + 1) * self.spacing * np.cos(self.leader_target.azimuth + np.pi / 2)
                offset_y = (i + 1) * self.spacing * np.sin(self.leader_target.azimuth + np.pi / 2)
            elif self.formation_type == "v":
                angle = self.leader_target.azimuth + (-1) ** (i + 1) * np.pi / 6
                offset_x = (i + 1) * self.spacing * np.cos(angle)
                offset_y = (i + 1) * self.spacing * np.sin(angle)
            else:
                offset_x = offset_y = 0

            member.position[0] = self.leader_target.position[0] + offset_x
            member.position[1] = self.leader_target.position[1] + offset_y
            member.position[2] = self.leader_target.position[2]
            member.velocity = self.leader_target.velocity.copy()


import random
from typing import List, Dict, Any


class RandomTargetGenerator:
    """随机目标生成器"""

    def __init__(self):
        # 预设的参数范围
        self.altitude_ranges = {
            'low': (50, 500),  # 低空：50-500m
            'medium': (500, 3000),  # 中空：500-3000m
            'high': (3000, 12000),  # 高空：3000-12000m
            'mixed': (50, 12000)  # 混合：50-12000m
        }

        self.velocity_ranges = {
            'slow': (50, 200),  # 慢速：50-200 m/s
            'medium': (200, 500),  # 中速：200-500 m/s
            'fast': (500, 1000),  # 快速：500-1000 m/s
            'mixed': (50, 1000)  # 混合：50-1000 m/s
        }

        self.rcs_ranges = {
            'small': (-20, 0),  # 小目标：-20到0 dBsm
            'medium': (0, 20),  # 中等目标：0到20 dBsm
            'large': (20, 40),  # 大目标：20到40 dBsm
            'mixed': (-20, 40)  # 混合：-20到40 dBsm
        }

        # 目标类型及其特性
        self.target_types = {
            'fighter': {
                'rcs_range': (0, 10),
                'velocity_range': (200, 800),
                'altitude_range': (500, 12000),
                'maneuver_capability': 'high'
            },
            'bomber': {
                'rcs_range': (20, 40),
                'velocity_range': (150, 400),
                'altitude_range': (3000, 12000),
                'maneuver_capability': 'low'
            },
            'transport': {
                'rcs_range': (30, 50),
                'velocity_range': (100, 300),
                'altitude_range': (8000, 12000),
                'maneuver_capability': 'low'
            },
            'helicopter': {
                'rcs_range': (5, 15),
                'velocity_range': (50, 120),
                'altitude_range': (50, 1000),
                'maneuver_capability': 'medium'
            },
            'drone': {
                'rcs_range': (-15, 5),
                'velocity_range': (80, 200),
                'altitude_range': (100, 5000),
                'maneuver_capability': 'medium'
            },
            'cruise_missile': {
                'rcs_range': (-10, 5),
                'velocity_range': (200, 600),
                'altitude_range': (50, 500),
                'maneuver_capability': 'high'
            }
        }

        # 编队类型
        self.formation_types = {
            'line': {'spacing': (500, 2000), 'max_size': 8},
            'v': {'spacing': (300, 1500), 'max_size': 6},
            'diamond': {'spacing': (400, 1200), 'max_size': 4},
            'column': {'spacing': (200, 800), 'max_size': 10},
            'wedge': {'spacing': (300, 1000), 'max_size': 5}
        }

    def generate_random_targets(self, num_targets: int,
                                altitude_type: str = 'mixed',
                                velocity_type: str = 'mixed',
                                rcs_type: str = 'mixed',
                                radar_range: float = 100000,
                                enable_formation: bool = True,
                                specific_types: List[str] = None) -> List[Dict[str, Any]]:
        """
        生成随机目标

        Args:
            num_targets: 目标数量
            altitude_type: 高度类型 ('low', 'medium', 'high', 'mixed')
            velocity_type: 速度类型 ('slow', 'medium', 'fast', 'mixed')
            rcs_type: RCS类型 ('small', 'medium', 'large', 'mixed')
            radar_range: 雷达探测范围
            enable_formation: 是否启用编队
            specific_types: 指定目标类型列表

        Returns:
            目标列表
        """
        targets = []
        target_id = 1
        formation_id = 1

        # 如果启用编队，决定编队分布
        formations_to_create = []
        remaining_targets = num_targets

        if enable_formation and num_targets > 1:
            # 随机生成编队
            while remaining_targets > 0:
                if remaining_targets >= 2 and random.random() > 0.3:
                    # 30%概率不组编队，70%概率组编队
                    formation_type = random.choice(list(self.formation_types.keys()))
                    max_formation_size = self.formation_types[formation_type]['max_size']
                    formation_size = random.randint(2, min(max_formation_size, remaining_targets))

                    formations_to_create.append({
                        'type': formation_type,
                        'size': formation_size,
                        'id': formation_id
                    })
                    formation_id += 1
                    remaining_targets -= formation_size
                else:
                    # 单独目标
                    formations_to_create.append({
                        'type': 'single',
                        'size': 1,
                        'id': None
                    })
                    remaining_targets -= 1
        else:
            # 全部单独目标
            for _ in range(num_targets):
                formations_to_create.append({
                    'type': 'single',
                    'size': 1,
                    'id': None
                })

        # 生成目标
        for formation in formations_to_create:
            if formation['type'] == 'single':
                # 单独目标
                target = self._generate_single_target(
                    target_id, altitude_type, velocity_type, rcs_type,
                    radar_range, specific_types
                )
                targets.append(target)
                target_id += 1
            else:
                # 编队目标
                formation_targets = self._generate_formation_targets(
                    formation, target_id, altitude_type, velocity_type,
                    rcs_type, radar_range, specific_types
                )
                targets.extend(formation_targets)
                target_id += formation['size']

        return targets

    def _generate_single_target(self, target_id: int, altitude_type: str,
                                velocity_type: str, rcs_type: str, radar_range: float,
                                specific_types: List[str] = None) -> Dict[str, Any]:
        """生成单个随机目标"""

        # 选择目标类型
        if specific_types:
            target_type = random.choice(specific_types)
        else:
            target_type = random.choice(list(self.target_types.keys()))

        target_spec = self.target_types[target_type]

        # 生成位置（极坐标）
        range_val = random.uniform(5000, radar_range * 0.9)  # 5km到最大范围的90%
        azimuth = random.uniform(-np.pi, np.pi)  # 全方位角度

        # 生成高度
        if altitude_type == 'mixed':
            altitude = random.uniform(*target_spec['altitude_range'])
        else:
            altitude_range = self.altitude_ranges[altitude_type]
            # 与目标类型特性结合
            effective_range = (
                max(altitude_range[0], target_spec['altitude_range'][0]),
                min(altitude_range[1], target_spec['altitude_range'][1])
            )
            if effective_range[0] >= effective_range[1]:
                effective_range = altitude_range
            altitude = random.uniform(*effective_range)

        # 转换为直角坐标
        elevation = np.arcsin(altitude / range_val) if altitude < range_val else np.pi / 6
        x = range_val * np.cos(azimuth) * np.cos(elevation)
        y = range_val * np.sin(azimuth) * np.cos(elevation)
        z = altitude

        # 生成速度
        if velocity_type == 'mixed':
            velocity_magnitude = random.uniform(*target_spec['velocity_range'])
        else:
            velocity_range = self.velocity_ranges[velocity_type]
            effective_range = (
                max(velocity_range[0], target_spec['velocity_range'][0]),
                min(velocity_range[1], target_spec['velocity_range'][1])
            )
            if effective_range[0] >= effective_range[1]:
                effective_range = velocity_range
            velocity_magnitude = random.uniform(*effective_range)

        # 随机速度方向
        velocity_azimuth = random.uniform(-np.pi, np.pi)
        velocity_elevation = random.uniform(-np.pi / 12, np.pi / 12)  # 小的垂直速度分量

        vx = velocity_magnitude * np.cos(velocity_azimuth) * np.cos(velocity_elevation)
        vy = velocity_magnitude * np.sin(velocity_azimuth) * np.cos(velocity_elevation)
        vz = velocity_magnitude * np.sin(velocity_elevation)

        # 生成RCS
        if rcs_type == 'mixed':
            rcs = random.uniform(*target_spec['rcs_range'])
        else:
            rcs_range = self.rcs_ranges[rcs_type]
            effective_range = (
                max(rcs_range[0], target_spec['rcs_range'][0]),
                min(rcs_range[1], target_spec['rcs_range'][1])
            )
            if effective_range[0] >= effective_range[1]:
                effective_range = rcs_range
            rcs = random.uniform(*effective_range)

        # 生成面向角度（目标姿态）
        aspect_angle = random.uniform(-np.pi, np.pi)

        return {
            'id': target_id,
            'type': target_type,
            'position': [x, y, z],
            'velocity': [vx, vy, vz],
            'rcs': rcs,
            'altitude': altitude,
            'aspect_angle': aspect_angle,
            'range': range_val,
            'azimuth': azimuth,
            'elevation': elevation,
            'maneuver_capability': target_spec['maneuver_capability'],
            'formation_id': None,
            'is_leader': False
        }

    def _generate_formation_targets(self, formation_info: Dict, start_id: int,
                                    altitude_type: str, velocity_type: str,
                                    rcs_type: str, radar_range: float,
                                    specific_types: List[str] = None) -> List[Dict[str, Any]]:
        """生成编队目标"""

        formation_type = formation_info['type']
        formation_size = formation_info['size']
        formation_id = formation_info['id']

        # 生成长机
        leader = self._generate_single_target(
            start_id, altitude_type, velocity_type, rcs_type,
            radar_range, specific_types
        )
        leader['formation_id'] = formation_id
        leader['is_leader'] = True
        leader['role'] = 'leader'

        targets = [leader]

        # 编队间距
        spacing_range = self.formation_types[formation_type]['spacing']
        spacing = random.uniform(*spacing_range)

        # 生成僚机
        for i in range(1, formation_size):
            wingman = self._generate_formation_wingman(
                leader, i, formation_type, spacing, start_id + i
            )
            wingman['formation_id'] = formation_id
            wingman['is_leader'] = False
            wingman['role'] = 'wingman'
            targets.append(wingman)

        return targets

    def _generate_formation_wingman(self, leader: Dict, position_index: int,
                                    formation_type: str, spacing: float,
                                    wingman_id: int) -> Dict[str, Any]:
        """生成编队中的僚机"""

        wingman = leader.copy()
        wingman['id'] = wingman_id

        # 根据编队类型计算相对位置
        leader_pos = np.array(leader['position'])
        leader_vel = np.array(leader['velocity'])

        # 计算长机的航向角
        leader_heading = np.arctan2(leader_vel[1], leader_vel[0])

        # 根据编队类型确定僚机位置
        if formation_type == 'line':
            # 横队：僚机在长机两侧
            side = 1 if position_index % 2 == 1 else -1
            offset_distance = (position_index + 1) // 2 * spacing
            offset_x = -offset_distance * np.sin(leader_heading) * side
            offset_y = offset_distance * np.cos(leader_heading) * side
            offset_z = random.uniform(-50, 50)  # 高度微调

        elif formation_type == 'v':
            # V字形：僚机在长机后方两侧
            side = 1 if position_index % 2 == 1 else -1
            wing_distance = (position_index + 1) // 2 * spacing
            back_distance = wing_distance * 0.8

            offset_x = (-back_distance * np.cos(leader_heading) +
                        -wing_distance * np.sin(leader_heading) * side)
            offset_y = (-back_distance * np.sin(leader_heading) +
                        wing_distance * np.cos(leader_heading) * side)
            offset_z = random.uniform(-30, 30)

        elif formation_type == 'diamond':
            # 菱形编队
            if position_index == 1:  # 前方僚机
                offset_x = spacing * 0.7 * np.cos(leader_heading)
                offset_y = spacing * 0.7 * np.sin(leader_heading)
                offset_z = random.uniform(-20, 50)
            elif position_index == 2:  # 左侧僚机
                offset_x = -spacing * 0.7 * np.sin(leader_heading)
                offset_y = spacing * 0.7 * np.cos(leader_heading)
                offset_z = random.uniform(-50, 20)
            else:  # 右侧僚机
                offset_x = spacing * 0.7 * np.sin(leader_heading)
                offset_y = -spacing * 0.7 * np.cos(leader_heading)
                offset_z = random.uniform(-50, 20)

        elif formation_type == 'column':
            # 纵队：僚机在长机后方
            back_distance = position_index * spacing
            offset_x = -back_distance * np.cos(leader_heading)
            offset_y = -back_distance * np.sin(leader_heading)
            offset_z = random.uniform(-100, 100)

        elif formation_type == 'wedge':
            # 楔形编队
            side = 1 if position_index % 2 == 1 else -1
            row = (position_index + 1) // 2
            back_distance = row * spacing * 0.8
            wing_distance = row * spacing * 0.6

            offset_x = (-back_distance * np.cos(leader_heading) +
                        -wing_distance * np.sin(leader_heading) * side)
            offset_y = (-back_distance * np.sin(leader_heading) +
                        wing_distance * np.cos(leader_heading) * side)
            offset_z = random.uniform(-50, 50)
        else:
            # 默认后方跟随
            offset_x = -position_index * spacing * np.cos(leader_heading)
            offset_y = -position_index * spacing * np.sin(leader_heading)
            offset_z = random.uniform(-100, 100)

        # 更新僚机位置
        wingman_pos = leader_pos + np.array([offset_x, offset_y, offset_z])
        wingman['position'] = wingman_pos.tolist()

        # 僚机保持与长机相似的速度，但有小幅差异
        velocity_variation = np.random.normal(0, 0.05, 3)  # 5%的速度变化
        wingman_velocity = leader_vel * (1 + velocity_variation)
        wingman['velocity'] = wingman_velocity.tolist()

        # 更新相关参数
        wingman['altitude'] = wingman_pos[2]
        wingman['range'] = np.linalg.norm(wingman_pos)
        wingman['azimuth'] = np.arctan2(wingman_pos[1], wingman_pos[0])
        wingman['elevation'] = np.arcsin(wingman_pos[2] / max(wingman['range'], 1))

        # RCS有小幅变化
        rcs_variation = random.uniform(0.8, 1.2)
        wingman['rcs'] = leader['rcs'] + np.log10(rcs_variation) * 10

        # 面向角度有小幅差异
        wingman['aspect_angle'] = leader['aspect_angle'] + random.uniform(-np.pi / 6, np.pi / 6)

        return wingman

    def generate_threat_scenario(self, scenario_type: str = 'mixed') -> List[Dict[str, Any]]:
        """生成威胁场景"""

        scenarios = {
            'air_raid': {
                'bombers': random.randint(2, 4),
                'escorts': random.randint(4, 8),
                'formation_probability': 0.9
            },
            'patrol': {
                'fighters': random.randint(1, 3),
                'formation_probability': 0.7
            },
            'low_altitude_penetration': {
                'cruise_missiles': random.randint(3, 6),
                'helicopters': random.randint(1, 2),
                'formation_probability': 0.4
            },
            'swarm_attack': {
                'drones': random.randint(8, 15),
                'formation_probability': 0.8
            },
            'mixed': {
                'fighters': random.randint(1, 3),
                'bombers': random.randint(1, 2),
                'transports': random.randint(0, 1),
                'drones': random.randint(2, 5),
                'formation_probability': 0.6
            }
        }

        if scenario_type not in scenarios:
            scenario_type = 'mixed'

        scenario = scenarios[scenario_type]
        all_targets = []

        for target_type, count in scenario.items():
            if target_type == 'formation_probability':
                continue

            if count > 0:
                enable_formation = random.random() < scenario['formation_probability']

                targets = self.generate_random_targets(
                    num_targets=count,
                    altitude_type='mixed',
                    velocity_type='mixed',
                    rcs_type='mixed',
                    enable_formation=enable_formation,
                    specific_types=[target_type]
                )
                all_targets.extend(targets)

        return all_targets


# 添加便捷函数
def generate_random_enemies(count: int, **kwargs) -> List[Target]:
    """便捷的随机敌人生成函数"""
    generator = RandomTargetGenerator()
    target_data = generator.generate_random_targets(count, **kwargs)

    enemies = []
    for data in target_data:
        target = Target(
            target_id=data['id'],
            initial_position=data['position'],
            initial_velocity=data['velocity'],
            rcs=data['rcs']
        )

        # 添加额外属性
        target.target_type = data['type']
        target.altitude = data['altitude']
        target.aspect_angle = data['aspect_angle']
        target.formation_id = data.get('formation_id')
        target.is_leader = data.get('is_leader', False)
        target.role = data.get('role', 'single')
        target.maneuver_capability = data['maneuver_capability']

        enemies.append(target)

    return enemies
