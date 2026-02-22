# 坐标变换模块 (Coordinate Transform Module)
# 本模块提供各种坐标系之间的转换函数

import numpy as np
from typing import Tuple, Union
from radar.common.constants import MathConstants


# ==================== 笛卡尔坐标与球坐标转换 ====================

def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """将笛卡尔坐标转换为球坐标
    
    Args:
        x: X坐标 (m)
        y: Y坐标 (m)
        z: Z坐标 (m)
    
    Returns:
        (r, theta, phi): 距离(m), 方位角(rad), 俯仰角(rad)
        - r: 原点到点的距离
        - theta: 方位角（从X轴起算，逆时针）
        - phi: 俯仰角（从XY平面向上）
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arcsin(z / r) if r > 0 else 0.0
    return r, theta, phi


def spherical_to_cartesian(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """将球坐标转换为笛卡尔坐标
    
    Args:
        r: 距离 (m)
        theta: 方位角 (rad)
        phi: 俯仰角 (rad)
    
    Returns:
        (x, y, z): 笛卡尔坐标 (m)
    """
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.cos(phi) * np.sin(theta)
    z = r * np.sin(phi)
    return x, y, z


# ==================== ENU与NED坐标系转换 ====================

def enu_to_ned(enu: np.ndarray) -> np.ndarray:
    """将ENU坐标转换为NED坐标
    
    Args:
        enu: ENU坐标系下的位置向量 [E, N, U]
    
    Returns:
        NED坐标系下的位置向量 [N, E, D]
    """
    return np.array([enu[1], enu[0], -enu[2]], dtype=np.float64)


def ned_to_enu(ned: np.ndarray) -> np.ndarray:
    """将NED坐标转换为ENU坐标"""
    return np.array([ned[1], ned[0], -ned[2]], dtype=np.float64)


# ==================== 雷达坐标与ENU坐标转换 ====================

def radar_to_enu(
    range_val: float,
    azimuth: float,
    elevation: float
) -> np.ndarray:
    """将雷达坐标转换为ENU坐标
    
    Args:
        range_val: 距离 (m)
        azimuth: 方位角 (rad)，正北为0，顺时针为正
        elevation: 俯仰角 (rad)，水平面为0，向上为正
    
    Returns:
        ENU坐标向量 [E, N, U] (m)
    
    转换公式:
        E = R * cos(el) * sin(az)
        N = R * cos(el) * cos(az)
        U = R * sin(el)
    """
    e = range_val * np.cos(elevation) * np.sin(azimuth)
    n = range_val * np.cos(elevation) * np.cos(azimuth)
    u = range_val * np.sin(elevation)
    return np.array([e, n, u], dtype=np.float64)


def enu_to_radar(position: np.ndarray) -> Tuple[float, float, float]:
    """将ENU坐标转换为雷达坐标
    
    Args:
        position: ENU坐标向量 [E, N, U] (m)
    
    Returns:
        (range, azimuth, elevation): 距离(m), 方位角(rad), 俯仰角(rad)
    """
    e, n, u = position[0], position[1], position[2]
    
    range_val = np.sqrt(e**2 + n**2 + u**2)
    azimuth = np.arctan2(e, n)
    elevation = np.arcsin(u / range_val) if range_val > 0 else 0.0
    
    return range_val, azimuth, elevation


# ==================== NED坐标与雷达坐标转换 ====================

def radar_to_ned(
    range_val: float,
    azimuth: float,
    elevation: float
) -> np.ndarray:
    """将雷达坐标转换为NED坐标
    
    Args:
        range_val: 距离 (m)
        azimuth: 方位角 (rad)，正北为0，顺时针为正
        elevation: 俯仰角 (rad)，水平面为0，向上为正
    
    Returns:
        NED坐标向量 [N, E, D] (m)
    """
    enu = radar_to_enu(range_val, azimuth, elevation)
    return enu_to_ned(enu)


def ned_to_radar(position: np.ndarray) -> Tuple[float, float, float]:
    """将NED坐标转换为雷达坐标"""
    enu = ned_to_enu(position)
    return enu_to_radar(enu)


# ==================== 位置向量旋转 ====================

def rotate_about_z(position: np.ndarray, angle: float) -> np.ndarray:
    """绕Z轴旋转向量（偏航旋转）
    
    Args:
        position: 位置向量 [x, y, z]
        angle: 旋转角度 (rad)，逆时针为正
    
    Returns:
        旋转后的位置向量
    """
    c, s = np.cos(angle), np.sin(angle)
    R_z = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    return R_z @ position


def rotate_about_y(position: np.ndarray, angle: float) -> np.ndarray:
    """绕Y轴旋转向量（俯仰旋转）
    
    Args:
        position: 位置向量 [x, y, z]
        angle: 旋转角度 (rad)
    
    Returns:
        旋转后的位置向量
    """
    c, s = np.cos(angle), np.sin(angle)
    R_y = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    return R_y @ position


def rotate_about_x(position: np.ndarray, angle: float) -> np.ndarray:
    """绕X轴旋转向量（横滚旋转）
    
    Args:
        position: 位置向量 [x, y, z]
        angle: 旋转角度 (rad)
    
    Returns:
        旋转后的位置向量
    """
    c, s = np.cos(angle), np.sin(angle)
    R_x = np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])
    return R_x @ position


# ==================== 欧拉角到旋转矩阵 ====================

def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """将欧拉角转换为旋转矩阵 (ZYX顺序)
    
    Args:
        roll: 横滚角 (rad)
        pitch: 俯仰角 (rad)
        yaw: 偏航角 (rad)
    
    Returns:
        3x3旋转矩阵
    """
    # 分别计算各轴旋转矩阵
    c_r, s_r = np.cos(roll), np.sin(roll)
    c_p, s_p = np.cos(pitch), np.sin(pitch)
    c_y, s_y = np.cos(yaw), np.sin(yaw)
    
    R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
    R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
    
    return R_z @ R_y @ R_x


# ==================== 地球坐标转换 ====================

def lla_to_enu(
    lat: float, lon: float, alt: float,
    ref_lat: float, ref_lon: float, ref_alt: float
) -> np.ndarray:
    """将经纬度转换为ENU坐标
    
    Args:
        lat: 目标纬度 (rad)
        lon: 目标经度 (rad)
        alt: 目标高度 (m)
        ref_lat: 参考点纬度 (rad)
        ref_lon: 参考点经度 (rad)
        ref_alt: 参考点高度 (m)
    
    Returns:
        ENU坐标 [E, N, U] (m)
    """
    from radar.common.constants import PhysicsConstants
    
    R = PhysicsConstants.RE
    
    # 计算相对于参考点的角度差
    dlat = lat - ref_lat
    dlon = lon - ref_lon
    
    # 计算ENU坐标
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    sin_ref_lat, cos_ref_lat = np.sin(ref_lat), np.cos(ref_lat)
    cos_dlon = np.cos(dlon)
    
    n = (R + alt) * dlat
    e = (R + alt) * cos_ref_lat * dlon
    u = alt - ref_alt
    
    return np.array([e, n, u], dtype=np.float64)


def enu_to_lla(
    enu: np.ndarray,
    ref_lat: float,
    ref_lon: float,
    ref_alt: float
) -> Tuple[float, float, float]:
    """将ENU坐标转换为经纬度"""
    from radar.common.constants import PhysicsConstants
    
    R = PhysicsConstants.RE
    e, n, u = enu[0], enu[1], enu[2]
    
    lat = ref_lat + n / (R + ref_alt)
    lon = ref_lon + e / ((R + ref_alt) * np.cos(ref_lat))
    alt = ref_alt + u
    
    return lat, lon, alt


__all__ = [
    'cartesian_to_spherical', 'spherical_to_cartesian',
    'enu_to_ned', 'ned_to_enu',
    'radar_to_enu', 'enu_to_radar',
    'radar_to_ned', 'ned_to_radar',
    'rotate_about_x', 'rotate_about_y', 'rotate_about_z',
    'euler_to_rotation_matrix',
    'lla_to_enu', 'enu_to_lla',
    'enu_to_azel', 'azel_to_enu',  # 别名
    'enu_to_ecef', 'ecef_to_enu',  # ECEF转换
    'rotation_matrix_x', 'rotation_matrix_y', 'rotation_matrix_z', 'rotation_matrix_zyx',  # 旋转矩阵别名
    'geodetic_to_ecef', 'ecef_to_geodetic',  # 大地坐标转换
]

# 别名
enu_to_azel = enu_to_radar
azel_to_enu = radar_to_enu
rotation_matrix_x = rotate_about_x
rotation_matrix_y = rotate_about_y
rotation_matrix_z = rotate_about_z
rotation_matrix_zyx = euler_to_rotation_matrix
geodetic_to_ecef = lla_to_enu
ecef_to_geodetic = enu_to_lla


# ==================== ECEF坐标转换 ====================

def enu_to_ecef(enu: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    """ENU转ECEF"""
    sin_lat, cos_lat = np.sin(ref_lat), np.cos(ref_lat)
    sin_lon, cos_lon = np.sin(ref_lon), np.cos(ref_lon)

    R = np.array([
        [-sin_lon, -sin_lat*cos_lon, cos_lat*cos_lon],
        [cos_lon,  -sin_lat*sin_lon, cos_lat*sin_lon],
        [0,        cos_lat,          sin_lat]
    ])

    return R @ enu


def ecef_to_enu(ecef: np.ndarray, ref_lat: float, ref_lon: float) -> np.ndarray:
    """ECEF转ENU"""
    sin_lat, cos_lat = np.sin(ref_lat), np.cos(ref_lat)
    sin_lon, cos_lon = np.sin(ref_lon), np.cos(ref_lon)

    R = np.array([
        [-sin_lon, cos_lon, 0],
        [-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat],
        [cos_lat*cos_lon, cos_lat*sin_lon, sin_lat]
    ])

    return R @ ecef
