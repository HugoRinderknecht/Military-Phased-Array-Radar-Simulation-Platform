# 类型定义模块
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime
import numpy as np


class TargetType(Enum):
    AIRCRAFT = 'aircraft'
    MISSILE = 'missile'
    SHIP = 'ship'
    GROUND_VEHICLE = 'ground'
    HELICOPTER = 'helicopter'
    UAV = 'uav'
    SATELLITE = 'satellite'
    UNKNOWN = 'unknown'


class MotionModel(Enum):
    CONSTANT_VELOCITY = 'cv'
    CONSTANT_ACCELERATION = 'ca'
    COORDINATED_TURN = 'ct'
    CIRCULAR = 'circular'
    SIX_DOF = '6dof'
    WAYPOINT = 'waypoint'
    PREDEFINED = 'predefined'


class SwerlingModel(IntEnum):
    I = 1
    II = 2
    III = 3
    IV = 4


class WorkMode(Enum):
    TWS = 'tws'
    TAS = 'tas'
    TRACK_ONLY = 'track_only'
    SEARCH_ONLY = 'search_only'
    GUIDANCE = 'guidance'


class TaskType(Enum):
    SEARCH = 'search'
    TRACK = 'track'
    VERIFY = 'verify'
    ACQUIRE = 'acquire'
    GUIDANCE = 'guidance'
    LOSS = 'loss'
    MISSILE = 'missile'
    CALIBRATE = 'calibrate'


class BeamType(Enum):
    SUM = 'sum'
    DELTA_AZ = 'delta_az'
    DELTA_EL = 'delta_el'


class TrackState(Enum):
    INITIATING = 'initiating'
    CONFIRMED = 'confirmed'
    TENTATIVE = 'tentative'
    DROPPED = 'dropped'
    TERMINATED = 'terminated'
    MERGED = 'merged'


class CFARType(Enum):
    CA = 'ca'
    GO = 'go'
    SO = 'so'
    OS = 'os'
    TM = 'tm'


class FilterType(Enum):
    KF = 'kf'
    EKF = 'ekf'
    UKF = 'ukf'
    IMM = 'imm'
    PHD = 'phd'


@dataclass
class Position3D:
    x: float
    y: float
    z: float
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)
    
    def distance_to(self, other: 'Position3D') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


@dataclass
class Velocity3D:
    vx: float
    vy: float
    vz: float
    
    def magnitude(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz], dtype=np.float64)


@dataclass
class Attitude:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    
    def to_rotation_matrix(self) -> np.ndarray:
        c_r, s_r = np.cos(self.roll), np.sin(self.roll)
        c_p, s_p = np.cos(self.pitch), np.sin(self.pitch)
        c_y, s_y = np.cos(self.yaw), np.sin(self.yaw)
        
        R_x = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
        R_y = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
        R_z = np.array([[c_y, -s_y, 0], [s_y, c_y, 0], [0, 0, 1]])
        
        return R_z @ R_y @ R_x


@dataclass
class SphericalCoord:
    range: float
    azimuth: float
    elevation: float


@dataclass
class WindowParams:
    front: int
    back: int
    
    @property
    def width(self) -> int:
        return self.back - self.front


@dataclass
class Plot:
    id: int
    timestamp: int
    position: Position3D
    velocity: Optional[Velocity3D] = None
    range_val: float = 0.0
    azimuth: float = 0.0
    elevation: float = 0.0
    doppler_freq: float = 0.0
    doppler_vel: float = 0.0
    snr: float = 0.0
    rcs: float = 1.0
    quality: float = 1.0
    beam_id: int = 0
    source_task_id: int = 0


@dataclass
class Track:
    id: int
    state: TrackState
    position: Position3D
    velocity: Velocity3D
    position_cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    velocity_cov: np.ndarray = field(default_factory=lambda: np.eye(3))
    last_update: int = 0
    creation_time: int = 0
    update_count: int = 0
    missed_count: int = 0
    associated_plots: list = field(default_factory=list)


__all__ = [
    'TargetType', 'MotionModel', 'SwerlingModel', 'WorkMode', 'TaskType',
    'BeamType', 'TrackState', 'CFARType', 'FilterType',
    'Position3D', 'Velocity3D', 'Attitude', 'SphericalCoord',
    'WindowParams', 'Plot', 'Track'
]
