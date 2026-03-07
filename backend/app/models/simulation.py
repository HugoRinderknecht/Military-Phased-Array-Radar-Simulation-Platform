"""
仿真相关数据模型
"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlotData(BaseModel):
    """点迹数据"""

    plot_id: str
    time: float
    range: float
    azimuth: float
    elevation: float
    amplitude: float
    snr: float
    doppler: Optional[float] = None
    target_id: Optional[str] = None


class TrackData(BaseModel):
    """航迹数据"""

    track_id: str
    time: float
    position: Dict[str, float]  # x, y, z
    velocity: Dict[str, float]  # vx, vy, vz
    covariance: Optional[List[List[float]]] = None
    quality: float = Field(default=1.0, ge=0, le=1)
    target_type: Optional[str] = None
    iff_confidence: float = Field(default=0.5, ge=0, le=1)


class SimulationEvent(BaseModel):
    """仿真事件"""

    event_id: str
    time: float
    event_type: str
    description: str
    data: Optional[Dict[str, Any]] = None


class SimulationStatus(BaseModel):
    """仿真状态"""

    simulation_id: str
    status: Literal["idle", "running", "paused", "completed", "error"]
    progress: float = Field(default=0, ge=0, le=100)
    current_time: float = Field(default=0, ge=0)
    total_time: float = Field(default=100, gt=0)
    plots_count: int = Field(default=0, ge=0)
    tracks_count: int = Field(default=0, ge=0)
    events_count: int = Field(default=0, ge=0)
    started_at: Optional[str] = None
    updated_at: Optional[str] = None
    error_message: Optional[str] = None


class SimulationControl(BaseModel):
    """仿真控制命令"""

    command: Literal["start", "pause", "resume", "stop", "step"]
    scene_id: Optional[str] = None  # start命令需要
    speed: Optional[float] = Field(
        None, ge=0.1, le=100
    )  # 速度倍率，1为实时，>1加速，<1减速
    step_size: Optional[float] = Field(None, gt=0)  # 单步执行的时间步长


class SimulationRequest(BaseModel):
    """仿真启动请求"""

    scene_id: str
    save_results: bool = Field(default=True, description="是否保存结果")
    save_interval: float = Field(default=1.0, gt=0, description="保存间隔(s)")
    enable_clutter: bool = Field(default=True, description="是否启用杂波")
    enable_interference: bool = Field(default=False, description="是否启用干扰")


class SimulationResult(BaseModel):
    """仿真结果"""

    simulation_id: str
    scene_id: str
    status: str
    duration: float
    plots_file: Optional[str] = None  # HDF5文件路径
    tracks_file: Optional[str] = None
    events_file: Optional[str] = None
    metadata_file: Optional[str] = None
    created_at: str
