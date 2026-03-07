"""
场景模型数据结构
参考文档 4.5.2 节
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class TrajectoryPoint(BaseModel):
    """轨迹点模型"""

    time: float = Field(..., ge=0, description="时间(s)")
    x: float = Field(..., description="X坐标(m)")
    y: float = Field(..., description="Y坐标(m)")
    z: float = Field(..., description="Z坐标(m)")
    vx: Optional[float] = Field(None, description="X速度(m/s)")
    vy: Optional[float] = Field(None, description="Y速度(m/s)")
    vz: Optional[float] = Field(None, description="Z速度(m/s)")


class TargetModel(BaseModel):
    """目标模型"""

    id: str = Field(..., description="目标ID")
    name: str = Field(..., min_length=1, max_length=100, description="目标名称")
    type: Literal["fighter", "bomber", "helicopter", "uav", "missile", "ship", "civil"] = Field(
        default="fighter", description="目标类型"
    )
    iff_code: Optional[str] = Field(None, description="IFF识别码")

    # RCS 模型
    rcs_model: Literal["swerling1", "swerling2", "swerling3", "swerling4", "constant"] = Field(
        default="swerling1", description="RCS起伏模型"
    )
    rcs_mean: float = Field(..., gt=0, description="平均RCS(m²)")

    # 轨迹
    trajectory: List[TrajectoryPoint] = Field(..., min_length=1, description="轨迹点列表")

    # 其他属性
    altitude: Optional[float] = Field(None, description="高度(m)")
    speed: Optional[float] = Field(None, ge=0, description="速度(m/s)")
    heading: Optional[float] = Field(None, ge=0, lt=360, description="航向(度)")


class ClutterModel(BaseModel):
    """杂波模型"""

    type: Literal["ground", "sea", "rain", "chaff"] = Field(
        default="ground", description="杂波类型"
    )
    pdf: Literal["rayleigh", "lognormal", "weibull", "k"] = Field(
        default="rayleigh", description="概率分布"
    )
    psd: Literal["gaussian", "exponential", "cauchy"] = Field(
        default="gaussian", description="功率谱密度"
    )
    strength: float = Field(default=0.1, ge=0, description="杂波强度")
    correlation_time: float = Field(default=0.01, gt=0, description="相关时间(s)")


class InterferenceModel(BaseModel):
    """干扰模型"""

    type: Literal["noise", "sweep", "pulse"] = Field(default="noise", description="干扰类型")
    frequency: float = Field(..., gt=0, description="干扰频率(Hz)")
    power: float = Field(..., ge=0, description="干扰功率(W)")
    bandwidth: float = Field(default=10e6, gt=0, description="干扰带宽(Hz)")


class AtmosphereModel(BaseModel):
    """大气效应模型"""

    temperature: float = Field(default=15, description="温度(°C)")
    pressure: float = Field(default=101325, description="气压(Pa)")
    humidity: float = Field(default=50, ge=0, le=100, description="湿度(%)")
    rain_rate: float = Field(default=0, ge=0, description="降雨率(mm/h)")


class EnvironmentModel(BaseModel):
    """环境参数模型"""

    # 噪声
    noise_power: float = Field(default=1e-12, gt=0, description="噪声功率(W)")

    # 杂波
    clutter: Optional[ClutterModel] = None

    # 干扰
    interferences: List[InterferenceModel] = Field(default_factory=list)

    # 大气
    atmosphere: AtmosphereModel = Field(default_factory=AtmosphereModel)


class SceneModel(BaseModel):
    """场景模型完整结构"""

    id: str = Field(..., description="场景ID")
    name: str = Field(..., min_length=1, max_length=100, description="场景名称")
    description: Optional[str] = Field(None, description="场景描述")

    # 关联的雷达模型
    radar_model_id: str = Field(..., description="雷达模型ID")

    # 目标列表
    targets: List[TargetModel] = Field(default_factory=list, description="目标列表")

    # 环境参数
    environment: EnvironmentModel = Field(default_factory=EnvironmentModel)

    # 元数据
    created_by: Optional[str] = Field(None, description="创建用户")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")

    # 仿真参数
    duration: float = Field(default=100, gt=0, description="仿真时长(s)")
    time_step: float = Field(default=0.1, gt=0, description="时间步长(s)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "scene-001",
                "name": "两架敌机突防",
                "description": "模拟两架战斗机低空突防场景",
                "radar_model_id": "radar-001",
                "targets": [
                    {
                        "id": "t1",
                        "name": "Target-1",
                        "type": "fighter",
                        "rcs_model": "swerling1",
                        "rcs_mean": 5,
                        "trajectory": [
                            {"time": 0, "x": 100000, "y": 0, "z": 5000},
                            {"time": 10, "x": 90000, "y": 5000, "z": 4800},
                        ],
                    }
                ],
                "environment": {
                    "noise_power": 1e-12,
                    "clutter": {
                        "type": "ground",
                        "pdf": "rayleigh",
                        "psd": "gaussian",
                        "strength": 0.1,
                    },
                    "atmosphere": {"temperature": 15, "pressure": 101325, "humidity": 50},
                },
                "created_by": "admin",
                "created_at": "2026-02-23T10:00:00",
                "updated_at": "2026-02-23T10:00:00",
                "duration": 100,
                "time_step": 0.1,
            }
        }
