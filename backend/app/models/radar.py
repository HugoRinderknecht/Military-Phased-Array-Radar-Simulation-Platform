"""
雷达模型数据结构
参考文档 4.5.3 节
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AntennaModel(BaseModel):
    """天线参数模型"""

    num_h: int = Field(..., ge=10, description="水平方向阵元数")
    num_v: int = Field(..., ge=10, description="垂直方向阵元数")
    d_h: float = Field(..., gt=0, le=0.5, description="水平阵元间距（波长倍数）")
    d_v: float = Field(..., gt=0, le=0.5, description="垂直阵元间距（波长倍数）")
    arrangement: Literal["rect", "triangular", "circular"] = Field(
        default="rect", description="阵元排列方式"
    )
    taper: Literal["uniform", "taylor", "hamming", "hanning", "blackman"] = Field(
        default="hamming", description="加权函数"
    )
    taylor_sidelobe_level: Optional[float] = Field(None, ge=20, le=50, description="Taylor副瓣电平(dB)")
    taylor_nbar: Optional[int] = Field(None, ge=1, le=10, description="Taylor参数")

    @field_validator("d_h", "d_v")
    @classmethod
    def validate_spacing(cls, v):
        """验证阵元间距应 ≤ 0.5λ"""
        if v > 0.5:
            raise ValueError("阵元间距必须 ≤ 0.5λ 以避免栅瓣")
        return v


class TransmitterModel(BaseModel):
    """发射机参数模型"""

    power: float = Field(..., gt=0, description="发射功率(W)")
    frequency: float = Field(..., gt=0, description="工作频率(Hz)")
    waveform: Literal["pulse", "lfm", "barker", "mseq", "frank"] = Field(
        default="lfm", description="波形类型"
    )
    pulse_width: float = Field(..., gt=0, description="脉冲宽度(s)")
    bandwidth: float = Field(..., gt=0, description="信号带宽(Hz)")
    prf: float = Field(..., gt=0, description="脉冲重复频率(Hz)")
    duty_cycle: Optional[float] = Field(None, ge=0, le=1, description="占空比")


class ReceiverModel(BaseModel):
    """接收机参数模型"""

    noise_figure: float = Field(..., ge=0, description="噪声系数(dB)")
    bandwidth: float = Field(..., gt=0, description="接收带宽(Hz)")
    temperature: float = Field(default=290, ge=0, description="噪声温度(K)")
    gain: float = Field(default=0, description="接收增益(dB)")


class ScanModel(BaseModel):
    """扫描策略模型"""

    az_min: float = Field(..., ge=-180, le=180, description="方位角起始(度)")
    az_max: float = Field(..., ge=-180, le=180, description="方位角结束(度)")
    el_min: float = Field(..., ge=-90, le=90, description="俯仰角起始(度)")
    el_max: float = Field(..., ge=-90, le=90, description="俯仰角结束(度)")
    range_max: float = Field(..., gt=0, description="最大作用距离(m)")
    beam_pattern: Literal["staggered", "raster", "spiral"] = Field(
        default="staggered", description="波束扫描模式"
    )
    scan_period: float = Field(..., gt=0, description="扫描周期(s)")


class TrackingModel(BaseModel):
    """跟踪参数模型"""

    filter: Literal["KF", "EKF", "UKF"] = Field(default="EKF", description="滤波器类型")
    association: Literal["NN", "GNN", "PDA", "JPDA"] = Field(
        default="JPDA", description="关联算法"
    )
    m_n_start: List[int] = Field(default=[3, 4], description="M/N航起始准则")
    track_lost: int = Field(default=3, ge=1, description="航迹丢失确认帧数")
    gate_probability: float = Field(default=0.99, ge=0.5, le=1)
    process_noise: float = Field(default=1.0, gt=0)
    measurement_noise: float = Field(default=10.0, gt=0)


class RadarModel(BaseModel):
    """雷达模型完整结构"""

    id: str = Field(..., description="雷达模型ID")
    name: str = Field(..., min_length=1, max_length=100, description="雷达名称")
    type: Literal["PESA", "AESA", "Mechanical"] = Field(
        default="AESA", description="雷达类型"
    )
    material: str = Field(..., description="天线材料(GaAs/GaN/Ga2O3)")
    antenna: AntennaModel
    transmitter: TransmitterModel
    receiver: ReceiverModel
    scan: ScanModel
    tracking: TrackingModel

    created_by: Optional[str] = Field(None, description="创建用户")
    created_at: Optional[str] = Field(None, description="创建时间")
    updated_at: Optional[str] = Field(None, description="更新时间")
    description: Optional[str] = Field(None, description="描述")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "radar-001",
                "name": "AN/APG-77",
                "type": "AESA",
                "material": "GaN",
                "antenna": {
                    "num_h": 1000,
                    "num_v": 200,
                    "d_h": 0.5,
                    "d_v": 0.5,
                    "arrangement": "rect",
                    "taper": "hamming",
                },
                "transmitter": {
                    "power": 20000,
                    "frequency": 10e9,
                    "waveform": "lfm",
                    "pulse_width": 1e-6,
                    "bandwidth": 10e6,
                    "prf": 10000,
                },
                "receiver": {"noise_figure": 3, "bandwidth": 10e6, "temperature": 290},
                "scan": {
                    "az_min": -60,
                    "az_max": 60,
                    "el_min": -10,
                    "el_max": 10,
                    "range_max": 200000,
                    "beam_pattern": "staggered",
                    "scan_period": 2.0,
                },
                "tracking": {
                    "filter": "EKF",
                    "association": "JPDA",
                    "m_n_start": [3, 4],
                    "track_lost": 3,
                },
                "created_by": "admin",
                "created_at": "2026-02-23T10:00:00",
                "updated_at": "2026-02-23T10:00:00",
            }
        }
