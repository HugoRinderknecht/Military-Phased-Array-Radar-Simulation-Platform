# messages.py - 通信协议消息定义
"""
本模块定义前后端通信的消息结构。

使用消息类型进行分类：
- 指令消息：前端发送给后端的控制命令
- 数据消息：后端推送给前端的实时数据
- 状态消息：系统状态更新
- 事件消息：系统事件通知
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any
from datetime import datetime
import numpy as np

from radar.common.types import (
    Position3D, Velocity3D, Plot, Track
)


# Define missing types locally
class AzimuthElevation:
    def __init__(self, azimuth: float = 0.0, elevation: float = 0.0):
        self.azimuth = azimuth
        self.elevation = elevation


class BeamStatus:
    def __init__(self):
        self.azimuth = 0.0
        self.elevation = 0.0
        self.beam_id = 0
        self.task_id = 0


class SystemState:
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class SimulationState:
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


# ============================================================================
# 消息类型枚举
# ============================================================================

class MessageType:
    """消息类型常量"""
    # 前端->后端：控制指令
    CMD_START = "cmd_start"
    CMD_STOP = "cmd_stop"
    CMD_PAUSE = "cmd_pause"
    CMD_RESUME = "cmd_resume"
    CMD_SET_PARAM = "cmd_set_param"
    CMD_LOAD_SCENARIO = "cmd_load_scenario"
    CMD_MANUAL_TRACK = "cmd_manual_track"
    CMD_CANCEL_TRACK = "cmd_cancel_track"

    # 后端->前端：实时数据推送
    DATA_PLOT = "data_plot"
    DATA_TRACK = "data_track"
    DATA_BEAM = "data_beam"
    DATA_WAVEFORM = "data_waveform"
    DATA_SPECTRUM = "data_spectrum"

    # 后端->前端：状态更新
    STATE_SIMULATION = "state_simulation"
    STATE_SYSTEM = "state_system"
    STATE_RESOURCE = "state_resource"

    # 后端->前端：事件通知
    EVENT_DETECTION = "event_detection"
    EVENT_TRACK_INIT = "event_track_init"
    EVENT_TRACK_LOST = "event_track_lost"
    EVENT_ERROR = "event_error"
    EVENT_WARNING = "event_warning"

    # 心跳
    HEARTBEAT = "heartbeat"

    # 响应
    RESPONSE = "response"


# ============================================================================
# 基础消息类
# ============================================================================

@dataclass
class BaseMessage:
    """
    消息基类

    所有消息都必须继承此类，包含消息的基本元信息。

    Attributes:
        type: 消息类型
        timestamp: 时间戳 [微秒]
        sequence_id: 序列号（用于消息排序和可靠性）
    """
    type: str
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1e6))
    sequence_id: int = 0


# ============================================================================
# 指令消息
# ============================================================================

@dataclass
class StartCommand(BaseMessage):
    """
    开始仿真指令

    Args:
        time_scale: 时间缩放因子
    """
    type: str = MessageType.CMD_START
    time_scale: float = 1.0


@dataclass
class StopCommand(BaseMessage):
    """停止仿真指令"""
    type: str = MessageType.CMD_STOP


@dataclass
class PauseCommand(BaseMessage):
    """暂停仿真指令"""
    type: str = MessageType.CMD_PAUSE


@dataclass
class ResumeCommand(BaseMessage):
    """恢复仿真指令"""
    type: str = MessageType.CMD_RESUME


@dataclass
class SetParameterCommand(BaseMessage):
    """
    设置参数指令

    Args:
        param_path: 参数路径，如 "radar.frequency"
        value: 参数值
    """
    type: str = MessageType.CMD_SET_PARAM
    param_path: str = ""
    value: Any = None


@dataclass
class LoadScenarioCommand(BaseMessage):
    """
    加载场景指令

    Args:
        scenario_data: 场景数据（JSON字符串或dict）
    """
    type: str = MessageType.CMD_LOAD_SCENARIO
    scenario_data: dict = field(default_factory=dict)


@dataclass
class ManualTrackCommand(BaseMessage):
    """
    手动跟踪指令

    Args:
        plot_id: 要跟踪的点迹ID
    """
    type: str = MessageType.CMD_MANUAL_TRACK
    plot_id: int = 0


@dataclass
class CancelTrackCommand(BaseMessage):
    """
    取消跟踪指令

    Args:
        track_id: 要取消的航迹ID
    """
    type: str = MessageType.CMD_CANCEL_TRACK
    track_id: int = 0


# ============================================================================
# 数据消息
# ============================================================================

@dataclass
class PlotDataMessage(BaseMessage):
    """
    点迹数据消息

    包含当前帧检测到的所有点迹。

    Attributes:
        plots: 点迹列表
        frame_number: 帧号
    """
    type: str = MessageType.DATA_PLOT
    plots: List[Plot] = field(default_factory=list)
    frame_number: int = 0


@dataclass
class TrackDataMessage(BaseMessage):
    """
    航迹数据消息

    包含当前所有跟踪的航迹。

    Attributes:
        tracks: 航迹列表
        frame_number: 帧号
    """
    type: str = MessageType.DATA_TRACK
    tracks: List[Track] = field(default_factory=list)
    frame_number: int = 0


@dataclass
class BeamDataMessage(BaseMessage):
    """
    波束数据消息

    包含当前波束指向和状态。

    Attributes:
        beam_status: 波束状态
        frame_number: 帧号
    """
    type: str = MessageType.DATA_BEAM
    beam_status: BeamStatus = field(default_factory=BeamStatus)
    frame_number: int = 0


@dataclass
class WaveformDataMessage(BaseMessage):
    """
    波形数据消息

    用于调试和可视化，包含发射波形信息。

    Attributes:
        waveform: 波形数据（复数数组）
        sample_rate: 采样率 [Hz]
        pulse_width: 脉冲宽度 [秒]
    """
    type: str = MessageType.DATA_WAVEFORM
    waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.complex128))
    sample_rate: float = 20e6
    pulse_width: float = 10e-6


@dataclass
class SpectrumDataMessage(BaseMessage):
    """
    频谱数据消息

    用于显示信号频谱，便于调试。

    Attributes:
        spectrum: 频谱数据（幅度）
        frequencies: 频率轴 [Hz]
    """
    type: str = MessageType.DATA_SPECTRUM
    spectrum: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    frequencies: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))


# ============================================================================
# 状态消息
# ============================================================================

@dataclass
class SimulationStateMessage(BaseMessage):
    """
    仿真状态消息

    包含仿真的运行状态信息。

    Attributes:
        is_running: 是否正在运行
        is_paused: 是否暂停
        current_time: 当前仿真时间 [微秒]
        frame_count: 帧计数
        fps: 帧率
    """
    type: str = MessageType.STATE_SIMULATION
    is_running: bool = False
    is_paused: bool = False
    current_time: int = 0
    frame_count: int = 0
    fps: float = 60.0


@dataclass
class SystemStateMessage(BaseMessage):
    """
    系统状态消息

    包含系统级别的状态信息。

    Attributes:
        state: 系统状态枚举
        message: 状态描述
        error_code: 错误代码
        uptime: 运行时间 [秒]
    """
    type: str = MessageType.STATE_SYSTEM
    state: SystemState = SystemState.IDLE
    message: str = ""
    error_code: int = 0
    uptime: float = 0.0


@dataclass
class ResourceStateMessage(BaseMessage):
    """
    资源状态消息

    包含系统资源使用情况。

    Attributes:
        cpu_usage: CPU使用率 [%]
        memory_usage: 内存使用 [MB]
        time_utilization: 雷达时间资源利用率 [%]
        task_queue_size: 任务队列长度
    """
    type: str = MessageType.STATE_RESOURCE
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    time_utilization: float = 0.0
    task_queue_size: int = 0


# ============================================================================
# 事件消息
# ============================================================================

@dataclass
class DetectionEventMessage(BaseMessage):
    """
    检测事件消息

    当检测到新目标时发送。

    Attributes:
        plot: 检测到的点迹
        snr: 信噪比 [dB]
    """
    type: str = MessageType.EVENT_DETECTION
    plot: Plot = field(default_factory=lambda: Plot(
        id=0, timestamp=0, position=Position3D(0, 0, 0)
    ))
    snr: float = 0.0


@dataclass
class TrackInitEventMessage(BaseMessage):
    """
    航迹起始事件消息

    当新航迹起始时发送。

    Attributes:
        track_id: 航迹ID
        initial_position: 初始位置
    """
    type: str = MessageType.EVENT_TRACK_INIT
    track_id: int = 0
    initial_position: Position3D = field(default_factory=lambda: Position3D(0, 0, 0))


@dataclass
class TrackLostEventMessage(BaseMessage):
    """
    航迹丢失事件消息

    当航迹丢失时发送。

    Attributes:
        track_id: 航迹ID
        last_position: 最后已知位置
        reason: 丢失原因
    """
    type: str = MessageType.EVENT_TRACK_LOST
    track_id: int = 0
    last_position: Position3D = field(default_factory=lambda: Position3D(0, 0, 0))
    reason: str = ""


@dataclass
class ErrorEventMessage(BaseMessage):
    """
    错误事件消息

    当系统发生错误时发送。

    Attributes:
        error_code: 错误代码
        error_message: 错误消息
        severity: 严重程度 (error/warning/critical)
    """
    type: str = MessageType.EVENT_ERROR
    error_code: int = 0
    error_message: str = ""
    severity: str = "error"


@dataclass
class WarningEventMessage(BaseMessage):
    """
    警告事件消息

    当系统产生警告时发送。

    Attributes:
        warning_code: 警告代码
        warning_message: 警告消息
    """
    type: str = MessageType.EVENT_WARNING
    warning_code: int = 0
    warning_message: str = ""


# ============================================================================
# 心跳消息
# ============================================================================

@dataclass
class HeartbeatMessage(BaseMessage):
    """
    心跳消息

    用于保持连接活跃和检测超时。

    Attributes:
        server_time: 服务器当前时间 [微秒]
    """
    type: str = MessageType.HEARTBEAT
    server_time: int = field(default_factory=lambda: int(datetime.now().timestamp() * 1e6))


# ============================================================================
# 响应消息
# ============================================================================

@dataclass
class ResponseMessage(BaseMessage):
    """
    通用响应消息

    用于确认指令执行结果。

    Attributes:
        success: 是否成功
        request_id: 对应的请求ID
        data: 响应数据
        error_message: 错误消息（如果失败）
    """
    type: str = MessageType.RESPONSE
    success: bool = True
    request_id: int = 0
    data: Any = None
    error_message: str = ""


# ============================================================================
# 导出所有消息类型
# ============================================================================

__all__ = [
    # 消息类型
    "MessageType",

    # 基础消息
    "BaseMessage",

    # 指令消息
    "StartCommand",
    "StopCommand",
    "PauseCommand",
    "ResumeCommand",
    "SetParameterCommand",
    "LoadScenarioCommand",
    "ManualTrackCommand",
    "CancelTrackCommand",

    # 数据消息
    "PlotDataMessage",
    "TrackDataMessage",
    "BeamDataMessage",
    "WaveformDataMessage",
    "SpectrumDataMessage",

    # 状态消息
    "SimulationStateMessage",
    "SystemStateMessage",
    "ResourceStateMessage",

    # 事件消息
    "DetectionEventMessage",
    "TrackInitEventMessage",
    "TrackLostEventMessage",
    "ErrorEventMessage",
    "WarningEventMessage",

    # 心跳和响应
    "HeartbeatMessage",
    "ResponseMessage",
]
