# protocol - 通信协议模块
"""
通信协议模块定义前后端通信的消息格式和序列化方式。

包括：
- messages: 消息类型定义
- commands: 指令定义和处理
- serializer: 消息序列化/反序列化
- constants: 协议常量
"""

from radar.protocol.messages import (
    MessageType,
    BaseMessage,
    StartCommand,
    StopCommand,
    PauseCommand,
    ResumeCommand,
    SetParameterCommand,
    LoadScenarioCommand,
    ManualTrackCommand,
    CancelTrackCommand,
    PlotDataMessage,
    TrackDataMessage,
    BeamDataMessage,
    WaveformDataMessage,
    SpectrumDataMessage,
    SimulationStateMessage,
    SystemStateMessage,
    ResourceStateMessage,
    DetectionEventMessage,
    TrackInitEventMessage,
    TrackLostEventMessage,
    ErrorEventMessage,
    WarningEventMessage,
    HeartbeatMessage,
    ResponseMessage,
)

__all__ = [
    # Messages
    "MessageType",
    "BaseMessage",
    "StartCommand",
    "StopCommand",
    "PauseCommand",
    "ResumeCommand",
    "SetParameterCommand",
    "LoadScenarioCommand",
    "ManualTrackCommand",
    "CancelTrackCommand",
    "PlotDataMessage",
    "TrackDataMessage",
    "BeamDataMessage",
    "WaveformDataMessage",
    "SpectrumDataMessage",
    "SimulationStateMessage",
    "SystemStateMessage",
    "ResourceStateMessage",
    "DetectionEventMessage",
    "TrackInitEventMessage",
    "TrackLostEventMessage",
    "ErrorEventMessage",
    "WarningEventMessage",
    "HeartbeatMessage",
    "ResponseMessage",
]
