# 异常定义模块 (Exception Definitions Module)
# 本模块定义雷达仿真系统中使用的所有自定义异常

from typing import Optional, Any


class RadarError(Exception):
    """
    雷达系统基础异常类 (Base Radar Exception Class)
    
    所有雷达系统自定义异常的基类
    """
    
    def __init__(self, message: str, error_code: Optional[int] = None):
        """
        初始化异常
        
        Args:
            message: 错误信息
            error_code: 错误码（可选）
        """
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.error_code is not None:
            return f'[Error {self.error_code}] {self.message}'
        return self.message


class ConfigError(RadarError):
    """配置异常 (Configuration Error)"""
    pass


class InitializationError(RadarError):
    """初始化异常 (Initialization Error)"""
    pass


class SimulationError(RadarError):
    """仿真异常 (Simulation Error)"""
    pass


class TargetError(RadarError):
    """目标相关异常 (Target-related Error)"""
    pass


class AntennaError(RadarError):
    """天线相关异常 (Antenna-related Error)"""
    pass


class SignalProcessingError(RadarError):
    """信号处理异常 (Signal Processing Error)"""
    pass


class DataProcessingError(RadarError):
    """数据处理异常 (Data Processing Error)"""
    pass


class SchedulingError(RadarError):
    """调度异常 (Scheduling Error)"""
    pass


class NetworkError(RadarError):
    """网络通信异常 (Network Communication Error)"""
    pass


class ProtocolError(RadarError):
    """协议异常 (Protocol Error)"""
    pass


class ValidationError(RadarError):
    """数据验证异常 (Data Validation Error)"""
    pass


class ResourceError(RadarError):
    """资源异常 (Resource Error)"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None):
        """
        初始化资源异常
        
        Args:
            message: 错误信息
            resource_type: 资源类型（如'memory', 'cpu', 'time'）
        """
        self.resource_type = resource_type
        full_message = f'{resource_type}: {message}' if resource_type else message
        super().__init__(full_message)


class TimeoutError(RadarError):
    """超时异常 (Timeout Error)"""
    pass


class CalculationError(RadarError):
    """计算异常 (Calculation Error)"""
    pass


class ArrayError(AntennaError):
    """阵列天线异常 (Array Antenna Error)"""
    pass


class BeamError(AntennaError):
    """波束相关异常 (Beam-related Error)"""
    pass


class FilterError(SignalProcessingError):
    """滤波器异常 (Filter Error)"""
    pass


class DetectionError(SignalProcessingError):
    """检测异常 (Detection Error)"""
    pass


class TrackError(DataProcessingError):
    """航迹相关异常 (Track-related Error)"""
    pass


class AssociationError(DataProcessingError):
    """数据关联异常 (Data Association Error)"""
    pass


class ClutterModelError(SimulationError):
    """杂波模型异常 (Clutter Model Error)"""
    pass


class PropagationError(SimulationError):
    """传播模型异常 (Propagation Model Error)"""
    pass


class JammerError(SimulationError):
    """干扰模型异常 (Jammer Model Error)"""
    pass


# ==================== 导出所有异常 ====================

__all__ = [
    'RadarError',
    'ConfigError',
    'InitializationError',
    'SimulationError',
    'TargetError',
    'AntennaError',
    'SignalProcessingError',
    'DataProcessingError',
    'SchedulingError',
    'NetworkError',
    'ProtocolError',
    'ValidationError',
    'ResourceError',
    'TimeoutError',
    'CalculationError',
    'ArrayError',
    'BeamError',
    'FilterError',
    'DetectionError',
    'TrackError',
    'AssociationError',
    'ClutterModelError',
    'PropagationError',
    'JammerError',
]
