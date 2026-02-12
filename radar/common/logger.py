# 日志系统模块

import sys
import logging
from pathlib import Path
from typing import Optional
from loguru import logger as loguru_logger
from datetime import datetime


class RadarLogger:
    """雷达系统日志管理器"""
    
    def __init__(
        self,
        log_path: str = './logs',
        log_level: str = 'INFO',
        enable_console: bool = True,
        enable_file: bool = True,
        rotation: str = '100 MB',
        retention: str = '30 days'
    ):
        """初始化日志系统"""
        self.log_path = Path(log_path)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.log_level = log_level
        
        # 移除默认处理器
        loguru_logger.remove()
        
        # 添加控制台处理器
        if enable_console:
            loguru_logger.add(
                sys.stderr,
                format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
                level=log_level,
                colorize=True
            )
        
        # 添加文件处理器
        if enable_file:
            # 所有日志
            loguru_logger.add(
                self.log_path / 'radar_{time:YYYY-MM-DD}.log',
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level=log_level,
                rotation=rotation,
                retention=retention,
                encoding='utf-8'
            )
            
            # 错误日志
            loguru_logger.add(
                self.log_path / 'radar_error_{time:YYYY-MM-DD}.log',
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
                level='ERROR',
                rotation=rotation,
                retention=retention,
                encoding='utf-8'
            )
        
        self._logger = loguru_logger
    
    def get_logger(self):
        """获取logger实例"""
        return self._logger
    
    def set_level(self, level: str) -> None:
        """设置日志级别"""
        self.log_level = level
        # 这里可以更新所有处理器的级别
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """调试日志"""
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """信息日志"""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """警告日志"""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """错误日志"""
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """严重错误日志"""
        self._logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """异常日志（包含堆栈跟踪）"""
        self._logger.exception(message, *args, **kwargs)


# 全局日志实例
_radar_logger: Optional[RadarLogger] = None


def get_logger(
    log_path: str = './logs',
    log_level: str = 'INFO',
    enable_console: bool = True,
    enable_file: bool = True
) -> RadarLogger:
    """获取全局日志实例"""
    global _radar_logger
    if _radar_logger is None:
        _radar_logger = RadarLogger(
            log_path=log_path,
            log_level=log_level,
            enable_console=enable_console,
            enable_file=enable_file
        )
    return _radar_logger


def init_logger(
    log_path: str = './logs',
    log_level: str = 'INFO',
    enable_console: bool = True,
    enable_file: bool = True
) -> RadarLogger:
    """初始化全局日志系统"""
    global _radar_logger
    _radar_logger = RadarLogger(
        log_path=log_path,
        log_level=log_level,
        enable_console=enable_console,
        enable_file=enable_file
    )
    return _radar_logger


# 便捷函数
def debug(message: str, *args, **kwargs) -> None:
    """调试日志"""
    if _radar_logger is not None:
        _radar_logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs) -> None:
    """信息日志"""
    if _radar_logger is not None:
        _radar_logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    """警告日志"""
    if _radar_logger is not None:
        _radar_logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    """错误日志"""
    if _radar_logger is not None:
        _radar_logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs) -> None:
    """严重错误日志"""
    if _radar_logger is not None:
        _radar_logger.critical(message, *args, **kwargs)


def exception(message: str, *args, **kwargs) -> None:
    """异常日志"""
    if _radar_logger is not None:
        _radar_logger.exception(message, *args, **kwargs)


__all__ = [
    'RadarLogger',
    'get_logger',
    'init_logger',
    'debug',
    'info',
    'warning',
    'error',
    'critical',
    'exception',
]
