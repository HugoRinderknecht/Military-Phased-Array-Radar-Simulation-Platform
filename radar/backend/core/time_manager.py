# time_manager.py - 时间管理器
"""
本模块实现仿真时间的管理。

时间管理器负责：
- 维护仿真时钟
- 处理时间缩放（加速/减速）
- 时间步进控制
- 帧同步
"""

import asyncio
from typing import Optional
from datetime import datetime, timedelta

from radar.common.logger import get_logger


class TimeManager:
    """
    仿真时间管理器

    管理仿真时间的推进和缩放。
    """

    def __init__(self, time_scale: float = 1.0, frame_rate: float = 60.0):
        """
        初始化时间管理器

        Args:
            time_scale: 时间缩放因子 (1.0 = 实时)
            frame_rate: 目标帧率 [Hz]
        """
        self._logger = get_logger("time_manager")

        # 时间配置
        self._time_scale = time_scale
        self._frame_rate = frame_rate
        self._frame_interval = 1.0 / frame_rate if frame_rate > 0 else 0.0

        # 仿真时间 [微秒]
        self._simulation_time_us = 0

        # 实际时间基准
        self._real_start_time: Optional[datetime] = None

        # 帧计数
        self._frame_count = 0

        # 运行状态
        self._is_paused = False
        self._is_running = False

        self._logger.info(f"时间管理器初始化: scale={time_scale}x, fps={frame_rate}")

    @property
    def simulation_time(self) -> int:
        """获取当前仿真时间 [微秒]"""
        return self._simulation_time_us

    @property
    def simulation_time_seconds(self) -> float:
        """获取当前仿真时间 [秒]"""
        return self._simulation_time_us / 1e6

    @property
    def time_scale(self) -> float:
        """获取时间缩放因子"""
        return self._time_scale

    @time_scale.setter
    def time_scale(self, value: float) -> None:
        """设置时间缩放因子"""
        if value <= 0:
            self._logger.warning(f"无效的时间缩放因子: {value}, 使用1.0")
            self._time_scale = 1.0
        else:
            self._time_scale = value
            self._logger.info(f"时间缩放因子设置为: {value}x")

    @property
    def frame_rate(self) -> float:
        """获取帧率"""
        return self._frame_rate

    @frame_rate.setter
    def frame_rate(self, value: float) -> None:
        """设置帧率"""
        if value <= 0:
            self._logger.warning(f"无效的帧率: {value}, 使用60Hz")
            self._frame_rate = 60.0
        else:
            self._frame_rate = value
            self._frame_interval = 1.0 / value
            self._logger.info(f"帧率设置为: {value}Hz")

    @property
    def frame_interval(self) -> float:
        """获取帧间隔 [秒]"""
        return self._frame_interval

    @property
    def frame_count(self) -> int:
        """获取帧计数"""
        return self._frame_count

    @property
    def is_paused(self) -> bool:
        """是否暂停"""
        return self._is_paused

    @property
    def is_running(self) -> bool:
        """是否运行"""
        return self._is_running

    def start(self) -> None:
        """开始时间管理"""
        self._is_running = True
        self._is_paused = False
        self._real_start_time = datetime.now()
        self._logger.info("时间管理器已启动")

    def pause(self) -> None:
        """暂停时间推进"""
        self._is_paused = True
        self._logger.info("时间管理器已暂停")

    def resume(self) -> None:
        """恢复时间推进"""
        self._is_paused = False
        self._logger.info("时间管理器已恢复")

    def stop(self) -> None:
        """停止时间管理"""
        self._is_running = False
        self._is_paused = False
        elapsed = (datetime.now() - self._real_start_time).total_seconds() if self._real_start_time else 0
        self._logger.info(f"时间管理器已停止. 运行时间: {elapsed:.2f}秒")

    def advance(self, delta_seconds: float) -> int:
        """
        推进仿真时间

        Args:
            delta_seconds: 实际流逝时间 [秒]

        Returns:
            推进后的仿真时间 [微秒]
        """
        if not self._is_running or self._is_paused:
            return self._simulation_time_us

        # 根据时间缩放计算仿真时间增量
        sim_delta_us = int(delta_seconds * self._time_scale * 1e6)
        self._simulation_time_us += sim_delta_us
        self._frame_count += 1

        return self._simulation_time_us

    def advance_frame(self) -> int:
        """
        按帧推进时间

        Returns:
            推进后的仿真时间 [微秒]
        """
        return self.advance(self._frame_interval)

    def reset(self) -> None:
        """重置时间"""
        self._simulation_time_us = 0
        self._frame_count = 0
        self._logger.info("时间已重置")

    def get_elapsed_real_time(self) -> float:
        """
        获取实际流逝时间 [秒]

        Returns:
            从启动到现在的实际时间 [秒]
        """
        if self._real_start_time is None:
            return 0.0
        return (datetime.now() - self._real_start_time).total_seconds()

    def get_elapsed_simulation_time(self) -> float:
        """
        获取仿真流逝时间 [秒]

        Returns:
            仿真系统内部经过的时间 [秒]
        """
        return self.simulation_time_seconds

    async def wait_for_frame(self) -> None:
        """
        等待下一帧

        使用异步sleep实现帧同步。
        """
        if self._frame_interval > 0:
            await asyncio.sleep(self._frame_interval)

    def __repr__(self) -> str:
        """字符串表示"""
        return (f"TimeManager(sim_time={self.simulation_time_seconds:.3f}s, "
                f"scale={self._time_scale}x, frame={self._frame_count})")


__all__ = [
    "TimeManager",
]
