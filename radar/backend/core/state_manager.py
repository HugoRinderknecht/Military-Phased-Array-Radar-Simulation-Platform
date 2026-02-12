# state_manager.py - 状态管理器
"""
本模块实现系统状态的管理。

状态管理器负责：
- 维护系统全局状态
- 提供状态查询接口
- 状态变化通知
- 线程安全的状态访问
"""

import asyncio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from radar.common.logger import get_logger
from radar.common.types import SystemState


class StateEvent(Enum):
    """状态事件类型"""
    ENTER = "enter"     # 进入状态
    EXIT = "exit"       # 退出状态
    CHANGE = "change"     # 状态变化


@dataclass
class StateTransition:
    """
    状态转换记录

    Attributes:
        from_state: 源状态
        to_state: 目标状态
        timestamp: 时间戳 [微秒]
        reason: 转换原因
    """
    from_state: SystemState
    to_state: SystemState
    timestamp: int
    reason: str = ""


class StateManager:
    """
    系统状态管理器

    管理雷达仿真系统的全局状态。
    """

    def __init__(self):
        """初始化状态管理器"""
        self._logger = get_logger("state_manager")

        # 当前状态
        self._current_state: SystemState = SystemState.IDLE

        # 状态转换历史
        self._transition_history: list = field(default_factory=list)
        self._max_history = 100

        # 状态变化回调
        self._state_callbacks: Dict[SystemState, list] = {}
        self._any_state_callbacks: list = []

        # 状态锁
        self._lock = asyncio.Lock()

        # 状态数据存储
        self._state_data: Dict[str, Any] = {}

        self._logger.info(f"状态管理器初始化: 初始状态={self._current_state.value}")

    async def get_state(self) -> SystemState:
        """
        获取当前状态

        Returns:
            当前系统状态
        """
        async with self._lock:
            return self._current_state

    async def set_state(self, new_state: SystemState, reason: str = "") -> bool:
        """
        设置新状态

        Args:
            new_state: 新状态
            reason: 状态转换原因

        Returns:
            是否成功转换
        """
        async with self._lock:
            if self._current_state == new_state:
                self._logger.debug(f"状态未变化: {new_state.value}")
                return False

            old_state = self._current_state

            # 检查状态转换是否合法
            if not self._is_valid_transition(old_state, new_state):
                self._logger.warning(
                    f"非法状态转换: {old_state.value} -> {new_state.value}"
                )
                return False

            # 记录转换
            from radar.common.types import SimulationState
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                timestamp=int(SimulationState.current_time),
                reason=reason
            )
            self._transition_history.append(transition)

            # 限制历史长度
            if len(self._transition_history) > self._max_history:
                self._transition_history.pop(0)

            # 更新状态
            self._current_state = new_state

            self._logger.info(f"状态转换: {old_state.value} -> {new_state.value}, 原因: {reason}")

            # 触发回调
            await self._trigger_callbacks(old_state, new_state, reason)

            return True

    def _is_valid_transition(self, from_state: SystemState, to_state: SystemState) -> bool:
        """
        检查状态转换是否合法

        Args:
            from_state: 源状态
            to_state: 目标状态

        Returns:
            是否合法
        """
        # 定义允许的状态转换
        valid_transitions = {
            SystemState.IDLE: [SystemState.INITIALIZING, SystemState.ERROR],
            SystemState.INITIALIZING: [SystemState.RUNNING, SystemState.ERROR],
            SystemState.RUNNING: [SystemState.PAUSED, SystemState.SHUTTING_DOWN, SystemState.ERROR],
            SystemState.PAUSED: [SystemState.RUNNING, SystemState.SHUTTING_DOWN],
            SystemState.SHUTTING_DOWN: [SystemState.IDLE],
            SystemState.ERROR: [SystemState.IDLE, SystemState.SHUTTING_DOWN],
        }

        allowed = valid_transitions.get(from_state, [])
        return to_state in allowed

    async def register_callback(self, state: SystemState, callback: Callable) -> None:
        """
        注册状态变化回调

        Args:
            state: 要监听的状态（None表示所有状态）
            callback: 回调函数，签名为 async def callback(old_state, new_state, reason)
        """
        if state is None:
            self._any_state_callbacks.append(callback)
        else:
            if state not in self._state_callbacks:
                self._state_callbacks[state] = []
            self._state_callbacks[state].append(callback)

        self._logger.debug(f"注册状态回调: state={state.value if state else 'any'}")

    async def _trigger_callbacks(self, old_state: SystemState,
                                new_state: SystemState, reason: str) -> None:
        """
        触发状态变化回调

        Args:
            old_state: 旧状态
            new_state: 新状态
            reason: 转换原因
        """
        # 触发特定状态回调
        if new_state in self._state_callbacks:
            for callback in self._state_callbacks[new_state]:
                try:
                    await callback(old_state, new_state, reason)
                except Exception as e:
                    self._logger.error(f"状态回调执行失败: {e}")

        # 触发任意状态回调
        for callback in self._any_state_callbacks:
            try:
                await callback(old_state, new_state, reason)
            except Exception as e:
                self._logger.error(f"状态回调执行失败: {e}")

    def get_state_data(self, key: str, default: Any = None) -> Any:
        """
        获取状态相关数据

        Args:
            key: 数据键
            default: 默认值

        Returns:
            数据值
        """
        return self._state_data.get(key, default)

    async def set_state_data(self, key: str, value: Any) -> None:
        """
        设置状态相关数据

        Args:
            key: 数据键
            value: 数据值
        """
        self._state_data[key] = value
        self._logger.debug(f"设置状态数据: {key}={value}")

    def get_transition_history(self, limit: int = 10) -> list:
        """
        获取状态转换历史

        Args:
            limit: 返回的最大记录数

        Returns:
            状态转换记录列表
        """
        return self._transition_history[-limit:]

    def clear_history(self) -> None:
        """清空历史记录"""
        self._transition_history.clear()
        self._logger.info("状态转换历史已清空")

    async def wait_for_state(self, state: SystemState, timeout: float = 10.0) -> bool:
        """
        等待指定状态

        Args:
            state: 目标状态
            timeout: 超时时间 [秒]

        Returns:
            是否成功等到状态
        """
        if self._current_state == state:
            return True

        try:
            # 创建事件等待
            event = asyncio.Event()
            async def callback(old, new, reason):
                if new == state:
                    event.set()
            await self.register_callback(state, callback)
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    @property
    def current_state(self) -> SystemState:
        """获取当前状态（同步接口）"""
        return self._current_state

    def is_state(self, state: SystemState) -> bool:
        """检查是否处于指定状态"""
        return self._current_state == state

    def is_running(self) -> bool:
        """检查是否正在运行"""
        return self._current_state == SystemState.RUNNING

    def is_paused(self) -> bool:
        """检查是否暂停"""
        return self._current_state == SystemState.PAUSED

    def is_idle(self) -> bool:
        """检查是否空闲"""
        return self._current_state == SystemState.IDLE

    def is_error(self) -> bool:
        """检查是否错误状态"""
        return self._current_state == SystemState.ERROR

    def __repr__(self) -> str:
        """字符串表示"""
        return f"StateManager(state={self._current_state.value})"


__all__ = [
    "StateManager",
    "StateEvent",
    "StateTransition",
]
