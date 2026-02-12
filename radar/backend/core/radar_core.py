# radar_core.py - 雷达核心控制器
"""
本模块实现雷达仿真平台的核心控制器。

RadarCore是整个系统的协调者，负责：
- 初始化所有子系统
- 运行仿真主循环
- 协调各模块间的数据流
- 处理前端指令
- 发布仿真数据
"""

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from radar.common.logger import get_logger
from radar.common.types import (
    Position3D, Velocity3D, Plot, Track, SystemState, TaskType, WorkMode
)
from radar.common.exceptions import RadarError

from radar.backend.core.time_manager import TimeManager
from radar.backend.core.state_manager import StateManager


class IRadarCore:
    """雷达核心控制器接口"""

    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化雷达系统"""
        pass

    async def run(self) -> None:
        """运行仿真主循环"""
        pass

    async def shutdown(self) -> None:
        """关闭系统"""
        pass

    async def pause(self) -> None:
        """暂停仿真"""
        pass

    async def resume(self) -> None:
        """恢复仿真"""
        pass

    def get_state(self) -> SystemState:
        """获取当前状态"""
        pass


@dataclass
class RadarConfig:
    """
    雷达配置

    Attributes:
        # 基本参数
        frequency: 工作频率 [Hz]
        bandwidth: 信号带宽 [Hz]
        sample_rate: 采样率 [Hz]
        pulse_width: 脉冲宽度 [秒]
        prf: 脉冲重复频率 [Hz]

        # 天线参数
        array_size: 阵列大小 (M, N)
        element_spacing: 阵元间距 [米]
        antenna_gain: 天线增益 [dB]

        # 发射参数
        transmit_power: 发射功率 [W]

        # 接收参数
        noise_figure: 噪声系数 [dB]
        system_losses: 系统损耗 [dB]

        # 处理参数
        cfar_threshold: CFAR检测门限
        integration_gain: 积累增益 [dB]

        # 调度参数
        schedule_interval: 调度间隔 [秒]

        # 工作模式
        work_mode: 工作模式
    """
    # 基本参数
    frequency: float = 10e9           # X波段
    bandwidth: float = 10e6           # 10 MHz
    sample_rate: float = 20e6         # 20 MHz
    pulse_width: float = 10e-6       # 10 微秒
    prf: float = 2000.0             # 2 kHz

    # 天线参数
    array_size: tuple = (32, 32)     # 32x32阵列
    element_spacing: float = 0.015     # 半波长
    antenna_gain: float = 40.0        # 40 dB

    # 发射参数
    transmit_power: float = 100e3     # 100 kW

    # 接收参数
    noise_figure: float = 3.0          # 3 dB
    system_losses: float = 8.0          # 8 dB

    # 处理参数
    cfar_threshold: float = 12.0       # 线性值
    integration_pulses: int = 10         # 10脉冲

    # 调度参数
    schedule_interval: float = 0.05    # 50ms

    # 工作模式
    work_mode: WorkMode = WorkMode.TWS


class RadarCore(IRadarCore):
    """
    雷达核心控制器

    协调所有子系统，运行仿真主循环。
    """

    def __init__(self, config: Optional[RadarConfig] = None):
        """
        初始化雷达核心

        Args:
            config: 雷达配置，None则使用默认配置
        """
        self._logger = get_logger("radar_core")
        self._config = config or RadarConfig()

        # 核心组件
        self._time_manager = TimeManager(
            time_scale=1.0,
            frame_rate=1.0 / self._config.schedule_interval
        )
        self._state_manager = StateManager()

        # 子系统（将在initialize中创建）
        self._environment = None
        self._antenna = None
        self._signal_processor = None
        self._data_processor = None
        self._scheduler = None
        self._evaluator = None

        # 网络服务
        self._websocket_server = None
        self._http_server = None

        # 数据队列
        self._plot_queue: asyncio.Queue = None
        self._track_queue: asyncio.Queue = None
        self._beam_queue: asyncio.Queue = None
        self._command_queue: asyncio.Queue = None

        # 运行标志
        self._running = False
        self._paused = False

        # 性能统计
        self._stats = {
            'frames_processed': 0,
            'plots_detected': 0,
            'tracks_active': 0,
            'average_frame_time': 0.0,
        }

    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        初始化雷达系统

        Args:
            config: 配置字典，将覆盖默认配置

        Returns:
            是否初始化成功
        """
        try:
            self._logger.info("开始初始化雷达系统...")

            # 更新配置
            if config:
                self._update_config(config)

            # 更新状态
            await self._state_manager.set_state(SystemState.INITIALIZING, "系统初始化")

            # 初始化数据队列
            self._plot_queue = asyncio.Queue(maxsize=1000)
            self._track_queue = asyncio.Queue(maxsize=1000)
            self._beam_queue = asyncio.Queue(maxsize=1000)
            self._command_queue = asyncio.Queue(maxsize=100)

            # 初始化子系统
            await self._initialize_subsystems()

            # 初始化网络服务
            await self._initialize_network()

            self._logger.success("雷达系统初始化完成")

            # 更新状态
            await self._state_manager.set_state(SystemState.IDLE, "初始化完成")

            return True

        except Exception as e:
            self._logger.error(f"初始化失败: {e}")
            await self._state_manager.set_state(SystemState.ERROR, f"初始化失败: {e}")
            return False

    async def _initialize_subsystems(self) -> None:
        """初始化各子系统"""
        self._logger.info("初始化子系统...")

        # TODO: 创建各子系统实例
        # from radar.backend.environment import EnvironmentSimulator
        # from radar.backend.antenna import AntennaSystem
        # from radar.backend.signal import SignalProcessor
        # from radar.backend.dataproc import DataProcessor
        # from radar.backend.scheduler import Scheduler
        # from radar.backend.evaluation import Evaluator

        # self._environment = EnvironmentSimulator(...)
        # self._antenna = AntennaSystem(...)
        # ...

        self._logger.info("子系统初始化完成")

    async def _initialize_network(self) -> None:
        """初始化网络服务"""
        self._logger.info("初始化网络服务...")

        # TODO: 创建网络服务器
        # from radar.backend.network import WebSocketServer, HTTPServer

        # self._websocket_server = WebSocketServer(...)
        # self._http_server = HTTPServer(...)

        # await self._websocket_server.start()
        # await self._http_server.start()

        self._logger.info("网络服务初始化完成")

    def _update_config(self, config: Dict[str, Any]) -> None:
        """更新配置"""
        for key, value in config.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                self._logger.debug(f"配置更新: {key}={value}")

    async def run(self) -> None:
        """
        运行仿真主循环

        这是系统的主要运行循环。
        """
        if self._running:
            self._logger.warning("系统已在运行中")
            return

        self._running = True
        self._paused = False

        await self._state_manager.set_state(SystemState.RUNNING, "开始仿真")
        self._logger.info("开始运行仿真主循环")

        # 启动时间管理器
        self._time_manager.start()

        try:
            while self._running:
                # 检查是否暂停
                if self._paused:
                    await asyncio.sleep(0.1)
                    continue

                # 执行一帧仿真
                await self._simulation_frame()

                # 帧同步
                await self._time_manager.wait_for_frame()

        except asyncio.CancelledError:
            self._logger.info("仿真循环被取消")
        except Exception as e:
            self._logger.error(f"仿真循环错误: {e}")
            await self._state_manager.set_state(SystemState.ERROR, str(e))

    async def _simulation_frame(self) -> None:
        """
        执行单帧仿真

        这是仿真的核心处理流程。
        """
        import time
        frame_start = time.time()

        # Step 1: 处理前端指令
        await self._process_commands()

        # Step 2: 更新时间
        self._time_manager.advance_frame()

        # Step 3: 调度决策
        schedule_result = await self._scheduler.schedule(
            self._time_manager.simulation_time
        )

        # Step 4: 更新环境
        await self._environment.update(self._time_manager.frame_interval)

        # Step 5: 执行调度任务
        for task in schedule_result.execute_queue:
            await self._execute_task(task)

        # Step 6: 信号处理
        plots = await self._signal_processor.process_current_frame()

        # Step 7: 数据处理
        tracks = await self._data_processor.process(plots)

        # Step 8: 效能评估
        await self._evaluator.evaluate(tracks)

        # Step 9: 发布数据
        await self._publish_data(plots, tracks)

        # 更新统计
        frame_time = time.time() - frame_start
        self._stats['frames_processed'] += 1
        self._stats['average_frame_time'] = (
            0.9 * self._stats['average_frame_time'] + 0.1 * frame_time
        )

    async def _process_commands(self) -> None:
        """处理前端指令"""
        try:
            while True:
                # 非阻塞获取指令
                cmd = self._command_queue.get_nowait()
                await self._handle_command(cmd)
        except asyncio.QueueEmpty:
            pass

    async def _handle_command(self, cmd: Any) -> None:
        """处理单个指令"""
        from radar.protocol.messages import (
            StartCommand, StopCommand, PauseCommand, ResumeCommand,
            SetParameterCommand, LoadScenarioCommand
        )

        if isinstance(cmd, StartCommand):
            await self.resume()
        elif isinstance(cmd, StopCommand):
            await self.shutdown()
        elif isinstance(cmd, PauseCommand):
            await self.pause()
        elif isinstance(cmd, ResumeCommand):
            await self.resume()
        elif isinstance(cmd, SetParameterCommand):
            self._update_config({cmd.param_path: cmd.value})
        elif isinstance(cmd, LoadScenarioCommand):
            await self._load_scenario(cmd.scenario_data)

    async def _execute_task(self, task: Any) -> None:
        """执行调度任务"""
        # 波束控制
        beam_info = await self._antenna.steer_beam(
            task.direction_az, task.direction_el
        )

        # 生成回波
        echo_data = await self._environment.generate_echo(beam_info)

        # 信号处理
        plots = await self._signal_processor.process(echo_data)

        return plots

    async def _publish_data(self, plots: List[Plot], tracks: List[Track]) -> None:
        """发布仿真数据"""
        from radar.protocol.messages import PlotDataMessage, TrackDataMessage, BeamDataMessage

        # 发布点迹
        if plots:
            plot_msg = PlotDataMessage(
                plots=plots,
                frame_number=self._time_manager.frame_count
            )
            try:
                self._plot_queue.put_nowait(plot_msg)
            except asyncio.QueueFull:
                self._logger.warning("点迹队列已满，丢弃数据")

        # 发布航迹
        if tracks:
            track_msg = TrackDataMessage(
                tracks=tracks,
                frame_number=self._time_manager.frame_count
            )
            try:
                self._track_queue.put_nowait(track_msg)
            except asyncio.QueueFull:
                self._logger.warning("航迹队列已满，丢弃数据")

    async def pause(self) -> None:
        """暂停仿真"""
        if not self._running:
            self._logger.warning("系统未运行，无法暂停")
            return

        self._paused = True
        self._time_manager.pause()
        await self._state_manager.set_state(SystemState.PAUSED, "用户暂停")
        self._logger.info("仿真已暂停")

    async def resume(self) -> None:
        """恢复仿真"""
        if not self._running:
            self._logger.warning("系统未运行，无法恢复")
            return

        self._paused = False
        self._time_manager.resume()
        await self._state_manager.set_state(SystemState.RUNNING, "用户恢复")
        self._logger.info("仿真已恢复")

    async def shutdown(self) -> None:
        """关闭系统"""
        self._logger.info("开始关闭系统...")

        # 设置状态
        await self._state_manager.set_state(SystemState.SHUTTING_DOWN, "系统关闭")

        # 停止运行
        self._running = False

        # 停止时间管理器
        self._time_manager.stop()

        # 关闭子系统
        # await self._environment.shutdown()
        # ...

        # 关闭网络服务
        # await self._websocket_server.stop()
        # await self._http_server.stop()

        # 清空队列
        await self._plot_queue.put(None)
        await self._track_queue.put(None)

        await self._state_manager.set_state(SystemState.IDLE, "关闭完成")

        self._logger.info("系统已关闭")

    def get_state(self) -> SystemState:
        """获取当前系统状态"""
        return self._state_manager.current_state

    def get_statistics(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            **self._stats,
            'simulation_time': self._time_manager.simulation_time_seconds,
            'frame_rate': self._time_manager.frame_rate,
            'active_tracks': self._stats['tracks_active'],
        }


__all__ = [
    "IRadarCore",
    "RadarConfig",
    "RadarCore",
]
