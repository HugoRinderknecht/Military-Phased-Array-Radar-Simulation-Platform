# test_system.py - 系统集成测试
"""
相控阵雷达仿真平台 - 系统集成测试

验证系统各模块的基本功能。
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from radar.common.logger import get_logger
from radar.common.types import (
    Position3D, Velocity3D, TargetType,
    Plot, Track, TaskType, SystemState
)
from radar.common.constants import PhysicsConstants, MathConstants
from radar.common.utils import (
    deg_to_rad, rad_to_deg,
    generate_lfm_pulse, pulse_compression,
    enu_to_azel, azel_to_enu
)
from radar.common.containers import RingBuffer


class TestCommonModules:
    """测试公共模块"""

    @pytest.mark.asyncio
    async def test_types(self):
        """测试类型定义"""
        from radar.common.types import TargetType, MotionModel

        # 测试枚举
        assert TargetType.AIRCRAFT.value == "aircraft"
        assert MotionModel.CONSTANT_VELOCITY.value == "cv"

        # 测试数据类
        pos = Position3D(100.0, 200.0, 300.0)
        vel = Velocity3D(10.0, 20.0, 30.0)

        assert pos.x == 100.0
        assert vel.vx == 10.0
        assert vel.magnitude() == np.sqrt(10**2 + 20**2 + 30**2)

    def test_constants(self):
        """测试物理常数"""
        assert PhysicsConstants.C == 299792458.0
        assert MathConstants.PI == np.pi

    def test_coord_transform(self):
        """测试坐标变换"""
        # ENU到球坐标
        r, az, el = enu_to_azel(100.0, 0.0, 0.0)
        assert abs(r - 100.0) < 0.01

        # 球坐标到ENU
        x, y, z = azel_to_enu(r, az, el)
        assert abs(x - 100.0) < 0.01
        assert abs(y) < 0.01
        assert abs(z) < 0.01

    def test_math_utils(self):
        """测试数学工具"""
        # 角度转换
        assert abs(deg_to_rad(180) - np.pi) < 1e-10
        assert abs(rad_to_deg(np.pi) - 180) < 1e-10

    def test_signal_utils(self):
        """测试信号工具"""
        # 生成LFM波形
        signal = generate_lfm_pulse(
            sample_rate=20e6,
            pulse_width=10e-6,
            bandwidth=10e6
        )

        assert len(signal) == 200  # 20MHz * 10us = 200 samples
        assert np.allclose(np.abs(signal), 1.0, atol=0.1)

    def test_ring_buffer(self):
        """测试环形缓冲区"""
        buf = RingBuffer(capacity=10, dtype=np.float64)

        # 写入数据
        for i in range(15):
            buf.write(i)

        # 检查容量和内容
        assert buf.size == 10
        data = buf.to_list()
        assert data == list(range(5, 15))


class TestProtocol:
    """测试通信协议"""

    def test_message_creation(self):
        """测试消息创建"""
        from radar.protocol.messages import (
            StartCommand, PlotDataMessage, BeamStatus
        )
        from radar.common.types import SimulationState

        # 测试指令消息
        cmd = StartCommand(time_scale=1.0)
        assert cmd.type == "cmd_start"
        assert cmd.time_scale == 1.0

        # 测试数据消息
        plot = Plot(
            id=1,
            timestamp=0,
            position=Position3D(0, 0, 0),
            snr=20.0
        )
        msg = PlotDataMessage(plots=[plot])
        assert msg.type == "plot_update"
        assert len(msg.plots) == 1

    def test_message_serialization(self):
        """测试消息序列化"""
        from radar.protocol.messages import StartCommand
        from radar.protocol.serializer import serialize_message, deserialize_message

        # 序列化
        cmd = StartCommand()
        data = serialize_message(cmd, format='json')

        # 反序列化
        restored_cmd = deserialize_message(data, StartCommand, format='json')

        assert restored_cmd.type == cmd.type


class TestBackendCore:
    """测试后端核心模块"""

    @pytest.mark.asyncio
    async def test_time_manager(self):
        """测试时间管理器"""
        from radar.backend.core.time_manager import TimeManager

        tm = TimeManager(time_scale=1.0, frame_rate=60.0)

        # 测试时间推进
        tm.start()
        initial_time = tm.simulation_time
        tm.advance_frame()
        assert tm.simulation_time > initial_time
        assert tm.frame_count == 1
        tm.stop()

    @pytest.mark.asyncio
    async def test_state_manager(self):
        """测试状态管理器"""
        from radar.backend.core.state_manager import StateManager
        from radar.common.types import SystemState

        sm = StateManager()

        # 测试状态转换
        state = await sm.get_state()
        assert state == SystemState.IDLE

        # 测试状态设置
        await sm.set_state(SystemState.RUNNING, "test")
        state = await sm.get_state()
        assert state == SystemState.RUNNING


class TestEnvironment:
    """测试环境模拟模块"""

    def test_target_creation(self):
        """测试目标创建"""
        from radar.backend.environment.target import Target

        target = Target(
            target_id=1,
            target_type=TargetType.AIRCRAFT,
            initial_position=Position3D(1000, 2000, 3000),
            initial_velocity=Velocity3D(100, 200, 50)
        )

        assert target.id == 1
        assert target.type == TargetType.AIRCRAFT

    def test_motion_model_prediction(self):
        """测试运动模型预测"""
        from radar.backend.environment.target import (
            ConstantVelocityModel, TargetStateEstimate
        )

        model = ConstantVelocityModel()
        state = TargetStateEstimate(
            position=np.array([0.0, 0.0, 0.0]),
            velocity=np.array([100.0, 200.0, 50.0]),
            timestamp=0.0
        )

        # 预测下一步
        dt = 0.1
        predicted = model.predict(state, dt)

        # 验证预测
        expected = state.position + state.velocity * dt
        assert np.allclose(predicted.position, expected)


class TestAntenna:
    """测试天线模块"""

    def test_array_factor(self):
        """测试阵列因子计算"""
        from radar.backend.antenna.antenna_system import (
            PhasedArrayAntenna, ArrayGeometry
        )

        antenna = PhasedArrayAntenna(
            geometry=ArrayGeometry(shape='planar', size=(4, 4))
        )

        # 计算阵列因子
        af = antenna.compute_array_factor(0.0, 0.0)  # 正方向
        assert np.abs(af) > 0.9  # 应该接近最大值

    def test_beam_steer(self):
        """测试波束控制"""
        from radar.backend.antenna.antenna_system import (
            PhasedArrayAntenna, ArrayGeometry, BeamParameters
        )

        antenna = PhasedArrayAntenna(
            geometry=ArrayGeometry(shape='planar', size=(8, 8))
        )

        # 波束控制
        beam_params = antenna.steer_beam(
            az=np.deg2rad(45),
            el=np.deg2rad(10)
        )

        assert beam_params.azimuth == np.deg2rad(45)
        assert beam_params.elevation == np.deg2rad(10)


class TestSignalProcessing:
    """测试信号处理模块"""

    def test_lfm_generation(self):
        """测试LFM波形生成"""
        from radar.common.utils import generate_lfm_pulse

        signal = generate_lfm_pulse(
            sample_rate=20e6,
            pulse_width=10e-6,
            bandwidth=10e6
        )

        assert signal is not None
        assert len(signal) == 200

    def test_pulse_compression(self):
        """测试脉冲压缩"""
        from radar.common.utils import pulse_compression

        # 参考信号
        ref = generate_lfm_pulse(20e6, 10e-6, 10e6)

        # 脉冲压缩
        received = ref.copy()
        compressed = pulse_compression(received, ref)

        assert len(compressed) == len(ref)
        # 峰值应该接近中心
        peak_idx = np.argmax(compressed)
        assert peak_idx == len(ref) // 2


class TestDataProcessing:
    """测试数据处理模块"""

    def test_track_creation(self):
        """测试航迹创建"""
        track = Track(
            id=1,
            state=TrackState.TENTATIVE,
            position=Position3D(100.0, 200.0, 300.0),
            velocity=Velocity3D(10.0, 20.0, 30.0)
        )

        assert track.id == 1
        assert track.state == TrackState.TENTATIVE

    def test_association(self):
        """测试数据关联"""
        from radar.backend.dataproc.data_processor import DataProcessor

        processor = DataProcessor()

        # 创建测试航迹
        track = Track(
            id=1,
            state=TrackState.CONFIRMED,
            position=Position3D(0, 0, 0),
            velocity=Velocity3D(100, 0, 0)
        )

        # 创建测试点迹
        plot = Plot(
            id=1,
            timestamp=0,
            position=Position3D(10, 10, 10),
            velocity=Velocity3D(100, 0, 0)
        )

        # 处理
        tracks = processor.process([plot])

        assert len(tracks) >= 1


class TestScheduler:
    """测试调度模块"""

    def test_task_creation(self):
        """测试任务创建"""
        from radar.backend.scheduler.scheduler import ScheduleTask, TaskStatus, TaskType

        task = ScheduleTask(
            id=1,
            type=TaskType.SEARCH,
            priority=100.0,
            dwell_time=50000,  # 50ms
            deadline=int(1e9),  # 1秒
            direction_az=0.0,
            direction_el=0.0
        )

        assert task.id == 1
        assert task.status == TaskStatus.PENDING
        assert task.priority == 100.0

    def test_priority_calculation(self):
        """测试优先级计算"""
        from radar.backend.scheduler.scheduler import PriorityCalculator

        # 静态优先级
        static_prio = PriorityCalculator.static_priority(TaskType.SEARCH)
        assert static_prio == 50.0

        # EDF优先级
        task = ScheduleTask(
            id=1,
            type=TaskType.TRACK,
            priority=100.0,
            dwell_time=50000,
            deadline=int(1e9),
            direction_az=0.0,
            direction_el=0.0
        )
        edf_prio = PriorityCalculator.edf_priority(task, 0)
        assert edf_prio > 0


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
