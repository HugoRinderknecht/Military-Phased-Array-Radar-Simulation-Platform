# simple_usage.py - 简单使用示例
"""
相控阵雷达仿真平台 - 简单使用示例

演示如何使用雷达系统的各个模块。
"""

import asyncio
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from radar.common.logger import get_logger, init_logger
from radar.common.types import (
    Position3D, Velocity3D, TargetType,
    Plot, Track, TaskType, SystemState
)
from radar.common.constants import PhysicsConstants

from radar.backend.environment.simulator import EnvironmentSimulator
from radar.backend.antenna.antenna_system import PhasedArrayAntenna, ArrayGeometry
from radar.backend.signal.signal_processor import SignalProcessor
from radar.backend.dataproc.data_processor import DataProcessor
from radar.backend.scheduler.scheduler import Scheduler, PriorityCalculator
from radar.backend.evaluation.evaluator import Evaluator


async def example_basic_usage():
    """示例1：基本使用"""
    print("\n=== 示例1：基本使用 ===\n")

    # 1. 创建环境模拟器
    env = EnvironmentSimulator({
        'max_range': 100000,  # 100km
        'clutter_enabled': True
    })
    await env.initialize()

    # 2. 添加目标
    await env.add_target(
        target_type=TargetType.AIRCRAFT,
        position=Position3D(50000, 30000, 10000),  # 50km处
        velocity=Velocity3D(200, 150, 0),      # 200m/s
        rcs=10.0                                  # 10m²
    )

    print(f"✓ 目标已添加")
    print(f"✓ 目标总数: {len(env.get_all_targets())}")

    # 3. 更新环境
    await env.update(0.1)  # 推进100ms
    print(f"✓ 环境已更新")

    targets = env.get_active_targets()
    for target in targets:
        print(f"  目标 {target.id}: "
              f"pos=({target.position.x:.0f}, {target.position.y:.0f}, {target.position.z:.0f}), "
              f"vel=({target.velocity.vx:.1f}, {target.velocity.vy:.1f}, {target.velocity.vz:.1f})")


async def example_antenna_system():
    """示例2：天线系统"""
    print("\n=== 示例2：天线系统 ===\n")

    # 创建相控阵天线
    antenna = PhasedArrayAntenna(
        geometry=ArrayGeometry(
            shape='planar',
            size=(16, 16)  # 16x16阵列
        )
    )

    print(f"✓ 天线系统创建: {antenna.geometry.shape}阵列")
    print(f"✓ 阵元数量: {antenna.geometry.size[0] * antenna.geometry.size[1]}")

    # 波束控制
    beam_params = antenna.steer_beam(
        az=0.0,          # 正北
        el=np.deg2rad(15)  # 15度仰角
    )

    print(f"✓ 波束指向: az={np.rad2deg(beam_params.azimuth):.1f}°, "
          f"el={np.rad2deg(beam_params.elevation):.1f}°")
    print(f"✓ 波束增益: {20*np.log10(beam_params.gain):.1f} dB")

    # 计算方向图
    pattern = antenna.compute_beam_pattern(
        az_range=(-np.pi/2, np.pi/2),
        el_range=(-np.pi/4, np.pi/4),
        n_az=91,
        n_el=45
    )

    print(f"✓ 方向图计算完成: {pattern.shape}")
    max_idx = np.unravel_index(np.argmax(pattern))
    print(f"  最大增益方向: [{np.rad2deg(np.unravel_index(pattern[max_idx])/180-90:.1f}°, "
          f"{np.rad2deg(max_idx % 91 * 1 - 45)}]")


async def example_signal_processing():
    """示例3：信号处理"""
    print("\n=== 示例3：信号处理 ===\n")

    # 创建信号处理器
    processor = SignalProcessor()

    # 生成测试回波
    from radar.backend.environment.simulator import BeamInfo, EchoData
    import numpy as np

    beam_info = BeamInfo(
        azimuth=0.0,
        elevation=0.0
    )

    # 生成简单回波（单频信号）
    n_samples = 1000
    echo_signal = np.ones(n_samples, dtype=np.complex128)

    echo_data = EchoData(
        beam_info=beam_info,
        echo_signal=echo_signal,
        combined_signal=echo_signal
    )

    # 处理回波
    plots = processor.process_pulse(
        received_signal=echo_data.combined_signal,
        beam_az=beam_info.azimuth,
        beam_el=beam_info.elevation
    )

    print(f"✓ 检测到 {len(plots)}个点迹")
    for plot in plots[:5]:  # 显示前5个
        print(f"  点迹 {plot.id}: "
              f"R={plot.range_val:.1f}m, "
              f"SNR={plot.snr:.1f}dB")


async def example_data_processing():
    """示例4：数据处理"""
    print("\n=== 示例4：数据处理 ===\n")

    # 创建数据处理器
    processor = DataProcessor()

    # 创建测试点迹
    plots = [
        Plot(
            id=i,
            timestamp=i * 1000,
            position=Position3D(
                10000 + i * 100,
                20000 + i * 50,
                10000
            ),
            velocity=Velocity3D(100, 50, 0),
            snr=20.0 - i
        )
        for i in range(1, 11)
    ]

    print(f"✓ 输入 {len(plots)} 个点迹")

    # 处理点迹
    tracks = processor.process(plots)

    print(f"✓ 输出 {len(tracks)} 条航迹")
    for track in tracks[:5]:
        print(f"  航迹 {track.id}: "
              f"state={track.state.value}, "
              f"pos=({track.position.x:.0f}, {track.position.y:.0f}, {track.position.z:.0f})")


async def example_scheduler():
    """示例5：资源调度"""
    print("\n=== 示例5：资源调度 ===\n")

    # 创建调度器
    scheduler = Scheduler(schedule_period=50000)  # 50ms

    # 添加任务
    task_id = await scheduler.add_task(
        task_type=TaskType.SEARCH,
        priority=PriorityCalculator.static_priority(TaskType.SEARCH),
        dwell_time=10000,  # 10ms
        direction_az=0.0,
        direction_el=np.deg2rad(10),
        deadline=int(1e6)  # 1秒后
    )

    print(f"✓ 添加任务: ID={task_id}")

    # 执行调度
    result = await scheduler.schedule(0)

    print(f"✓ 调度结果:")
    print(f"  执行队列: {len(result.execute_queue)} 个任务")
    print(f"  时间利用率: {result.utilization*100:.1f}%")
    print(f"  剩余时间: {result.time_available/1000:.1f}ms")


async def example_full_system():
    """示例6：完整系统"""
    print("\n=== 示例6：完整系统仿真 ===\n")

    # 初始化日志
    init_logger(log_path="./logs", log_level="INFO")

    logger = get_logger("example")

    logger.info("初始化完整系统...")

    # 创建组件
    env = EnvironmentSimulator()
    await env.initialize()

    antenna = PhasedArrayAntenna(
        geometry=ArrayGeometry(shape='planar', size=(32, 32))
    )

    signal_proc = SignalProcessor()

    data_proc = DataProcessor()

    scheduler = Scheduler()

    evaluator = Evaluator()

    # 添加目标
    await env.add_target(
        target_type=TargetType.AIRCRAFT,
        position=Position3D(100000, 0, 0),  # 正北100km
        velocity=Velocity3D(250, 0, 0),   # 向东250m/s
        rcs=10.0
    )

    logger.info("开始仿真循环...")

    # 模拟10帧
    for frame in range(10):
        logger.info(f"--- 帧 {frame + 1} ---")

        # 更新环境
        await env.update(0.05)

        # 获取波束指向
        beam_params = antenna.steer_beam(0.0, 0.0)

        # 生成回波并处理
        from radar.backend.environment.simulator import BeamInfo, EchoData
        beam_info = BeamInfo(
            azimuth=beam_params.azimuth,
            elevation=beam_params.elevation
        )

        # 简化：直接创建点迹
        import numpy as np
        plots = [Plot(
            id=1,
            timestamp=frame * 50000,
            position=Position3D(100000, 0, 100000),
            velocity=Velocity3D(250, 0, 0),
            snr=25.0
        )]

        # 数据处理
        tracks = data_proc.process(plots)

        # 调度
        schedule_result = await scheduler.schedule(frame * 50000)

        # 评估
        evaluation_result = await evaluator.evaluate(tracks, {})

        logger.info(f"点迹: {len(plots)}, "
                    f"航迹: {len(tracks)}, "
                    f"利用率: {schedule_result.utilization*100:.1f}%")

    logger.success("仿真循环完成")


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("相控阵雷达仿真平台 - 使用示例")
    print("="*60 + "\n")

    # 运行所有示例
    await example_basic_usage()
    await example_antenna_system()
    await example_signal_processing()
    await example_data_processing()
    await example_scheduler()
    await example_full_system()

    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
