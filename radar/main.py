# main.py - 后端主入口
"""
相控阵雷达仿真平台 - 后端主入口

本文件是整个后端系统的启动点。
"""

import asyncio
import sys
from pathlib import Path
import argparse
import toml

from radar.common.logger import get_logger, init_logger
from radar.common.types import SystemState
from radar.common.exceptions import RadarError

from radar.backend.core.radar_core import RadarCore, RadarConfig
from radar.backend.network.network_manager import NetworkManager
from radar.protocol.messages import StartCommand, StopCommand


def load_config(config_path: str) -> dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置字典
    """
    config_file = Path(config_path)

    if not config_file.exists():
        print(f"警告: 配置文件不存在: {config_path}")
        print("使用默认配置")
        return {}

    try:
        with open(config_file, 'r', encoding='utf-8') as toml.load
        print(f"已加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"错误: 加载配置文件失败: {e}")
        return {}


def create_radar_config(config_dict: dict) -> RadarConfig:
    """
    从配置字典创建雷达配置

    Args:
        config_dict: 配置字典

    Returns:
        RadarConfig对象
    """
    config = RadarConfig()

    # 基本参数
    if 'radar' in config_dict:
        radar_cfg = config_dict['radar']
        if 'frequency' in radar_cfg:
            config.frequency = radar_cfg['frequency']
        if 'bandwidth' in radar_cfg:
            config.bandwidth = radar_cfg['bandwidth']
        if 'pulse_width' in radar_cfg:
            config.pulse_width = radar_cfg['pulse_width']
        if 'prf' in radar_cfg:
            config.prf = radar_cfg['prf']

    # 天线参数
    if 'antenna' in config_dict:
        antenna_cfg = config_dict['antenna']
        if 'array_size' in antenna_cfg:
            config.array_size = tuple(antenna_cfg['array_size'])
        if 'element_spacing' in antenna_cfg:
            config.element_spacing = antenna_cfg['element_spacing']
        if 'antenna_gain' in antenna_cfg:
            config.antenna_gain = antenna_cfg['antenna_gain']

    # 发射参数
    if 'transmitter' in config_dict:
        tx_cfg = config_dict['transmitter']
        if 'power' in tx_cfg:
            config.transmit_power = tx_cfg['power']

    # 接收参数
    if 'receiver' in config_dict:
        rx_cfg = config_dict['receiver']
        if 'noise_figure' in rx_cfg:
            config.noise_figure = rx_cfg['noise_figure']
        if 'system_losses' in rx_cfg:
            config.system_losses = rx_cfg['system_losses']

    # 调度参数
    if 'scheduler' in config_dict:
        sched_cfg = config_dict['scheduler']
        if 'schedule_interval' in sched_cfg:
            config.schedule_interval = sched_cfg['schedule_interval']

    return config


async def main_coroutine(config: RadarConfig,
                        log_config: dict) -> None:
    """
    主协程

    Args:
        config: 雷达配置
        log_config: 日志配置
    """
    # 初始化日志
    log_path = log_config.get('path', './logs')
    log_level = log_config.get('level', 'INFO')
    init_logger(log_path=log_path, log_level=log_level)

    logger = get_logger("main")
    logger.info("=" * 60)
    logger.info("相控阵雷达仿真平台 - 后端系统")
    logger.info("版本: 2.0")
    logger.info("=" * 60)

    # 创建网络管理器
    network = NetworkManager(
        host=config.get('host', '0.0.0.0'),
        port=config.get('port', 8000)
    )

    # 创建雷达核心
    radar_core = RadarCore(config)

    # 设置网络的数据队列
    # TODO: 从radar_core获取队列
    # network.set_data_queues(...)

    # 设置指令处理器
    async def handle_command(command):
        logger.info(f"收到指令: {command.type}")

        if isinstance(command, StartCommand):
            await radar_core.run()
        elif isinstance(command, StopCommand):
            await radar_core.shutdown()
        # ... 其他指令处理

    network.set_command_handler(handle_command)

    try:
        # 初始化雷达核心
        logger.info("初始化雷达系统...")
        init_success = await radar_core.initialize()

        if not init_success:
            logger.error("雷达系统初始化失败")
            return

        logger.success("雷达系统初始化成功")

        # 启动网络服务
        logger.info("启动网络服务...")
        # 网络服务在network.start()中启动

        # 创建数据发布任务
        async def data_publisher():
            """数据发布任务"""
            while radar_core.get_state() == SystemState.RUNNING:
                # TODO: 从队列获取数据并发布
                # plots = await plot_queue.get()
                # await network.publish_plots(plots)
                await asyncio.sleep(0.05)  # 20Hz

        # 启动数据发布
        publisher_task = asyncio.create_task(data_publisher())

        logger.info("系统启动完成")
        logger.info(f"监听地址: http://{config.get('host', '0.0.0.0')}:{config.get('port', 8000)}")
        logger.info("按 Ctrl+C 停止系统")

        # 等待取消信号
        try:
            # 启动网络服务器（这会阻塞）
            await network.start()
        except asyncio.CancelledError:
            pass

        # 清理
        logger.info("正在关闭系统...")
        publisher_task.cancel()

        # 关闭雷达核心
        await radar_core.shutdown()

        # 关闭网络服务
        await network.stop()

        logger.success("系统已安全关闭")

    except RadarError as e:
        logger.error(f"雷达错误: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"未预期的错误: {e}")
        sys.exit(1)


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='相控阵雷达仿真平台 - 后端系统',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '-c', '--config',
        type=str,
        default='radar_config.toml',
        help='配置文件路径 (默认: radar_config.toml)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='日志级别 (默认: INFO)'
    )

    parser.add_argument(
        '--log-path',
        type=str,
        default='./logs',
        help='日志文件路径 (默认: ./logs)'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='服务器监听地址 (默认: 0.0.0.0)'
    )

    parser.add_argument(
        '-p', '--port',
        type=int,
        default=8000,
        help='服务器监听端口 (默认: 8000)'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0'
    )

    args = parser.parse_args()

    # 加载配置
    config_dict = load_config(args.config)

    # 创建雷达配置
    radar_config = create_radar_config(config_dict)

    # 日志配置
    log_config = {
        'path': args.log_path,
        'level': args.log_level
    }

    # 覆盖host和port
    radar_config.__dict__['host'] = args.host
    radar_config.__dict__['port'] = args.port

    # 运行主协程
    try:
        asyncio.run(main_coroutine(radar_config, log_config))
    except KeyboardInterrupt:
        print("\n收到中断信号，正在退出...")
        sys.exit(0)


if __name__ == '__main__':
    main()
