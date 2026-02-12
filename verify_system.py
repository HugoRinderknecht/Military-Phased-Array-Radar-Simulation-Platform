#!/usr/bin/env python3
# verify_system.py - 系统验证脚本
"""
相控阵雷达仿真平台 - 系统验证脚本

验证所有模块是否正确导入和初始化。
"""

import sys
from pathlib import Path


def test_imports():
    """测试所有模块导入"""
    print("="*70)
    print("相控阵雷达仿真平台 - 系统验证")
    print("="*70)
    print()

    success_count = 0
    fail_count = 0

    # 测试公共模块
    print("测试公共模块...")
    try:
        from radar.common import types, constants, config
        from radar.common.utils import math_utils, coord_transform, signal_utils
        from radar.common.containers import ring_buffer, object_pool
        from radar.common.logger import get_logger
        print("  ✓ 公共模块导入成功")
        success_count += 6
    except Exception as e:
        print(f"  ✗ 公共模块导入失败: {e}")
        fail_count += 1

    # 测试协议模块
    print("测试通信协议...")
    try:
        from radar.protocol import messages, serializer
        print("  ✓ 通信协议导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 通信协议导入失败: {e}")
        fail_count += 1

    # 测试核心模块
    print("测试核心模块...")
    try:
        from radar.backend.core import time_manager, state_manager
        from radar.backend.core.radar_core import RadarCore, RadarConfig
        print("  ✓ 核心模块导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 核心模块导入失败: {e}")
        fail_count += 1

    # 测试环境模拟
    print("测试环境模拟...")
    try:
        from radar.backend.environment import simulator
        from radar.backend.environment.target import Target
        print("  ✓ 环境模拟导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 环境模拟导入失败: {e}")
        fail_count += 1

    # 测试天线模块
    print("测试天线模块...")
    try:
        from radar.backend.antenna import antenna_system
        print("  ✓ 天线模块导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 天线模块导入失败: {e}")
        fail_count += 1

    # 测试信号处理
    print("测试信号处理...")
    try:
        from radar.backend.signal import signal_processor
        print("  ✓ 信号处理导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 信号处理导入失败: {e}")
        fail_count += 1

    # 测试数据处理
    print("测试数据处理...")
    try:
        from radar.backend.dataproc import data_processor
        print("  ✓ 数据处理导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 数据处理导入失败: {e}")
        fail_count += 1

    # 测试调度器
    print("测试资源调度...")
    try:
        from radar.backend.scheduler import scheduler
        print("  ✓ 资源调度导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 资源调度导入失败: {e}")
        fail_count += 1

    # 测试网络通信
    print("测试网络通信...")
    try:
        from radar.backend.network import network_manager
        print("  ✓ 网络通信导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 网络通信导入失败: {e}")
        fail_count += 1

    # 测试效能评估
    print("测试效能评估...")
    try:
        from radar.backend.evaluation import evaluator
        print("  ✓ 效能评估导入成功")
        success_count += 1
    except Exception as e:
        print(f"  ✗ 效能评估导入失败: {e}")
        fail_count += 1

    print()
    print("="*70)
    print(f"验证完成！")
    print(f"  成功: {success_count} 个模块")
    print(f"  失败: {fail_count} 个模块")
    print(f"  成功率: {success_count/(success_count+fail_count)*100:.1f}%")
    print("="*70)

    return fail_count == 0


def test_file_structure():
    """测试文件结构"""
    print()
    print("检查文件结构...")

    required_files = [
        "radar/__init__.py",
        "radar/main.py",
        "radar/common/__init__.py",
        "radar/common/types.py",
        "radar/common/constants.py",
        "radar/common/logger.py",
        "radar/common/config.py",
        "radar/common/utils/__init__.py",
        "radar/common/utils/math_utils.py",
        "radar/common/utils/coord_transform.py",
        "radar/common/utils/signal_utils.py",
        "radar/common/containers/__init__.py",
        "radar/protocol/__init__.py",
        "radar/protocol/messages.py",
        "radar/protocol/serializer.py",
        "radar/backend/__init__.py",
        "radar/backend/core/__init__.py",
        "radar/backend/core/time_manager.py",
        "radar/backend/core/state_manager.py",
        "radar/backend/core/radar_core.py",
        "radar/backend/environment/__init__.py",
        "radar/backend/environment/simulator.py",
        "radar/backend/environment/target/target.py",
        "radar/backend/antenna/__init__.py",
        "radar/backend/antenna/antenna_system.py",
        "radar/backend/signal/__init__.py",
        "radar/backend/signal/signal_processor.py",
        "radar/backend/dataproc/__init__.py",
        "radar/backend/dataproc/data_processor.py",
        "radar/backend/scheduler/__init__.py",
        "radar/backend/scheduler/scheduler.py",
        "radar/backend/network/__init__.py",
        "radar/backend/network/network_manager.py",
        "radar/backend/evaluation/__init__.py",
        "radar/backend/evaluation/evaluator.py",
        "radar_config.toml",
        "requirements.txt",
        "README.md",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"  ✗ 缺失 {len(missing_files)} 个文件:")
        for f in missing_files[:5]:
            print(f"    - {f}")
        if len(missing_files) > 5:
            print(f"    ... 还有 {len(missing_files)-5} 个文件")
        return False
    else:
        print(f"  ✓ 所需文件都存在 ({len(required_files)} 个)")
        return True


def show_quick_start():
    """显示快速开始指南"""
    print()
    print("="*70)
    print("快速开始指南")
    print("="*70)
    print()
    print("1. 安装依赖:")
    print("   pip install -r requirements.txt")
    print()
    print("2. 配置系统:")
    print("   编辑 radar_config.toml")
    print()
    print("3. 启动后端:")
    print("   python -m radar.main")
    print()
    print("4. 访问API:")
    print("   浏览器打开 http://localhost:8000/docs")
    print()
    print("="*70)


if __name__ == "__main__":
    # 执行导入测试
    imports_ok = test_imports()

    # 执行文件结构测试
    files_ok = test_file_structure()

    # 显示快速开始
    show_quick_start()

    # 返回状态
    if imports_ok and files_ok:
        print("\n🎉 系统验证通过！可以开始使用。")
        sys.exit(0)
    else:
        print("\n⚠️  系统验证失败！请检查错误。")
        sys.exit(1)
