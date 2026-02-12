# pytest配置文件

import pytest
import asyncio
import sys
sys.path.insert(0, '.')


@pytest.fixture(scope='session')
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def sample_config():
    """示例配置"""
    from radar.common.config import SystemConfig
    return SystemConfig()


@pytest.fixture
async def radar_core(sample_config):
    """雷达核心实例"""
    from radar.backend.core.radar_core import RadarCore
    core = RadarCore()
    await core.initialize(sample_config)
    yield core
    await core.shutdown()
