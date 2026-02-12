# 核心模块测试

import pytest
import asyncio
import sys
sys.path.insert(0, '.')


@pytest.mark.asyncio
async def test_radar_core_initialization(radar_core):
    """测试雷达核心初始化"""
    assert radar_core is not None
    assert not radar_core.is_running


@pytest.mark.asyncio
async def test_time_manager():
    """测试时间管理器"""
    from radar.backend.core.time_manager import TimeManager
    
    tm = TimeManager()
    assert tm.get_time() == 0.0
    
    tm.update(0.1)
    assert abs(tm.get_time() - 0.1) < 0.001
    assert tm.get_frame_count() == 1


@pytest.mark.asyncio
async def test_state_manager():
    """测试状态管理器"""
    from radar.backend.core.state_manager import StateManager, SimulationState
    
    sm = StateManager()
    state = sm.get_state()
    
    assert state['simulation_state'] == 'stopped'
    assert state['simulation_time'] == 0.0
    
    await sm.set_simulation_state(SimulationState.RUNNING)
    state = sm.get_state()
    assert state['simulation_state'] == 'running'


@pytest.mark.asyncio
async def test_simulation_engine():
    """测试仿真引擎"""
    from radar.backend.core.simulation_engine import SimulationEngine
    
    engine = SimulationEngine()
    assert not engine._running
    
    await engine.start()
    assert engine.is_running
    
    await asyncio.sleep(0.2)
    
    await engine.stop()
    assert not engine.is_running


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
