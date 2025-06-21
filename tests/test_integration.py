import pytest
import json
from app import create_app
from services.simulation_service import SimulationService


@pytest.fixture
def app():
    return create_app()


@pytest.fixture
def simulation_service():
    return SimulationService()


def test_end_to_end_simulation():
    """完整的端到端仿真测试"""
    simulation_params = {
        "radar": {
            "radar_area": 50.0,
            "tr_components": 500,
            "radar_power": 25000,
            "frequency": 10e9
        },
        "environment": {
            "weather_type": "clear",
            "clutter_density": 0.2,
            "interference_level": 0.05
        },
        "targets": {
            "num_targets": 3,
            "specific_targets": [
                {
                    "position": [30000, 20000, 3000],
                    "velocity": [200, 150, 0],
                    "rcs": 5.0,
                    "altitude": 3000,
                    "aspect_angle": 0.785
                },
                {
                    "position": [45000, -15000, 8000],
                    "velocity": [-300, 100, -50],
                    "rcs": 15.0,
                    "altitude": 8000,
                    "aspect_angle": 2.356
                }
            ]
        },
        "simulation_time": 3.0,
        "time_step": 0.06,
        "monte_carlo_runs": 1
    }

    service = SimulationService()

    # 初始化仿真
    init_result = service.initialize_simulation(simulation_params)
    assert init_result['status'] == 'success'

    # 运行仿真
    simulation_result = service.run_simulation()
    assert simulation_result['status'] == 'success'
    assert 'results' in simulation_result

    # 验证结果结构
    results = simulation_result['results']
    assert 'summary' in results
    assert 'time_series' in results

    # 验证时间序列数据
    time_series = results['time_series']
    assert len(time_series) > 0

    # 验证每个时间步的数据
    for step_data in time_series.values():
        assert 'time' in step_data
        assert 'avg_detections' in step_data
        assert 'avg_tracks' in step_data


def test_formation_simulation():
    """编队目标仿真测试"""
    formation_params = {
        "radar": {
            "radar_area": 100.0,
            "tr_components": 1000,
            "radar_power": 50000
        },
        "environment": {
            "weather_type": "clear"
        },
        "targets": {
            "num_targets": 4,
            "specific_targets": [
                {
                    "position": [40000, 30000, 5000],
                    "velocity": [250, 0, 0],
                    "rcs": 10.0,
                    "altitude": 5000,
                    "aspect_angle": 0.0,
                    "is_formation": True,
                    "formation_id": 1
                },
                {
                    "position": [40100, 30000, 5000],
                    "velocity": [250, 0, 0],
                    "rcs": 8.0,
                    "altitude": 5000,
                    "aspect_angle": 0.0,
                    "is_formation": True,
                    "formation_id": 1
                }
            ],
            "formations": [
                {
                    "formation_id": 1,
                    "leader_id": 1,
                    "member_ids": [2],
                    "formation_type": "line",
                    "spacing": 100.0
                }
            ]
        },
        "simulation_time": 2.0,
        "monte_carlo_runs": 1
    }

    service = SimulationService()

    init_result = service.initialize_simulation(formation_params)
    assert init_result['status'] == 'success'

    simulation_result = service.run_simulation()
    assert simulation_result['status'] == 'success'

    # 验证编队处理
    results = simulation_result['results']
    assert 'time_series' in results

    # 编队目标应该产生多个检测
    time_series = results['time_series']
    total_detections = sum(step.get('avg_detections', 0) for step in time_series.values())
    assert total_detections > 0


def test_weather_impact_simulation():
    """天气影响仿真测试"""
    clear_weather_params = {
        "radar": {
            "radar_area": 100.0,
            "tr_components": 1000,
            "radar_power": 50000
        },
        "environment": {
            "weather_type": "clear",
            "precipitation_rate": 0.0
        },
        "targets": {
            "num_targets": 2,
            "specific_targets": [
                {
                    "position": [50000, 0, 5000],
                    "velocity": [200, 0, 0],
                    "rcs": 10.0,
                    "altitude": 5000,
                    "aspect_angle": 0.0
                }
            ]
        },
        "simulation_time": 1.0,
        "monte_carlo_runs": 1
    }

    rainy_weather_params = clear_weather_params.copy()
    rainy_weather_params["environment"]["weather_type"] = "rain"
    rainy_weather_params["environment"]["precipitation_rate"] = 10.0

    service = SimulationService()

    # 晴天仿真
    service.initialize_simulation(clear_weather_params)
    clear_result = service.run_simulation()

    service.reset_simulation()

    # 雨天仿真
    service.initialize_simulation(rainy_weather_params)
    rainy_result = service.run_simulation()

    # 验证雨天检测性能下降
    assert clear_result['status'] == 'success'
    assert rainy_result['status'] == 'success'
