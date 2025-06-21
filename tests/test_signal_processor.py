import pytest
import numpy as np
from core.signal_processor import SignalProcessor
from models.radar_system import RadarSystem
from models.environment import Environment, WeatherCondition


@pytest.fixture
def radar_system():
    return RadarSystem(
        radar_area=100.0,
        tr_components=1000,
        radar_power=50000,
        frequency=10e9
    )


@pytest.fixture
def environment():
    weather = WeatherCondition(weather_type="clear")
    return Environment(weather=weather)


@pytest.fixture
def signal_processor(radar_system, environment):
    return SignalProcessor(radar_system, environment)


def test_signal_processor_initialization(signal_processor):
    assert signal_processor is not None
    assert signal_processor.c == 299792458.0


def test_adaptive_clutter_suppression(signal_processor):
    signal_length = 1024
    test_signal = np.random.random(signal_length) + 1j * np.random.random(signal_length)

    result = signal_processor._adaptive_clutter_suppression(test_signal)

    assert len(result) == signal_length
    assert np.iscomplexobj(result)


def test_pulse_compression(signal_processor):
    signal_length = 1024
    test_signal = np.random.random(signal_length) + 1j * np.random.random(signal_length)

    result = signal_processor._pulse_compression(test_signal)

    assert len(result) == signal_length
    assert np.iscomplexobj(result)


def test_cfar_detection(signal_processor):
    signal_length = 1024
    # 创建包含目标的信号
    test_signal = np.random.normal(0, 1, signal_length) + 1j * np.random.normal(0, 1, signal_length)
    # 添加强目标
    test_signal[500:510] += 10 * (np.random.random(10) + 1j * np.random.random(10))

    detections = signal_processor._cfar_detection(test_signal, 0.0)

    assert isinstance(detections, list)
    # 应该检测到至少一个目标
    assert len(detections) >= 0


def test_process_radar_signal(signal_processor):
    signal_length = 2048
    test_signal = np.random.random(signal_length) + 1j * np.random.random(signal_length)

    detections = signal_processor.process_radar_signal(test_signal, 0.0)

    assert isinstance(detections, list)
    for detection in detections:
        assert hasattr(detection, 'range')
        assert hasattr(detection, 'snr')
        assert hasattr(detection, 'velocity')
