import pytest
import numpy as np
import time
from unittest.mock import Mock
from core.signal_processor import SignalProcessor
import logging

# 设置中文日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)


# 创建模拟的雷达系统和环境类
class MockRadarSystem:
    def __init__(self, radar_area=100.0, tr_components=1000, radar_power=50000, frequency=10e9):
        self.radar_area = radar_area
        self.tr_components = tr_components
        self.radar_power = radar_power
        self.frequency = frequency


class MockWeatherCondition:
    def __init__(self, weather_type="clear"):
        self.weather_type = weather_type


class MockEnvironment:
    def __init__(self, weather=None):
        self.weather = weather or MockWeatherCondition()


@pytest.fixture
def radar_system():
    """雷达系统测试夹具"""
    return MockRadarSystem(
        radar_area=100.0,
        tr_components=1000,
        radar_power=50000,
        frequency=10e9
    )


@pytest.fixture
def environment():
    """环境测试夹具"""
    weather = MockWeatherCondition(weather_type="clear")
    return MockEnvironment(weather=weather)


@pytest.fixture
def signal_processor(radar_system, environment):
    """信号处理器测试夹具"""
    return SignalProcessor(radar_system, environment)


@pytest.fixture
def performance_logger():
    """性能日志记录器"""
    logger = logging.getLogger("性能测试")
    return logger


def test_signal_processor_initialization(signal_processor):
    """测试信号处理器初始化"""
    print("\n🔧 正在测试信号处理器初始化...")
    assert signal_processor is not None, "信号处理器初始化失败"
    assert signal_processor.c == 299792458.0, "光速常数设置错误"
    assert signal_processor.fft_size == 1024, "FFT大小设置错误"
    print("✅ 信号处理器初始化测试通过")


def test_adaptive_clutter_suppression(signal_processor, performance_logger):
    """测试自适应杂波抑制功能和性能"""
    print("\n🎯 正在测试自适应杂波抑制...")

    # 测试不同信号长度的性能
    test_sizes = [512, 1024, 2048, 4096]
    performance_results = {}

    for size in test_sizes:
        test_signal = np.random.random(size) + 1j * np.random.random(size)

        # 性能测试
        start_time = time.perf_counter()
        result = signal_processor._adaptive_clutter_suppression(test_signal)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000  # 转换为毫秒
        performance_results[size] = processing_time

        # 功能验证
        assert len(result) == size, f"输出信号长度不匹配，期望 {size}，实际 {len(result)}"
        assert np.iscomplexobj(result), "输出信号应为复数类型"
        assert not np.any(np.isnan(result)), "输出信号包含NaN值"

        performance_logger.info(f"杂波抑制处理 {size} 个样本耗时: {processing_time:.2f} 毫秒")

    # 性能基准检查
    for size, time_ms in performance_results.items():
        expected_max_time = size * 0.01  # 每个样本最多0.01毫秒
        assert time_ms < expected_max_time, f"处理 {size} 样本耗时过长: {time_ms:.2f}ms > {expected_max_time:.2f}ms"

    print("✅ 自适应杂波抑制测试通过")
    print(f"📊 性能结果: {performance_results}")


def test_pulse_compression(signal_processor, performance_logger):
    """测试脉冲压缩功能和性能"""
    print("\n🔄 正在测试脉冲压缩...")

    test_sizes = [1024, 2048, 4096]
    performance_results = {}

    for size in test_sizes:
        test_signal = np.random.random(size) + 1j * np.random.random(size)

        # 性能测试
        start_time = time.perf_counter()
        result = signal_processor._pulse_compression(test_signal)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000
        performance_results[size] = processing_time

        # 功能验证
        assert len(result) == size, f"脉冲压缩输出长度错误"
        assert np.iscomplexobj(result), "脉冲压缩输出应为复数类型"
        assert not np.any(np.isnan(result)), "脉冲压缩输出包含NaN值"

        # 检查信号增益
        input_power = np.mean(np.abs(test_signal) ** 2)
        output_power = np.mean(np.abs(result) ** 2)
        gain_db = 10 * np.log10(output_power / input_power) if input_power > 0 else 0

        performance_logger.info(f"脉冲压缩处理 {size} 样本耗时: {processing_time:.2f} 毫秒, 增益: {gain_db:.1f} dB")

    print("✅ 脉冲压缩测试通过")
    print(f"📊 性能结果: {performance_results}")


def test_cfar_detection(signal_processor, performance_logger):
    """测试CFAR检测功能和性能"""
    print("\n🎯 正在测试CFAR检测...")

    test_sizes = [1024, 2048, 4096]
    detection_results = {}

    for size in test_sizes:
        # 创建包含已知目标的测试信号
        test_signal = np.random.normal(0, 1, size) + 1j * np.random.normal(0, 1, size)

        # 在信号中插入强目标
        target_positions = [size // 4, size // 2, 3 * size // 4]
        target_strength = 10.0

        for pos in target_positions:
            if pos + 10 < size:
                test_signal[pos:pos + 10] += target_strength * (
                        np.random.random(10) + 1j * np.random.random(10)
                )

        # 性能测试
        start_time = time.perf_counter()
        detections = signal_processor._cfar_detection(test_signal, 0.0)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000
        detection_results[size] = {
            'time_ms': processing_time,
            'detections': len(detections)
        }

        # 功能验证
        assert isinstance(detections, list), "CFAR检测结果应为列表类型"

        # 验证检测结果的属性
        for detection in detections:
            assert hasattr(detection, 'range'), "检测结果缺少距离属性"
            assert hasattr(detection, 'snr'), "检测结果缺少信噪比属性"
            assert hasattr(detection, 'velocity'), "检测结果缺少速度属性"
            assert detection.range >= 0, "检测距离不能为负数"
            assert not np.isnan(detection.snr), "信噪比不能为NaN"

        performance_logger.info(
            f"CFAR检测处理 {size} 样本耗时: {processing_time:.2f} 毫秒, "
            f"检测到 {len(detections)} 个目标"
        )

    print("✅ CFAR检测测试通过")
    print(f"📊 检测结果: {detection_results}")


def test_process_radar_signal(signal_processor, performance_logger):
    """测试完整雷达信号处理链路和性能"""
    print("\n🔍 正在测试完整雷达信号处理...")

    test_sizes = [2048, 4096, 8192]
    processing_results = {}

    for size in test_sizes:
        # 创建复杂的测试信号
        test_signal = np.random.random(size) + 1j * np.random.random(size)

        # 添加多个目标
        num_targets = 3
        for i in range(num_targets):
            target_pos = (i + 1) * size // (num_targets + 1)
            target_width = 20
            target_amplitude = 5.0 + i * 2.0

            if target_pos + target_width < size:
                test_signal[target_pos:target_pos + target_width] += target_amplitude * (
                        np.random.random(target_width) + 1j * np.random.random(target_width)
                )

        # 性能测试
        start_time = time.perf_counter()
        detections = signal_processor.process_radar_signal(test_signal, 0.0)
        end_time = time.perf_counter()

        processing_time = (end_time - start_time) * 1000
        processing_results[size] = {
            'time_ms': processing_time,
            'detections': len(detections),
            'throughput_samples_per_sec': size / (processing_time / 1000) if processing_time > 0 else 0
        }

        # 功能验证
        assert isinstance(detections, list), "处理结果应为检测列表"

        for detection in detections:
            assert hasattr(detection, 'range'), "检测结果必须包含距离信息"
            assert hasattr(detection, 'snr'), "检测结果必须包含信噪比信息"
            assert hasattr(detection, 'velocity'), "检测结果必须包含速度信息"

            # 数值合理性检查
            assert 0 <= detection.range <= 1e6, f"距离值异常: {detection.range}"
            assert -50 <= detection.snr <= 100, f"信噪比值异常: {detection.snr}"
            assert -1000 <= detection.velocity <= 1000, f"速度值异常: {detection.velocity}"

        throughput = processing_results[size]['throughput_samples_per_sec']
        performance_logger.info(
            f"完整处理 {size} 样本耗时: {processing_time:.2f} 毫秒, "
            f"检测到 {len(detections)} 个目标, "
            f"吞吐量: {throughput:.0f} 样本/秒"
        )

    print("✅ 完整雷达信号处理测试通过")
    print(f"📊 处理结果: {processing_results}")


def test_performance_benchmark(signal_processor, performance_logger):
    """性能基准测试"""
    print("\n⚡ 正在进行性能基准测试...")

    # 大规模信号处理测试
    large_signal_size = 16384
    test_signal = np.random.random(large_signal_size) + 1j * np.random.random(large_signal_size)

    # 多次运行取平均值
    num_runs = 5
    total_times = []

    for run in range(num_runs):
        start_time = time.perf_counter()
        detections = signal_processor.process_radar_signal(test_signal, 0.0)
        end_time = time.perf_counter()

        run_time = (end_time - start_time) * 1000
        total_times.append(run_time)

        performance_logger.info(f"第 {run + 1} 次运行耗时: {run_time:.2f} 毫秒")

    # 统计分析
    avg_time = np.mean(total_times)
    std_time = np.std(total_times)
    min_time = np.min(total_times)
    max_time = np.max(total_times)

    # 性能要求检查
    max_acceptable_time = 1000.0  # 1秒内处理16K样本
    assert avg_time < max_acceptable_time, f"平均处理时间过长: {avg_time:.2f}ms > {max_acceptable_time}ms"

    # 计算实时处理能力
    sample_rate = large_signal_size / (avg_time / 1000)  # 样本/秒
    real_time_factor = sample_rate / 1e6  # 相对于1MHz采样率的实时倍数

    performance_summary = {
        '平均处理时间': f"{avg_time:.2f} 毫秒",
        '标准偏差': f"{std_time:.2f} 毫秒",
        '最短时间': f"{min_time:.2f} 毫秒",
        '最长时间': f"{max_time:.2f} 毫秒",
        '处理吞吐量': f"{sample_rate:.0f} 样本/秒",
        '实时处理倍数': f"{real_time_factor:.1f}x"
    }

    print("✅ 性能基准测试通过")
    print("📊 性能摘要:")
    for key, value in performance_summary.items():
        print(f"   {key}: {value}")


def test_memory_usage(signal_processor):
    """内存使用测试"""
    print("\n💾 正在测试内存使用...")

    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # 处理大量信号
    large_signals = []
    for i in range(10):
        signal_size = 8192
        test_signal = np.random.random(signal_size) + 1j * np.random.random(signal_size)
        detections = signal_processor.process_radar_signal(test_signal, float(i))
        large_signals.append((test_signal, detections))

    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = peak_memory - initial_memory

    # 清理
    del large_signals

    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_recovered = peak_memory - final_memory

    print(f"📊 内存使用情况:")
    print(f"   初始内存: {initial_memory:.1f} MB")
    print(f"   峰值内存: {peak_memory:.1f} MB")
    print(f"   内存增长: {memory_increase:.1f} MB")
    print(f"   回收内存: {memory_recovered:.1f} MB")
    print(f"   最终内存: {final_memory:.1f} MB")

    # 内存泄漏检查
    assert memory_increase < 500, f"内存使用过多: {memory_increase:.1f} MB"

    print("✅ 内存使用测试通过")


if __name__ == "__main__":
    print("🚀 开始运行雷达信号处理器测试套件...")
    pytest.main([__file__, "-v", "--tb=short"])
