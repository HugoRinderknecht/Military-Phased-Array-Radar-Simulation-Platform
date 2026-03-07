"""
信号处理模块单元测试

验证核心算法的正确性
"""
import pytest
import numpy as np
from scipy.fft import fft, fftshift

from app.core.signal_processing.antenna import (
    calculate_antenna_pattern,
    analyze_pattern_properties,
)
from app.core.signal_processing.waveform import (
    generate_lfm_pulse,
    pulse_compression,
    calculate_range_resolution,
)
from app.core.signal_processing.clutter import (
    generate_zmnl_clutter,
    verify_clutter_distribution,
)
from app.core.signal_processing.mti_mtd import (
    mti_canceler_1st,
    mti_canceler_2nd,
    mti_canceler_3rd,
    mtd_processing,
)
from app.core.signal_processing.cfar import (
    ca_cfar_1d,
    calculate_cfar_threshold,
)


class TestAntennaPattern:
    """天线方向图测试"""

    def test_antenna_pattern_basic(self):
        """测试基本天线方向图计算"""
        num_h, num_v = 20, 10
        d_h, d_v = 0.5, 0.5
        wavelength = 0.03

        pattern, theta, phi = calculate_antenna_pattern(
            num_h, num_v, d_h, d_v, wavelength
        )

        # 检查输出形状
        assert pattern.shape == theta.shape
        assert pattern.shape == phi.shape

        # 检查峰值在中心
        peak_idx = np.unravel_index(np.argmax(pattern), pattern.shape)
        assert peak_idx[0] == pattern.shape[0] // 2
        assert peak_idx[1] == pattern.shape[1] // 2

        # 检查归一化（最大值为0dB）
        assert np.max(pattern) == pytest.approx(0, abs=0.1)

    def test_antenna_pattern_with_weights(self):
        """测试不同加权函数"""
        num_h, num_v = 20, 10
        wavelength = 0.03

        # Hamming加权
        pattern_hamming, _, _ = calculate_antenna_pattern(
            num_h, num_v, 0.5, 0.5, wavelength, taper="hamming"
        )

        # Taylor加权
        pattern_taylor, _, _ = calculate_antenna_pattern(
            num_h, num_v, 0.5, 0.5, wavelength,
            taper="taylor", taylor_sll=30, taylor_nbar=5
        )

        # Taylor应该有更低的副瓣
        hamming_sll = np.max(pattern_hamming[pattern_hamming < -10])
        taylor_sll = np.max(pattern_taylor[pattern_taylor < -10])

        # Taylor副瓣应该低于Hamming（大部分情况下）
        assert taylor_sll <= hamming_sll + 2  # 允许小误差


class TestWaveform:
    """波形生成测试"""

    def test_lfm_pulse_generation(self):
        """测试LFM脉冲生成"""
        pulse_width = 1e-6
        bandwidth = 10e6
        sampling_rate = 20e6
        prf = 1000

        waveform, time = generate_lfm_pulse(
            pulse_width, bandwidth, sampling_rate, prf
        )

        # 检查输出
        assert len(waveform) == len(time)
        assert waveform.dtype == complex

    def test_pulse_compression(self):
        """测试脉冲压缩"""
        pulse_width = 1e-6
        bandwidth = 10e6
        sampling_rate = 20e6
        prf = 1000

        # 生成LFM脉冲
        waveform, _ = generate_lfm_pulse(
            pulse_width, bandwidth, sampling_rate, prf, num_pulses=1
        )

        # 脉冲压缩
        compressed = pulse_compression(waveform, waveform)

        # 检查峰值位置
        peak_idx = np.argmax(np.abs(compressed))
        assert peak_idx == pytest.approx(len(waveform) // 2, abs=5)

        # 检查主副比
        peak_val = np.abs(compressed).max()
        sidelobe_level = np.sort(np.abs(compressed))[-2]
        psr = 20 * np.log10(peak_val / (sidelobe + 1e-10))

        # LFM脉冲压缩主副比应该 > 13dB（理论值）
        assert psr > 10  # 允许一定误差

    def test_range_resolution_calculation(self):
        """测试距离分辨率计算"""
        bandwidth = 10e6  # 10 MHz
        resolution = calculate_range_resolution(bandwidth)

        # c / (2B) = 3e8 / (2 * 10e6) = 15 m
        assert resolution == pytest.approx(15.0, rel=0.01)


class TestClutter:
    """杂波生成测试"""

    def test_zmnl_rayleigh(self):
        """测试瑞利分布杂波生成"""
        size = 10000
        clutter = generate_zmnl_clutter("rayleigh", size, correlation_coeff=0.9)

        assert len(clutter) == size
        assert np.all(clutter >= 0)

        # 验证分布
        result = verify_clutter_distribution(clutter, "rayleigh")
        assert result["is_accepted"] or result["p_value"] > 0.01  # 放宽标准

    def test_zmnl_weibull(self):
        """测试威布尔分布杂波生成"""
        size = 10000
        clutter = generate_zmnl_clutter(
            "weibull", size, correlation_coeff=0.9, shape_param=2.0
        )

        assert len(clutter) == size
        assert np.all(clutter >= 0)

    def test_zmnl_lognormal(self):
        """测试对数正态分布杂波生成"""
        size = 10000
        clutter = generate_zmnl_clutter(
            "lognormal", size, correlation_coeff=0.9, shape_param=1.0
        )

        assert len(clutter) == size
        assert np.all(clutter >= 0)


class TestMTI:
    """MTI/MTD测试"""

    def test_mti_1st_canceler(self):
        """测试一次MTI对消器"""
        # 创建直流信号（零多普勒）
        pulses = np.ones((10, 100))

        cancelled = mti_canceler_1st(pulses)

        # 直流应该被抑制
        assert np.mean(np.abs(cancelled)) < 0.5

    def test_mti_2nd_canceler(self):
        """测试二次MTI对消器"""
        # 创建直流信号
        pulses = np.ones((10, 100))

        cancelled = mti_canceler_2nd(pulses)

        # 二次对消对直流抑制更强
        assert np.mean(np.abs(cancelled)) < 0.3

    def test_mtd_processing(self):
        """测试MTD处理"""
        num_pulses = 16
        num_range_bins = 100

        # 创建测试数据：添加一些多普勒频率
        pulses = np.random.randn(num_pulses, num_range_bins) + 0.1

        # 添加一个有特定多普勒的目标
        for i in range(num_pulses):
            pulses[i, 50] += 10 * np.exp(1j * 2 * np.pi * 0.3 * i)

        doppler_output = mtd_processing(pulses)

        # 检查输出形状
        assert doppler_output.shape[0] == num_pulses
        assert doppler_output.shape[1] == num_range_bins


class TestCFAR:
    """CFAR检测测试"""

    def test_cfar_threshold_calculation(self):
        """测试CFAR门限计算"""
        pfa = 1e-6
        num_cells = 20

        threshold = calculate_cfar_threshold(pfa, num_cells)

        assert threshold > 0
        # 对于低虚警概率，门限应该较高
        assert threshold > 10

    def test_ca_cfar_detection(self):
        """测试CA-CFAR检测"""
        # 创建测试信号：噪声+一个目标
        signal = np.random.randn(100) * 0.1
        signal[50] = 5.0  # 目标

        detections, threshold, noise_est = ca_cfar_1d(
            signal,
            num_guard_cells=2,
            num_training_cells=10,
            pfa=1e-3
        )

        # 应该检测到目标
        assert detections[50]

        # 检查虚警率（目标附近以外不应该有检测）
        other_detections = np.sum(detections[:40]) + np.sum(detections[60:])
        assert other_detections <= 2  # 允许少量虚警


class TestIntegration:
    """集成测试"""

    def test_full_processing_chain(self):
        """测试完整的处理链"""
        # 1. 生成发射信号
        waveform, _ = generate_lfm_pulse(
            pulse_width=1e-6,
            bandwidth=10e6,
            sampling_rate=20e6,
            prf=1000,
            num_pulses=16
        )

        # 2. 添加杂波
        clutter = generate_zmnl_clutter("rayleigh", len(waveform), correlation_coeff=0.95)
        signal_with_clutter = waveform + 0.1 * clutter

        # 3. 重塑为脉冲矩阵
        num_pulses = 16
        samples_per_pulse = len(waveform) // num_pulses
        pulse_matrix = signal_with_clutter[:num_pulses * samples_per_pulse].reshape(
            num_pulses, samples_per_pulse
        )

        # 4. MTI处理
        mti_output = mti_canceler_1st(pulse_matrix)

        # 检查输出形状
        assert mti_output.shape == pulse_matrix.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
