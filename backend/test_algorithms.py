"""
测试核心算法是否正确工作
"""
import sys
import numpy as np

print("=" * 60)
print("雷达仿真平台 - 核心算法测试")
print("=" * 60)

# 测试信号处理模块
print("\n[1/8] 测试天线方向图...")
try:
    from app.core.signal_processing.antenna import calculate_antenna_pattern
    pattern, theta, phi = calculate_antenna_pattern(20, 10, 0.5, 0.5, 0.03)
    print(f"   OK - 方向图形状: {pattern.shape}")
    print(f"   OK - 最大值: {np.max(pattern):.2f} dB")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试波形生成
print("\n[2/8] 测试发射信号生成...")
try:
    from app.core.signal_processing.waveform import generate_lfm_pulse, pulse_compression
    waveform, time = generate_lfm_pulse(1e-6, 10e6, 20e6, 1000)
    compressed = pulse_compression(waveform, waveform)
    peak_idx = np.argmax(np.abs(compressed))
    print(f"   OK - 波形长度: {len(waveform)}")
    print(f"   OK - 峰值位置: {peak_idx}")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试ZMNL杂波
print("\n[3/8] 测试ZMNL杂波生成...")
try:
    from app.core.signal_processing.clutter import generate_zmnl_clutter, verify_clutter_distribution
    clutter = generate_zmnl_clutter("rayleigh", 1000, 0.9)
    result = verify_clutter_distribution(clutter, "rayleigh")
    print(f"   OK - 杂波样本数: {len(clutter)}")
    print(f"   OK - 分布验证: {'通过' if result['is_accepted'] else '未通过'}")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试MTI/MTD
print("\n[4/8] 测试MTI/MTD...")
try:
    from app.core.signal_processing.mti_mtd import mti_canceler_1st, mti_canceler_2nd, mtd_processing
    pulses = np.random.randn(16, 100)
    cancelled_1st = mti_canceler_1st(pulses)
    cancelled_2nd = mti_canceler_2nd(pulses)
    doppler_output = mtd_processing(pulses)
    print(f"   OK - MTI对消完成")
    print(f"   OK - MTD输出形状: {doppler_output.shape}")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试CFAR
print("\n[5/8] 测试CFAR检测...")
try:
    from app.core.signal_processing.cfar import ca_cfar_1d, calculate_cfar_threshold
    signal = np.random.randn(100) * 0.1
    signal[50] = 5.0  # 目标
    detections, threshold, noise_est = ca_cfar_1d(signal, 2, 10, 1e-3)
    print(f"   OK - 检测到目标: {np.sum(detections)}")
    print(f"   OK - 检测位置: {np.where(detections)[0]}")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试Kalman滤波
print("\n[6/8] 测试Kalman滤波...")
try:
    from app.core.data_processing.filter import KalmanFilter
    kf = KalmanFilter(dt=0.1, process_noise=1.0, measurement_noise=10.0)
    state = np.array([0, 0, 0, 10, 0, 0])  # x, y, z, vx, vy, vz
    covariance = np.eye(6) * 100
    measurement = np.array([100, 50, 20])
    updated_state, updated_cov = kf.filter_step(state, covariance, measurement)
    print(f"   OK - 滤波器状态更新")
    print(f"   OK - 更新后位置: ({updated_state[0]:.1f}, {updated_state[1]:.1f}, {updated_state[2]:.1f})")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试点迹关联
print("\n[7/8] 测试点迹关联...")
try:
    from app.core.data_processing.association import NNAassociator
    from app.core.data_processing.track_init import ConfirmedTrack, Plot
    associator = NNAassociator()
    track = ConfirmedTrack(
        track_id="test-001",
        plots=[],
        state=np.array([1000, 2000, 500, 100, 50, 0]),
        covariance=np.eye(6) * 10
    )
    plot = Plot(plot_id="p1", time=0, x=1050, y=2050, z=510, amplitude=1.0)
    associations = associator.associate([track], [plot])
    print(f"   OK - 关联结果: {len(associations)}")
except Exception as e:
    print(f"   ERROR: {e}")

# 测试参数估计
print("\n[8/8] 测试参数估计...")
try:
    from app.core.signal_processing.measurement import estimate_range, calculate_range_accuracy
    detections = np.array([0, 0, 1, 0, 0, 0, 0])
    range_bins = np.arange(100) * 10  # 0-990m
    ranges, indices = estimate_range(detections, range_bins)
    print(f"   OK - 估计距离: {ranges}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)
