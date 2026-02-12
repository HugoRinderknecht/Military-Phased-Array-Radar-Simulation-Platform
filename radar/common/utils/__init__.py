# utils - 工具函数模块
"""
工具函数模块提供各种实用的数学和信号处理工具函数。

包括：
- math_utils: 数学工具函数
- coord_transform: 坐标系转换
- signal_utils: 信号处理工具
"""

from radar.common.utils.math_utils import (
    # 基础数学函数
    deg_to_rad,
    rad_to_deg,
    normalize_angle,
    angle_difference,
    wrap_angle,

    # 向量运算
    normalize_vector,
    vector_magnitude,
    vector_distance,
    dot_product,
    cross_product,

    # 统计函数
    mean,
    std,
    variance,
    median,

    # 插值函数
    linear_interpolate,
    spline_interpolate,
    nearest_neighbor,

    # 窗函数
    get_window,
    apply_window,

    # FFT相关
    fft_shift,
    fft_frequency,
    next_power_of_2,
)

from radar.common.utils.coord_transform import (
    # 坐标转换
    enu_to_azel,
    azel_to_enu,
    enu_to_ecef,
    ecef_to_enu,
    ned_to_enu,
    enu_to_ned,

    # 旋转矩阵
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotation_matrix_zyx,

    # 转换辅助
    geodetic_to_ecef,
    ecef_to_geodetic,
)

from radar.common.utils.signal_utils import (
    # 信号生成
    generate_complex_pulse,
    generate_lfm_pulse,
    generate_phase_code,
    generate_noise,

    # 信号处理
    resample_signal,
    decimate,
    interpolate,

    # 频谱分析
    compute_fft,
    compute_psd,
    compute_spectrogram,

    # 滤波
    apply_fir_filter,
    apply_iir_filter,
    design_fir_filter,
    design_iir_filter,

    # 相关
    correlate,
    convolve,

    # 其他
    db_to_linear,
    linear_to_db,
    complex_to_db,
    envelope,
    instantaneous_phase,
    instantaneous_frequency,
)

__all__ = [
    # math_utils
    "deg_to_rad",
    "rad_to_deg",
    "normalize_angle",
    "angle_difference",
    "wrap_angle",
    "normalize_vector",
    "vector_magnitude",
    "vector_distance",
    "dot_product",
    "cross_product",
    "mean",
    "std",
    "variance",
    "median",
    "linear_interpolate",
    "spline_interpolate",
    "nearest_neighbor",
    "get_window",
    "apply_window",
    "fft_shift",
    "fft_frequency",
    "next_power_of_2",

    # coord_transform
    "enu_to_azel",
    "azel_to_enu",
    "enu_to_ecef",
    "ecef_to_enu",
    "ned_to_enu",
    "enu_to_ned",
    "rotation_matrix_x",
    "rotation_matrix_y",
    "rotation_matrix_z",
    "rotation_matrix_zyx",
    "geodetic_to_ecef",
    "ecef_to_geodetic",

    # signal_utils
    "generate_complex_pulse",
    "generate_lfm_pulse",
    "generate_phase_code",
    "generate_noise",
    "resample_signal",
    "decimate",
    "interpolate",
    "compute_fft",
    "compute_psd",
    "compute_spectrogram",
    "apply_fir_filter",
    "apply_iir_filter",
    "design_fir_filter",
    "design_iir_filter",
    "correlate",
    "convolve",
    "db_to_linear",
    "linear_to_db",
    "complex_to_db",
    "envelope",
    "instantaneous_phase",
    "instantaneous_frequency",
]
