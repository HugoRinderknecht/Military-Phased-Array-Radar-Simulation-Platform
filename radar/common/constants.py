# 物理常数模块 (Physical Constants Module)
# 本模块定义雷达仿真系统使用的所有物理常数和系统常量

from enum import Enum
import numpy as np


class PhysicsConstants:
    """
    物理常数类 (Physical Constants Class)
    
    包含物理学基本常数，用于雷达方程计算
    """
    
    # ==================== 基本物理常数 ====================
    
    # 光速 (Speed of Light) - m/s
    C = 299792458.0
    
    # 玻尔兹曼常数 (Boltzmann Constant) - J/K
    K = 1.380649e-23
    
    # 标准大气温度 (Standard Temperature) - K
    T0 = 290.0
    
    # 标准大气压 (Standard Pressure) - Pa
    P0 = 101325.0
    
    # 地球半径 (Earth Radius) - m (WGS84)
    RE = 6378137.0
    
    # 地球自转角速度 (Earth Rotation Rate) - rad/s
    OMEGA_E = 7.292115e-5
    
    # 重力加速度 (Gravitational Acceleration) - m/s²
    G = 9.80665


class RadarConstants:
    """
    雷达常数类 (Radar Constants Class)
    
    包含雷达系统设计常用常量
    """
    
    # ==================== 雷达频段 ====================
    
    # 雷达频段定义 (中心频率 Hz)
    BAND_L = 1e9         # L波段: 1-2 GHz
    BAND_S = 3e9         # S波段: 2-4 GHz
    BAND_C = 5e9         # C波段: 4-8 GHz
    BAND_X = 10e9        # X波段: 8-12 GHz
    BAND_KU = 15e9       # Ku波段: 12-18 GHz
    BAND_K = 20e9        # K波段: 18-27 GHz
    BAND_KA = 35e9       # Ka波段: 27-40 GHz
    BAND_MM = 95e9       # 毫米波段: 40-300 GHz
    
    # 频段名称映射
    BAND_NAMES = {
        'L': BAND_L,
        'S': BAND_S,
        'C': BAND_C,
        'X': BAND_X,
        'Ku': BAND_KU,
        'K': BAND_K,
        'Ka': BAND_KA,
        'mm': BAND_MM
    }
    
    # ==================== 常见雷达参数 ====================
    
    # 典型RCS值 (Typical RCS Values) - m²
    RCS_LARGE_AIRCRAFT = 100.0      # 大型飞机
    RCS_MEDIUM_AIRCRAFT = 10.0      # 中型飞机
    RCS_SMALL_AIRCRAFT = 2.0        # 小型飞机
    RCS_FIGHTER = 5.0               # 战斗机
    RCS_STEALTH_FIGHTER = 0.01      # 隐身战斗机
    RCS_HELICOPTER = 3.0            # 直升机
    RCS_MISSILE = 0.1               # 导弹
    RCS_CRUISE_MISSILE = 0.01       # 巡航导弹
    RCS_DRONE = 0.05                # 无人机
    RCS_SHIP = 10000.0              # 舰船
    RCS_SMALL_BOAT = 50.0           # 小艇
    RCS_GROUND_VEHICLE = 20.0       # 地面车辆
    
    # 目标分类RCS阈值 - m²
    RCS_THRESHOLD_LARGE = 10.0
    RCS_THRESHOLD_MEDIUM = 1.0
    RCS_THRESHOLD_SMALL = 0.1
    RCS_THRESHOLD_STEALTH = 0.01
    
    # ==================== 天线常数 ====================
    
    # 半波长间距 (Half-Wavelength Spacing)
    # 不同频段的半波长间距 - m
    HALF_WAVELENGTH_L = PhysicsConstants.C / (2 * BAND_L)
    HALF_WAVELENGTH_S = PhysicsConstants.C / (2 * BAND_S)
    HALF_WAVELENGTH_C = PhysicsConstants.C / (2 * BAND_C)
    HALF_WAVELENGTH_X = PhysicsConstants.C / (2 * BAND_X)
    HALF_WAVELENGTH_KU = PhysicsConstants.C / (2 * BAND_KU)
    HALF_WAVELENGTH_K = PhysicsConstants.C / (2 * BAND_K)
    HALF_WAVELENGTH_KA = PhysicsConstants.C / (2 * BAND_KA)
    
    # ==================== 系统损耗 ====================
    
    # 典型系统损耗 (dB)
    LOSS_TRANSMIT = 1.0              # 发射损耗
    LOSS_RECEIVE = 1.5               # 接收损耗
    LOSS_BEAMSHAPE = 1.5            # 波束形状损耗
    LOSS_SIGNAL_PROCESSING = 1.0    # 信号处理损耗
    LOSS_CFAR = 1.0                 # CFAR损耗
    LOSSPropagation = 2.0           # 传播损耗（大气）
    LOSS_SYSTEM_TOTAL = 7.0          # 系统总损耗（典型值）
    
    # LNA噪声系数 (dB)
    NOISE_FIGURE_LNA_TYPICAL = 1.0
    NOISE_FIGURE_LNA_GOOD = 0.5
    NOISE_FIGURE_LNA_POOR = 3.0


class MathConstants:
    """
    数学常数类 (Math Constants Class)
    
    包含数值计算中常用的数学常数
    """
    
    # 圆周率
    PI = np.pi
    
    # 2π
    TWO_PI = 2 * np.pi
    
    # π/2
    HALF_PI = np.pi / 2
    
    # 度转弧度
    DEG_TO_RAD = np.pi / 180.0
    
    # 弧度转度
    RAD_TO_DEG = 180.0 / np.pi
    
    # 分贝转线性
    DB_TO_LINEAR = 10.0 ** (1.0 / 10.0)
    
    # 自然对数底
    E = np.e


class DefaultConfig:
    """
    默认配置类 (Default Configuration Class)
    
    定义系统的默认参数值
    """
    
    # ==================== 仿真参数 ====================
    
    # 默认仿真时间步长 - 秒
    TIME_STEP = 0.001
    
    # 默认调度周期 - 秒
    SCHEDULE_PERIOD = 0.05
    
    # 最大仿真加速倍数
    MAX_SIMULATION_SPEED = 10.0
    
    # ==================== 默认雷达参数 ====================
    
    # 默认频段: X波段
    DEFAULT_FREQUENCY = RadarConstants.BAND_X
    
    # 默认带宽: 10 MHz
    DEFAULT_BANDWIDTH = 10e6
    
    # 默认采样率: 20 MHz
    DEFAULT_SAMPLE_RATE = 20e6
    
    # 默认脉冲宽度: 10 μs
    DEFAULT_PULSE_WIDTH = 10e-6
    
    # 默认PRF: 2000 Hz
    DEFAULT_PRF = 2000
    
    # 默认发射功率: 100 kW
    DEFAULT_POWER = 100000.0
    
    # 默认天线增益: 40 dB
    DEFAULT_ANTENNA_GAIN = 40.0
    
    # ==================== 默认信号处理参数 ====================
    
    # 默认FFT点数
    DEFAULT_FFT_SIZE = 1024
    
    # 默认脉冲压缩比
    DEFAULT_COMPRESSION_RATIO = 100
    
    # 默认CFAR虚警率
    DEFAULT_FALSE_ALARM_RATE = 1e-6
    
    # 默认CFAR训练单元数
    DEFAULT_CFAR_TRAIN = 20
    
    # 默认CFAR保护单元数
    DEFAULT_CFAR_GUARD = 2
    
    # 默认加窗类型
    DEFAULT_WINDOW = 'hamming'
    
    # ==================== 默认跟踪参数 ====================
    
    # 默认M/N逻辑法的M值
    DEFAULT_M_VALUE = 3
    
    # 默认M/N逻辑法的N值
    DEFAULT_N_VALUE = 3
    
    # 默认航迹确认扫描数
    DEFAULT_CONFIRM_SCANS = 3
    
    # 默认航迹终止扫描数
    DEFAULT_TERMINATE_SCANS = 5
    
    # 最大航迹数
    DEFAULT_MAX_TRACKS = 100
    
    # ==================== 默认天线参数 ====================
    
    # 默认阵列大小: 32x32
    DEFAULT_ARRAY_SIZE = 32
    
    # 默认阵元间距: 半波长
    DEFAULT_ELEMENT_SPACING = PhysicsConstants.C / (2 * RadarConstants.BAND_X)
    
    # 默认旁瓣电平: -25 dB
    DEFAULT_SIDELOBE_LEVEL = -25.0
    
    # 默认扫描范围
    DEFAULT_SCAN_RANGE_AZ = (-60, 60)      # 方位: ±60度
    DEFAULT_SCAN_RANGE_EL = (-10, 60)      # 俯仰: -10到60度


class PerformanceMetrics:
    """
    性能指标类 (Performance Metrics Class)
    
    定义系统性能的目标指标
    """
    
    # ==================== 实时性指标 ====================
    
    # 最大处理延迟 - ms
    MAX_LATENCY = 50.0
    
    # 数据更新频率 - Hz
    PLOT_UPDATE_RATE = 20.0
    TRACK_UPDATE_RATE = 10.0
    BEAM_UPDATE_RATE = 20.0
    HEARTBEAT_RATE = 1.0
    
    # ==================== 容量指标 ====================
    
    # 跟踪容量
    MAX_TRACKING_TARGETS = 30
    
    # 搜索容量
    MAX_SEARCH_TARGETS = 100
    
    # ==================== 精度指标 ====================
    
    # 测距精度 (典型值，取决于带宽) - m
    # 理论精度 = c / (2 * B)
    RANGE_RESOLUTION_X_BAND = PhysicsConstants.C / (2 * 10e6)    # 15 m
    RANGE_RESOLUTION_KA_BAND = PhysicsConstants.C / (2 * 100e6)  # 1.5 m
    
    # 测角精度 (典型值，波束宽度的1/10)
    # 实际精度取决于SNR和测角算法
    ANGLE_ACCURACY_RATIO = 0.1  # 波束宽度的10%
    
    # 测速精度 (典型值) - m/s
    VELOCITY_RESOLUTION = 1.0
    
    # ==================== 调度指标 ====================
    
    # 高优先级任务调度成功率
    SCHEDULING_SUCCESS_RATE = 0.99
    
    # 时间资源利用率上限
    TIME_UTILIZATION_LIMIT = 0.9
    
    # ==================== 检测指标 ====================
    
    # 虚警率
    FALSE_ALARM_RATE = 1e-6
    
    # 典型检测概率 (针对1m²目标，100km距离)
    DETECTION_PROBABILITY_TYPICAL = 0.9


# ==================== 导出所有常数 ====================

__all__ = [
    # Physics
    'PhysicsConstants',
    # Radar
    'RadarConstants',
    # Math
    'MathConstants',
    # Config
    'DefaultConfig',
    # Performance
    'PerformanceMetrics',
]

# 便捷访问
c = PhysicsConstants.C
k = PhysicsConstants.K
t0 = PhysicsConstants.T0
pi = MathConstants.PI
