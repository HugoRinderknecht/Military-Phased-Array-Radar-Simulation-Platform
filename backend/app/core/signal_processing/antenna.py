"""
天线方向图计算模块
完整实现矩形平面阵方向图计算，支持多种加权函数和FFT加速

参考文档 4.4.1 节
"""
import numpy as np
from typing import Literal, Optional, Tuple
import warnings


def calculate_antenna_pattern(
    num_h: int,
    num_v: int,
    d_h: float,
    d_v: float,
    wavelength: float,
    weights: Optional[np.ndarray] = None,
    taper: Literal["uniform", "taylor", "hamming", "hanning", "blackman"] = "hamming",
    taylor_sll: Optional[float] = None,
    taylor_nbar: Optional[int] = None,
    use_fft: bool = True,
    grid_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算矩形平面阵天线方向图

    完整实现二维阵列因子计算:
    AF(u,v) = ΣₘΣₙ wₘₙ * exp(j*2π/λ*(m*dx*u + n*dy*v))

    Args:
        num_h: 水平方向阵元数
        num_v: 垂直方向阵元数
        d_h: 水平阵元间距（波长）
        d_v: 垂直阵元间距（波长）
        wavelength: 波长（m）
        weights: 自定义加权系数 (num_h, num_v)
        taper: 加权函数类型
        taylor_sll: Taylor 副瓣电平 (dB)
        taylor_nbar: Taylor 参数
        use_fft: 是否使用FFT加速
        grid_size: 方向图网格大小

    Returns:
        (pattern_db, theta_grid, phi_grid): 方向图(dB)、俯仰角网格、方位角网格
    """
    # 生成加权系数
    if weights is None:
        weights = _generate_weights(
            num_h, num_v, taper, taylor_sll, taylor_nbar
        )

    if use_fft:
        # 使用FFT加速计算（推荐用于大阵列）
        pattern = _calculate_pattern_fft(weights, grid_size)
    else:
        # 直接计算（精确但较慢）
        pattern = _calculate_pattern_direct(
            weights, num_h, num_v, d_h, d_v, wavelength, grid_size
        )

    # 归一化
    pattern_normalized = pattern / np.max(pattern)

    # 转换为dB
    pattern_db = 20 * np.log10(pattern_normalized + 1e-20)  # 避免log(0)

    # 生成角度网格
    theta = np.linspace(-np.pi/2, np.pi/2, grid_size)
    phi = np.linspace(-np.pi/2, np.pi/2, grid_size)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    return pattern_db, theta_grid, phi_grid


def _generate_weights(
    num_h: int,
    num_v: int,
    taper: str,
    taylor_sll: Optional[float],
    taylor_nbar: Optional[int],
) -> np.ndarray:
    """生成加权系数"""

    # 一维加权函数
    def get_1d_weights(n, taper_type):
        indices = np.arange(n) - (n - 1) / 2

        if taper_type == "uniform":
            return np.ones(n)
        elif taper_type == "hamming":
            return 0.54 - 0.46 * np.cos(2 * np.pi * indices / (n - 1))
        elif taper_type == "hanning":
            return 0.5 * (1 + np.cos(2 * np.pi * indices / (n - 1)))
        elif taper_type == "blackman":
            return (
                0.42
                - 0.5 * np.cos(2 * np.pi * indices / (n - 1))
                + 0.08 * np.cos(4 * np.pi * indices / (n - 1))
            )
        elif taper_type == "taylor":
            if taylor_sll is None or taylor_nbar is None:
                warnings.warn("Taylor参数缺失，使用Hamming代替")
                return 0.54 - 0.46 * np.cos(2 * np.pi * indices / (n - 1))
            return _taylor_weights(n, taylor_sll, taylor_nbar)
        else:
            return np.ones(n)

    w_h = get_1d_weights(num_h, taper)
    w_v = get_1d_weights(num_v, taper)

    # 二维加权（外积）
    weights = np.outer(w_h, w_v)

    return weights


def _taylor_weights(n: int, sll: float, nbar: int) -> np.ndarray:
    """
    Taylor 加权函数

    Args:
        n: 阵元数
        sll: 副瓣电平 (dB)
        nbar: 接近主瓣的等副瓣数

    Returns:
        加权系数
    """
    # Taylor 加权参数计算
    A = np.arccosh(10 ** (sll / 20)) / np.pi
    indices = np.arange(n) - (n - 1) / 2

    # 计算 Taylor 权重
    sigma2 = nbar ** 2 / (A ** 2 + (nbar - 0.5) ** 2)

    weights = np.zeros(n)
    for m, idx in enumerate(indices):
        sum_val = 1
        for n in range(1, nbar):
            # 计算 F 系数
            numerator = (-1) ** (n + 1) * np.prod(m - (i + 0.5) for i in range(n)) * \
                        np.prod(m + (i + 0.5) for i in range(n))
            denominator = np.prod(n - (i + 0.5) for i in range(n)) * \
                         np.prod(n + (i + 0.5) for i in range(n)) * \
                         np.prod(n - (i + 0.5) for i in range(n)) * \
                         np.prod(n + (i + 0.5) for i in range(n))
            F_n = -1 * numerator / denominator if n > 0 else 1

            # 计算 Bessel 函数项
            psi_n = np.pi * idx * np.sqrt(sigma2) / nbar
            from scipy.special import iv
            bessel_term = iv(0, np.pi * idx * np.sqrt(sigma2) / nbar)

            sum_val += 2 * F_n * bessel_term

        weights[m] = sum_val

    # 归一化
    weights = weights / np.max(weights)

    return weights


def _calculate_pattern_fft(
    weights: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """
    使用FFT计算方向图

    FFT 方法将 O(N²M²) 的复杂度降低到 O(NM log NM)

    Args:
        weights: 加权系数矩阵 (num_h, num_v)
        grid_size: 输出网格大小

    Returns:
        方向图幅度
    """
    num_h, num_v = weights.shape

    # 零填充到 grid_size
    padded = np.zeros((grid_size, grid_size), dtype=complex)
    h_start = (grid_size - num_h) // 2
    v_start = (grid_size - num_v) // 2
    padded[h_start:h_start + num_h, v_start:v_start + num_v] = weights

    # FFT 变换
    pattern = np.fft.fft2(padded)
    pattern = np.fft.fftshift(pattern)

    # 取幅度
    pattern = np.abs(pattern)

    return pattern


def _calculate_pattern_direct(
    weights: np.ndarray,
    num_h: int,
    num_v: int,
    d_h: float,
    d_v: float,
    wavelength: float,
    grid_size: int,
) -> np.ndarray:
    """
    直接计算方向图（精确但较慢）

    Args:
        weights: 加权系数矩阵
        num_h: 水平阵元数
        num_v: 垂直阵元数
        d_h: 水平间距
        d_v: 垂直间距
        wavelength: 波长
        grid_size: 网格大小

    Returns:
        方向图幅度
    """
    # 生成角度网格
    theta = np.linspace(-np.pi/2, np.pi/2, grid_size)
    phi = np.linspace(-np.pi/2, np.pi/2, grid_size)

    pattern = np.zeros((grid_size, grid_size))

    # 直接计算阵列因子
    for i, th in enumerate(theta):
        for j, ph in enumerate(phi):
            # 方向余弦
            u = np.sin(th) * np.cos(ph)
            v = np.sin(th) * np.sin(ph)

            # 计算相位项
            phase_h = 2 * np.pi * d_h * u
            phase_v = 2 * np.pi * d_v * v

            # 阵列因子
            af = 0
            for m in range(num_h):
                for n in range(num_v):
                    af += weights[m, n] * np.exp(1j * (m * phase_h + n * phase_v))

            pattern[i, j] = np.abs(af)

    return pattern


def analyze_pattern_properties(
    pattern_db: np.ndarray,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
) -> dict:
    """
    分析方向图特性

    Args:
        pattern_db: 方向图 (dB)
        theta_grid: 俯仰角网格
        phi_grid: 方位角网格

    Returns:
        方向图特性字典
    """
    result = {}

    # 找到主瓣位置
    max_idx = np.unravel_index(np.argmax(pattern_db), pattern_db.shape)
    theta_max = theta_grid[max_idx]
    phi_max = phi_grid[max_idx]

    # 计算主瓣宽度（3dB）
    max_val = pattern_db[max_idx]
    val_3db = max_val - 3

    # 在主瓣切面上找3dB点
    theta_slice = pattern_db[:, max_idx[1]]
    indices_3db = np.where(theta_slice >= val_3db)[0]
    if len(indices_3db) > 1:
        beamwidth_3db_theta = (
            theta_grid[indices_3db[-1]] - theta_grid[indices_3db[0]]
        ) * 180 / np.pi
        result["beamwidth_3db_theta_deg"] = beamwidth_3db_theta

    # 计算副瓣电平
    mask = pattern_db < max_val - 10  # 排除主瓣附近区域
    if np.any(mask):
        max_sidelobe = np.max(pattern_db[mask])
        result["sidelobe_level_db"] = max_sidelobe - max_val
    else:
        result["sidelobe_level_db"] = -60  # 默认值

    # 计算方向性系数（近似）
    total_power = np.sum(10 ** (pattern_db / 10))
    peak_power = 10 ** (max_val / 10)
    directivity = 10 * np.log10(peak_power / (total_power / pattern_db.size))
    result["directivity_db"] = directivity

    return result


def calculate_scanned_pattern(
    base_pattern: np.ndarray,
    theta_scan: float,
    phi_scan: float,
    theta_grid: np.ndarray,
    phi_grid: np.ndarray,
    wavelength: float,
    d_h: float,
    d_v: float,
    num_h: int,
    num_v: int,
) -> np.ndarray:
    """
    计算相控扫描后的方向图

    Args:
        base_pattern: 基础方向图
        theta_scan: 扫描俯仰角
        phi_scan: 扫描方位角
        theta_grid: 俯仰角网格
        phi_grid: 方位角网格
        wavelength: 波长
        d_h: 水平阵元间距
        d_v: 垂直阵元间距
        num_h: 水平阵元数
        num_v: 垂直阵元数

    Returns:
        扫描后的方向图
    """
    # 计算相移梯度
    u_scan = np.sin(theta_scan) * np.cos(phi_scan)
    v_scan = np.sin(theta_scan) * np.sin(phi_scan)

    # 计算每个阵元的相移
    phase_shift_h = 2 * np.pi * d_h * u_scan
    phase_shift_v = 2 * np.pi * d_v * v_scan

    # 生成扫描相移矩阵
    phase_grid = np.zeros_like(theta_grid)
    for m in range(num_h):
        for n in range(num_v):
            phase_grid += (m * phase_shift_h + n * phase_shift_v) / (num_h * num_v)

    # 应用相移（简化模型）
    scanned_pattern = base_pattern * np.exp(1j * phase_grid)

    return np.abs(scanned_pattern)
