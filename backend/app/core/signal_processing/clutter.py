"""
ZMNL杂波生成模块
完整实现基于零记忆非线性变换(ZMNL)的杂波生成

参考文档 4.4.4 节
"""
import numpy as np
from typing import Literal, Optional, Tuple
from scipy import signal
from scipy.stats import kstest, norm, lognorm, weibull_min


def generate_zmnl_clutter(
    pdf_type: Literal["rayleigh", "lognormal", "weibull", "k"],
    size: int,
    correlation_coeff: float = 0.9,
    power: float = 1.0,
    shape_param: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    使用ZMNL方法生成相关杂波

    ZMNL方法步骤：
    1. 生成相关的高斯白噪声序列
    2. 设计相关滤波器
    3. 应用无记忆非线性变换

    Args:
        pdf_type: 概率分布类型
            - rayleigh: 瑞利分布（适用于地杂波）
            - lognormal: 对数正态分布（高分辨率海杂波）
            - weibull: 威布尔分布（通用杂波模型）
            - k: K分布（高分辨率杂波）
        size: 生成样本数
        correlation_coeff: 相关系数（0-1）
        power: 杂波功率
        shape_param: 形状参数（lognormal的sigma，weibull的形状参数）
        rng: 随机数生成器

    Returns:
        杂波幅度序列
    """
    if rng is None:
        rng = np.random.default_rng()

    # 步骤1: 生成相关的高斯序列
    correlated_gaussian = _generate_correlated_gaussian(
        size, correlation_coeff, rng
    )

    # 步骤2 & 3: 应用非线性变换
    if pdf_type == "rayleigh":
        clutter = _rayleigh_transform(correlated_gaussian, power)
    elif pdf_type == "lognormal":
        clutter = _lognormal_transform(correlated_gaussian, power, shape_param)
    elif pdf_type == "weibull":
        clutter = _weibull_transform(correlated_gaussian, power, shape_param)
    elif pdf_type == "k":
        clutter = _k_distribution_transform(correlated_gaussian, power, shape_param, rng)
    else:
        raise ValueError(f"不支持的分布类型: {pdf_type}")

    return clutter


def _generate_correlated_gaussian(
    size: int,
    correlation_coeff: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    生成相关的高斯序列

    使用AR(1)模型：x[n] = ρ*x[n-1] + w[n]

    Args:
        size: 序列长度
        correlation_coeff: 相关系数ρ
        rng: 随机数生成器

    Returns:
        相关高斯序列
    """
    # 白噪声
    white_noise = rng.standard_normal(size)

    # AR(1)滤波器
    if abs(correlation_coeff) < 0.01:
        return white_noise

    correlated = np.zeros(size)
    correlated[0] = white_noise[0]

    for n in range(1, size):
        correlated[n] = correlation_coeff * correlated[n-1] + \
                       np.sqrt(1 - correlation_coeff**2) * white_noise[n]

    return correlated


def _rayleigh_transform(
    gaussian: np.ndarray,
    power: float,
) -> np.ndarray:
    """
    瑞利分布变换

    对于高斯变量 x, y ~ N(0, σ²)
    r = √(x² + y²) 服从瑞利分布

    Args:
        gaussian: 高斯序列
        power: 目标功率

    Returns:
        瑞利分布序列
    """
    # 生成独立的高斯序列对
    n = len(gaussian)
    gaussian2 = np.random.standard_normal(n)

    # 瑞利变换
    rayleigh = np.sqrt(gaussian**2 + gaussian2**2)

    # 调整功率
    current_power = np.mean(rayleigh**2)
    if current_power > 0:
        rayleigh = rayleigh * np.sqrt(power / current_power)

    return rayleigh


def _lognormal_transform(
    gaussian: np.ndarray,
    power: float,
    sigma: float,
) -> np.ndarray:
    """
    对数正态分布变换

    如果 ln(x) ~ N(μ, σ²)，则 x 服从对数正态分布

    Args:
        gaussian: 高斯序列
        power: 目标功率
        sigma: 对数标准差

    Returns:
        对数正态分布序列
    """
    # 对数正态变换
    lognormal = np.exp(sigma * gaussian)

    # 调整功率
    current_power = np.mean(lognormal**2)
    if current_power > 0:
        lognormal = lognormal * np.sqrt(power / current_power)

    return lognormal


def _weibull_transform(
    gaussian: np.ndarray,
    power: float,
    shape: float,
) -> np.ndarray:
    """
    威布尔分布变换

    威布尔分位数函数：F^(-1)(p) = λ * (-ln(1-p))^(1/k)

    Args:
        gaussian: 高斯序列
        power: 目标功率
        shape: 威布尔形状参数k

    Returns:
        威布尔分布序列
    """
    # 将高斯转换为均匀分布（使用CDF）
    uniform = 0.5 * (1 + np.erf(gaussian / np.sqrt(2)))

    # 威布尔分位数变换
    # 先计算尺度参数λ使功率匹配
    # E[X²] = λ² * Γ(1 + 2/k)
    from scipy.special import gamma
    lambda_scale = np.sqrt(power / gamma(1 + 2/shape))

    weibull = lambda_scale * (-np.log(1 - uniform + 1e-10)) ** (1/shape)

    return weibull


def _k_distribution_transform(
    gaussian: np.ndarray,
    power: float,
    shape: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    K分布变换

    K分布可以看作是两个独立随机变量的乘积：
    - 瑞利分布（快起伏）
    - Gamma分布（慢起伏，纹理分量）

    Args:
        gaussian: 高斯序列
        power: 目标功率
        shape: K分布形状参数
        rng: 随机数生成器

    Returns:
        K分布序列
    """
    n = len(gaussian)

    # 纹理分量（Gamma分布）
    # Gamma(ν, 1)
    texture = rng.gamma(shape, 1.0, n)

    # 斑点分量（瑞利分布）
    speckle = np.abs(gaussian + 1j * rng.standard_normal(n))

    # K分布 = 斑点 * sqrt(纹理)
    k_dist = speckle * np.sqrt(texture)

    # 调整功率
    current_power = np.mean(k_dist**2)
    if current_power > 0:
        k_dist = k_dist * np.sqrt(power / current_power)

    return k_dist


def design_correlation_filter(
    desired_psd: Literal["gaussian", "exponential", "cauchy"],
    correlation_coeff: float,
    filter_length: int = 64,
) -> np.ndarray:
    """
    设计杂波相关滤波器

    Args:
        desired_psd: 期望功率谱类型
        correlation_coeff: 相关系数
        filter_length: 滤波器长度

    Returns:
        FIR滤波器系数
    """
    n = np.arange(filter_length) - filter_length // 2

    if desired_psd == "gaussian":
        # 高斯功率谱
        h = np.exp(-0.5 * (n / (correlation_coeff * 10))**2)

    elif desired_psd == "exponential":
        # 指数功率谱
        h = correlation_coeff ** np.abs(n)

    elif desired_psd == "cauchy":
        # 柯西功率谱
        h = 1 / (1 + (n / (correlation_coeff * 5))**2)

    else:
        raise ValueError(f"不支持的功率谱类型: {desired_psd}")

    # 归一化
    h = h / np.sum(h)

    return h


def apply_psd_shaping(
    clutter: np.ndarray,
    filter_coeffs: np.ndarray,
) -> np.ndarray:
    """
    对杂波应用功率谱成形

    Args:
        clutter: 输入杂波序列
        filter_coeffs: FIR滤波器系数

    Returns:
    功率谱成形后的杂波
    """
    # 使用FIR滤波器
    shaped_clutter = signal.lfilter(filter_coeffs, 1.0, clutter)

    # 归一化功率
    shaped_clutter = shaped_clutter / np.sqrt(np.mean(np.abs(shaped_clutter)**2))

    return shaped_clutter


def generate_2d_clutter_map(
    pdf_type: Literal["rayleigh", "lognormal", "weibull", "k"],
    map_size: Tuple[int, int],
    correlation_coeff_range: float = 0.9,
    correlation_coeff_doppler: float = 0.95,
    power: float = 1.0,
    shape_param: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    生成二维杂波图（距离-多普勒）

    Args:
        pdf_type: 概率分布类型
        map_size: (range_bins, doppler_bins)
        correlation_coeff_range: 距离向相关系数
        correlation_coeff_doppler: 多普勒向相关系数
        power: 杂波功率
        shape_param: 形状参数
        rng: 随机数生成器

    Returns:
        二维杂波图
    """
    if rng is None:
        rng = np.random.default_rng()

    range_bins, doppler_bins = map_size

    # 生成距离相关杂波
    clutter_map = np.zeros(map_size)

    for d in range(doppler_bins):
        # 每个多普勒通道生成距离维杂波
        range_clutter = generate_zmnl_clutter(
            pdf_type,
            range_bins,
            correlation_coeff_range,
            power,
            shape_param,
            rng,
        )
        clutter_map[:, d] = range_clutter

    # 多普勒向相关（沿多普勒维平滑）
    if correlation_coeff_doppler > 0:
        for r in range(range_bins):
            doppler_profile = clutter_map[r, :]
            # AR(1)相关
            for d in range(1, doppler_bins):
                doppler_profile[d] = (
                    correlation_coeff_doppler * doppler_profile[d-1] +
                    np.sqrt(1 - correlation_coeff_doppler**2) * doppler_profile[d]
                )
            clutter_map[r, :] = doppler_profile

    return clutter_map


def verify_clutter_distribution(
    clutter_samples: np.ndarray,
    target_distribution: Literal["rayleigh", "lognormal", "weibull"],
    significance_level: float = 0.05,
) -> dict:
    """
    验证生成杂波的分布特性

    使用KS检验验证样本是否服从目标分布

    Args:
        clutter_samples: 杂波样本
        target_distribution: 目标分布类型
        significance_level: 显著性水平

    Returns:
        检验结果字典
    """
    result = {}

    # 归一化样本
    normalized_samples = clutter_samples / np.mean(clutter_samples)

    if target_distribution == "rayleigh":
        # KS检验对比瑞利分布
        # 瑞利分布CDF: F(x) = 1 - exp(-x²/(2σ²))
        # 估计参数
        sigma_est = np.sqrt(np.mean(normalized_samples**2) / 2)

        def rayleigh_cdf(x):
            return 1 - np.exp(-x**2 / (2 * sigma_est**2))

        # KS检验
        ks_statistic, p_value = kstest(normalized_samples, rayleigh_cdf)

        result["distribution"] = "rayleigh"
        result["ks_statistic"] = ks_statistic
        result["p_value"] = p_value
        result["is_accepted"] = p_value > significance_level
        result["estimated_sigma"] = sigma_est

    elif target_distribution == "lognormal":
        # 对数正态分布检验
        log_samples = np.log(normalized_samples + 1e-10)
        sigma_est = np.std(log_samples)
        mu_est = np.mean(log_samples)

        # KS检验
        ks_statistic, p_value = kstest(log_samples, 'norm', args=(mu_est, sigma_est))

        result["distribution"] = "lognormal"
        result["ks_statistic"] = ks_statistic
        result["p_value"] = p_value
        result["is_accepted"] = p_value > significance_level
        result["estimated_mu"] = mu_est
        result["estimated_sigma"] = sigma_est

    elif target_distribution == "weibull":
        # 威布尔分布检验
        # 使用scipy的fit
        params = weibull_min.fit(normalized_samples, f0=1.0)

        # KS检验
        ks_statistic, p_value = kstest(
            normalized_samples,
            lambda x: weibull_min.cdf(x, *params)
        )

        result["distribution"] = "weibull"
        result["ks_statistic"] = ks_statistic
        result["p_value"] = p_value
        result["is_accepted"] = p_value > significance_level
        result["estimated_params"] = params

    return result


def calculate_acf(
    clutter: np.ndarray,
    max_lag: int = 50,
) -> np.ndarray:
    """
    计算杂波的自相关函数

    Args:
        clutter: 杂波序列
        max_lag: 最大滞后数

    Returns:
        自相关函数
    """
    acf = np.zeros(max_lag + 1)

    # 归一化
    clutter = clutter - np.mean(clutter)
    variance = np.var(clutter)

    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            correlated = np.mean(clutter[lag:] * clutter[:-lag])
            acf[lag] = correlated / variance

    return acf
