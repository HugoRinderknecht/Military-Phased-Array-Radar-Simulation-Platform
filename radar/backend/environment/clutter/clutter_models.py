# clutter_models.py - 完整的杂波模型实现
"""
雷达杂波模型

完整的杂波统计模型实现:
- K分布杂波
- Log-Normal分布杂波
- Weibull分布杂波
- Rayleigh分布杂波
- 相关杂波生成
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
from scipy import signal


class ClutterType(Enum):
    """杂波类型"""
    RAYLEIGH = "rayleigh"       # 瑞利分布 (点杂波)
    LOGNORMAL = "lognormal"     # 对数正态分布 (地杂波)
    WEIBULL = "weibull"         # 威布尔分布 (海杂波)
    K_DISTRIBUTION = "k"        # K分布 (高分辨率杂波)


@dataclass
class ClutterParameters:
    """杂波参数"""
    clutter_type: ClutterType
    mean_power: float          # 平均功率 (线性值)
    shape_param: float = 1.0   # 形状参数 (K: ν, Weibull: α)
    scale_param: float = 1.0   # 尺度参数 (Log-Normal: μσ, Weibull: β)
    correlation_time: float = 0.0  # 相关时间 (s)
    spectrum_type: str = "gaussian"  # 谱类型


class ClutterGenerator:
    """
    杂波生成器

    生成各种类型的雷达杂波
    """

    def __init__(
        self,
        params: ClutterParameters,
        sample_rate: float = 20e6,
        prf: float = 2000.0
    ):
        """
        Args:
            params: 杂波参数
            sample_rate: 采样率
            prf: 脉冲重复频率
        """
        self.params = params
        self.sample_rate = sample_rate
        self.prf = prf

    def generate_clutter(
        self,
        num_samples: int,
        num_pulses: int = 1
    ) -> np.ndarray:
        """
        生成杂波数据

        Args:
            num_samples: 距离单元数
            num_pulses: 脉冲数

        Returns:
            杂波数据 [num_pulses, num_samples]
        """
        if self.params.clutter_type == ClutterType.RAYLEIGH:
            return self._generate_rayleigh_clutter(num_samples, num_pulses)
        elif self.params.clutter_type == ClutterType.LOGNORMAL:
            return self._generate_lognormal_clutter(num_samples, num_pulses)
        elif self.params.clutter_type == ClutterType.WEIBULL:
            return self._generate_weibull_clutter(num_samples, num_pulses)
        elif self.params.clutter_type == ClutterType.K_DISTRIBUTION:
            return self._generate_k_clutter(num_samples, num_pulses)
        else:
            raise ValueError(f"Unknown clutter type: {self.params.clutter_type}")

    def _generate_rayleigh_clutter(
        self,
        num_samples: int,
        num_pulses: int
    ) -> np.ndarray:
        """
        瑞利分布杂波

        适用于点杂波，最简单的杂波模型
        """
        # 瑞利分布参数
        sigma = np.sqrt(self.params.mean_power / 2.0)

        # 生成瑞利随机变量
        clutter = np.random.rayleigh(sigma, (num_pulses, num_samples))

        # 添加时间相关性
        if self.params.correlation_time > 0:
            clutter = self._add_correlation(clutter)

        return clutter

    def _generate_lognormal_clutter(
        self,
        num_samples: int,
        num_pulses: int
    ) -> np.ndarray:
        """
        对数正态分布杂波

        适用于地杂波，特别是高角度分辨率情况
        """
        # Log-Normal参数
        mu = self.params.scale_param  # 均值
        sigma = self.params.shape_param  # 标准差

        # 计算对数正态参数以匹配期望功率
        # E[X] = exp(μ + σ²/2)
        # 调整μ使得E[X] = mean_power
        mu = np.log(self.params.mean_power) - sigma**2 / 2.0

        # 生成对数正态随机变量
        normal_samples = np.random.normal(mu, sigma, (num_pulses, num_samples))
        clutter = np.exp(normal_samples)

        # 添加时间相关性
        if self.params.correlation_time > 0:
            clutter = self._add_correlation(clutter)

        return clutter

    def _generate_weibull_clutter(
        self,
        num_samples: int,
        num_pulses: int
    ) -> np.ndarray:
        """
        威布尔分布杂波

        适用于海杂波，尾部比瑞利分布重
        """
        # Weibull参数
        alpha = self.params.shape_param  # 形状参数
        beta = self.params.scale_param    # 尺度参数

        # 计算beta以匹配期望功率
        # E[X] = β * Γ(1 + 1/α)
        gamma_factor = np.math.gamma(1.0 + 1.0 / alpha)
        beta = self.params.mean_power / gamma_factor

        # 生成威布尔随机变量
        clutter = np.random.weibull(alpha, (num_pulses, num_samples)) * beta

        # 添加时间相关性
        if self.params.correlation_time > 0:
            clutter = self._add_correlation(clutter)

        return clutter

    def _generate_k_clutter(
        self,
        num_samples: int,
        num_pulses: int
    ) -> np.ndarray:
        """
        K分布杂波

        适用于高分辨率雷达杂波，复合高斯模型
        """
        # K分布参数
        nu = self.params.shape_param  # 形状参数
        sigma = self.params.scale_param  # 尺度参数

        # 计算sigma以匹配期望功率
        # E[X] = σ
        sigma = self.params.mean_power

        # K分布是复合分布: Z = τ * X
        # 其中 τ 服从 Gamma(ν, 1)，X 服从瑞利分布

        # 生成Gamma随机变量
        tau = np.random.gamma(nu, 1.0, (num_pulses, num_samples))

        # 生成条件瑞利随机变量
        # 条件方差是 τ * σ² / ν
        conditional_sigma = np.sqrt(tau * sigma**2 / nu)
        clutter = np.random.rayleigh(conditional_sigma)

        # 添加时间相关性
        if self.params.correlation_time > 0:
            clutter = self._add_correlation(clutter)

        return clutter

    def _add_correlation(self, data: np.ndarray) -> np.ndarray:
        """
        添加时间相关性

        使用指定的功率谱模型
        """
        num_pulses, num_samples = data.shape

        # 生成相关系数
        if self.params.correlation_time > 0:
            # 指数相关模型
            rho = np.exp(-1.0 / (self.params.correlation_time * self.prf))

            # AR(1)模型
            correlated_data = np.zeros_like(data)
            correlated_data[0, :] = data[0, :]

            for i in range(1, num_pulses):
                correlated_data[i, :] = (
                    rho * correlated_data[i-1, :] +
                    np.sqrt(1 - rho**2) * data[i, :]
                )

            return correlated_data
        else:
            return data

    def generate_clutter_map(
        self,
        range_bins: np.ndarray,
        azimuth_bins: np.ndarray
    ) -> np.ndarray:
        """
        生成杂波图

        Args:
            range_bins: 距离单元
            azimuth_bins: 方位单元

        Returns:
            杂波图 [azimuth, range]
        """
        num_azimuth = len(azimuth_bins)
        num_range = len(range_bins)

        clutter_map = np.zeros((num_azimuth, num_range))

        for i in range(num_azimuth):
            # 每个方位生成独立杂波
            clutter_row = self.generate_clutter(num_range, 1)
            clutter_map[i, :] = clutter_row[0, :]

        return clutter_map


class CorrelatedClutterGenerator:
    """
    相关杂波生成器

    使用指定的功率谱生成时间相关杂波
    """

    def __init__(
        self,
        clutter_type: ClutterType,
        mean_power: float,
        spectrum_type: str = "gaussian",
        spectrum_width: float = 100.0,
        correlation_length: int = 10
    ):
        """
        Args:
            clutter_type: 杂波类型
            mean_power: 平均功率
            spectrum_type: 功率谱类型 ("gaussian", "exponential")
            spectrum_width: 谱宽度
            correlation_length: 相关长度
        """
        self.clutter_type = clutter_type
        self.mean_power = mean_power
        self.spectrum_type = spectrum_type
        self.spectrum_width = spectrum_width
        self.correlation_length = correlation_length

    def generate(
        self,
        num_samples: int,
        num_pulses: int = 1
    ) -> np.ndarray:
        """
        生成相关杂波

        Args:
            num_samples: 距离单元数
            num_pulses: 脉冲数

        Returns:
            杂波数据
        """
        # 1. 生成白噪声
        white_noise = np.random.randn(num_pulses, num_samples)

        # 2. 设计滤波器
        filter_coeff = self._design_correlation_filter()

        # 3. 应用滤波器
        correlated_noise = np.zeros((num_pulses, num_samples))
        for i in range(num_samples):
            correlated_noise[:, i] = np.convolve(
                white_noise[:, i],
                filter_coeff,
                mode='same'
            )

        # 4. 调整功率
        current_power = np.mean(correlated_noise**2)
        scale_factor = np.sqrt(self.mean_power / current_power)
        correlated_noise *= scale_factor

        return correlated_noise

    def _design_correlation_filter(self) -> np.ndarray:
        """设计相关滤波器"""
        # 简化实现：使用指数相关模型
        rho = np.exp(-1.0 / self.correlation_length)

        # AR(1) 滤波器
        # y[n] = rho * y[n-1] + sqrt(1-rho^2) * x[n]
        filter_length = 10
        filter_coeff = np.zeros(filter_length)

        for i in range(filter_length):
            filter_coeff[i] = (1 - rho**2) * rho**i

        # 归一化
        filter_coeff /= np.sqrt(np.sum(filter_coeff**2))

        return filter_coeff


class ClutterSuppressor:
    """
    杂波抑制器

    使用MTI和脉冲多普勒技术抑制杂波
    """

    def __init__(
        self,
        num_pulses: int = 64,
        clutter_type: ClutterType = ClutterType.RAYLEIGH
    ):
        """
        Args:
            num_pulses: 处理脉冲数
            clutter_type: 杂波类型
        """
        self.num_pulses = num_pulses
        self.clutter_type = clutter_type

    def suppress_mti(
        self,
        data: np.ndarray,
        mti_order: int = 2
    ) -> np.ndarray:
        """
        MTI杂波抑制

        Args:
            data: 输入数据 [num_pulses, num_range_bins]
            mti_order: MTI阶数 (2或3)

        Returns:
            抑制后的数据
        """
        if mti_order == 2:
            # 双脉冲对消器
            suppressed = np.zeros_like(data)
            suppressed[0, :] = data[0, :]
            suppressed[1:, :] = data[1:, :] - data[:-1, :]
        else:
            # 三脉冲对消器
            suppressed = np.zeros_like(data)
            suppressed[:2, :] = data[:2, :]
            suppressed[2:, :] = data[2:, :] - 2*data[1:-1, :] + data[:-2, :]

        return suppressed

    def suppress_adaptive(
        self,
        data: np.ndarray,
        clutter_estimate: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        自适应杂波抑制

        Args:
            data: 输入数据
            clutter_estimate: 杂波估计 (None则自动估计)

        Returns:
            抑制后的数据
        """
        if clutter_estimate is None:
            # 估计杂波 (简单平均)
            clutter_estimate = np.mean(data, axis=0, keepdims=True)

        # 减去杂波
        suppressed = data - clutter_estimate

        return suppressed


def create_clutter_generator(
    clutter_type: Union[str, ClutterType],
    mean_power: float,
    shape_param: float = 1.0,
    scale_param: float = 1.0,
    correlation_time: float = 0.0
) -> ClutterGenerator:
    """
    创建杂波生成器

    Args:
        clutter_type: 杂波类型
        mean_power: 平均功率
        shape_param: 形状参数
        scale_param: 尺度参数
        correlation_time: 相关时间

    Returns:
        杂波生成器
    """
    if isinstance(clutter_type, str):
        clutter_type = ClutterType(clutter_type.lower())

    params = ClutterParameters(
        clutter_type=clutter_type,
        mean_power=mean_power,
        shape_param=shape_param,
        scale_param=scale_param,
        correlation_time=correlation_time
    )

    return ClutterGenerator(params)


# 便捷函数
def generate_sea_clutter(
    num_samples: int,
    num_pulses: int = 1,
    mean_power: float = 1.0,
    sea_state: int = 3
) -> np.ndarray:
    """
    生成海杂波

    Args:
        num_samples: 距离单元数
        num_pulses: 脉冲数
        mean_power: 平均功率
        sea_state: 海况等级 (1-5)

    Returns:
        海杂波数据
    """
    # 根据海况选择参数
    # 海况越高，形状参数越大，拖尾越重
    shape_params = {1: 1.2, 2: 1.5, 3: 1.8, 4: 2.2, 5: 2.5}
    shape_param = shape_params.get(sea_state, 2.0)

    gen = create_clutter_generator(
        clutter_type=ClutterType.WEIBULL,
        mean_power=mean_power,
        shape_param=shape_param,
        correlation_time=0.1
    )

    return gen.generate_clutter(num_samples, num_pulses)


def generate_land_clutter(
    num_samples: int,
    num_pulses: int = 1,
    mean_power: float = 1.0,
    terrain_type: str = "flat"
) -> np.ndarray:
    """
    生成地杂波

    Args:
        num_samples: 距离单元数
        num_pulses: 脉冲数
        mean_power: 平均功率
        terrain_type: 地形类型

    Returns:
        地杂波数据
    """
    # 根据地形选择参数
    if terrain_type == "mountainous":
        clutter_type = ClutterType.LOGNORMAL
        shape_param = 0.5
    elif terrain_type == "hilly":
        clutter_type = ClutterType.WEIBULL
        shape_param = 1.5
    else:  # flat
        clutter_type = ClutterType.RAYLEIGH
        shape_param = 1.0

    gen = create_clutter_generator(
        clutter_type=clutter_type,
        mean_power=mean_power,
        shape_param=shape_param,
        correlation_time=0.05
    )

    return gen.generate_clutter(num_samples, num_pulses)
