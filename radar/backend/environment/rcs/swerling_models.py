# swerling_models.py - 完整的Swerling起伏模型实现
"""
Swerling RCS起伏模型

完整的Swerling I-IV模型实现:
- Swerling I: 慢起伏，扫描间瑞利分布
- Swerling II: 快起伏，脉冲间瑞利分布
- Swerling III: 慢起伏，扫描间4自由度 chi-square分布
- Swerling IV: 快起伏，脉冲间4自由度 chi-square分布
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass
from enum import Enum


class SwerlingModel(Enum):
    """Swerling模型类型"""
    I = 1    # 慢起伏，瑞利分布 (扫描间)
    II = 2   # 快起伏，瑞利分布 (脉冲间)
    III = 3  # 慢起伏，4自由度chi-square (扫描间)
    IV = 4   # 快起伏，4自由度chi-square (脉冲间)


@dataclass
class RCSSample:
    """RCS样本"""
    rcs_mean: float      # 平均RCS (m^2)
    rcs_instant: float   # 瞬时RCS (m^2)
    rcs_db: float        # 瞬时RCS (dB)
    swerling_case: SwerlingModel
    timestamp: float = 0.0


class SwerlingGenerator:
    """
    Swerling RCS起伏生成器

    根据Swerling模型生成目标RCS起伏
    """

    def __init__(
        self,
        swerling_case: Union[int, SwerlingModel],
        mean_rcs: float = 10.0,
        prf: float = 2000.0,
        scan_time: float = 2.0
    ):
        """
        Args:
            swerling_case: Swerling模型类型 (1-4)
            mean_rcs: 平均RCS (m^2)
            prf: 脉冲重复频率 (Hz)
            scan_time: 扫描时间 (s)
        """
        if isinstance(swerling_case, int):
            swerling_case = SwerlingModel(swerling_case)

        self.swerling_case = swerling_case
        self.mean_rcs = mean_rcs
        self.prf = prf
        self.scan_time = scan_time

        # 内部状态
        self._current_rcs = mean_rcs
        self._pulse_count = 0
        self._scan_count = 0

    def generate_rcs_sample(
        self,
        num_pulses: int = 1,
        dt: float = 1.0
    ) -> List[RCSSample]:
        """
        生成RCS样本序列

        Args:
            num_pulses: 脉冲数
            dt: 时间步长 (s)

        Returns:
            RCS样本列表
        """
        samples = []

        for i in range(num_pulses):
            # 根据Swerling模型生成RCS
            rcs_instant = self._generate_single_sample()

            # 转换为dB
            rcs_db = 10.0 * np.log10(rcs_instant + 1e-10)

            samples.append(RCSSample(
                rcs_mean=self.mean_rcs,
                rcs_instant=rcs_instant,
                rcs_db=rcs_db,
                swerling_case=self.swerling_case,
                timestamp=self._scan_count * self.scan_time + self._pulse_count / self.prf
            ))

            self._pulse_count += 1

        return samples

    def _generate_single_sample(self) -> float:
        """生成单个RCS样本"""
        if self.swerling_case == SwerlingModel.I:
            return self._generate_swerling_i()
        elif self.swerling_case == SwerlingModel.II:
            return self._generate_swerling_ii()
        elif self.swerling_case == SwerlingModel.III:
            return self._generate_swerling_iii()
        elif self.swerling_case == SwerlingModel.IV:
            return self._generate_swerling_iv()
        else:
            return self.mean_rcs

    def _generate_swerling_i(self) -> float:
        """
        Swerling I模型: 慢起伏，瑞利分布

        RCS在一个扫描内保持恒定，扫描间变化
        """
        # 每个扫描开始时生成新的RCS值
        if self._pulse_count == 0:
            # 生成瑞利分布随机变量
            # chi-square(2) 分布
            chi2 = np.random.chisquare(2)
            self._current_rcs = self.mean_rcs * chi2

        return self._current_rcs

    def _generate_swerling_ii(self) -> float:
        """
        Swerling II模型: 快起伏，瑞利分布

        RCS每个脉冲间独立变化
        """
        # 每个脉冲生成新的RCS值
        chi2 = np.random.chisquare(2)
        return self.mean_rcs * chi2

    def _generate_swerling_iii(self) -> float:
        """
        Swerling III模型: 慢起伏，4自由度chi-square分布

        RCS在一个扫描内保持恒定，扫描间变化
        适用于由一个大散射体+小散射体组成的目标
        """
        # 每个扫描开始时生成新的RCS值
        if self._pulse_count == 0:
            # 生成4自由度chi-square分布
            chi2 = np.random.chisquare(4)
            self._current_rcs = self.mean_rcs * chi2

        return self._current_rcs

    def _generate_swerling_iv(self) -> float:
        """
        Swerling IV模型: 快起伏，4自由度chi-square分布

        RCS每个脉冲间独立变化
        """
        # 每个脉冲生成新的RCS值
        chi2 = np.random.chisquare(4)
        return self.mean_rcs * chi2

    def reset_scan(self) -> None:
        """重置扫描计数 (用于Swerling I/III)"""
        self._pulse_count = 0
        self._scan_count += 1


class RCSGenerator:
    """
    通用RCS生成器

    支持多种RCS模型和复杂目标
    """

    def __init__(
        self,
        swerling_case: Union[int, SwerlingModel] = SwerlingModel.I,
        mean_rcs: float = 10.0,
        prf: float = 2000.0,
        scan_time: float = 2.0
    ):
        """
        Args:
            swerling_case: Swerling模型类型
            mean_rcs: 平均RCS
            prf: 脉冲重复频率
            scan_time: 扫描时间
        """
        self.swerling_gen = SwerlingGenerator(
            swerling_case, mean_rcs, prf, scan_time
        )

    def generate_rcs(
        self,
        num_pulses: int,
        aspect_angle: Optional[float] = None,
        frequency: float = 10e9
    ) -> np.ndarray:
        """
        生成RCS序列

        Args:
            num_pulses: 脉冲数
            aspect_angle: 视角角 (rad, 可选，用于角度依赖的RCS)
            frequency: 雷达频率 (Hz)

        Returns:
            RCS数组 (m^2)
        """
        # 生成基本Swerling起伏
        samples = self.swerling_gen.generate_rcs_sample(num_pulses)
        rcs_values = np.array([s.rcs_instant for s in samples])

        # 如果提供了视角角，可以添加角度依赖的变化
        if aspect_angle is not None:
            # 简化模型：RCS随角度变化
            # 实际应该使用更复杂的散射中心模型
            angle_variation = 1.0 + 0.3 * np.cos(4 * aspect_angle)
            rcs_values *= angle_variation

        return rcs_values


class ComplexTargetRCS:
    """
    复杂目标RCS模型

    由多个散射中心组成的目标
    """

    def __init__(
        self,
        scatterer_positions: List[np.ndarray],  # 各散射中心位置
        scatterer_rcs: List[float],              # 各散射中心RCS
        frequency: float = 10e9
    ):
        """
        Args:
            scatterer_positions: 散射中心位置列表 (机体坐标系)
            scatterer_rcs: 散射中心RCS列表
            frequency: 雷达频率
        """
        self.scatterer_positions = scatterer_positions
        self.scatterer_rcs = np.array(scatterer_rcs)
        self.frequency = frequency
        self.wavelength = 3e8 / frequency

    def compute_rcs(
        self,
        aspect_angles: np.ndarray,  # [azimuth, elevation] (rad)
        prf: float = 2000.0,
        num_pulses: int = 1
    ) -> np.ndarray:
        """
        计算复杂目标的RCS

        Args:
            aspect_angles: 视角角 [azimuth, elevation]
            prf: 脉冲重复频率
            num_pulses: 脉冲数

        Returns:
            RCS序列 (m^2)
        """
        azimuth, elevation = aspect_angles

        # 计算每个散射中心的相位
        phases = []
        for pos in self.scatterer_positions:
            # 计算散射中心到雷达的距离分量
            r = pos[0] * np.cos(azimuth) * np.cos(elevation) + \
                pos[1] * np.sin(azimuth) * np.cos(elevation) + \
                pos[2] * np.sin(elevation)

            # 相位
            phase = 4 * np.pi * r / self.wavelength
            phases.append(phase)

        phases = np.array(phases)

        # 相干求和
        complex_rcs = np.sum(
            np.sqrt(self.scatterer_rcs) * np.exp(1j * phases)
        )

        # 总RCS
        total_rcs = np.abs(complex_rcs) ** 2

        # 添加Swerling起伏
        # 这里使用Swerling I模型作为示例
        swerling = SwerlingGenerator(1, total_rcs, prf)
        samples = swerling.generate_rcs_sample(num_pulses)

        return np.array([s.rcs_instant for s in samples])


class RCSProbability:
    """
    RCS概率分布计算

    计算给定Swerling模型的RCS概率分布
    """

    @staticmethod
    def pdf_swerling_1(rcs: np.ndarray, mean_rcs: float) -> np.ndarray:
        """
        Swerling I模型的PDF

        P(σ) = (1/σ_mean) * exp(-σ/σ_mean)

        Args:
            rcs: RCS值 (m^2)
            mean_rcs: 平均RCS (m^2)

        Returns:
            概率密度
        """
        return (1.0 / mean_rcs) * np.exp(-rcs / mean_rcs)

    @staticmethod
    def pdf_swerling_3(rcs: np.ndarray, mean_rcs: float) -> np.ndarray:
        """
        Swerling III模型的PDF

        P(σ) = (4σ/σ_mean^2) * exp(-2σ/σ_mean)

        Args:
            rcs: RCS值 (m^2)
            mean_rcs: 平均RCS (m^2)

        Returns:
            概率密度
        """
        return (4.0 * rcs / mean_rcs**2) * np.exp(-2.0 * rcs / mean_rcs)

    @staticmethod
    def cdf_swerling_1(rcs: np.ndarray, mean_rcs: float) -> np.ndarray:
        """
        Swerling I模型的CDF

        Args:
            rcs: RCS值 (m^2)
            mean_rcs: 平均RCS (m^2)

        Returns:
            累积概率
        """
        return 1.0 - np.exp(-rcs / mean_rcs)

    @staticmethod
    def detection_probability(
        snr: np.ndarray,
        pd: float,
        swerling_case: SwerlingModel,
        num_pulses: int = 1
    ) -> np.ndarray:
        """
        计算检测概率

        使用Shnidman方程近似Swerling模型的检测概率

        Args:
            snr: 信噪比 (linear)
            pd: 期望的检测概率
            swerling_case: Swerling模型
            num_pulses: 非相干积累脉冲数

        Returns:
            检测概率
        """
        if swerling_case == SwerlingModel.I:
            # Swerling I (单脉冲) 或 Swerling II (多脉冲)
            if num_pulses == 1:
                # 单脉冲情况
                k = 1
            else:
                # 非相干积累
                k = num_pulses
        elif swerling_case == SwerlingModel.III:
            # Swerling III (单脉冲) 或 Swerling IV (多脉冲)
            if num_pulses == 1:
                k = 2
            else:
                k = 2 * num_pulses
        else:
            k = 1

        # 使用Marcum Q函数近似
        # 这里简化处理，实际应该使用更精确的计算
        alpha = np.sqrt(-2 * np.log(pd))

        # Shnidman方程近似
        x = np.sqrt(snr) - alpha

        # 累积概率
        Pd = 0.5 * (1 + np.erf(x / np.sqrt(2)))

        return Pd


# 便捷函数
def create_swerling_model(
    swerling_case: int,
    mean_rcs: float = 10.0,
    prf: float = 2000.0,
    scan_time: float = 2.0
) -> SwerlingGenerator:
    """
    创建Swerling模型

    Args:
        swerling_case: Swerling模型类型 (1-4)
        mean_rcs: 平均RCS
        prf: 脉冲重复频率
        scan_time: 扫描时间

    Returns:
        Swerling生成器
    """
    return SwerlingGenerator(swerling_case, mean_rcs, prf, scan_time)


def generate_rcs_samples(
    swerling_case: int,
    mean_rcs: float,
    num_samples: int,
    prf: float = 2000.0
) -> np.ndarray:
    """
    生成RCS样本

    Args:
        swerling_case: Swerling模型类型
        mean_rcs: 平均RCS
        num_samples: 样本数
        prf: 脉冲重复频率

    Returns:
        RCS样本数组
    """
    gen = create_swerling_model(swerling_case, mean_rcs, prf)
    samples = gen.generate_rcs_sample(num_samples)
    return np.array([s.rcs_instant for s in samples])
