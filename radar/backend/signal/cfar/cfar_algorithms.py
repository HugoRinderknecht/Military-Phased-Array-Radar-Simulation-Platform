# cfar_algorithms.py - 完整的CFAR检测算法实现
"""
CFAR (Constant False Alarm Rate) 检测算法完整实现

包括多种CFAR变体:
- CA-CFAR: 单元平均CFAR
- GO-CFAR: 最大选择CFAR
- SO-CFAR: 最小选择CFAR
- OS-CFAR: 有序统计CFAR
- TM-CFAR: 削减平均CFAR
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from enum import Enum
from dataclasses import dataclass


class CFARType(Enum):
    """CFAR类型"""
    CA = "ca"       # Cell Averaging
    GO = "go"       # Greatest Of
    SO = "so"       # Smallest Of
    OS = "os"       # Ordered Statistics
    TM = "tm"       # Trimmed Mean


@dataclass
class CFARDetection:
    """CFAR检测结果"""
    range_idx: int
    doppler_idx: int
    snr: float
    threshold: float
    cell_value: float
    noise_level: float


class BaseCFAR:
    """CFAR检测器基类"""

    def __init__(
        self,
        num_train: int = 20,
        num_guard: int = 2,
        pfa: float = 1e-6
    ):
        """
        Args:
            num_train: 训练单元数 (每侧)
            num_guard: 保护单元数 (每侧)
            pfa: 虚警概率
        """
        self.num_train = num_train
        self.num_guard = num_guard
        self.pfa = pfa
        self._compute_threshold_factor()

    def _compute_threshold_factor(self) -> None:
        """计算门限因子"""
        # 对于指数分布的噪声，门限因子 α = N * (Pfa^(-1/N) - 1)
        # 其中 N = 2 * num_train (两侧训练单元)
        N = 2 * self.num_train
        self.alpha = N * (self.pfa ** (-1.0 / N) - 1.0)

    def _get_training_cells(
        self,
        data: np.ndarray,
        cell_idx: int,
        axis: int = 0
    ) -> List[np.ndarray]:
        """
        获取训练单元

        Args:
            data: 输入数据 (2D: range-Doppler)
            cell_idx: 待检测单元索引
            axis: 处理轴 (0: range, 1: Doppler)

        Returns:
            训练单元列表 [左侧, 右侧]
        """
        num_train = self.num_train
        num_guard = self.num_guard

        if axis == 0:  # Range维
            # 前部训练单元
            start_left = max(0, cell_idx - num_guard - num_train)
            end_left = max(0, cell_idx - num_guard)
            left_cells = data[start_left:end_left, :]

            # 后部训练单元
            start_right = min(data.shape[0], cell_idx + num_guard + 1)
            end_right = min(data.shape[0], cell_idx + num_guard + 1 + num_train)
            right_cells = data[start_right:end_right, :]

        else:  # Doppler维
            # 前部训练单元
            start_left = max(0, cell_idx - num_guard - num_train)
            end_left = max(0, cell_idx - num_guard)
            left_cells = data[:, start_left:end_left]

            # 后部训练单元
            start_right = min(data.shape[1], cell_idx + num_guard + 1)
            end_right = min(data.shape[1], cell_idx + num_guard + 1 + num_train)
            right_cells = data[:, start_right:end_right]

        return [left_cells, right_cells]

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """
        执行CFAR检测

        Args:
            data: 输入数据 (range-Doppler矩阵)
            axis: 滑动窗方向 (0: range, 1: Doppler, 或 2: 2D)
            threshold_offset: 门限偏置 (dB)

        Returns:
            检测列表
        """
        raise NotImplementedError


class CACFAR(BaseCFAR):
    """
    单元平均CFAR (CA-CFAR)

    最基本的CFAR算法，使用周围单元的平均值估计噪声水平
    """

    def _estimate_noise_level(
        self,
        training_cells: List[np.ndarray]
    ) -> float:
        """估计噪声水平 (平均值)"""
        all_cells = np.concatenate([cell.flatten() for cell in training_cells])
        return float(np.mean(all_cells))

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """执行CA-CFAR检测"""
        detections = []

        # 确定处理维度
        if axis == 2 or axis == -1:  # 2D检测
            return self._detect_2d(data, threshold_offset)

        # 1D检测
        if axis == 0:
            num_cells = data.shape[0]
        else:
            num_cells = data.shape[1]

        for i in range(self.num_guard, num_cells - self.num_guard):
            # 获取训练单元
            training_cells = self._get_training_cells(data, i, axis)

            # 估计噪声水平
            noise_level = self._estimate_noise_level(training_cells)

            # 计算门限
            threshold = self.alpha * noise_level

            # 检测单元
            if axis == 0:
                cell_value = data[i, 0] if data.ndim == 2 else data[i]
            else:
                cell_value = data[0, i] if data.ndim == 2 else data[i]

            # 判断是否检测到目标
            if cell_value > threshold:
                # 计算SNR
                snr = 10.0 * np.log10(cell_value / (noise_level + 1e-10))

                if axis == 0:
                    range_idx = i
                    doppler_idx = 0
                else:
                    range_idx = 0
                    doppler_idx = i

                detections.append(CFARDetection(
                    range_idx=range_idx,
                    doppler_idx=doppler_idx,
                    snr=snr,
                    threshold=threshold,
                    cell_value=float(cell_value),
                    noise_level=noise_level
                ))

        return detections

    def _detect_2d(
        self,
        data: np.ndarray,
        threshold_offset: float
    ) -> List[CFARDetection]:
        """2D CA-CFAR检测"""
        detections = []

        # 对每个Doppler单元进行range维CA-CFAR
        for doppler_idx in range(self.num_guard, data.shape[1] - self.num_guard):
            doppler_slice = data[:, doppler_idx]

            for range_idx in range(self.num_guard, len(doppler_slice) - self.num_guard):
                # Range维训练单元
                left_train = doppler_slice[range_idx - self.num_guard - self.num_train:range_idx - self.num_guard]
                right_train = doppler_slice[range_idx + self.num_guard + 1:range_idx + self.num_guard + 1 + self.num_train]

                # Doppler维训练单元
                if doppler_idx >= self.num_guard + self.num_train:
                    up_train = data[doppler_idx - self.num_guard - self.num_train:doppler_idx - self.num_guard, range_idx]
                else:
                    up_train = np.array([])

                if doppler_idx + self.num_guard + self.num_train < data.shape[1]:
                    down_train = data[doppler_idx + self.num_guard + 1:doppler_idx + self.num_guard + 1 + self.num_train, range_idx]
                else:
                    down_train = np.array([])

                # 合并所有训练单元
                all_train = np.concatenate([left_train, right_train, up_train, down_train])

                # 估计噪声
                noise_level = float(np.mean(all_train))

                # 门限
                threshold = self.alpha * noise_level

                # 检测
                cell_value = data[range_idx, doppler_idx]

                if cell_value > threshold:
                    snr = 10.0 * np.log10(cell_value / (noise_level + 1e-10))

                    detections.append(CFARDetection(
                        range_idx=range_idx,
                        doppler_idx=doppler_idx,
                        snr=snr,
                        threshold=threshold,
                        cell_value=float(cell_value),
                        noise_level=noise_level
                    ))

        return detections


class GOCFAR(BaseCFAR):
    """
    最大选择CFAR (GO-CFAR)

    选择两侧训练单元中较大的平均值，适用于杂波边缘环境
    """

    def _estimate_noise_level(
        self,
        training_cells: List[np.ndarray]
    ) -> float:
        """估计噪声水平 (最大平均值)"""
        if len(training_cells) != 2:
            return float(np.mean(np.concatenate([cell.flatten() for cell in training_cells])))

        left_mean = float(np.mean(training_cells[0])) if training_cells[0].size > 0 else 0.0
        right_mean = float(np.mean(training_cells[1])) if training_cells[1].size > 0 else 0.0

        return max(left_mean, right_mean)

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """执行GO-CFAR检测"""
        # 重用CA-CFAR的检测逻辑，但使用不同的噪声估计
        cfar = CACFAR(self.num_train, self.num_guard, self.pfa)
        cfar._estimate_noise_level = self._estimate_noise_level
        return cfar.detect(data, axis, threshold_offset)


class SOCFAR(BaseCFAR):
    """
    最小选择CFAR (SO-CFAR)

    选择两侧训练单元中较小的平均值，适用于多目标情况
    """

    def _estimate_noise_level(
        self,
        training_cells: List[np.ndarray]
    ) -> float:
        """估计噪声水平 (最小平均值)"""
        if len(training_cells) != 2:
            return float(np.mean(np.concatenate([cell.flatten() for cell in training_cells])))

        left_mean = float(np.mean(training_cells[0])) if training_cells[0].size > 0 else float('inf')
        right_mean = float(np.mean(training_cells[1])) if training_cells[1].size > 0 else float('inf')

        return min(left_mean, right_mean)

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """执行SO-CFAR检测"""
        cfar = CACFAR(self.num_train, self.num_guard, self.pfa)
        cfar._estimate_noise_level = self._estimate_noise_level
        return cfar.detect(data, axis, threshold_offset)


class OSCFAR(BaseCFAR):
    """
    有序统计CFAR (OS-CFAR)

    使用训练单元排序后的第k个值作为噪声估计
    对非均匀环境和多目标情况鲁棒
    """

    def __init__(
        self,
        num_train: int = 20,
        num_guard: int = 2,
        pfa: float = 1e-6,
        k_order: Optional[int] = None
    ):
        """
        Args:
            num_train: 训练单元数 (每侧)
            num_guard: 保护单元数 (每侧)
            pfa: 虚警概率
            k_order: 排序索引 (None则使用3N/4)
        """
        super().__init__(num_train, num_guard, pfa)
        self.k_order = k_order
        self._compute_os_threshold_factor()

    def _compute_os_threshold_factor(self) -> None:
        """计算OS-CFAR门限因子"""
        if self.k_order is None:
            self.k_order = int(3 * 2 * self.num_train / 4)

        # OS-CFAR的门限因子需要查表或近似计算
        # 这里使用近似公式
        N = 2 * self.num_train
        k = self.k_order

        # 近似: α ≈ N * (Pfa^(-1/(N-k+1)) - 1) / (scaling_factor)
        scaling_factor = k / N
        self.alpha = (N / scaling_factor) * (self.pfa ** (-1.0 / (N - k + 1)) - 1.0)

    def _estimate_noise_level(
        self,
        training_cells: List[np.ndarray]
    ) -> float:
        """估计噪声水平 (有序统计)"""
        all_cells = np.concatenate([cell.flatten() for cell in training_cells])
        sorted_cells = np.sort(all_cells)

        k = min(self.k_order, len(sorted_cells) - 1)
        return float(sorted_cells[k])

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """执行OS-CFAR检测"""
        cfar = CACFAR(self.num_train, self.num_guard, self.pfa)
        cfar._estimate_noise_level = self._estimate_noise_level
        cfar.alpha = self.alpha
        return cfar.detect(data, axis, threshold_offset)


class TMCFAR(BaseCFAR):
    """
    削减平均CFAR (TM-CFAR)

    剔除训练单元中最高和最低的值后取平均
    对异常值鲁棒
    """

    def __init__(
        self,
        num_train: int = 20,
        num_guard: int = 2,
        pfa: float = 1e-6,
        trim_ratio: float = 0.25
    ):
        """
        Args:
            num_train: 训练单元数 (每侧)
            num_guard: 保护单元数 (每侧)
            pfa: 虚警概率
            trim_ratio: 削减比例 (0-0.5)
        """
        super().__init__(num_train, num_guard, pfa)
        self.trim_ratio = trim_ratio

    def _estimate_noise_level(
        self,
        training_cells: List[np.ndarray]
    ) -> float:
        """估计噪声水平 (削减平均)"""
        all_cells = np.concatenate([cell.flatten() for cell in training_cells])
        sorted_cells = np.sort(all_cells)

        # 剔除最高和最低的值
        n = len(sorted_cells)
        n_trim = int(n * self.trim_ratio)

        if n_trim > 0 and n > 2 * n_trim:
            trimmed_cells = sorted_cells[n_trim:n - n_trim]
        else:
            trimmed_cells = sorted_cells

        return float(np.mean(trimmed_cells))

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """执行TM-CFAR检测"""
        cfar = CACFAR(self.num_train, self.num_guard, self.pfa)
        cfar._estimate_noise_level = self._estimate_noise_level
        return cfar.detect(data, axis, threshold_offset)


class AdaptiveCFAR(BaseCFAR):
    """
    自适应CFAR

    根据局部环境特性自动调整CFAR类型
    """

    def __init__(
        self,
        num_train: int = 20,
        num_guard: int = 2,
        pfa: float = 1e-6
    ):
        super().__init__(num_train, num_guard, pfa)

        # 创建各种CFAR实例
        self.cfar_ca = CACFAR(num_train, num_guard, pfa)
        self.cfar_go = GOCFAR(num_train, num_guard, pfa)
        self.cfar_so = SOCFAR(num_train, num_guard, pfa)
        self.cfar_os = OSCFAR(num_train, num_guard, pfa)
        self.cfar_tm = TMCFAR(num_train, num_guard, pfa)

    def _detect_homogeneity(
        self,
        training_cells: List[np.ndarray]
    ) -> float:
        """
        检测训练单元的均匀性

        Returns:
            0-1之间的值，接近1表示均匀
        """
        all_cells = np.concatenate([cell.flatten() for cell in training_cells])

        # 计算变异系数
        mean = np.mean(all_cells)
        std = np.std(all_cells)

        if mean == 0:
            return 0.0

        cv = std / mean
        homogeneity = 1.0 / (1.0 + cv)

        return float(homogeneity)

    def detect(
        self,
        data: np.ndarray,
        axis: int = 0,
        threshold_offset: float = 0.0
    ) -> List[CFARDetection]:
        """执行自适应CFAR检测"""
        # 根据均匀性选择CFAR类型
        # 这里简化处理：实际应该对每个检测单元分别判断

        # 默认使用CA-CFAR
        return self.cfar_ca.detect(data, axis, threshold_offset)


def create_cfar(
    cfar_type: Union[str, CFARType],
    num_train: int = 20,
    num_guard: int = 2,
    pfa: float = 1e-6,
    **kwargs
) -> BaseCFAR:
    """
    工厂函数：创建指定类型的CFAR检测器

    Args:
        cfar_type: CFAR类型
        num_train: 训练单元数
        num_guard: 保护单元数
        pfa: 虚警概率
        **kwargs: 其他参数

    Returns:
        CFAR检测器实例
    """
    if isinstance(cfar_type, str):
        cfar_type = CFARType(cfar_type.lower())

    if cfar_type == CFARType.CA:
        return CACFAR(num_train, num_guard, pfa)
    elif cfar_type == CFARType.GO:
        return GOCFAR(num_train, num_guard, pfa)
    elif cfar_type == CFARType.SO:
        return SOCFAR(num_train, num_guard, pfa)
    elif cfar_type == CFARType.OS:
        k_order = kwargs.get('k_order', None)
        return OSCFAR(num_train, num_guard, pfa, k_order)
    elif cfar_type == CFARType.TM:
        trim_ratio = kwargs.get('trim_ratio', 0.25)
        return TMCFAR(num_train, num_guard, pfa, trim_ratio)
    else:
        raise ValueError(f"Unknown CFAR type: {cfar_type}")


# 便捷函数
def detect_with_cfar(
    data: np.ndarray,
    cfar_type: Union[str, CFARType] = CFARType.CA,
    num_train: int = 20,
    num_guard: int = 2,
    pfa: float = 1e-6,
    axis: int = 0,
    **kwargs
) -> List[CFARDetection]:
    """
    使用指定CFAR类型进行检测

    Args:
        data: 输入数据
        cfar_type: CFAR类型
        num_train: 训练单元数
        num_guard: 保护单元数
        pfa: 虚警概率
        axis: 处理轴

    Returns:
        检测列表
    """
    cfar = create_cfar(cfar_type, num_train, num_guard, pfa, **kwargs)
    return cfar.detect(data, axis)
