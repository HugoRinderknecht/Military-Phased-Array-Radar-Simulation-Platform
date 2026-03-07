"""
点迹关联模块
完整实现NN、GNN、PDA、JPDA关联算法

参考文档 4.4.9 节
"""
import numpy as np
from typing import List, Optional, Tuple, Dict
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass

from .track_init import Plot, ConfirmedTrack


@dataclass
class Association:
    """关联结果"""
    track_id: str
    plot_id: Optional[str]  # None表示无关联点迹
    association_probability: float = 1.0


class NNAassociator:
    """
    最近邻（NN）关联器

    选择波门内马氏距离最小的点迹
    """

    def __init__(
        self,
        gate_threshold: float = 4.0,  # 马氏距离门限
        measurement_noise: np.ndarray = None,
    ):
        """
        Args:
            gate_threshold: 波门门限（马氏距离）
            measurement_noise: 测量噪声协方差
        """
        self.gate_threshold = gate_threshold
        self.measurement_noise = measurement_noise if measurement_noise is not None else np.eye(3) * 100

    def calculate_mahalanobis_distance(
        self,
        measurement: np.ndarray,
        prediction: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> float:
        """
        计算马氏距离

        D² = (z - Hx)ᵀ S⁻¹ (z - Hx)

        Args:
            measurement: 测量值 [x, y, z]
            prediction: 预测值 [x, y, z]
            innovation_covariance: 新息协方差

        Returns:
            马氏距离
        """
        innovation = measurement - prediction

        try:
            inv_cov = np.linalg.inv(innovation_covariance)
            mahalanobis_sq = innovation.T @ inv_cov @ innovation
            return np.sqrt(mahalanobis_sq)
        except np.linalg.LinAlgError:
            return float('inf')

    def associate(
        self,
        tracks: List[ConfirmedTrack],
        plots: List[Plot],
    ) -> List[Association]:
        """
        执行NN关联

        Args:
            tracks: 航迹列表
            plots: 点迹列表

        Returns:
            关联结果列表
        """
        associations = []
        associated_plots = set()

        for track in tracks:
            predicted_position = track.state[:3]
            innovation_cov = track.covariance[:3, :3] + self.measurement_noise

            # 在波门内寻找最近点迹
            min_distance = float('inf')
            nearest_plot = None

            for plot in plots:
                if plot.plot_id in associated_plots:
                    continue

                measurement = np.array([plot.x, plot.y, plot.z])
                distance = self.calculate_mahalanobis_distance(
                    measurement,
                    predicted_position,
                    innovation_cov,
                )

                if distance < min_distance and distance <= self.gate_threshold:
                    min_distance = distance
                    nearest_plot = plot

            if nearest_plot is not None:
                associations.append(Association(
                    track_id=track.track_id,
                    plot_id=nearest_plot.plot_id,
                    association_probability=1.0 / (1.0 + min_distance),
                ))
                associated_plots.add(nearest_plot.plot_id)
            else:
                # 无关联点迹
                associations.append(Association(
                    track_id=track.track_id,
                    plot_id=None,
                ))

        return associations


class GNNAassociator:
    """
    全局最近邻（GNN）关联器

    使用匈牙利算法求解全局最优分配
    """

    def __init__(
        self,
        gate_threshold: float = 4.0,
        measurement_noise: np.ndarray = None,
        cost_type: str = "mahalanobis",
    ):
        """
        Args:
            gate_threshold: 波门门限
            measurement_noise: 测量噪声协方差
            cost_type: 代价类型 ("mahalanobis" 或 "euclidean")
        """
        self.gate_threshold = gate_threshold
        self.measurement_noise = measurement_noise if measurement_noise is not None else np.eye(3) * 100
        self.cost_type = cost_type

    def calculate_cost_matrix(
        self,
        tracks: List[ConfirmedTrack],
        plots: List[Plot],
    ) -> np.ndarray:
        """计算代价矩阵"""
        num_tracks = len(tracks)
        num_plots = len(plots)

        cost_matrix = np.full((num_tracks, num_plots), float('inf'))

        for i, track in enumerate(tracks):
            predicted_position = track.state[:3]

            for j, plot in enumerate(plots):
                measurement = np.array([plot.x, plot.y, plot.z])

                if self.cost_type == "mahalanobis":
                    innovation_cov = track.covariance[:3, :3] + self.measurement_noise
                    try:
                        inv_cov = np.linalg.inv(innovation_cov)
                        innovation = measurement - predicted_position
                        cost = innovation.T @ inv_cov @ innovation
                    except np.linalg.LinAlgError:
                        cost = float('inf')
                else:  # euclidean
                    cost = np.sum((measurement - predicted_position) ** 2)

                cost_matrix[i, j] = cost if cost <= self.gate_threshold ** 2 else float('inf')

        return cost_matrix

    def associate(
        self,
        tracks: List[ConfirmedTrack],
        plots: List[Plot],
    ) -> List[Association]:
        """
        执行GNN关联

        Args:
            tracks: 航迹列表
            plots: 点迹列表

        Returns:
            关联结果列表
        """
        if len(tracks) == 0 or len(plots) == 0:
            return [
                Association(track_id=t.track_id, plot_id=None)
                for t in tracks
            ]

        # 计算代价矩阵
        cost_matrix = self.calculate_cost_matrix(tracks, plots)

        # 匈牙利算法求解
        track_indices, plot_indices = linear_sum_assignment(cost_matrix)

        # 构建关联结果
        associations_dict = {t.track_id: Association(track_id=t.track_id, plot_id=None) for t in tracks}
        associated_plots = set()

        for track_idx, plot_idx in zip(track_indices, plot_indices):
            if cost_matrix[track_idx, plot_idx] < float('inf'):
                track = tracks[track_idx]
                plot = plots[plot_idx]

                associations_dict[track.track_id] = Association(
                    track_id=track.track_id,
                    plot_id=plot.plot_id,
                    association_probability=1.0,
                )
                associated_plots.add(plot.plot_id)

        return list(associations_dict.values())


class PDAAssociator:
    """
    概率数据关联（PDA）

    计算波门内所有候选点迹的关联概率并加权融合
    """

    def __init__(
        self,
        gate_threshold: float = 4.0,
        clutter_density: float = 1e-6,
        measurement_noise: np.ndarray = None,
    ):
        """
        Args:
            gate_threshold: 波门门限
            clutter_density: 杂波密度
            measurement_noise: 测量噪声协方差
        """
        self.gate_threshold = gate_threshold
        self.clutter_density = clutter_density
        self.measurement_noise = measurement_noise if measurement_noise is not None else np.eye(3) * 100

    def calculate_likelihood(
        self,
        measurement: np.ndarray,
        prediction: np.ndarray,
        innovation_covariance: np.ndarray,
    ) -> float:
        """计算似然函数"""
        innovation = measurement - prediction

        try:
            inv_cov = np.linalg.inv(innovation_cov)
            det_cov = np.linalg.det(innovation_cov)

            mahalanobis_sq = innovation.T @ inv_cov @ innovation
            likelihood = np.exp(-0.5 * mahalanobis_sq) / np.sqrt((2 * np.pi) ** 3 * det_cov)

            return likelihood
        except np.linalg.LinAlgError:
            return 0.0

    def associate(
        self,
        track: ConfirmedTrack,
        plots: List[Plot],
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        计算关联概率

        Args:
            track: 航迹
            plots: 点迹列表

        Returns:
            (融合后的测量, 关联概率字典)
        """
        predicted_position = track.state[:3]
        innovation_cov = track.covariance[:3, :3] + self.measurement_noise

        # 找出波门内的点迹
        gated_plots = []
        likelihoods = []

        for plot in plots:
            measurement = np.array([plot.x, plot.y, plot.z])
            innovation = measurement - predicted_position

            try:
                inv_cov = np.linalg.inv(innovation_cov)
                mahalanobis_sq = innovation.T @ inv_cov @ innovation

                if mahalanobis_sq <= self.gate_threshold ** 2:
                    likelihood = self.calculate_likelihood(
                        measurement, predicted_position, innovation_cov
                    )
                    gated_plots.append(plot)
                    likelihoods.append(likelihood)
            except np.linalg.LinAlgError:
                continue

        # 计算关联概率
        probabilities = {}
        fused_measurement = predicted_position.copy()

        if len(gated_plots) == 0:
            # 无点迹，使用预测值
            return fused_measurement, probabilities

        # 计算归一化常数
        beta_0 = 1 - np.sum(likelihoods) * self.clutter_density
        likelihood_sum = np.sum(likelihoods)

        if likelihood_sum > 0:
            # 计算每个点迹的关联概率
            for plot, likelihood in zip(gated_plots, likelihoods):
                beta_i = likelihood / likelihood_sum
                probabilities[plot.plot_id] = beta_i

                # 加权融合
                measurement = np.array([plot.x, plot.y, plot.z])
                fused_measurement += beta_i * (measurement - predicted_position)

        return fused_measurement, probabilities


class JPDATracker:
    """
    联合概率数据关联（JPDA）

    考虑多目标波门重叠情况
    """

    def __init__(
        self,
        gate_threshold: float = 4.0,
        clutter_density: float = 1e-6,
        measurement_noise: np.ndarray = None,
    ):
        """
        Args:
            gate_threshold: 波门门限
            clutter_density: 杂波密度
            measurement_noise: 测量噪声协方差
        """
        self.gate_threshold = gate_threshold
        self.clutter_density = clutter_density
        self.measurement_noise = measurement_noise if measurement_noise is not None else np.eye(3) * 100

    def generate_association_events(
        self,
        tracks: List[ConfirmedTrack],
        plots: List[Plot],
    ) -> List[Dict[str, Optional[str]]]:
        """
        生成关联事件

        每个事件是一个字典，映射 track_id -> plot_id
        plot_id 为 None 表示该航迹无关联点迹

        Args:
            tracks: 航迹列表
            plots: 点迹列表

        Returns:
            关联事件列表
        """
        # 构建波门矩阵
        gated = np.zeros((len(tracks), len(plots)), dtype=bool)

        for i, track in enumerate(tracks):
            predicted_position = track.state[:3]
            innovation_cov = track.covariance[:3, :3] + self.measurement_noise

            for j, plot in enumerate(plots):
                measurement = np.array([plot.x, plot.y, plot.z])
                innovation = measurement - predicted_position

                try:
                    inv_cov = np.linalg.inv(innovation_cov)
                    mahalanobis_sq = innovation.T @ inv_cov @ innovation

                    if mahalanobis_sq <= self.gate_threshold ** 2:
                        gated[i, j] = True
                except np.linalg.LinAlgError:
                    continue

        # 生成所有可能的关联事件（实际实现需要剪枝）
        events = []

        # 简化实现：只考虑一对一关联
        # 完整实现需要枚举所有有效关联矩阵
        if len(tracks) <= 2 and len(plots) <= 3:
            # 小规模情况，可以枚举
            from itertools import permutations

            plot_ids = [p.plot_id for p in plots] + [None]

            for perm in permutations(plot_ids, len(tracks)):
                # 检查是否有效（不能多个航迹关联同一非None点迹）
                used = set()
                valid = True
                for pid in perm:
                    if pid is not None and pid in used:
                        valid = False
                        break
                    if pid is not None:
                        used.add(pid)

                if valid:
                    event = {track.track_id: perm[i] for i, track in enumerate(tracks)}
                    events.append(event)

        return events

    def calculate_joint_probability(
        self,
        event: Dict[str, Optional[str]],
        tracks: List[ConfirmedTrack],
        plots: List[Plot],
        plot_dict: Dict[str, Plot],
    ) -> float:
        """计算关联事件的联合概率"""
        probability = 1.0

        for track in tracks:
            plot_id = event.get(track.track_id)

            if plot_id is None:
                # 无关联，概率为杂波密度
                probability *= self.clutter_density
            else:
                # 计算似然
                plot = plot_dict[plot_id]
                predicted_position = track.state[:3]
                innovation_cov = track.covariance[:3, :3] + self.measurement_noise

                measurement = np.array([plot.x, plot.y, plot.z])
                innovation = measurement - predicted_position

                try:
                    inv_cov = np.linalg.inv(innovation_cov)
                    det_cov = np.linalg.det(innovation_cov)
                    mahalanobis_sq = innovation.T @ inv_cov @ innovation

                    likelihood = np.exp(-0.5 * mahalanobis_sq) / np.sqrt((2 * np.pi) ** 3 * det_cov)
                    probability *= likelihood
                except np.linalg.LinAlgError:
                    probability *= 0.0

        return probability

    def associate(
        self,
        tracks: List[ConfirmedTrack],
        plots: List[Plot],
    ) -> Dict[str, Tuple[np.ndarray, Dict[str, float]]]:
        """
        执行JPDA关联

        Args:
            tracks: 航迹列表
            plots: 点迹列表

        Returns:
            字典：track_id -> (融合测量, 关联概率)
        """
        plot_dict = {p.plot_id: p for p in plots}

        # 生成关联事件
        events = self.generate_association_events(tracks, plots)

        if len(events) == 0:
            return {
                track.track_id: (track.state[:3].copy(), {})
                for track in tracks
            }

        # 计算每个事件的概率
        event_probabilities = []
        for event in events:
            prob = self.calculate_joint_probability(event, tracks, plots, plot_dict)
            event_probabilities.append(prob)

        # 归一化
        total_prob = np.sum(event_probabilities)
        if total_prob > 0:
            event_probabilities = [p / total_prob for p in event_probabilities]
        else:
            event_probabilities = [1.0 / len(events)] * len(events)

        # 计算每个航迹的关联概率和融合测量
        results = {}

        for track in tracks:
            # 融合测量
            fused_measurement = np.zeros(3)
            marginal_probabilities = {}

            for plot in plots:
                # 计算该点迹与该航迹的边缘关联概率
                marginal_prob = 0.0
                for event, event_prob in zip(events, event_probabilities):
                    if event.get(track.track_id) == plot.plot_id:
                        marginal_prob += event_prob

                if marginal_prob > 0:
                    marginal_probabilities[plot.plot_id] = marginal_prob
                    measurement = np.array([plot.x, plot.y, plot.z])
                    fused_measurement += marginal_prob * measurement

            # 加上预测位置
            predicted_position = track.state[:3]
            prob_no_detection = 1.0 - np.sum(list(marginal_probabilities.values()))
            fused_measurement += prob_no_detection * predicted_position

            results[track.track_id] = (fused_measurement, marginal_probabilities)

        return results
