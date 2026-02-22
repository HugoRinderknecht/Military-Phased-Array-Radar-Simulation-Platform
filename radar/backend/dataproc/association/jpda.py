# jpda.py - 联合概率数据关联实现
"""
JPDA (Joint Probabilistic Data Association) - 联合概率数据关联

JPDA处理多目标多测量的数据关联问题：
- 联合关联事件生成
- 关联概率计算
- 加权状态更新
- 杂波和虚警处理
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from itertools import product
import math


@dataclass
class AssociationEvent:
    """关联事件"""
    track_to_measurement: Dict[int, Optional[int]]  # 航迹->测量映射 (None表示未关联)
    probability: float  # 事件概率
    likelihood: float  # 事件似然


class JPDAAssociator:
    """
    联合概率数据关联器

    计算多个航迹与多个测量之间的关联概率
    """

    def __init__(
        self,
        gating_threshold: float = 4.0,
        pd: float = 0.9,
        lambda_c: float = 0.001
    ):
        """
        Args:
            gating_threshold: 波门阈值 (马氏距离)
            pd: 检测概率
            lambda_c: 杂波密度 (单位体积内的杂波数)
        """
        self.gating_threshold = gating_threshold
        self.pd = pd
        self.lambda_c = lambda_c

    def compute_gates(
        self,
        tracks: List[np.ndarray],
        track_covs: List[np.ndarray],
        measurements: List[np.ndarray],
        meas_cov: Optional[np.ndarray] = None
    ) -> List[List[Tuple[int, float]]]:
        """
        计算每个航迹的有效测量

        Args:
            tracks: 航迹状态列表
            track_covs: 航迹协方差列表
            measurements: 测量列表
            meas_cov: 测量噪声协方差

        Returns:
            gates[i]: 航迹i的有效测量列表 [(测量索引, 马氏距离), ...]
        """
        gates = []
        dim = len(measurements[0])

        for i, (track, P_track) in enumerate(zip(tracks, track_covs)):
            valid_measurements = []

            # 预测协方差 + 测量噪声
            if meas_cov is not None:
                S = P_track + meas_cov
            else:
                S = P_track

            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            for j, meas in enumerate(measurements):
                # 新息
                y = meas - track

                # 马氏距离
                mahalanobis = float(np.sqrt(y.T @ S_inv @ y))

                if mahalanobis < self.gating_threshold:
                    valid_measurements.append((j, mahalanobis))

            gates.append(valid_measurements)

        return gates

    def compute_association_probabilities(
        self,
        tracks: List[np.ndarray],
        track_covs: List[np.ndarray],
        measurements: List[np.ndarray],
        meas_cov: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        计算关联概率矩阵

        Args:
            tracks: 航迹状态列表
            track_covs: 航迹协方差列表
            measurements: 测量列表
            meas_cov: 测量噪声协方差

        Returns:
            beta[i, j]: 航迹i与测量j的关联概率
            beta[i, -1]: 航迹i未关联到任何测量的概率
        """
        num_tracks = len(tracks)
        num_meas = len(measurements)

        # 初始化似然矩阵
        likelihoods = np.zeros((num_tracks, num_meas + 1))  # +1 表示未关联

        # 计算每个航迹-测量对的似然
        for i, (track, P_track) in enumerate(zip(tracks, track_covs)):
            # 预测协方差 + 测量噪声
            if meas_cov is not None:
                S = P_track + meas_cov
            else:
                S = P_track

            try:
                S_det = np.linalg.det(S)
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_det = np.linalg.det(S + np.eye(S.shape[0]) * 1e-6)
                S_inv = np.linalg.pinv(S)

            # 未关联似然 (基于杂波)
            likelihoods[i, -1] = self.lambda_c * (1 - self.pd)

            for j, meas in enumerate(measurements):
                y = meas - track
                mahalanobis_sq = float(y.T @ S_inv @ y)

                # 检查是否在波门内
                if mahalanobis_sq < self.gating_threshold**2:
                    # 多变量正态分布
                    exponent = -0.5 * mahalanobis_sq
                    normalization = (2 * np.pi) ** (len(meas) / 2) * np.sqrt(S_det)
                    likelihoods[i, j] = (self.pd / normalization) * np.exp(exponent)
                else:
                    likelihoods[i, j] = 0.0

        # 生成所有可能的关联事件
        events = self._generate_joint_events(num_tracks, num_meas, likelihoods)

        # 计算联合事件的概率
        total_likelihood = 0.0
        for event in events:
            # 计算事件似然
            event_likelihood = 1.0
            for i in range(num_tracks):
                j = event['associations'][i]
                event_likelihood *= likelihoods[i, j]

            # 计算事件概率
            event_prob = event_likelihood * event['feasibility']
            event['probability'] = event_prob
            total_likelihood += event_prob

        # 归一化事件概率
        for event in events:
            if total_likelihood > 0:
                event['probability'] /= total_likelihood

        # 计算边缘关联概率
        beta = np.zeros((num_tracks, num_meas + 1))
        for i in range(num_tracks):
            for event in events:
                j = event['associations'][i]
                beta[i, j] += event['probability']

        return beta

    def _generate_joint_events(
        self,
        num_tracks: int,
        num_meas: int,
        likelihoods: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        生成所有有效的联合关联事件

        Args:
            num_tracks: 航迹数量
            num_meas: 测量数量
            likelihoods: 似然矩阵 [num_tracks, num_meas+1]

        Returns:
            事件列表，每个事件包含关联和可行性标志
        """
        events = []

        # 生成所有可能的关联组合
        # 每个航迹可以选择: 未关联 或 关联到任意测量
        # 但需要确保一个测量最多被一个航迹关联

        def is_valid_association(assignments: List[int]) -> bool:
            """检查关联是否有效 (一个测量只被一个航迹关联)"""
            used_measurements = set()
            for i, j in enumerate(assignments):
                if j < num_meas:  # 不是未关联
                    if j in used_measurements:
                        return False
                    used_measurements.add(j)
            return True

        # 递归生成所有关联
        def generate_events_recursive(
            track_idx: int,
            current_assignments: List[int]
        ) -> None:
            if track_idx == num_tracks:
                # 检查是否有效
                if is_valid_association(current_assignments):
                    # 计算可行性
                    feasibility = 1.0
                    for i, j in enumerate(current_assignments):
                        if j < num_meas and likelihoods[i, j] == 0:
                            feasibility = 0.0
                            break

                    events.append({
                        'associations': list(current_assignments),
                        'feasibility': feasibility
                    })
                return

            # 尝试所有可能的关联
            for j in range(num_meas + 1):  # +1 包括未关联
                current_assignments.append(j)
                generate_events_recursive(track_idx + 1, current_assignments)
                current_assignments.pop()

        # 开始生成
        generate_events_recursive(0, [])

        return events

    def compute_combined_update(
        self,
        track_state: np.ndarray,
        track_cov: np.ndarray,
        measurements: List[np.ndarray],
        beta: np.ndarray,
        meas_cov: Optional[np.ndarray] = None,
        track_idx: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        使用JPDA概率计算组合状态更新

        Args:
            track_state: 航迹状态
            track_cov: 航迹协方差
            measurements: 测量列表
            beta: 关联概率矩阵 [num_tracks, num_meas+1]
            meas_cov: 测量噪声协方差
            track_idx: 当前航迹索引

        Returns:
            (x_updated, P_updated): 更新后的状态和协方差
        """
        num_meas = len(measurements)
        beta_i = beta[track_idx, :]  # 当前航迹的关联概率

        # 预测测量 (在当前状态下)
        z_pred = track_state

        # 组合新息
        v_combined = np.zeros_like(track_state)
        P_combined = np.zeros_like(track_cov)

        for j in range(num_meas + 1):
            if j < num_meas:
                # 关联到测量j
                z_j = measurements[j]
                v_j = z_j - z_pred  # 新息

                # 新息协方差
                if meas_cov is not None:
                    S_j = track_cov + meas_cov
                else:
                    S_j = track_cov

                v_combined += beta_i[j] * v_j

                # Pvv项
                if j < num_meas:
                    Pvv = v_j @ v_j.T
                    P_combined += beta_i[j] * (Pvv - S_j)
            else:
                # 未关联
                # 不贡献新息

        # 状态更新
        x_updated = track_state + v_combined

        # 协方差更新
        if meas_cov is not None:
            P_updated = track_cov - P_combined + meas_cov
        else:
            P_updated = track_cov - P_combined

        # 确保对称正定
        P_updated = 0.5 * (P_updated + P_updated.T)
        P_updated += np.eye(P_updated.shape[0]) * 1e-6

        return x_updated, P_updated

    def associate(
        self,
        tracks: List[np.ndarray],
        track_covs: List[np.ndarray],
        measurements: List[np.ndarray],
        meas_cov: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        执行JPDA关联

        Args:
            tracks: 航迹状态列表
            track_covs: 航迹协方差列表
            measurements: 测量列表
            meas_cov: 测量噪声协方差

        Returns:
            (beta, associations): 关联概率矩阵, 关联列表
        """
        # 计算关联概率
        beta = self.compute_association_probabilities(
            tracks, track_covs, measurements, meas_cov
        )

        # 确定每个航迹最可能的关联
        associations = []
        for i in range(len(tracks)):
            # 找到最大概率的测量
            best_j = np.argmax(beta[i, :-1])  # 排除未关联
            if beta[i, best_j] > beta[i, -1]:  # 比未关联概率高
                associations.append([best_j])
            else:
                associations.append([])  # 未关联

        return beta, associations


class JPDAFTracker:
    """
    JPDA (JPDAF) 跟踪器

    集成了JPDA关联和卡尔曼滤波的完整跟踪器
    """

    def __init__(
        self,
        gating_threshold: float = 4.0,
        pd: float = 0.9,
        lambda_c: float = 0.001,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0
    ):
        """
        Args:
            gating_threshold: 波门阈值
            pd: 检测概率
            lambda_c: 杂波密度
            process_noise: 过程噪声强度
            measurement_noise: 测量噪声强度
        """
        self.jpda = JPDAAssociator(gating_threshold, pd, lambda_c)
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # 跟踪状态
        self.tracks: List[np.ndarray] = []
        self.track_covs: List[np.ndarray] = []
        self.track_ids: List[int] = []
        self.next_id = 1

    def initiate_track(self, measurement: np.ndarray) -> int:
        """起始新航迹"""
        # CV模型状态: [x, y, z, vx, vy, vz]
        x = np.zeros(6)
        x[0:3] = measurement

        # 初始协方差
        P = np.eye(6) * 100.0
        P[0:3, 0:3] = np.eye(3) * self.measurement_noise**2

        self.tracks.append(x)
        self.track_covs.append(P)
        track_id = self.next_id
        self.track_ids.append(track_id)
        self.next_id += 1

        return track_id

    def predict(self, dt: float) -> None:
        """预测所有航迹"""
        for i in range(len(self.tracks)):
            # CV模型预测
            F = np.eye(6)
            F[0:3, 3:6] = np.eye(3) * dt

            x_pred = F @ self.tracks[i]
            P_pred = F @ self.track_covs[i] @ F.T + np.eye(6) * self.process_noise

            self.tracks[i] = x_pred
            self.track_covs[i] = P_pred

    def update(self, measurements: List[np.ndarray]) -> None:
        """使用JPDA更新所有航迹"""
        if not self.tracks:
            return

        # 计算JPDA关联概率
        meas_cov = np.eye(3) * self.measurement_noise**2
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        # 投影状态到测量空间
        track_meas = [H @ x for x in self.tracks]
        track_meas_covs = [H @ P @ H.T for P in self.track_covs]

        beta, _ = self.jpda.associate(
            track_meas, track_meas_covs, measurements, meas_cov
        )

        # 更新每个航迹
        for i in range(len(self.tracks)):
            self.tracks[i], self.track_covs[i] = self.jpda.compute_combined_update(
                self.tracks[i],
                self.track_covs[i],
                measurements,
                beta,
                meas_cov,
                i
            )

    def step(self, measurements: List[np.ndarray], dt: float) -> None:
        """完整步骤"""
        self.predict(dt)
        self.update(measurements)
