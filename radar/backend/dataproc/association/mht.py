# mht.py - 多假设跟踪实现
"""
MHT (Multiple Hypothesis Tracking) - 多假设跟踪

MHT维护多个跟踪假设，延迟硬决策：
- 假设树生成
- 假设剪枝
- N-scan回溯
- 假设评分
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import heapq


class HypothesisStatus(Enum):
    """假设状态"""
    ACTIVE = "active"
    PRUNED = "pruned"
    MERGED = "merged"


@dataclass
class TrackHypothesis:
    """单个航迹假设"""
    state: np.ndarray  # 状态向量
    covariance: np.ndarray  # 协方差矩阵
    score: float  # 对数似然比分数
    history: List[int]  # 关联历史 [scan_idx, meas_idx, ...]
    parent: Optional['TrackHypothesis'] = None
    id: int = 0

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, TrackHypothesis):
            return self.id == other.id
        return False


@dataclass
class GlobalHypothesis:
    """全局假设 - 一组兼容的航迹假设"""
    track_hypotheses: List[TrackHypothesis]  # 航迹假设列表
    score: float  # 总分数 (对数似然比和)
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    timestamp: float = 0.0

    def __lt__(self, other):
        """用于堆排序 (分数高的在前)"""
        return self.score > other.score


@dataclass
class Measurement:
    """测量"""
    position: np.ndarray
    covariance: np.ndarray
    timestamp: float
    scan_index: int
    id: int = 0


class MHTTracker:
    """
    多假设跟踪器

    维护多个跟踪假设并智能剪枝
    """

    def __init__(
        self,
        max_hypotheses: int = 100,
        pruning_threshold: float = 10.0,
        merging_threshold: float = 3.0,
        pd: float = 0.9,
        lambda_c: float = 0.001,
        process_noise: float = 1.0,
        measurement_noise: float = 10.0
    ):
        """
        Args:
            max_hypotheses: 最大保留假设数
            pruning_threshold: 剪枝阈值 (对数似然比差)
            merging_threshold: 合并阈值 (马氏距离)
            pd: 检测概率
            lambda_c: 杂波密度
            process_noise: 过程噪声强度
            measurement_noise: 测量噪声强度
        """
        self.max_hypotheses = max_hypotheses
        self.pruning_threshold = pruning_threshold
        self.merging_threshold = merging_threshold
        self.pd = pd
        self.lambda_c = lambda_c
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # 数据存储
        self.global_hypotheses: List[GlobalHypothesis] = []
        self.measurements: List[List[Measurement]] = []  # 按扫描存储
        self.track_id_counter = 1
        self.hypothesis_id_counter = 1

        # 当前扫描索引
        self.current_scan = 0

    def _generate_new_hypotheses(
        self,
        hypothesis: GlobalHypothesis,
        new_measurements: List[Measurement]
    ) -> List[GlobalHypothesis]:
        """
        为给定全局假设生成新假设

        Args:
            hypothesis: 当前全局假设
            new_measurements: 新测量列表

        Returns:
            新生成的全局假设列表
        """
        new_hypotheses = []
        num_tracks = len(hypothesis.track_hypotheses)
        num_meas = len(new_measurements)

        # 为每个航迹生成可能的关联
        # 包括: 关联到任意测量, 未关联, 新航迹起始

        def generate_assignments(
            track_idx: int,
            current_assignments: List[Optional[int]],
            used_measurements: Set[int],
            new_tracks: List[Tuple[int, np.ndarray]]
        ) -> None:
            """递归生成所有有效的关联分配"""
            if track_idx == num_tracks:
                # 生成分配完成，创建新假设
                new_track_hyps = []

                # 更新现有航迹
                for i, track_hyp in enumerate(hypothesis.track_hypotheses):
                    meas_idx = current_assignments[i]

                    if meas_idx is not None:
                        # 关联到测量
                        meas = new_measurements[meas_idx]
                        updated_state, updated_cov, score_delta = self._update_track(
                            track_hyp, meas
                        )

                        new_hyp = TrackHypothesis(
                            state=updated_state,
                            covariance=updated_cov,
                            score=track_hyp.score + score_delta,
                            history=track_hyp.history + [meas_idx],
                            parent=track_hyp,
                            id=self.hypothesis_id_counter
                        )
                        self.hypothesis_id_counter += 1
                    else:
                        # 未关联
                        score_delta = np.log(1 - self.pd)  # 漏检惩罚
                        new_hyp = TrackHypothesis(
                            state=track_hyp.state.copy(),
                            covariance=track_hyp.covariance.copy(),
                            score=track_hyp.score + score_delta,
                            history=track_hyp.history + [-1],
                            parent=track_hyp,
                            id=self.hypothesis_id_counter
                        )
                        self.hypothesis_id_counter += 1

                    new_track_hyps.append(new_hyp)

                # 添加新航迹
                for meas_idx, state, cov in new_tracks:
                    new_track_hyp = TrackHypothesis(
                        state=state,
                        covariance=cov,
                        score=np.log(self.lambda_c / len(new_measurements)),  # 新航迹先验
                        history=[meas_idx],
                        parent=None,
                        id=self.hypothesis_id_counter
                    )
                    self.hypothesis_id_counter += 1
                    new_track_hyps.append(new_track_hyp)

                # 计算总分
                total_score = sum(hyp.score for hyp in new_track_hyps)

                new_global_hyp = GlobalHypothesis(
                    track_hypotheses=new_track_hyps,
                    score=total_score,
                    status=HypothesisStatus.ACTIVE,
                    timestamp=self.current_scan
                )

                new_hypotheses.append(new_global_hyp)
                return

            # 尝试所有可能的关联
            track_hyp = hypothesis.track_hypotheses[track_idx]

            # 选项1: 未关联
            current_assignments.append(None)
            generate_assignments(track_idx + 1, current_assignments, used_measurements, new_tracks)
            current_assignments.pop()

            # 选项2: 关联到任意未使用的测量
            for meas_idx in range(num_meas):
                if meas_idx not in used_measurements:
                    # 检查是否在波门内
                    if self._in_gate(track_hyp, new_measurements[meas_idx]):
                        current_assignments.append(meas_idx)
                        used_measurements.add(meas_idx)
                        generate_assignments(track_idx + 1, current_assignments, used_measurements, new_tracks)
                        used_measurements.remove(meas_idx)
                        current_assignments.pop()

            # 选项3: 起始新航迹 (仅对第一个航迹循环)
            if track_idx == 0:
                for meas_idx in range(num_meas):
                    if meas_idx not in used_measurements:
                        meas = new_measurements[meas_idx]
                        # 初始化新航迹
                        new_state, new_cov = self._init_track(meas)
                        new_tracks.append((meas_idx, new_state, new_cov))

                        used_measurements.add(meas_idx)
                        generate_assignments(track_idx + 1, current_assignments, used_measurements, new_tracks)
                        used_measurements.remove(meas_idx)
                        new_tracks.pop()

        # 开始生成
        generate_assignments(0, [], set(), [])

        return new_hypotheses

    def _in_gate(self, track_hyp: TrackHypothesis, measurement: Measurement) -> bool:
        """检查测量是否在波门内"""
        # 预测测量
        z_pred = track_hyp.state[0:3]  # 假设前3个是位置

        # 新息
        y = measurement.position - z_pred

        # 新息协方差
        S = track_hyp.covariance[0:3, 0:3] + measurement.covariance

        # 马氏距离
        try:
            mahalanobis_sq = float(y.T @ np.linalg.inv(S) @ y)
            return mahalanobis_sq < 16.0  # 4-sigma波门
        except:
            return True

    def _update_track(
        self,
        track_hyp: TrackHypothesis,
        measurement: Measurement
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """更新航迹"""
        # 卡尔曼更新
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 2] = 1.0

        # 预测测量
        z_pred = H @ track_hyp.state

        # 新息
        y = measurement.position - z_pred

        # 新息协方差
        S = H @ track_hyp.covariance @ H.T + measurement.covariance

        # 卡尔曼增益
        K = track_hyp.covariance @ H.T @ np.linalg.inv(S)

        # 状态更新
        x_updated = track_hyp.state + K @ y

        # 协方差更新
        P_updated = track_hyp.covariance - K @ S @ K.T

        # 计算似然
        try:
            S_det = np.linalg.det(S)
            S_inv = np.linalg.inv(S)
            mahalanobis_sq = float(y.T @ S_inv @ y)

            exponent = -0.5 * mahalanobis_sq
            normalization = (2 * np.pi) ** (1.5) * np.sqrt(S_det)
            likelihood = (self.pd / normalization) * np.exp(exponent)
        except:
            likelihood = 1e-10

        # 对数似然比
        log_likelihood_ratio = np.log(likelihood / self.lambda_c)

        return x_updated, P_updated, log_likelihood_ratio

    def _init_track(self, measurement: Measurement) -> Tuple[np.ndarray, np.ndarray]:
        """初始化新航迹"""
        # CV模型状态: [x, y, z, vx, vy, vz]
        x = np.zeros(6)
        x[0:3] = measurement.position

        # 初始协方差
        P = np.eye(6) * 100.0
        P[0:3, 0:3] = measurement.covariance

        return x, P

    def _predict_track(self, state: np.ndarray, covariance: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """预测航迹"""
        # CV模型
        F = np.eye(6)
        F[0:3, 3:6] = np.eye(3) * dt

        Q = np.eye(6) * self.process_noise

        x_pred = F @ state
        P_pred = F @ covariance @ F.T + Q

        return x_pred, P_pred

    def _prune_hypotheses(self) -> None:
        """剪枝低概率假设"""
        if len(self.global_hypotheses) <= self.max_hypotheses:
            return

        # 按分数排序
        self.global_hypotheses.sort(key=lambda h: h.score, reverse=True)

        # 找到最佳假设的分数
        best_score = self.global_hypotheses[0].score

        # 剪枝低于阈值的假设
        pruned = []
        for hyp in self.global_hypotheses:
            if hyp.score < best_score - self.pruning_threshold:
                hyp.status = HypothesisStatus.PRUNED
                pruned.append(hyp)

        # 移除剪枝的假设
        self.global_hypotheses = [h for h in self.global_hypotheses if h.status == HypothesisStatus.ACTIVE]

        # 确保不超过最大数量
        if len(self.global_hypotheses) > self.max_hypotheses:
            self.global_hypotheses = self.global_hypotheses[:self.max_hypotheses]

    def _merge_hypotheses(self) -> None:
        """合并相似假设"""
        if len(self.global_hypotheses) < 2:
            return

        merged = []
        used = set()

        for i, hyp1 in enumerate(self.global_hypotheses):
            if i in used:
                continue

            merged_hyp = hyp1
            used.add(i)

            # 查找相似假设
            for j, hyp2 in enumerate(self.global_hypotheses[i+1:], start=i+1):
                if j in used:
                    continue

                # 检查是否相似
                if self._are_similar(hyp1, hyp2):
                    # 合并
                    merged_hyp = self._merge_two(merged_hyp, hyp2)
                    used.add(j)
                    hyp2.status = HypothesisStatus.MERGED

            merged.append(merged_hyp)

        self.global_hypotheses = merged

    def _are_similar(self, hyp1: GlobalHypothesis, hyp2: GlobalHypothesis) -> bool:
        """检查两个假设是否相似"""
        if len(hyp1.track_hypotheses) != len(hyp2.track_hypotheses):
            return False

        # 比较每个航迹
        for t1, t2 in zip(hyp1.track_hypotheses, hyp2.track_hypotheses):
            # 计算状态之间的马氏距离
            diff = t1.state - t2.state
            P_avg = (t1.covariance + t2.covariance) / 2

            try:
                mahalanobis_sq = float(diff.T @ np.linalg.inv(P_avg) @ diff)
                if mahalanobis_sq > self.merging_threshold**2:
                    return False
            except:
                return False

        return True

    def _merge_two(self, hyp1: GlobalHypothesis, hyp2: GlobalHypothesis) -> GlobalHypothesis:
        """合并两个假设"""
        # 简单合并: 选择分数高的
        if hyp1.score >= hyp2.score:
            return hyp1
        else:
            return hyp2

    def process_scan(self, measurements: List[Measurement], dt: float) -> None:
        """
        处理新扫描

        Args:
            measurements: 测量列表
            dt: 时间步长
        """
        self.current_scan += 1
        self.measurements.append(measurements)

        # 如果没有假设，创建初始假设
        if not self.global_hypotheses:
            # 为每个测量创建一个新航迹假设
            for meas in measurements:
                new_state, new_cov = self._init_track(meas)
                track_hyp = TrackHypothesis(
                    state=new_state,
                    covariance=new_cov,
                    score=np.log(self.lambda_c / len(measurements)),
                    history=[meas.id],
                    id=self.hypothesis_id_counter
                )
                self.hypothesis_id_counter += 1

                global_hyp = GlobalHypothesis(
                    track_hypotheses=[track_hyp],
                    score=track_hyp.score,
                    status=HypothesisStatus.ACTIVE,
                    timestamp=self.current_scan
                )

                self.global_hypotheses.append(global_hyp)
            return

        # 生成新假设
        new_hypotheses = []
        for hyp in self.global_hypotheses:
            if hyp.status == HypothesisStatus.ACTIVE:
                # 预测所有航迹
                for track_hyp in hyp.track_hypotheses:
                    track_hyp.state, track_hyp.covariance = self._predict_track(
                        track_hyp.state, track_hyp.covariance, dt
                    )

                # 生成新假设
                generated = self._generate_new_hypotheses(hyp, measurements)
                new_hypotheses.extend(generated)

        # 更新全局假设列表
        self.global_hypotheses = new_hypotheses

        # 剪枝
        self._prune_hypotheses()

        # 合并
        self._merge_hypotheses()

    def get_best_hypothesis(self) -> Optional[GlobalHypothesis]:
        """获取最佳假设"""
        if not self.global_hypotheses:
            return None
        return max(self.global_hypotheses, key=lambda h: h.score)

    def get_all_tracks(self) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        从最佳假设获取所有航迹

        Returns:
            List of (state, covariance, score)
        """
        best_hyp = self.get_best_hypothesis()
        if best_hyp is None:
            return []

        return [
            (track.state, track.covariance, track.score)
            for track in best_hyp.track_hypotheses
        ]
