# imm_filter.py - 交互多模型滤波器实现
"""
交互多模型滤波器 (IMM) - 多模型自适应估计

IMM算法通过混合多个运动模型来适应目标机动：
- 模型概率估计
- 混合初始化
- 模型交互
- 组合估计
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from .kalman_filter import KalmanFilter, KalmanState
from .extended_kalman_filter import ExtendedKalmanFilter
from .unscented_kalman_filter import UnscentedKalmanFilter


@dataclass
class ModelFilter:
    """单个模型滤波器包装"""
    filter: Any  # KalmanFilter, ExtendedKalmanFilter, 或 UnscentedKalmanFilter
    prior_probability: float  # 先验概率
    predicted_probability: float = 0.0  # 预测概率
    likelihood: float = 0.0  # 似然
    posterior_probability: float = 0.0  # 后验概率


class InteractingMultipleModel:
    """
    交互多模型 (IMM) 滤波器

    通过混合多个运动模型来处理目标机动
    """

    def __init__(
        self,
        filters: List[Any],
        transition_matrix: np.ndarray,
        initial_probabilities: Optional[np.ndarray] = None
    ):
        """
        Args:
            filters: 滤波器列表 (KF, EKF, 或 UKF)
            transition_matrix: 模型转移矩阵 [r x r]
            initial_probabilities: 初始模型概率 [r]
        """
        self.num_models = len(filters)
        self.transition_matrix = transition_matrix  # Markov链转移概率

        # 初始化模型
        self.models: List[ModelFilter] = []
        for i, f in enumerate(filters):
            prob = initial_probabilities[i] if initial_probabilities is not None else 1.0 / self.num_models
            self.models.append(ModelFilter(filter=f, prior_probability=prob))

        self.current_time: float = 0.0

    def init(self, x0: np.ndarray, P0: np.ndarray, timestamp: float = 0.0) -> None:
        """初始化所有滤波器"""
        for model in self.models:
            if hasattr(model.filter, 'init'):
                model.filter.init(x0.copy(), P0.copy(), timestamp)
            else:
                # 对于没有init方法的滤波器，设置状态
                model.filter.state = KalmanState(x0.copy(), P0.copy(), timestamp)

        self.current_time = timestamp

    def _compute_mixing_probabilities(self) -> np.ndarray:
        """计算混合概率"""
        # Omega[i,j] = P(M_j|M_i) * P(M_i) / P(M_j)
        omega = np.zeros((self.num_models, self.num_models))

        for i in range(self.num_models):
            for j in range(self.num_models):
                numerator = self.transition_matrix[i, j] * self.models[i].prior_probability
                denominator = sum(self.transition_matrix[k, j] * self.models[k].prior_probability
                                for k in range(self.num_models))
                if denominator > 0:
                    omega[i, j] = numerator / denominator

        return omega

    def _mix_states(self, omega: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """混合状态和协方差"""
        mixed_states = []

        for j in range(self.num_models):
            # 计算混合状态
            x_mixed = np.zeros_like(self.models[0].filter.state.x)
            for i in range(self.num_models):
                x_mixed += omega[i, j] * self.models[i].filter.state.x

            # 计算混合协方差
            P_mixed = np.zeros_like(self.models[0].filter.state.P)
            for i in range(self.num_models):
                x_i = self.models[i].filter.state.x
                P_i = self.models[i].filter.state.P
                diff = x_i - x_mixed
                P_mixed += omega[i, j] * (P_i + diff @ diff.T)

            mixed_states.append((x_mixed, P_mixed))

        return mixed_states

    def predict(self, dt: float) -> None:
        """预测步骤"""
        # 1. 计算混合概率
        omega = self._compute_mixing_probabilities()

        # 2. 混合状态
        mixed_states = self._mix_states(omega)

        # 3. 每个模型进行预测
        for j, model in enumerate(self.models):
            # 更新滤波器状态为混合状态
            if hasattr(model.filter, 'state'):
                model.filter.state = KalmanState(
                    mixed_states[j][0],
                    mixed_states[j][1],
                    self.current_time
                )

            # 执行预测
            if hasattr(model.filter, 'predict'):
                model.filter.predict(dt)
            else:
                # 对于没有predict方法的，使用step的预测部分
                pass

            # 更新预测概率
            model.predicted_probability = sum(
                self.transition_matrix[i, j] * model.prior_probability
                for i in range(self.num_models)
            )

        self.current_time += dt

    def update(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        测量更新

        Returns:
            (x_combined, P_combined, info): 组合状态、组合协方差、辅助信息
        """
        total_likelihood = 0.0
        likelihoods = []

        # 1. 每个模型进行更新
        for j, model in enumerate(self.models):
            try:
                if hasattr(model.filter, 'update'):
                    state, info = model.filter.update(z)
                    model.likelihood = info.get('likelihood', 1.0)
                else:
                    model.likelihood = 1.0
            except Exception as e:
                # 更新失败，使用小似然
                model.likelihood = 1e-10

            likelihoods.append(model.likelihood)

        # 2. 计算模型概率
        for j, model in enumerate(self.models):
            numerator = model.predicted_probability * model.likelihood
            total_likelihood += numerator

        for j, model in enumerate(self.models):
            if total_likelihood > 0:
                model.posterior_probability = (
                    model.predicted_probability * model.likelihood / total_likelihood
                )
            else:
                model.posterior_probability = 1.0 / self.num_models

        # 3. 组合估计
        x_combined = np.zeros_like(self.models[0].filter.state.x)
        for model in self.models:
            if hasattr(model.filter, 'state'):
                x_combined += model.posterior_probability * model.filter.state.x

        P_combined = np.zeros_like(self.models[0].filter.state.P)
        for model in self.models:
            if hasattr(model.filter, 'state'):
                x_m = model.filter.state.x
                P_m = model.filter.state.P
                diff = x_m - x_combined
                prob = model.posterior_probability
                P_combined += prob * (P_m + diff @ diff.T)

        # 更新先验概率为后验概率（下一次迭代）
        for model in self.models:
            model.prior_probability = model.posterior_probability

        # 准备返回信息
        info = {
            'model_probabilities': [m.posterior_probability for m in self.models],
            'model_likelihoods': likelihoods,
            'dominant_model': np.argmax([m.posterior_probability for m in self.models]),
            'total_likelihood': total_likelihood,
        }

        return x_combined, P_combined, info

    def step(self, z: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """完整步骤"""
        self.predict(dt)
        return self.update(z)

    def get_dominant_model(self) -> int:
        """获取当前主导模型的索引"""
        probabilities = [m.posterior_probability for m in self.models]
        return int(np.argmax(probabilities))


def create_imm_cv_ca(
    dt: float = 0.1,
    sigma_q_cv: float = 0.5,
    sigma_q_ca: float = 2.0,
    sigma_r: float = 10.0,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    transition_matrix: Optional[np.ndarray] = None
) -> InteractingMultipleModel:
    """
    创建CV-CA IMM滤波器

    Args:
        dt: 时间步长
        sigma_q_cv: CV模型过程噪声
        sigma_q_ca: CA模型过程噪声
        sigma_r: 测量噪声标准差
        x0: 初始状态
        P0: 初始协方差
        transition_matrix: 模型转移矩阵

    Returns:
        配置好的IMM滤波器
    """
    from .kalman_filter import create_cv_filter, create_ca_filter

    # 创建CV和CA滤波器
    cv_filter = create_cv_filter(dt, sigma_q_cv, sigma_r, x0, P0)
    ca_filter = create_ca_filter(dt, sigma_q_ca, sigma_r, x0, P0)

    # 默认转移矩阵 (低转移概率)
    if transition_matrix is None:
        transition_matrix = np.array([
            [0.95, 0.05],  # CV -> CV, CV -> CA
            [0.05, 0.95],  # CA -> CV, CA -> CA
        ])

    imm = InteractingMultipleModel(
        filters=[cv_filter, ca_filter],
        transition_matrix=transition_matrix,
        initial_probabilities=np.array([0.7, 0.3])  # 初始倾向于CV模型
    )

    return imm


def create_imm_cv_ct(
    dt: float = 0.1,
    sigma_q_cv: float = 0.5,
    sigma_q_ct: float = 1.0,
    sigma_r: float = 10.0,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    omega_turn: float = 0.1
) -> InteractingMultipleModel:
    """
    创建CV-CT IMM滤波器（处理协调转弯）

    Args:
        dt: 时间步长
        sigma_q_cv: CV模型过程噪声
        sigma_q_ct: CT模型过程噪声
        sigma_r: 测量噪声标准差
        x0: 初始状态
        P0: 初始协方差
        omega_turn: 转弯角速度

    Returns:
        配置好的IMM滤波器
    """
    from .kalman_filter import create_cv_filter, ConstantTurnModel, PositionMeasurement, KalmanFilter

    # 创建CV滤波器
    cv_filter = create_cv_filter(dt, sigma_q_cv, sigma_r, x0, P0)

    # 创建CT滤波器
    ct_process_model = ConstantTurnModel(dt, sigma_q_ct, omega_turn)
    ct_meas_model = PositionMeasurement(sigma_r)
    ct_filter = KalmanFilter(ct_process_model, ct_meas_model)
    if x0 is not None:
        # CT模型需要5维状态 [x, y, vx, vy, omega]
        x0_ct = np.array([x0[0], x0[1], x0[3], x0[4], omega_turn])
        P0_ct = np.eye(5) * 100.0 if P0 is None else P0[:5, :5]
        ct_filter.init(x0_ct, P0_ct)

    # 转移矩阵
    transition_matrix = np.array([
        [0.95, 0.05],  # CV -> CV, CV -> CT
        [0.05, 0.95],  # CT -> CV, CT -> CT
    ])

    imm = InteractingMultipleModel(
        filters=[cv_filter, ct_filter],
        transition_matrix=transition_matrix,
        initial_probabilities=np.array([0.8, 0.2])
    )

    return imm
