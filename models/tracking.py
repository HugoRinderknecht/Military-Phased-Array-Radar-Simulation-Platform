import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from sklearn.svm import SVC


@dataclass
class Detection:
    range: float
    azimuth: float
    elevation: float
    velocity: float
    snr: float
    rcs_estimate: float
    timestamp: float
    cell_index: int = 0
    low_altitude: bool = False


@dataclass
class Track:
    track_id: int
    state: np.ndarray = field(default_factory=lambda: np.zeros(6))
    covariance: np.ndarray = field(default_factory=lambda: np.eye(6) * 100)
    age: int = 0
    score: float = 1.0
    confirmed: bool = False
    high_mobility: bool = False
    rcs_history: List[float] = field(default_factory=list)
    detections_history: List[Detection] = field(default_factory=list)

    def predict(self, dt: float):
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])

        Q = np.eye(6) * 1.0
        Q[:3, :3] *= dt ** 2 / 2
        Q[3:, 3:] *= dt

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q

    def update(self, detection: Detection):
        # 修复：观测矩阵应该是 2x6，只观测 x, y 位置
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])

        R = np.eye(2) * 100

        # 将极坐标观测转换为直角坐标
        z = np.array([
            detection.range * np.cos(detection.azimuth),
            detection.range * np.sin(detection.azimuth)
        ])

        # 修复：H @ self.state 应该是完整的状态向量，然后取前2个元素
        predicted_measurement = H @ self.state  # 这将得到 2 维向量
        y = z - predicted_measurement  # 新息向量，2维

        S = H @ self.covariance @ H.T + R  # 新息协方差矩阵，2x2

        try:
            K = self.covariance @ H.T @ np.linalg.inv(S)  # 卡尔曼增益，6x2
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            K = self.covariance @ H.T @ np.linalg.pinv(S)

        # 状态更新：K是6x2，y是2x1，所以K@y是6x1
        self.state = self.state + K @ y

        # 协方差更新
        I = np.eye(6)
        self.covariance = (I - K @ H) @ self.covariance

        self.age += 1
        self.score += 1.0
        self.detections_history.append(detection)

        if len(self.rcs_history) < 10:
            self.rcs_history.append(detection.rcs_estimate)
        else:
            self.rcs_history.pop(0)
            self.rcs_history.append(detection.rcs_estimate)


@dataclass
class TOMHTNode:
    detection: Optional[Detection]
    parent: Optional['TOMHTNode']
    children: List['TOMHTNode'] = field(default_factory=list)
    likelihood_ratio: float = 0.0
    log_likelihood_ratio: float = 0.0


class SVMClutterFilter:
    def __init__(self):
        self.model = SVC(kernel='rbf', probability=True, gamma='scale')
        self.scaler = None
        self.is_trained = False
        self.feature_buffer = []
        self.label_buffer = []

    def extract_features(self, detection: Detection) -> np.ndarray:
        return np.array([
            detection.snr,
            detection.range / 100000,
            abs(detection.velocity) / 500,
            detection.azimuth / np.pi,
            detection.elevation / (np.pi / 2),
            (detection.rcs_estimate + 40) / 80,
            1.0 if detection.low_altitude else 0.0,
            1.0
        ])

    def train(self, detections: List[Detection], labels: List[int]):
        from sklearn.preprocessing import StandardScaler

        if len(detections) < 10:
            return

        features = np.array([self.extract_features(det) for det in detections])

        if self.scaler is None:
            self.scaler = StandardScaler()
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = self.scaler.transform(features)

        self.model.fit(features_scaled, labels)
        self.is_trained = True

    def predict(self, detection: Detection) -> int:
        if not self.is_trained:
            return 1 if detection.snr > 5 else 0

        features = self.extract_features(detection).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        return int(self.model.predict(features_scaled)[0])
