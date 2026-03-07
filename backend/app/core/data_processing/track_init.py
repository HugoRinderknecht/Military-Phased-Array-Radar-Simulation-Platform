"""
航迹起始模块
完整实现逻辑法航起始

参考文档 4.4.9 节
"""
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Plot:
    """点迹"""
    plot_id: str
    time: float
    x: float
    y: float
    z: float
    vx: Optional[float] = None
    vy: Optional[float] = None
    vz: Optional[float] = None
    snr: float = 0.0
    amplitude: float = 0.0


@dataclass
class TentativeTrack:
    """临时航迹"""
    track_id: str
    plots: List[Plot]
    predicted_state: Optional[np.ndarray] = None  # [x, y, z, vx, vy, vz]
    covariance: Optional[np.ndarray] = None
    quality: int = 0  # 累计检测次数
    missed_scans: int = 0


@dataclass
class ConfirmedTrack:
    """确认航迹"""
    track_id: str
    plots: List[Plot]
    state: np.ndarray  # [x, y, z, vx, vy, vz]
    covariance: np.ndarray
    target_type: Optional[str] = None
    quality: float = 1.0


class LogicTrackInitializer:
    """
    逻辑法航迹起始器

    实现M/N逻辑准则
    """

    def __init__(
        self,
        m: int = 3,
        n: int = 4,
        gate_probability: float = 0.99,
        max_velocity: float = 1000,  # m/s
        max_acceleration: float = 100,  # m/s²
    ):
        """
        Args:
            m: M/N准则中的M（确认所需检测数）
            n: M/N准则中的N（滑窗长度）
            gate_probability: 波门概率
            max_velocity: 最大速度
            max_acceleration: 最大加速度
        """
        self.m = m
        self.n = n
        self.gate_probability = gate_probability
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration

        self.tentative_tracks: List[TentativeTrack] = []
        self.confirmed_tracks: List[ConfirmedTrack] = []
        self._track_counter = 0

    def calculate_gate_size(
        self,
        time_diff: float,
        measurement_noise: float = 10.0,
    ) -> float:
        """
        计算波门大小

        Args:
            time_diff: 时间差
            measurement_noise: 测量噪声标准差

        Returns:
            波门半径（m）
        """
        # 考虑最大速度的不确定性
        velocity_uncertainty = self.max_velocity * time_diff

        # 测量噪声
        measurement_uncertainty = measurement_noise

        # 总不确定性（平方和根）
        gate_size = np.sqrt(velocity_uncertainty**2 + measurement_uncertainty**2)

        # 根据波门概率调整
        # 对于3D情况，使用卡方分布
        from scipy.stats import chi2
        chi2_value = chi2.ppf(self.gate_probability, df=3)
        gate_size *= np.sqrt(chi2_value)

        return gate_size

    def is_plot_in_gate(
        self,
        plot: Plot,
        predicted_position: np.ndarray,
        gate_size: float,
    ) -> bool:
        """
        判断点迹是否在波门内

        Args:
            plot: 点迹
            predicted_position: 预测位置 [x, y, z]
            gate_size: 波门大小

        Returns:
            是否在波门内
        """
        position = np.array([plot.x, plot.y, plot.z])
        distance = np.linalg.norm(position - predicted_position)

        return distance <= gate_size

    def initialize_from_single_plot(
        self,
        plot: Plot,
    ) -> TentativeTrack:
        """
        从单点迹启动临时航迹

        Args:
            plot: 初始点迹

        Returns:
            临时航迹
        """
        self._track_counter += 1
        track_id = f"track-{self._track_counter:04d}"

        tentative_track = TentativeTrack(
            track_id=track_id,
            plots=[plot],
            quality=1,
        )

        # 初始化状态（假设静止或使用默认速度）
        # 这里使用简化的初始化
        tentative_track.predicted_state = np.array([
            plot.x, plot.y, plot.z,  # 位置
            0.0, 0.0, 0.0,  # 速度
        ])

        # 初始化协方差
        tentative_track.covariance = np.diag([
            100.0, 100.0, 100.0,  # 位置方差
            50.0, 50.0, 50.0,  # 速度方差
        ])

        return tentative_track

    def update_tentative_track(
        self,
        track: TentativeTrack,
        new_plot: Plot,
        time_diff: float,
    ) -> TentativeTrack:
        """
        更新临时航迹

        Args:
            track: 临时航迹
            new_plot: 新点迹
            time_diff: 时间差

        Returns:
            更新后的临时航迹
        """
        track.plots.append(new_plot)
        track.quality += 1
        track.missed_scans = 0

        # 更新预测状态（简单的线性外推）
        if len(track.plots) >= 2:
            # 使用最近的两个点迹估计速度
            prev_plot = track.plots[-2]
            curr_plot = track.plots[-1]

            dt = curr_plot.time - prev_plot.time
            if dt > 0:
                vx = (curr_plot.x - prev_plot.x) / dt
                vy = (curr_plot.y - prev_plot.y) / dt
                vz = (curr_plot.z - prev_plot.z) / dt

                # 限制最大速度
                speed = np.sqrt(vx**2 + vy**2 + vz**2)
                if speed > self.max_velocity:
                    scale = self.max_velocity / speed
                    vx *= scale
                    vy *= scale
                    vz *= scale

                track.predicted_state = np.array([
                    curr_plot.x, curr_plot.y, curr_plot.z,
                    vx, vy, vz,
                ])

                # 预测下一个位置
                next_time = new_plot.time + time_diff
                predicted_x = curr_plot.x + vx * (next_time - curr_plot.time)
                predicted_y = curr_plot.y + vy * (next_time - curr_plot.time)
                predicted_z = curr_plot.z + vz * (next_time - curr_plot.time)

                track.predicted_state[:3] = [predicted_x, predicted_y, predicted_z]

        return track

    def confirm_track(
        self,
        tentative_track: TentativeTrack,
    ) -> ConfirmedTrack:
        """
        确认航迹

        Args:
            tentative_track: 临时航迹

        Returns:
            确认航迹
        """
        # 计算最终状态估计（最小二乘拟合）
        plots = tentative_track.plots

        if len(plots) >= 2:
            # 线性拟合估计速度
            times = np.array([p.time for p in plots])
            x_positions = np.array([p.x for p in plots])
            y_positions = np.array([p.y for p in plots])
            z_positions = np.array([p.z for p in plots])

            # 最小二乘拟合: position = v*t + p0
            A = np.vstack([times, np.ones_like(times)]).T

            vx, x0 = np.linalg.lstsq(A, x_positions, rcond=None)[0]
            vy, y0 = np.linalg.lstsq(A, y_positions, rcond=None)[0]
            vz, z0 = np.linalg.lstsq(A, z_positions, rcond=None)[0]

            state = np.array([x0, y0, z0, vx, vy, vz])
        else:
            state = tentative_track.predicted_state

        # 初始化协方差
        covariance = tentative_track.covariance if tentative_track.covariance is not None else np.eye(6) * 100

        confirmed_track = ConfirmedTrack(
            track_id=tentative_track.track_id,
            plots=plots,
            state=state,
            covariance=covariance,
            quality=tentative_track.quality / tentative_track.n,
        )

        return confirmed_track

    def process_scan(
        self,
        plots: List[Plot],
        current_time: float,
        scan_interval: float,
    ) -> Tuple[List[ConfirmedTrack], List[TentativeTrack]]:
        """
        处理一次扫描

        Args:
            plots: 本次扫描的点迹列表
            current_time: 当前时间
            scan_interval: 扫描间隔

        Returns:
            (新确认的航迹, 更新后的临时航迹列表)
        """
        new_confirmed_tracks = []

        # 标记已关联的点迹
        associated_plots = set()

        # 1. 更新现有临时航迹
        updated_tentative_tracks = []
        for track in self.tentative_tracks:
            gate_size = self.calculate_gate_size(scan_interval)
            predicted_position = track.predicted_state[:3]

            # 寻找关联点迹
            associated = False
            for plot in plots:
                if plot.plot_id in associated_plots:
                    continue

                if self.is_plot_in_gate(plot, predicted_position, gate_size):
                    track = self.update_tentative_track(track, plot, scan_interval)
                    associated_plots.add(plot.plot_id)
                    associated = True

                    # 检查是否满足确认条件
                    if track.quality >= self.m:
                        confirmed_track = self.confirm_track(track)
                        new_confirmed_tracks.append(confirmed_track)
                    else:
                        updated_tentative_tracks.append(track)
                    break

            if not associated:
                track.missed_scans += 1
                # 如果未超过N次丢失，保留临时航迹
                if track.missed_scans < (self.n - self.m):
                    updated_tentative_tracks.append(track)

        self.tentative_tracks = updated_tentative_tracks

        # 2. 从未关联的点迹启动新的临时航迹
        for plot in plots:
            if plot.plot_id not in associated_plots:
                new_track = self.initialize_from_single_plot(plot)
                self.tentative_tracks.append(new_track)

        # 3. 添加新确认的航迹
        self.confirmed_tracks.extend(new_confirmed_tracks)

        return new_confirmed_tracks, self.tentative_tracks

    def get_confirmed_tracks(self) -> List[ConfirmedTrack]:
        """获取所有确认的航迹"""
        return self.confirmed_tracks

    def get_tentative_tracks(self) -> List[TentativeTrack]:
        """获取所有临时航迹"""
        return self.tentative_tracks

    def reset(self):
        """重置跟踪器"""
        self.tentative_tracks = []
        self.confirmed_tracks = []
        self._track_counter = 0
