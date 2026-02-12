# data_processor.py - 数据处理器
"""
本模块实现雷达数据处理功能。

数据处理流程：
1. 点迹预处理（分类、过滤）
2. 航迹起始
3. 数据关联
4. 航迹滤波
5. 航迹管理
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

from radar.common.logger import get_logger
from radar.common.types import (
    Plot, Track, TrackState, Position3D, Velocity3D,
    FilterType
)
from radar.common.constants import MathConstants


class TrackInitiatorType(Enum):
    """航迹起始方法"""
    RULE_BASED = "rule"        # 规则法（M/N）
    LOGIC = "logic"            # 逻辑法
    HOUGH = "hough"            # Hough变换
    MLB = "mlb"                # MLB算法


class AssociatorType(Enum):
    """数据关联方法"""
    NN = "nn"                  # 最近邻
    PDA = "pda"                # 概率数据关联
    JPDA = "jpda"              # 联合概率数据关联
    GNN = "gnn"                # 全局最近邻
    JPDA_MHT = "jpda_mht"      # JPDA多假设跟踪


@dataclass
class TrackInitConfig:
    """航迹起始配置"""
    method: TrackInitiatorType = TrackInitiatorType.RULE_BASED
    m_value: int = 3              # M/N逻辑的M值
    n_value: int = 3              # M/N逻辑的N值
    max_init_scans: int = 5       # 最大起始扫描数
    gate_threshold: float = 3.0    # 波门大小（标准差倍数）


@dataclass
class AssociationConfig:
    """数据关联配置"""
    method: AssociatorType = AssociatorType.NN
    gate_size: float = 3.0        # 波门大小
    max_associations: int = 1      # 最大关联数
    use_velocity: bool = True      # 是否使用速度信息


@dataclass
class FilterConfig:
    """滤波器配置"""
    filter_type: FilterType = FilterType.KF
    process_noise: float = 1.0     # 过程噪声
    measurement_noise: float = 10.0  # 量测噪声


class DataProcessor:
    """
    数据处理器

    管理航迹的起始、关联、滤波和维护。
    """

    def __init__(self,
                 init_config: Optional[TrackInitConfig] = None,
                 assoc_config: Optional[AssociationConfig] = None,
                 filter_config: Optional[FilterConfig] = None):
        """
        初始化数据处理器

        Args:
            init_config: 航迹起始配置
            assoc_config: 数据关联配置
            filter_config: 滤波器配置
        """
        self._logger = get_logger("data_proc")

        self._init_config = init_config or TrackInitConfig()
        self._assoc_config = assoc_config or AssociationConfig()
        self._filter_config = filter_config or FilterConfig()

        # 航迹存储
        self._tracks: Dict[int, Track] = {}
        self._track_id_counter = 0

        # 待确认航迹（tentative tracks）
        self._tentative_tracks: Dict[int, Track] = {}

        # 航迹历史（用于数据关联）
        self._plot_history: deque = deque(maxlen=10)

        # 统计信息
        self._stats = {
            'total_plots_processed': 0,
            'tracks_initiated': 0,
            'tracks_confirmed': 0,
            'tracks_dropped': 0,
        }

        self._logger.info("数据处理器初始化完成")

    def process(self, plots: List[Plot]) -> List[Track]:
        """
        处理点迹数据

        Args:
            plots: 当前帧的点迹列表

        Returns:
            更新后的航迹列表
        """
        self._stats['total_plots_processed'] += len(plots)

        # 添加点迹到历史
        for plot in plots:
            self._plot_history.append(plot)

        # Step 1: 航迹起始
        tentative_tracks = self._initiate_tracks(plots)

        # Step 2: 数据关联
        associations = self._associate_plots_to_tracks(
            plots, list(self._tracks.values())
        )

        # Step 3: 航迹滤波和更新
        for track_id, associated_plots in associations.items():
            if track_id in self._tracks:
                track = self._tracks[track_id]
                updated_track = self._update_track(track, associated_plots)
                self._tracks[track_id] = updated_track

        # Step 4: 航迹管理
        self._manage_tracks()

        return list(self._tracks.values())

    def _initiate_tracks(self, plots: List[Plot]) -> List[Track]:
        """
        航迹起始

        使用M/N逻辑法起始新航迹。

        Args:
            plots: 点迹列表

        Returns:
            新起始的航迹列表
        """
        new_tracks = []

        for plot in plots:
            # 检查是否可以与待确认航迹关联
            if self._can_associate_with_tentative(plot):
                continue

            # 检查是否在已有航迹的波门内
            if self._is_in_existing_gate(plot):
                continue

            # 创建新的待确认航迹
            self._track_id_counter += 1
            track = Track(
                id=self._track_id_counter,
                state=TrackState.TENTATIVE,
                position=plot.position,
                velocity=plot.velocity if plot.velocity else Velocity3D(0, 0, 0),
                last_update=plot.timestamp,
                creation_time=plot.timestamp,
                update_count=1,
                missed_count=0,
                associated_plots=[plot]
            )

            self._tentative_tracks[track.id] = track
            new_tracks.append(track)

        self._logger.debug(f"起始 {len(new_tracks)}个新航迹")

        return new_tracks

    def _can_associate_with_tentative(self, plot: Plot) -> bool:
        """检查点迹是否可以与待确认航迹关联"""
        for track in self._tentative_tracks.values():
            if self._is_in_gate(plot, track):
                return True
        return False

    def _is_in_existing_gate(self, plot: Plot) -> bool:
        """检查点迹是否在任何已确认航迹的波门内"""
        for track in self._tracks.values():
            if self._is_in_gate(plot, track):
                return True
        return False

    def _associate_plots_to_tracks(self,
                                  plots: List[Plot],
                                  tracks: List[Track]) -> Dict[int, List[Plot]]:
        """
        数据关联

        使用最近邻方法进行点迹-航迹关联。

        Args:
            plots: 点迹列表
            tracks: 航迹列表

        Returns:
            关联字典 {track_id: [plots]}
        """
        associations = {track.id: [] for track in tracks}
        used_plots = set()

        # 最近邻关联
        for track in tracks:
            best_plot = None
            best_distance = float('inf')

            for plot in plots:
                if plot.id in used_plots:
                    continue

                distance = self._calculate_distance(plot, track)
                gate_threshold = self._assoc_config.gate_size * np.sqrt(
                    np.sum(np.diag(track.position_cov))
                )

                if distance < gate_threshold and distance < best_distance:
                    best_distance = distance
                    best_plot = plot

            if best_plot:
                associations[track.id].append(best_plot)
                used_plots.add(best_plot.id)

        return associations

    def _is_in_gate(self, plot: Plot, track: Track) -> bool:
        """检查点迹是否在航迹波门内"""
        distance = self._calculate_distance(plot, track)
        gate_size = self._assoc_config.gate_size * np.sqrt(
            np.sum(np.diag(track.position_cov))
        )
        return distance < gate_size

    def _calculate_distance(self, plot: Plot, track: Track) -> float:
        """
        计算点迹与航迹之间的距离

        使用马氏距离或欧氏距离。

        Args:
            plot: 点迹
            track: 航迹

        Returns:
            距离度量
        """
        # 位置差
        dx = plot.position.x - track.position.x
        dy = plot.position.y - track.position.y
        dz = plot.position.z - track.position.z

        # 简化：欧氏距离
        # 完整实现应使用协方差矩阵（马氏距离）
        distance = np.sqrt(dx**2 + dy**2 + dz**2)

        return distance

    def _update_track(self, track: Track, plots: List[Plot]) -> Track:
        """
        更新航迹

        使用滤波器更新航迹状态。

        Args:
            track: 航迹
            plots: 关联的点迹列表

        Returns:
            更新后的航迹
        """
        if not plots:
            # 没有关联点迹，增加丢失计数
            track.missed_count += 1
        else:
            # 使用最新的点迹更新
            plot = plots[-1]

            # 简化更新：直接使用点迹数据
            # 完整实现应使用卡尔曼滤波器
            if plot.velocity:
                # 点迹有速度信息
                track.position = plot.position
                track.velocity = plot.velocity
            else:
                # 点迹没有速度，需要估计
                # 使用简单的一阶差分
                if track.update_count > 1:
                    dt = (plot.timestamp - track.last_update) / 1e6  # 转换为秒
                    if dt > 0:
                        vx = (plot.position.x - track.position.x) / dt
                        vy = (plot.position.y - track.position.y) / dt
                        vz = (plot.position.z - track.position.z) / dt
                        track.velocity = Velocity3D(vx, vy, vz)

                track.position = plot.position

            track.last_update = plot.timestamp
            track.update_count += 1
            track.missed_count = 0
            track.associated_plots.append(plot)

            # 更新航迹状态
            if track.state == TrackState.TENTATIVE and track.update_count >= self._init_config.m_value:
                track.state = TrackState.CONFIRMED
                self._stats['tracks_confirmed'] += 1

        return track

    def _manage_tracks(self) -> None:
        """航迹管理：删除、终止、合并"""
        to_remove = []

        for track_id, track in self._tracks.items():
            # 检查是否应该终止航迹
            if track.missed_count >= 5:  # 连续5次丢失
                track.state = TrackState.DROPPED
                to_remove.append(track_id)
                self._stats['tracks_dropped'] += 1

            # 检查航迹寿命
            age = (track.last_update - track.creation_time) / 1e6  # 秒
            if age > 300 and track.update_count < 3:  # 5分钟且更新次数少
                track.state = TrackState.TERMINATED
                to_remove.append(track_id)

        # 删除航迹
        for track_id in to_remove:
            del self._tracks[track_id]
            self._logger.debug(f"删除航迹: ID={track_id}")

        # 确认待确认航迹
        for track_id, track in list(self._tentative_tracks.items()):
            if track.update_count >= self._init_config.n_value:
                # 确认航迹
                track.state = TrackState.CONFIRMED
                self._tracks[track_id] = track
                del self._tentative_tracks[track_id]
                self._stats['tracks_initiated'] += 1
            elif track.missed_count >= self._init_config.m_value:
                # 删除待确认航迹
                del self._tentative_tracks[track_id]

    def get_tracks(self) -> List[Track]:
        """获取所有航迹"""
        return list(self._tracks.values())

    def get_track(self, track_id: int) -> Optional[Track]:
        """获取指定ID的航迹"""
        return self._tracks.get(track_id)

    def cancel_track(self, track_id: int) -> bool:
        """
        取消航迹

        Args:
            track_id: 航迹ID

        Returns:
            是否成功取消
        """
        if track_id in self._tracks:
            track = self._tracks[track_id]
            track.state = TrackState.TERMINATED
            del self._tracks[track_id]
            self._logger.info(f"取消航迹: ID={track_id}")
            return True
        return False

    def get_statistics(self) -> dict:
        """获取处理统计"""
        return {
            **self._stats,
            'active_tracks': len(self._tracks),
            'tentative_tracks': len(self._tentative_tracks),
        }


__all__ = [
    "TrackInitiatorType",
    "AssociatorType",
    "TrackInitConfig",
    "AssociationConfig",
    "FilterConfig",
    "DataProcessor",
]
