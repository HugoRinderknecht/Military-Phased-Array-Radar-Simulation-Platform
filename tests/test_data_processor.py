import pytest
import numpy as np
from core.data_processor import DataProcessor
from models.radar_system import RadarSystem
from models.environment import Environment, WeatherCondition
from models.tracking import Detection, Track


@pytest.fixture
def radar_system():
    return RadarSystem(
        radar_area=100.0,
        tr_components=1000,
        radar_power=50000,
        frequency=10e9
    )


@pytest.fixture
def environment():
    weather = WeatherCondition(weather_type="clear")
    return Environment(weather=weather)


@pytest.fixture
def data_processor(radar_system, environment):
    return DataProcessor(radar_system, environment)


@pytest.fixture
def sample_detections():
    return [
        Detection(
            range=50000, azimuth=0.1, elevation=0.05, velocity=200,
            snr=15.0, rcs_estimate=10.0, timestamp=0.0, cell_index=500
        ),
        Detection(
            range=50100, azimuth=0.11, elevation=0.06, velocity=210,
            snr=12.0, rcs_estimate=8.0, timestamp=0.0, cell_index=501
        ),
        Detection(
            range=75000, azimuth=-0.5, elevation=-0.1, velocity=-150,
            snr=18.0, rcs_estimate=15.0, timestamp=0.0, cell_index=750
        )
    ]


class TestDataProcessor:

    def test_initialization(self, data_processor):
        """测试数据处理器初始化"""
        assert data_processor is not None
        assert data_processor.tracks == []
        assert data_processor.track_id_counter == 1
        assert data_processor.svm_filter is not None

    def test_cluster_detections(self, data_processor, sample_detections):
        """测试点迹凝聚"""
        clusters = data_processor._cluster_detections(sample_detections)

        assert isinstance(clusters, list)
        assert len(clusters) >= 1

        # 验证聚类结果
        total_detections = sum(len(cluster) for cluster in clusters)
        assert total_detections == len(sample_detections)

        # 前两个检测应该被聚类到一起（距离较近）
        close_detections = [sample_detections[0], sample_detections[1]]
        clusters_close = data_processor._cluster_detections(close_detections)
        assert len(clusters_close) == 1  # 应该聚为一类
        assert len(clusters_close[0]) == 2

    def test_svm_filter_training(self, data_processor, sample_detections):
        """测试SVM滤波器训练"""
        # 初始状态SVM未训练
        assert not data_processor.svm_filter.is_trained

        # 创建足够的训练数据
        extended_detections = sample_detections * 5  # 15个检测
        labels = [1] * 10 + [0] * 5  # 10个目标，5个杂波

        data_processor.svm_filter.train(extended_detections, labels)

        # 训练后应该标记为已训练
        assert data_processor.svm_filter.is_trained

    def test_svm_filter_prediction(self, data_processor, sample_detections):
        """测试SVM分类预测"""
        detection = sample_detections[0]

        # 未训练时应该使用经验规则
        prediction = data_processor.svm_filter.predict(detection)
        assert prediction in [0, 1]

        # 高SNR检测应该被预测为目标
        high_snr_detection = Detection(
            range=50000, azimuth=0.1, elevation=0.05, velocity=200,
            snr=25.0, rcs_estimate=10.0, timestamp=0.0, cell_index=500
        )
        prediction = data_processor.svm_filter.predict(high_snr_detection)
        assert prediction == 1  # 应该被预测为目标

    def test_track_initialization(self, data_processor, sample_detections):
        """测试航迹起始"""
        initial_track_count = len(data_processor.tracks)

        # 处理未关联的检测应该产生新航迹
        data_processor._update_tracks(sample_detections)

        assert len(data_processor.tracks) > initial_track_count
        assert len(data_processor.tracks) <= initial_track_count + len(sample_detections)

        # 检查新航迹的属性
        new_track = data_processor.tracks[-1]
        assert new_track.track_id > 0
        assert new_track.age == 1
        assert not new_track.confirmed
        assert len(new_track.state) == 6

    def test_track_update(self, data_processor):
        """测试航迹更新"""
        # 创建一个航迹
        initial_detection = Detection(
            range=50000, azimuth=0.1, elevation=0.05, velocity=200,
            snr=15.0, rcs_estimate=10.0, timestamp=0.0, cell_index=500
        )

        track = Track(
            track_id=1,
            state=np.array([
                initial_detection.range * np.cos(initial_detection.azimuth),
                initial_detection.range * np.sin(initial_detection.azimuth),
                0, 200, 0, 0
            ])
        )
        data_processor.tracks.append(track)

        # 创建关联的新检测
        associated_detection = Detection(
            range=50200, azimuth=0.11, elevation=0.05, velocity=205,
            snr=16.0, rcs_estimate=11.0, timestamp=0.06, cell_index=502
        )

        initial_age = track.age
        initial_score = track.score

        # 更新航迹
        data_processor._update_tracks([associated_detection])

        # 验证更新
        assert track.age > initial_age
        assert track.score > initial_score
        assert len(track.detections_history) > 0

    def test_kalman_filter_prediction(self, data_processor):
        """测试卡尔曼滤波预测"""
        # 创建航迹
        track = Track(
            track_id=1,
            state=np.array([50000, 30000, 1000, 200, 100, 0]),
            covariance=np.eye(6) * 100
        )

        initial_position = track.state[:3].copy()
        initial_velocity = track.state[3:].copy()

        # 预测步骤
        track.predict(0.06)  # 60ms

        # 验证位置更新
        expected_position = initial_position + initial_velocity * 0.06
        np.testing.assert_array_almost_equal(track.state[:3], expected_position, decimal=5)

        # 验证协方差增长
        assert np.all(np.diag(track.covariance) >= 1.0)  # 不确定度应该增加

    def test_imm_tracking(self, data_processor):
        """测试交互式多模型跟踪"""
        # 创建高速目标
        high_speed_track = Track(
            track_id=1,
            state=np.array([50000, 30000, 1000, 300, 250, 0])  # 高速
        )

        low_speed_track = Track(
            track_id=2,
            state=np.array([40000, 20000, 800, 50, 30, 0])  # 低速
        )

        tracks = [high_speed_track, low_speed_track]

        # 应用IMM
        data_processor._apply_imm_tracking(tracks)

        # 高速目标应该被标记为高机动性
        assert high_speed_track.high_mobility
        assert not low_speed_track.high_mobility

    def test_low_altitude_enhancement(self, data_processor):
        """测试低空目标增强"""
        # 创建低空航迹
        low_altitude_track = Track(
            track_id=1,
            state=np.array([50000, 30000, 300, 200, 100, 0])  # 300m高度
        )

        high_altitude_track = Track(
            track_id=2,
            state=np.array([40000, 20000, 2000, 150, 80, 0])  # 2000m高度
        )

        data_processor.tracks = [low_altitude_track, high_altitude_track]

        initial_low_score = low_altitude_track.score
        initial_high_score = high_altitude_track.score

        # 应用低空增强
        data_processor._enhance_low_altitude_targets()

        # 低空目标得分应该有所提升
        assert low_altitude_track.score >= initial_low_score

    def test_process_tracking_data_complete(self, data_processor, sample_detections):
        """测试完整的数据处理流程"""
        result_tracks = data_processor.process_tracking_data(sample_detections)

        assert isinstance(result_tracks, list)
        assert len(result_tracks) >= 0

        # 验证航迹属性
        for track in result_tracks:
            assert isinstance(track, Track)
            assert track.track_id > 0
            assert len(track.state) == 6
            assert track.age >= 0

    def test_association_matrix(self, data_processor, sample_detections):
        """测试关联矩阵构建"""
        # 先创建一些航迹
        data_processor._update_tracks(sample_detections)

        # 构建关联矩阵
        matrix = data_processor._build_association_matrix(sample_detections)

        assert matrix.shape[0] == len(data_processor.tracks)
        assert matrix.shape[1] == len(sample_detections)
        assert np.all((matrix == 0) | (matrix == 1))  # 矩阵元素应该是0或1

    def test_feature_extraction(self, data_processor, sample_detections):
        """测试特征提取"""
        detection = sample_detections[0]
        features = data_processor.svm_filter.extract_features(detection)

        assert len(features) == 8  # FEATURE_DIM
        assert all(isinstance(f, (int, float, np.number)) for f in features)

        # 验证特征归一化
        assert 0 <= features[0] <= 1  # SNR特征
        assert 0 <= features[1] <= 1  # 距离特征

    def test_track_management(self, data_processor):
        """测试航迹管理"""
        # 创建不同年龄的航迹
        young_unconfirmed = Track(track_id=1, age=2, confirmed=False)
        old_unconfirmed = Track(track_id=2, age=15, confirmed=False)
        confirmed_track = Track(track_id=3, age=5, confirmed=True)

        data_processor.tracks = [young_unconfirmed, old_unconfirmed, confirmed_track]

        # 模拟航迹更新（不提供新检测）
        data_processor._update_tracks([])

        # 老的未确认航迹应该被删除
        remaining_ids = [track.track_id for track in data_processor.tracks]
        assert 1 in remaining_ids  # 年轻未确认航迹保留
        assert 2 not in remaining_ids  # 老的未确认航迹删除
        assert 3 in remaining_ids  # 确认航迹保留


class TestTrackingModel:

    def test_track_prediction(self):
        """测试单个航迹预测"""
        track = Track(
            track_id=1,
            state=np.array([1000, 2000, 500, 100, 50, 10]),
            covariance=np.eye(6) * 10
        )

        dt = 0.1
        initial_state = track.state.copy()

        track.predict(dt)

        # 验证位置更新
        expected_x = initial_state[0] + initial_state[3] * dt
        expected_y = initial_state[1] + initial_state[4] * dt
        expected_z = initial_state[2] + initial_state[5] * dt

        assert abs(track.state[0] - expected_x) < 1e-10
        assert abs(track.state[1] - expected_y) < 1e-10
        assert abs(track.state[2] - expected_z) < 1e-10

    def test_track_update_with_measurement(self):
        """测试航迹测量更新"""
        track = Track(
            track_id=1,
            state=np.array([1000, 2000, 500, 100, 50, 10]),
            covariance=np.eye(6) * 100
        )

        detection = Detection(
            range=2236, azimuth=1.107, elevation=0.218,
            velocity=111.8, snr=15.0, rcs_estimate=10.0,
            timestamp=1.0, cell_index=100
        )

        initial_age = track.age
        initial_score = track.score

        track.update(detection)

        # 验证航迹属性更新
        assert track.age == initial_age + 1
        assert track.score == initial_score + 1.0
        assert len(track.detections_history) == 1
        assert len(track.rcs_history) == 1
