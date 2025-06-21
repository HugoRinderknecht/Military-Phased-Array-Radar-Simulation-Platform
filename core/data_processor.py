import numpy as np
from typing import List, Dict, Tuple
from models.tracking import Track, Detection, SVMClutterFilter, TOMHTNode
from models.radar_system import RadarSystem
from models.environment import Environment
from sklearn.cluster import DBSCAN


class DataProcessor:
    def __init__(self, radar_system: RadarSystem, environment: Environment):
        self.radar_system = radar_system
        self.environment = environment
        self.tracks = []
        self.svm_filter = SVMClutterFilter()
        self.track_id_counter = 1

    def process_tracking_data(self, detections: List[Detection]) -> List[Track]:
        clusters = self._cluster_detections(detections)
        filtered_clusters = self._tomht_svm_tracking(clusters)

        for track in self.tracks:
            track.predict(0.06)
            if self.environment.weather.weather_type in ["heavy_rain", "snow"]:
                self._apply_imm_tracking([track])

        self._enhance_low_altitude_targets()

        return self.tracks

    def _cluster_detections(self, detections: List[Detection]) -> List[List[Detection]]:
        if len(detections) < 2:
            return [[det] for det in detections]

        positions = np.array([
            [det.range * np.cos(det.azimuth), det.range * np.sin(det.azimuth)]
            for det in detections
        ])

        clustering = DBSCAN(eps=100, min_samples=1).fit(positions)
        labels = clustering.labels_

        clusters = {}
        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(detections[i])

        return list(clusters.values())

    def _tomht_svm_tracking(self, clusters: List[List[Detection]]) -> List[Detection]:
        all_detections = [det for cluster in clusters for det in cluster]

        if len(all_detections) > 10:
            labels = []
            for detection in all_detections:
                is_target = self._is_detection_associated_with_track(detection)
                labels.append(1 if is_target else 0)

            if len(set(labels)) > 1:
                self.svm_filter.train(all_detections, labels)

        filtered_detections = []
        for detection in all_detections:
            if self.svm_filter.predict(detection) == 1:
                filtered_detections.append(detection)

        self._update_tracks(filtered_detections)

        return filtered_detections

    def _is_detection_associated_with_track(self, detection: Detection) -> bool:
        det_x = detection.range * np.cos(detection.azimuth)
        det_y = detection.range * np.sin(detection.azimuth)

        for track in self.tracks:
            pred_x, pred_y = track.state[0], track.state[1]
            distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)

            if distance < 1000:
                return True

        return False

    def _update_tracks(self, detections: List[Detection]):
        association_matrix = self._build_association_matrix(detections)

        used_detections = set()

        for i, track in enumerate(self.tracks):
            best_det_idx = -1
            min_distance = float('inf')

            for j, detection in enumerate(detections):
                if j in used_detections:
                    continue

                if association_matrix[i, j] == 1:
                    det_x = detection.range * np.cos(detection.azimuth)
                    det_y = detection.range * np.sin(detection.azimuth)
                    pred_x, pred_y = track.state[0], track.state[1]
                    distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)

                    if distance < min_distance:
                        min_distance = distance
                        best_det_idx = j

            if best_det_idx >= 0:
                track.update(detections[best_det_idx])
                used_detections.add(best_det_idx)
                if track.age >= 3:
                    track.confirmed = True

        for j, detection in enumerate(detections):
            if j not in used_detections and len(self.tracks) < 500:
                new_track = Track(
                    track_id=self.track_id_counter,
                    state=np.array([
                        detection.range * np.cos(detection.azimuth),
                        detection.range * np.sin(detection.azimuth),
                        detection.range * np.sin(detection.elevation),
                        detection.velocity * np.cos(detection.azimuth),
                        detection.velocity * np.sin(detection.azimuth),
                        0.0
                    ])
                )
                new_track.update(detection)
                self.tracks.append(new_track)
                self.track_id_counter += 1

        self.tracks = [track for track in self.tracks if track.age < 10 or track.confirmed]

    def _build_association_matrix(self, detections: List[Detection]) -> np.ndarray:
        n_tracks = len(self.tracks)
        n_detections = len(detections)

        if n_tracks == 0 or n_detections == 0:
            return np.zeros((max(1, n_tracks), max(1, n_detections)))

        matrix = np.zeros((n_tracks, n_detections))
        gate_threshold = 1000.0

        for i, track in enumerate(self.tracks):
            for j, detection in enumerate(detections):
                det_x = detection.range * np.cos(detection.azimuth)
                det_y = detection.range * np.sin(detection.azimuth)
                pred_x, pred_y = track.state[0], track.state[1]

                distance = np.sqrt((pred_x - det_x) ** 2 + (pred_y - det_y) ** 2)

                if distance < gate_threshold:
                    matrix[i, j] = 1

        return matrix

    def _apply_imm_tracking(self, tracks: List[Track]):
        for track in tracks:
            velocity = np.sqrt(track.state[3] ** 2 + track.state[4] ** 2)

            if velocity > 200.0:
                track.high_mobility = True
                track.covariance *= 2.0
            else:
                track.high_mobility = False

    def _enhance_low_altitude_targets(self):
        for track in self.tracks:
            altitude = track.state[2]

            if altitude < 500.0:
                self._apply_stap_processing(track)
                self._apply_polarimetric_processing(track)

                if self.environment.terrain_type in ["hills", "urban"]:
                    self._fuse_with_ir_sensor(track)

    def _apply_stap_processing(self, track: Track):
        improvement_factor = 1.5 if track.state[2] < 200 else 1.2
        track.score *= improvement_factor

    def _apply_polarimetric_processing(self, track: Track):
        if len(track.rcs_history) > 0:
            rcs_variation = np.std(track.rcs_history)
            if rcs_variation < 2.0:
                track.score *= 1.2
                track.confirmed = True

    def _fuse_with_ir_sensor(self, track: Track):
        ir_confidence = 0.8
        current_range = np.sqrt(track.state[0] ** 2 + track.state[1] ** 2)

        noise_factor = np.random.normal(0, 10)
        ir_range = current_range + noise_factor

        fusion_weight = 0.6
        fused_range = fusion_weight * current_range + (1 - fusion_weight) * ir_range

        range_ratio = fused_range / current_range
        track.state[0] *= range_ratio
        track.state[1] *= range_ratio

        track.score *= (1.0 + ir_confidence * 0.5)
        track.confirmed = True
