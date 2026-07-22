from __future__ import annotations

import math
from dataclasses import dataclass
from time import monotonic

from app.core.config import settings
from app.schemas.inference import InferenceBox, InferenceRequest


@dataclass(slots=True)
class SnapshotTrack:
    track_id: str
    detection: InferenceBox
    observed_at: float
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


class TemporalBoxTracker:
    """Associate detections across independently submitted camera frames.

    Ultralytics trackers are designed for a continuous iterator. Browser frames
    arrive as separate HTTP requests, and batch inference also uses ``predict``,
    so their result indices are not durable track IDs. This small motion tracker
    gives those paths stable IDs without sharing tracker state between cameras.
    Raw detector coordinates remain unchanged; velocity is used only to predict
    the next association and therefore does not add visual smoothing delay.
    """

    _live_occurrence_hints = {"continuous_monitoring", "dashboard_live_scan"}

    def __init__(self) -> None:
        self._tracks: dict[str, list[SnapshotTrack]] = {}
        self._next_track_number: dict[str, int] = {}

    def update(
        self,
        payload: InferenceRequest,
        detections: list[InferenceBox],
    ) -> list[InferenceBox]:
        if (
            not settings.model_enable_tracking
            or payload.occurrence_hint not in self._live_occurrence_hints
        ):
            return detections

        now = monotonic()
        camera_key = str(payload.camera_id)
        previous = [
            track
            for track in self._tracks.get(camera_key, [])
            if now - track.observed_at <= settings.model_snapshot_track_max_age_seconds
        ]
        person_indices = [
            index for index, detection in enumerate(detections) if detection.label == "person"
        ]
        if not person_indices:
            self._tracks[camera_key] = previous
            self._prune(now)
            return detections

        assignments = self._associate(previous, detections, person_indices, now)
        updated = list(detections)
        next_tracks: list[SnapshotTrack] = []
        matched_track_indices: set[int] = set()

        for detection_index in person_indices:
            detection = detections[detection_index]
            track_index = assignments.get(detection_index)
            if track_index is None:
                track = SnapshotTrack(
                    track_id=self._new_track_id(camera_key),
                    detection=detection.model_copy(deep=True),
                    observed_at=now,
                )
            else:
                matched_track_indices.add(track_index)
                track = self._advance_track(previous[track_index], detection, now)

            tracked_detection = detection.model_copy(update={"track_id": track.track_id})
            track.detection = tracked_detection.model_copy(deep=True)
            updated[detection_index] = tracked_detection
            next_tracks.append(track)

        # Keep briefly missed tracks so a single detector dropout does not
        # replace a person's identity as soon as they reappear.
        next_tracks.extend(
            track for index, track in enumerate(previous) if index not in matched_track_indices
        )
        self._tracks[camera_key] = next_tracks
        self._prune(now)
        return updated

    def _associate(
        self,
        tracks: list[SnapshotTrack],
        detections: list[InferenceBox],
        detection_indices: list[int],
        now: float,
    ) -> dict[int, int]:
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            predicted = self._predict(track, now)
            for detection_index in detection_indices:
                detection = detections[detection_index]
                score = self._association_score(predicted, detection)
                if score is not None:
                    candidates.append((score, track_index, detection_index))

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, track_index, detection_index in sorted(candidates, reverse=True):
            if track_index in used_tracks or detection_index in assignments:
                continue
            used_tracks.add(track_index)
            assignments[detection_index] = track_index
        return assignments

    @staticmethod
    def _association_score(
        predicted: InferenceBox,
        detection: InferenceBox,
    ) -> float | None:
        overlap = TemporalBoxTracker._intersection_over_union(predicted, detection)
        center_distance = TemporalBoxTracker._normalized_center_distance(predicted, detection)
        first_area = TemporalBoxTracker._area(predicted)
        second_area = TemporalBoxTracker._area(detection)
        size_similarity = (
            min(first_area, second_area) / max(first_area, second_area)
            if first_area and second_area
            else 0.0
        )
        if overlap < settings.model_snapshot_track_min_iou and (
            center_distance > settings.model_snapshot_track_max_center_distance
            or size_similarity < 0.30
        ):
            return None
        distance_score = max(
            0.0,
            1.0 - center_distance / settings.model_snapshot_track_max_center_distance,
        )
        return (0.70 * overlap) + (0.20 * distance_score) + (0.10 * size_similarity)

    @staticmethod
    def _predict(track: SnapshotTrack, now: float) -> InferenceBox:
        elapsed = min(
            max(now - track.observed_at, 0.0),
            settings.model_snapshot_track_max_age_seconds,
        )
        return track.detection.model_copy(
            update={
                "x1": max(0.0, track.detection.x1 + track.velocity[0] * elapsed),
                "y1": max(0.0, track.detection.y1 + track.velocity[1] * elapsed),
                "x2": max(0.0, track.detection.x2 + track.velocity[2] * elapsed),
                "y2": max(0.0, track.detection.y2 + track.velocity[3] * elapsed),
            }
        )

    @staticmethod
    def _advance_track(
        previous: SnapshotTrack,
        detection: InferenceBox,
        now: float,
    ) -> SnapshotTrack:
        elapsed = max(now - previous.observed_at, 1e-3)
        measured_velocity = (
            (detection.x1 - previous.detection.x1) / elapsed,
            (detection.y1 - previous.detection.y1) / elapsed,
            (detection.x2 - previous.detection.x2) / elapsed,
            (detection.y2 - previous.detection.y2) / elapsed,
        )
        alpha = settings.model_snapshot_track_velocity_alpha
        velocity = (
            measured_velocity
            if not any(previous.velocity)
            else tuple(
                (alpha * measured) + ((1.0 - alpha) * existing)
                for measured, existing in zip(measured_velocity, previous.velocity)
            )
        )
        return SnapshotTrack(
            track_id=previous.track_id,
            detection=detection.model_copy(deep=True),
            observed_at=now,
            velocity=velocity,
        )

    def _new_track_id(self, camera_key: str) -> str:
        next_number = self._next_track_number.get(camera_key, 0) + 1
        self._next_track_number[camera_key] = next_number
        return f"pe-t{next_number}"

    @staticmethod
    def _normalized_center_distance(first: InferenceBox, second: InferenceBox) -> float:
        first_center = ((first.x1 + first.x2) / 2, (first.y1 + first.y2) / 2)
        second_center = ((second.x1 + second.x2) / 2, (second.y1 + second.y2) / 2)
        distance = math.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )
        first_diagonal = math.hypot(first.x2 - first.x1, first.y2 - first.y1)
        second_diagonal = math.hypot(second.x2 - second.x1, second.y2 - second.y1)
        return distance / max((first_diagonal + second_diagonal) / 2, 1.0)

    @staticmethod
    def _area(detection: InferenceBox) -> float:
        return max(0.0, detection.x2 - detection.x1) * max(0.0, detection.y2 - detection.y1)

    @staticmethod
    def _intersection_over_union(first: InferenceBox, second: InferenceBox) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = TemporalBoxTracker._area(first) + TemporalBoxTracker._area(second) - intersection
        return intersection / union if union else 0.0

    def _prune(self, now: float) -> None:
        for camera_key in list(self._tracks):
            fresh = [
                track
                for track in self._tracks[camera_key]
                if now - track.observed_at <= settings.model_snapshot_track_max_age_seconds
            ]
            if fresh:
                self._tracks[camera_key] = fresh
            else:
                del self._tracks[camera_key]
