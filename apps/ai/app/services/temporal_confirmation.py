from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from app.core.config import settings
from app.schemas.inference import InferenceBox, InferenceRequest
from app.services.backends import normalize_requested_detectors


@dataclass(slots=True)
class CandidateState:
    detection: InferenceBox
    observations: int
    last_seen_at: float
    missed_observations: int = 0


class TemporalDetectionConfirmation:
    """Require repeat evidence for borderline continuous-monitoring detections."""

    _live_occurrence_hints = {"continuous_monitoring", "dashboard_live_scan"}

    def __init__(self) -> None:
        self._candidates: dict[str, list[CandidateState]] = {}

    def filter(
        self,
        payload: InferenceRequest,
        detections: list[InferenceBox],
    ) -> tuple[list[InferenceBox], int]:
        if (
            not settings.temporal_confirmation_enabled
            or payload.occurrence_hint not in self._live_occurrence_hints
        ):
            return detections, 0

        now = monotonic()
        camera_key = str(payload.camera_id)
        previous = [
            state
            for state in self._candidates.get(camera_key, [])
            if now - state.last_seen_at <= settings.temporal_confirmation_max_gap_seconds
        ]
        requested_detectors = set(normalize_requested_detectors(payload.requested_detectors))
        # Fast person/weapon scans are interleaved with the lower-frequency
        # fire/smoke lane. A scan cannot disprove a candidate for a detector it
        # did not run, so preserve that lane's fresh confirmation state until
        # its next observation. Previously every fast scan erased hazard state,
        # making three-frame fire/smoke confirmation practically impossible.
        next_states: list[CandidateState] = [
            state
            for state in previous
            if self._detector_lane(state.detection.label) not in requested_detectors
        ]
        visible: list[InferenceBox] = []
        matched_state_ids: set[int] = set()
        suppressed = 0

        for detection in detections:
            required = self._required_observations(detection.label)
            if required == 1 or self._is_immediate(detection):
                visible.append(detection)
                continue

            match = max(
                (state for state in previous if self._same_object(state.detection, detection)),
                key=lambda state: self._intersection_over_union(state.detection, detection),
                default=None,
            )
            if match is not None:
                matched_state_ids.add(id(match))
            state = CandidateState(
                detection=detection,
                observations=(match.observations + 1) if match else 1,
                last_seen_at=now,
            )
            next_states.append(state)
            if state.observations >= required:
                visible.append(detection)
            else:
                if detection.label in {"weapon", "fire", "smoke"}:
                    visible.append(detection.model_copy(update={"provisional": True}))
                suppressed += 1

        # A low-confidence detector can flicker for one frame while an object is
        # moving or partly occluded. Preserve that evidence briefly so the next
        # nearby hit can confirm quickly, but never expose a missing box or turn
        # the miss itself into an event.
        for state in previous:
            if (
                self._detector_lane(state.detection.label) in requested_detectors
                and id(state) not in matched_state_ids
                and state.missed_observations < settings.temporal_confirmation_allowed_misses
            ):
                next_states.append(
                    CandidateState(
                        detection=state.detection,
                        observations=state.observations,
                        last_seen_at=state.last_seen_at,
                        missed_observations=state.missed_observations + 1,
                    )
                )

        self._candidates[camera_key] = next_states
        self._prune(now)
        return visible, suppressed

    @staticmethod
    def _required_observations(label: str) -> int:
        return {
            "weapon": settings.weapon_confirmation_frames,
            "fire": settings.fire_confirmation_frames,
            "smoke": settings.smoke_confirmation_frames,
            "known_person": settings.recognition_confirmation_frames,
        }.get(label, 1)

    @staticmethod
    def _detector_lane(label: str) -> str:
        if label in {"person", "known_person", "unknown_person"}:
            return "person"
        return label

    @staticmethod
    def _is_immediate(detection: InferenceBox) -> bool:
        if detection.label == "known_person":
            score = detection.recognition.match_confidence if detection.recognition else None
            return score is not None and score >= settings.recognition_immediate_confidence
        threshold = {
            "weapon": settings.weapon_immediate_confidence,
            "fire": settings.fire_immediate_confidence,
            "smoke": settings.smoke_immediate_confidence,
        }.get(detection.label, 0.0)
        return detection.confidence >= threshold

    @classmethod
    def _same_object(cls, first: InferenceBox, second: InferenceBox) -> bool:
        if first.label != second.label:
            return False
        if first.label == "known_person":
            first_id = first.recognition.identity_id if first.recognition else None
            second_id = second.recognition.identity_id if second.recognition else None
            return first_id is not None and first_id == second_id
        if cls._intersection_over_union(first, second) >= 0.25:
            return True
        if first.label not in {"weapon", "fire", "smoke"}:
            return False

        # Threat boxes are less stable than person boxes: flames and smoke grow,
        # while a hand-held weapon can move substantially between snapshots.
        # Keep confirmation spatially strict, but accept scale changes and short
        # movements that plain IoU would incorrectly treat as a new candidate.
        return (
            cls._smaller_box_coverage(first, second) >= 0.40
            or cls._normalized_center_distance(first, second) <= 0.35
        )

    @staticmethod
    def _smaller_box_coverage(first: InferenceBox, second: InferenceBox) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
        second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
        smaller_area = min(first_area, second_area)
        return intersection / smaller_area if smaller_area else 0.0

    @staticmethod
    def _normalized_center_distance(first: InferenceBox, second: InferenceBox) -> float:
        first_center_x = (first.x1 + first.x2) / 2
        first_center_y = (first.y1 + first.y2) / 2
        second_center_x = (second.x1 + second.x2) / 2
        second_center_y = (second.y1 + second.y2) / 2
        center_distance = (
            (first_center_x - second_center_x) ** 2
            + (first_center_y - second_center_y) ** 2
        ) ** 0.5
        first_diagonal = (
            max(0.0, first.x2 - first.x1) ** 2
            + max(0.0, first.y2 - first.y1) ** 2
        ) ** 0.5
        second_diagonal = (
            max(0.0, second.x2 - second.x1) ** 2
            + max(0.0, second.y2 - second.y1) ** 2
        ) ** 0.5
        scale = max(first_diagonal, second_diagonal, 1.0)
        return center_distance / scale

    @staticmethod
    def _intersection_over_union(first: InferenceBox, second: InferenceBox) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
        second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _prune(self, now: float) -> None:
        for camera_key in list(self._candidates):
            if not self._candidates[camera_key] or all(
                now - state.last_seen_at > settings.temporal_confirmation_max_gap_seconds
                for state in self._candidates[camera_key]
            ):
                del self._candidates[camera_key]
