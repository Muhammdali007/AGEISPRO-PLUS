from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from time import monotonic
from typing import Callable

from app.core.config import settings
from app.models.incident import DetectionType, IncidentPriority
from app.schemas.detections import DetectionEventIngestItem, RecognitionStatus


HAZARD_PRIORITIES: dict[DetectionType, IncidentPriority] = {
    DetectionType.weapon: IncidentPriority.critical,
    DetectionType.fire: IncidentPriority.critical,
    DetectionType.smoke: IncidentPriority.high,
}


@dataclass(slots=True)
class CameraSoundAlertState:
    unknown_scan_streak: int = 0
    unknown_last_alerted_at: float | None = None
    hazard_last_alerted_at: dict[DetectionType, float] = field(default_factory=dict)


class SoundAlertService:
    """Turn raw camera scans into rate-limited operator sound events.

    This service deliberately observes every inference result, including results
    that are later suppressed as duplicate incidents. Incident deduplication
    therefore cannot prevent the three-scan unknown-person confirmation gate.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = monotonic,
        unknown_scan_threshold: int | None = None,
        unknown_cooldown_seconds: float | None = None,
        hazard_cooldown_seconds: float | None = None,
    ) -> None:
        self._clock = clock
        self._unknown_scan_threshold = unknown_scan_threshold
        self._unknown_cooldown_seconds = unknown_cooldown_seconds
        self._hazard_cooldown_seconds = hazard_cooldown_seconds
        self._states: dict[str, CameraSoundAlertState] = {}
        self._lock = asyncio.Lock()

    async def observe_scan(
        self,
        *,
        camera_id: object,
        camera_name: str,
        detections: list[DetectionEventIngestItem],
        requested_detectors: set[str] | None = None,
    ) -> list[dict[str, object]]:
        requested = requested_detectors or {"weapon", "fire", "smoke", "person"}
        now = self._clock()

        async with self._lock:
            state = self._states.setdefault(str(camera_id), CameraSoundAlertState())
            events = self._observe_hazards(
                state=state,
                camera_id=camera_id,
                camera_name=camera_name,
                detections=detections,
                requested=requested,
                now=now,
            )
            unknown_event = self._observe_unknown_people(
                state=state,
                camera_id=camera_id,
                camera_name=camera_name,
                detections=detections,
                requested=requested,
                now=now,
            )
            if unknown_event is not None:
                events.append(unknown_event)
            return events

    def _observe_hazards(
        self,
        *,
        state: CameraSoundAlertState,
        camera_id: object,
        camera_name: str,
        detections: list[DetectionEventIngestItem],
        requested: set[str],
        now: float,
    ) -> list[dict[str, object]]:
        events: list[dict[str, object]] = []
        cooldown = (
            self._hazard_cooldown_seconds
            if self._hazard_cooldown_seconds is not None
            else settings.sound_alert_hazard_cooldown_seconds
        )

        for detection_type, priority in HAZARD_PRIORITIES.items():
            matching = [
                detection
                for detection in detections
                if self._resolved_type(detection) is detection_type
            ]
            if not matching:
                if detection_type.value in requested:
                    # The detector ran and the hazard disappeared. A future
                    # recurrence should alert immediately instead of waiting on
                    # the previous episode's cooldown.
                    state.hazard_last_alerted_at.pop(detection_type, None)
                continue

            last_alerted_at = state.hazard_last_alerted_at.get(detection_type)
            if last_alerted_at is not None and now - last_alerted_at < cooldown:
                continue

            strongest = max(matching, key=lambda item: item.confidence)
            state.hazard_last_alerted_at[detection_type] = now
            display_name = self._display_name(detection_type, strongest)
            events.append(
                {
                    "type": "sound.alert",
                    "camera_id": str(camera_id),
                    "camera_name": camera_name,
                    "detection_type": detection_type.value,
                    "priority": priority.value,
                    "confidence": strongest.confidence,
                    "scan_count": 1,
                    "message": f"{display_name} detected on camera {camera_name}.",
                }
            )
        return events

    def _observe_unknown_people(
        self,
        *,
        state: CameraSoundAlertState,
        camera_id: object,
        camera_name: str,
        detections: list[DetectionEventIngestItem],
        requested: set[str],
        now: float,
    ) -> dict[str, object] | None:
        if "person" not in requested and "unknown_person" not in requested:
            return None

        unknown_people = [
            detection
            for detection in detections
            if self._resolved_type(detection) is DetectionType.unknown_person
        ]
        if not unknown_people:
            state.unknown_scan_streak = 0
            state.unknown_last_alerted_at = None
            return None

        state.unknown_scan_streak += 1
        threshold = (
            self._unknown_scan_threshold
            if self._unknown_scan_threshold is not None
            else settings.sound_alert_unknown_scan_threshold
        )
        if state.unknown_scan_streak < threshold:
            return None

        cooldown = (
            self._unknown_cooldown_seconds
            if self._unknown_cooldown_seconds is not None
            else settings.sound_alert_unknown_cooldown_seconds
        )
        if (
            state.unknown_last_alerted_at is not None
            and now - state.unknown_last_alerted_at < cooldown
        ):
            return None

        strongest = max(unknown_people, key=lambda item: item.confidence)
        state.unknown_last_alerted_at = now
        return {
            "type": "sound.alert",
            "camera_id": str(camera_id),
            "camera_name": camera_name,
            "detection_type": DetectionType.unknown_person.value,
            "priority": IncidentPriority.medium.value,
            "confidence": strongest.confidence,
            "scan_count": state.unknown_scan_streak,
            "message": (
                f"Unknown person confirmed on camera {camera_name} "
                f"after {state.unknown_scan_streak} consecutive scans."
            ),
        }

    @staticmethod
    def _resolved_type(detection: DetectionEventIngestItem) -> DetectionType:
        if detection.detection_type is DetectionType.person:
            if detection.recognition_status is RecognitionStatus.known:
                return DetectionType.known_person
            if detection.recognition_status is RecognitionStatus.unknown:
                return DetectionType.unknown_person
        return detection.detection_type

    @staticmethod
    def _display_name(
        detection_type: DetectionType,
        detection: DetectionEventIngestItem,
    ) -> str:
        if detection_type is DetectionType.weapon and detection.bounding_box:
            label = (detection.bounding_box.label or "").strip().lower()
            if label and label not in {"weapon", "face"}:
                return label.replace("_", " ").title()
        return detection_type.value.replace("_", " ").title()


sound_alert_service = SoundAlertService()
