import pytest

from app.models.incident import DetectionType
from app.schemas.detections import (
    DetectionBoundingBox,
    DetectionEventIngestItem,
    RecognitionStatus,
)
from app.services.sound_alerts import SoundAlertService


class MutableClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def unknown_person(confidence: float = 0.88) -> DetectionEventIngestItem:
    return DetectionEventIngestItem(
        detection_type=DetectionType.person,
        confidence=confidence,
        track_id="person-1",
        recognition_status=RecognitionStatus.unknown,
    )


@pytest.mark.asyncio
async def test_unknown_person_alert_requires_three_consecutive_person_scans() -> None:
    clock = MutableClock()
    service = SoundAlertService(
        clock=clock,
        unknown_scan_threshold=3,
        unknown_cooldown_seconds=30,
        hazard_cooldown_seconds=10,
    )
    scan = {
        "camera_id": "camera-1",
        "camera_name": "North Gate",
        "requested_detectors": {"person"},
    }

    assert await service.observe_scan(detections=[unknown_person()], **scan) == []
    assert await service.observe_scan(detections=[unknown_person()], **scan) == []

    events = await service.observe_scan(detections=[unknown_person(0.93)], **scan)

    assert len(events) == 1
    assert events[0]["type"] == "sound.alert"
    assert events[0]["detection_type"] == "unknown_person"
    assert events[0]["scan_count"] == 3
    assert events[0]["confidence"] == 0.93
    assert "3 consecutive scans" in str(events[0]["message"])

    # A continuously visible person does not create a sound on every frame.
    assert await service.observe_scan(detections=[unknown_person()], **scan) == []


@pytest.mark.asyncio
async def test_unknown_person_streak_resets_after_a_clear_person_scan() -> None:
    service = SoundAlertService(
        unknown_scan_threshold=3,
        unknown_cooldown_seconds=30,
        hazard_cooldown_seconds=10,
    )
    scan = {
        "camera_id": "camera-2",
        "camera_name": "Lobby",
        "requested_detectors": {"person"},
    }

    assert await service.observe_scan(detections=[unknown_person()], **scan) == []
    assert await service.observe_scan(detections=[], **scan) == []
    assert await service.observe_scan(detections=[unknown_person()], **scan) == []
    assert await service.observe_scan(detections=[unknown_person()], **scan) == []
    events = await service.observe_scan(detections=[unknown_person()], **scan)

    assert len(events) == 1
    assert events[0]["scan_count"] == 3


@pytest.mark.asyncio
async def test_weapon_fire_and_smoke_alert_immediately_with_episode_cooldown() -> None:
    clock = MutableClock()
    service = SoundAlertService(
        clock=clock,
        unknown_scan_threshold=3,
        unknown_cooldown_seconds=30,
        hazard_cooldown_seconds=10,
    )
    hazards = [
        DetectionEventIngestItem(
            detection_type=DetectionType.weapon,
            confidence=0.94,
            bounding_box=DetectionBoundingBox(
                x1=1,
                y1=1,
                x2=10,
                y2=10,
                label="firearm",
            ),
        ),
        DetectionEventIngestItem(detection_type=DetectionType.fire, confidence=0.91),
        DetectionEventIngestItem(detection_type=DetectionType.smoke, confidence=0.86),
    ]
    scan = {
        "camera_id": "camera-3",
        "camera_name": "Warehouse",
        "requested_detectors": {"weapon", "fire", "smoke", "person"},
    }

    events = await service.observe_scan(detections=hazards, **scan)

    assert {event["detection_type"] for event in events} == {"weapon", "fire", "smoke"}
    assert all(event["scan_count"] == 1 for event in events)
    assert next(event for event in events if event["detection_type"] == "weapon")[
        "message"
    ].startswith("Firearm detected")

    clock.advance(1)
    assert await service.observe_scan(detections=hazards, **scan) == []

    # A clear detector pass closes the episode, so recurrence is immediate.
    assert await service.observe_scan(detections=[], **scan) == []
    repeated = await service.observe_scan(detections=hazards[:1], **scan)
    assert [event["detection_type"] for event in repeated] == ["weapon"]
