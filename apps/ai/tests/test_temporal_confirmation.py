from uuid import uuid4

from app.schemas.inference import InferenceBox, InferenceRecognition, InferenceRequest
from app.services.temporal_confirmation import TemporalDetectionConfirmation


def request() -> InferenceRequest:
    return InferenceRequest(
        camera_id=uuid4(),
        frame_reference="live-frame",
        source_type="rtsp",
        occurrence_hint="continuous_monitoring",
    )


def box(label: str, confidence: float, *, offset: float = 0) -> InferenceBox:
    return InferenceBox(
        x1=10 + offset,
        y1=10,
        x2=110 + offset,
        y2=110,
        confidence=confidence,
        label=label,
    )


def test_borderline_weapon_requires_two_matching_frames(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.weapon_immediate_confidence", 0.60)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.weapon_confirmation_frames", 2)
    confirmation = TemporalDetectionConfirmation()
    payload = request()

    first, first_suppressed = confirmation.filter(payload, [box("weapon", 0.40)])
    second, second_suppressed = confirmation.filter(payload, [box("weapon", 0.42, offset=5)])

    assert len(first) == 1
    assert first[0].provisional is True
    assert first_suppressed == 1
    assert len(second) == 1
    assert second_suppressed == 0


def test_borderline_fire_requires_configured_matching_frames(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_immediate_confidence", 0.65)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_confirmation_frames", 3)
    confirmation = TemporalDetectionConfirmation()
    payload = request()

    first, _ = confirmation.filter(payload, [box("fire", 0.25)])
    second, _ = confirmation.filter(payload, [box("fire", 0.27, offset=2)])
    third, third_suppressed = confirmation.filter(payload, [box("fire", 0.29, offset=4)])

    assert len(first) == 1 and first[0].provisional is True
    assert len(second) == 1 and second[0].provisional is True
    assert len(third) == 1
    assert third_suppressed == 0


def test_fire_confirmation_survives_interleaved_fast_lane_scans(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_immediate_confidence", 0.65)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_confirmation_frames", 3)
    confirmation = TemporalDetectionConfirmation()
    hazard_payload = request().model_copy(update={"requested_detectors": ["fire", "smoke"]})
    fast_payload = request().model_copy(update={"requested_detectors": ["weapon", "person"]})

    first, _ = confirmation.filter(hazard_payload, [box("fire", 0.25)])
    confirmation.filter(fast_payload, [box("person", 0.90)])
    second, _ = confirmation.filter(hazard_payload, [box("fire", 0.27, offset=2)])
    confirmation.filter(fast_payload, [box("person", 0.91, offset=1)])
    third, third_suppressed = confirmation.filter(
        hazard_payload,
        [box("fire", 0.29, offset=4)],
    )

    assert len(first) == 1 and first[0].provisional is True
    assert len(second) == 1 and second[0].provisional is True
    assert len(third) == 1
    assert third_suppressed == 0


def test_single_missing_detection_keeps_candidate_for_a_requested_lane(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_immediate_confidence", 0.65)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_confirmation_frames", 2)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_allowed_misses", 1)
    confirmation = TemporalDetectionConfirmation()
    hazard_payload = request().model_copy(update={"requested_detectors": ["fire", "smoke"]})

    confirmation.filter(hazard_payload, [box("fire", 0.25)])
    confirmation.filter(hazard_payload, [])
    after_gap, suppressed = confirmation.filter(hazard_payload, [box("fire", 0.27)])

    assert len(after_gap) == 1
    assert after_gap[0].provisional is False
    assert suppressed == 0


def test_dynamic_fire_box_confirms_after_scale_change(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_immediate_confidence", 0.65)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.fire_confirmation_frames", 2)
    confirmation = TemporalDetectionConfirmation()
    payload = request().model_copy(update={"requested_detectors": ["fire", "smoke"]})

    first, _ = confirmation.filter(payload, [box("fire", 0.20)])
    expanded = InferenceBox(
        x1=0,
        y1=0,
        x2=160,
        y2=160,
        confidence=0.22,
        label="fire",
    )
    second, suppressed = confirmation.filter(payload, [expanded])

    assert len(first) == 1 and first[0].provisional is True
    assert len(second) == 1
    assert suppressed == 0


def test_distant_threat_does_not_confirm_previous_candidate(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.smoke_immediate_confidence", 0.70)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.smoke_confirmation_frames", 2)
    confirmation = TemporalDetectionConfirmation()
    payload = request().model_copy(update={"requested_detectors": ["fire", "smoke"]})

    confirmation.filter(payload, [box("smoke", 0.20)])
    distant, suppressed = confirmation.filter(payload, [box("smoke", 0.22, offset=250)])

    assert len(distant) == 1
    assert distant[0].provisional is True
    assert suppressed == 1


def test_strong_weapon_is_immediate(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.weapon_immediate_confidence", 0.65)

    confirmed, suppressed = TemporalDetectionConfirmation().filter(request(), [box("weapon", 0.81)])

    assert len(confirmed) == 1
    assert suppressed == 0


def test_known_person_requires_repeat_identity_when_not_high_confidence(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.recognition_confirmation_frames", 2)
    identity_id = uuid4()
    detection = box("known_person", 0.95).model_copy(
        update={
            "recognition": InferenceRecognition(
                status="known",
                identity_id=identity_id,
                identity_label="Known Person",
                match_confidence=0.72,
            )
        }
    )
    confirmation = TemporalDetectionConfirmation()
    payload = request()

    first, _ = confirmation.filter(payload, [detection])
    second, _ = confirmation.filter(payload, [detection])

    assert first == []
    assert len(second) == 1


def test_manual_scan_is_not_delayed(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    payload = request().model_copy(update={"occurrence_hint": "manual_scan"})

    confirmed, suppressed = TemporalDetectionConfirmation().filter(payload, [box("smoke", 0.45)])

    assert len(confirmed) == 1
    assert suppressed == 0


def test_dashboard_live_scan_uses_temporal_confirmation(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_confirmation.settings.temporal_confirmation_enabled", True)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.weapon_confirmation_frames", 2)
    monkeypatch.setattr("app.services.temporal_confirmation.settings.weapon_immediate_confidence", 0.70)
    payload = request().model_copy(update={"occurrence_hint": "dashboard_live_scan"})
    confirmation = TemporalDetectionConfirmation()

    first, suppressed = confirmation.filter(payload, [box("weapon", 0.50)])
    second, _ = confirmation.filter(payload, [box("weapon", 0.52, offset=2)])

    assert len(first) == 1
    assert first[0].provisional is True
    assert suppressed == 1
    assert len(second) == 1
