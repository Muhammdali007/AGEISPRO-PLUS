from uuid import uuid4

from app.schemas.inference import InferenceBox, InferenceRequest
from app.services.temporal_tracking import TemporalBoxTracker


def _request(camera_id=None, occurrence_hint: str = "dashboard_live_scan") -> InferenceRequest:
    return InferenceRequest(
        camera_id=camera_id or uuid4(),
        frame_reference="browser-frame",
        source_type="usb",
        frame_content_base64="frame",
        occurrence_hint=occurrence_hint,
        requested_detectors=["person"],
    )


def _person(x1: float, x2: float) -> InferenceBox:
    return InferenceBox(
        x1=x1,
        y1=0,
        x2=x2,
        y2=200,
        confidence=0.9,
        label="person",
        track_id="pe-0",
    )


def _threat(label: str, x1: float, x2: float) -> InferenceBox:
    return InferenceBox(
        x1=x1,
        y1=20,
        x2=x2,
        y2=120,
        confidence=0.8,
        label=label,
    )


def test_snapshot_tracker_keeps_ids_during_fast_crossing_motion(monkeypatch) -> None:
    times = iter((0.0, 0.2, 0.4))
    monkeypatch.setattr("app.services.temporal_tracking.monotonic", lambda: next(times))
    tracker = TemporalBoxTracker()
    request = _request()

    first = tracker.update(request, [_person(0, 100), _person(300, 400)])
    second = tracker.update(request, [_person(80, 180), _person(220, 320)])
    third = tracker.update(request, [_person(180, 280), _person(120, 220)])

    assert [item.track_id for item in first] == ["pe-t1", "pe-t2"]
    assert [item.track_id for item in second] == ["pe-t1", "pe-t2"]
    assert [item.track_id for item in third] == ["pe-t1", "pe-t2"]


def test_snapshot_tracker_isolated_per_camera(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_tracking.monotonic", lambda: 1.0)
    tracker = TemporalBoxTracker()

    first_camera = tracker.update(_request(), [_person(0, 100)])
    second_camera = tracker.update(_request(), [_person(0, 100)])

    assert first_camera[0].track_id == "pe-t1"
    assert second_camera[0].track_id == "pe-t1"


def test_snapshot_tracker_does_not_change_manual_scan_ids(monkeypatch) -> None:
    monkeypatch.setattr("app.services.temporal_tracking.monotonic", lambda: 1.0)
    tracker = TemporalBoxTracker()

    result = tracker.update(_request(occurrence_hint="manual_scan"), [_person(0, 100)])

    assert result[0].track_id == "pe-0"


def test_snapshot_tracker_does_not_recycle_ids_after_track_expiry(monkeypatch) -> None:
    clock = [0.0]
    monkeypatch.setattr("app.services.temporal_tracking.monotonic", lambda: clock[0])
    tracker = TemporalBoxTracker()
    request = _request()

    first = tracker.update(request, [_person(0, 100)])
    clock[0] = 4.0
    tracker.update(request, [])
    replacement = tracker.update(request, [_person(0, 100)])

    assert first[0].track_id == "pe-t1"
    assert replacement[0].track_id == "pe-t2"


def test_snapshot_tracker_assigns_stable_ids_to_threat_objects(monkeypatch) -> None:
    times = iter((0.0, 0.2))
    monkeypatch.setattr("app.services.temporal_tracking.monotonic", lambda: next(times))
    tracker = TemporalBoxTracker()
    request = _request()

    first = tracker.update(
        request,
        [_person(0, 100), _threat("weapon", 150, 210), _threat("fire", 300, 390)],
    )
    second = tracker.update(
        request,
        [_person(8, 108), _threat("weapon", 158, 218), _threat("fire", 310, 400)],
    )

    assert [item.track_id for item in first] == ["pe-t1", "we-t2", "fi-t3"]
    assert [item.track_id for item in second] == ["pe-t1", "we-t2", "fi-t3"]
