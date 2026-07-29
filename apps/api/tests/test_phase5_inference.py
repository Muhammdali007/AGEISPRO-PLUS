from collections.abc import AsyncIterator
import asyncio
import base64

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.metadata import Base
from app.models.camera import CameraSourceType
from app.models.incident import DetectionType, IncidentPriority
from app.repositories.alerts import AlertRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.persons import PersonRepository
from app.schemas.cameras import CameraCreate, CameraDetectionScanRequest, CameraUpdate
from app.schemas.persons import PersonCreate
from app.schemas.detections import DetectionBoundingBox, DetectionEventIngest, DetectionEventIngestItem
from app.db.session import get_db
from app.main import app
from httpx import ASGITransport, AsyncClient
from app.services.camera_detection import CameraDetectionService
from app.services.continuous_detection import CameraJobState, ContinuousDetectionWorker
from app.services.detection_events import DetectionEventService
from app.services.media_agent import MediaFrame


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as test_session:
        yield test_session

    await engine.dispose()


@pytest.mark.asyncio
async def test_detection_ingest_creates_incidents_and_alerts(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    service = DetectionEventService(session)

    camera = await cameras.create(
        CameraCreate(name="Dock", source_type=CameraSourceType.http, source="http://camera/dock")
    )

    response = await service.ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="yolo11",
            model_version="phase5-sim",
            inference_fps=5,
            source_fps=25,
            snapshot_path="storage/incidents/dock/weapon.jpg",
            detections=[
                DetectionEventIngestItem(
                    detection_type=DetectionType.weapon,
                    confidence=0.96,
                    track_id="trk-9",
                    bounding_box=DetectionBoundingBox(x1=10, y1=20, x2=120, y2=180, label="weapon"),
                ),
                DetectionEventIngestItem(
                    detection_type=DetectionType.person,
                    confidence=0.88,
                    track_id="trk-10",
                    bounding_box=DetectionBoundingBox(x1=40, y1=10, x2=200, y2=220, label="person"),
                ),
            ],
            metadata={"pipeline": "phase5"},
        )
    )

    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    alerts = await AlertRepository(session).list()

    assert response.incident_count == 2
    assert response.alert_count == 1
    assert response.ignored_count == 0
    assert len(incidents) == 2
    assert len(alerts) == 1
    assert incidents[0].metadata_["model_name"] == "yolo11"
    assert incidents[0].metadata_["track_id"] in {"trk-9", "trk-10"}


@pytest.mark.asyncio
async def test_detection_ingest_ignores_disabled_cameras(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    service = DetectionEventService(session)

    camera = await cameras.create(
        CameraCreate(name="Vault", source_type=CameraSourceType.rtsp, source="rtsp://camera/vault")
    )
    await cameras.update(camera, CameraUpdate(detection_enabled=False))

    response = await service.ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="yolo11",
            detections=[
                DetectionEventIngestItem(
                    detection_type=DetectionType.fire,
                    confidence=0.91,
                )
            ],
        )
    )

    assert response.incident_count == 0
    assert response.alert_count == 0
    assert response.ignored_count == 1
    assert response.ignored_reasons == ["Camera detection is disabled."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("detection_type", "expected_priority"),
    [
        (DetectionType.weapon, IncidentPriority.critical),
        (DetectionType.fire, IncidentPriority.critical),
        (DetectionType.smoke, IncidentPriority.high),
    ],
    ids=["weapon", "fire", "smoke"],
)
async def test_detection_ingest_maps_emergency_detections_to_expected_priority(
    session: AsyncSession,
    detection_type: DetectionType,
    expected_priority: IncidentPriority,
) -> None:
    cameras = CameraRepository(session)
    service = DetectionEventService(session)

    camera = await cameras.create(
        CameraCreate(name="Plant", source_type=CameraSourceType.usb, source="0")
    )

    response = await service.ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="yolo11",
            detections=[
                DetectionEventIngestItem(
                    detection_type=detection_type,
                    confidence=0.73,
                )
            ],
        )
    )

    assert response.results[0].priority is expected_priority


@pytest.mark.asyncio
async def test_detection_ingest_suppresses_recent_duplicate_tracks(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    service = DetectionEventService(session)
    camera = await cameras.create(
        CameraCreate(name="Dedup Cam", source_type=CameraSourceType.http, source="http://camera")
    )
    payload = DetectionEventIngest(
        camera_id=camera.id,
        model_name="dedup-test",
        detections=[
            DetectionEventIngestItem(
                detection_type=DetectionType.fire,
                confidence=0.91,
                track_id="fi-42",
                bounding_box=DetectionBoundingBox(x1=10, y1=10, x2=100, y2=100),
            )
        ],
    )

    first = await service.ingest(payload)
    second = await service.ingest(payload)

    assert first.incident_count == 1
    assert second.incident_count == 0
    assert second.ignored_count == 1
    assert "Duplicate fire detection" in second.ignored_reasons[0]


@pytest.mark.asyncio
async def test_detection_ingest_accepts_configured_service_token(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    camera = await cameras.create(
        CameraCreate(name="Service Token Cam", source_type=CameraSourceType.http, source="http://camera/service")
    )

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield session

    monkeypatch.setattr("app.api.deps.settings.service_callback_token", "phase5-service-token")
    app.dependency_overrides[get_db] = override_get_db

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post(
                "/api/v1/detections/ingest",
                headers={"X-Service-Token": "phase5-service-token"},
                json={
                    "camera_id": str(camera.id),
                    "model_name": "phase5-service",
                    "detections": [{"detection_type": "fire", "confidence": 0.97}],
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 202
    assert response.json()["incident_count"] == 1


@pytest.mark.asyncio
async def test_camera_scan_runs_inference_and_ingests_alerts(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    persons = PersonRepository(session)
    service = CameraDetectionService(session)

    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-jpeg")

    camera = await cameras.create(
        CameraCreate(name="Scan Cam", source_type=CameraSourceType.file, source="frame.jpg")
    )
    await persons.create(
        PersonCreate(
            full_name="Dana Rivers",
            person_type="employee",
            is_active=True,
        )
    )

    async def fake_run_inference(*args, **kwargs):
        known_persons = kwargs["known_persons"]
        assert len(known_persons) == 1
        return {
            "camera_id": str(camera.id),
            "model_name": "yolo11",
            "model_version": "test-scan",
            "occurred_at": "2026-07-08T12:00:00Z",
            "inference_fps": 5.0,
            "source_fps": 15.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.93,
                    "label": "weapon",
                    "object_label": "kitchen_knife",
                    "track_id": "we-1",
                }
            ],
            "metadata": {"backend": "simulated"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)

    response = await service.run_scan(camera.id, payload=CameraDetectionScanRequest(include_evidence=True))

    alerts = await AlertRepository(session).list()
    incidents = await IncidentRepository(session).list(camera_id=camera.id)

    assert response.detection_count == 1
    assert response.incident_count == 1
    assert response.alert_count == 1
    assert response.backend == "simulated"
    assert response.detections[0].object_label == "kitchen_knife"
    assert response.detections[0].bounding_box is not None
    assert response.detections[0].bounding_box.label == "kitchen_knife"
    assert len(alerts) == 1
    assert alerts[0].title == "Kitchen Knife detected"
    assert alerts[0].message.startswith("Kitchen Knife detected on camera Scan Cam")
    assert len(incidents) == 1
    assert incidents[0].detection_type is DetectionType.weapon


@pytest.mark.asyncio
async def test_camera_scan_allows_manual_scan_when_detection_is_paused(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)

    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    image_path = tmp_path / "frame.jpg"
    image_path.write_bytes(b"fake-jpeg")

    camera = await cameras.create(
        CameraCreate(name="Paused Scan Cam", source_type=CameraSourceType.file, source="frame.jpg")
    )
    await cameras.update(camera, CameraUpdate(detection_enabled=False))

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "yolo11",
            "model_version": "test-scan",
            "occurred_at": "2026-07-08T12:00:00Z",
            "inference_fps": 5.0,
            "source_fps": 15.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.81,
                    "label": "smoke",
                    "track_id": "sm-1",
                }
            ],
            "metadata": {"backend": "simulated"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)

    response = await service.run_scan(camera.id, payload=CameraDetectionScanRequest(include_evidence=True))

    incidents = await IncidentRepository(session).list(camera_id=camera.id)

    assert response.detection_count == 1
    assert response.incident_count == 1
    assert len(incidents) == 1
    assert incidents[0].detection_type is DetectionType.smoke


@pytest.mark.asyncio
async def test_camera_scan_explains_unsupported_model_classes(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)
    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    (tmp_path / "frame.jpg").write_bytes(b"fake-jpeg")
    camera = await cameras.create(
        CameraCreate(name="Capability Cam", source_type=CameraSourceType.file, source="frame.jpg")
    )

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "yolo11n",
            "model_version": "coco",
            "inference_fps": 5.0,
            "detections": [],
            "metadata": {
                "backend": "ultralytics",
                "unsupported_requested_detectors": ["weapon", "fire", "smoke"],
            },
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    response = await service.run_scan(camera.id, CameraDetectionScanRequest())

    assert response.detection_count == 0
    assert response.ignored_reasons == [
        "The configured model has no classes for: weapon, fire, smoke."
    ]


@pytest.mark.asyncio
async def test_camera_scan_reads_usb_camera_without_browser_frame(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)

    camera = await cameras.create(
        CameraCreate(name="USB Cam", source_type=CameraSourceType.usb, source="0")
    )

    monkeypatch.setattr(
        service.media_agent,
        "capture_opencv_frame",
        lambda source, display_source=None: MediaFrame(content_base64="encoded-frame", content_type="image/jpeg"),
    )

    async def fake_run_inference(*args, **kwargs):
        assert kwargs["frame_content_base64"] == "encoded-frame"
        return {
            "camera_id": str(camera.id),
            "model_name": "usb-test",
            "model_version": "1",
            "inference_fps": 1.0,
            "detections": [],
            "metadata": {"backend": "ultralytics"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    result = await service.run_scan(camera.id, payload=CameraDetectionScanRequest())

    assert result.detection_count == 0


@pytest.mark.asyncio
async def test_recorded_video_scans_advance_through_playback(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)
    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    monkeypatch.setattr(
        "app.services.camera_detection.settings.file_video_scan_step_seconds",
        0.5,
    )
    source = tmp_path / "recording.mp4"
    source.write_bytes(b"fake-video")
    camera = await cameras.create(
        CameraCreate(
            name="Progressing Video",
            source_type=CameraSourceType.file,
            source="recording.mp4",
            inference_fps=5,
        )
    )
    positions: list[float | None] = []

    def capture_frame(source, display_source=None, position_seconds=None):  # type: ignore[no-untyped-def]
        positions.append(position_seconds)
        return MediaFrame(
            content_base64="encoded-frame",
            content_type="image/jpeg",
            source_position_seconds=position_seconds,
            source_duration_seconds=2.0,
        )

    monkeypatch.setattr(service.media_agent, "capture_opencv_frame", capture_frame)
    CameraDetectionService._video_file_positions.pop(str(camera.id), None)

    for _ in range(10):
        await service._load_frame_from_camera(camera)

    assert positions == pytest.approx([index * 0.2 for index in range(10)])
    assert CameraDetectionService._video_file_positions[str(camera.id)][1] == 0.0


@pytest.mark.asyncio
async def test_camera_scan_reads_http_mjpeg_camera_without_browser_frame(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)

    camera = await cameras.create(
        CameraCreate(
            name="Phone Cam",
            source_type=CameraSourceType.http,
            source="http://192.168.0.106:8080",
            metadata={"stream_url": "http://192.168.0.106:8080/video", "stream_format": "mjpeg"},
        )
    )

    jpeg_bytes = b"\xff\xd8fake-jpeg-frame\xff\xd9"
    monkeypatch.setattr(
        service.media_agent,
        "capture_http_frame",
        lambda **kwargs: MediaFrame(
            content_base64=base64.b64encode(jpeg_bytes).decode("utf-8"),
            content_type="image/jpeg",
        ),
    )

    async def fake_run_inference(*args, **kwargs):
        assert base64.b64decode(kwargs["frame_content_base64"]) == jpeg_bytes
        assert kwargs["frame_content_type"] == "image/jpeg"
        return {
            "camera_id": str(camera.id),
            "model_name": "http-mjpeg-test",
            "model_version": "1",
            "inference_fps": 1.0,
            "detections": [],
            "metadata": {"backend": "ultralytics"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    result = await service.run_scan(camera.id, payload=CameraDetectionScanRequest())

    assert result.detection_count == 0


@pytest.mark.asyncio
async def test_provisional_weapon_is_visible_and_creates_incident(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)
    camera = await cameras.create(
        CameraCreate(name="Fast Threat Cam", source_type=CameraSourceType.http, source="http://camera/live")
    )

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "fast-threat-test",
            "model_version": "1",
            "inference_fps": 5.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.18,
                    "label": "weapon",
                    "track_id": "we-candidate",
                    "provisional": True,
                }
            ],
            "metadata": {"backend": "ultralytics"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    result = await service.run_scan(
        camera.id,
        CameraDetectionScanRequest(
            frame_content_base64="encoded-frame",
            include_evidence=False,
            occurrence_hint="dashboard_live_scan",
        ),
    )

    incidents = await IncidentRepository(session).list()
    assert result.detection_count == 1
    assert len(result.detections) == 1
    assert result.detections[0].detection_type == "weapon"
    assert result.detections[0].metadata["provisional"] is True
    assert result.incident_count == 1
    assert result.alert_count == 1
    assert len(incidents) == 1
    assert incidents[0].detection_type is DetectionType.weapon
    assert incidents[0].metadata_["detection_metadata"]["provisional"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("detection_type", [DetectionType.fire, DetectionType.smoke])
async def test_provisional_hazard_types_create_incidents(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    detection_type: DetectionType,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)
    camera = await cameras.create(
        CameraCreate(
            name=f"{detection_type.value.title()} Cam",
            source_type=CameraSourceType.http,
            source="http://camera/live",
        )
    )

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "provisional-hazard-test",
            "model_version": "1",
            "inference_fps": 5.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.18,
                    "label": detection_type.value,
                    "track_id": f"{detection_type.value}-candidate",
                    "provisional": True,
                }
            ],
            "metadata": {"backend": "ultralytics"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    result = await service.run_scan(
        camera.id,
        CameraDetectionScanRequest(
            frame_content_base64="encoded-frame",
            include_evidence=False,
            occurrence_hint="dashboard_live_scan",
        ),
    )

    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    assert result.incident_count == 1
    assert len(incidents) == 1
    assert incidents[0].detection_type is detection_type
    assert incidents[0].metadata_["detection_metadata"]["provisional"] is True


def test_provisional_fire_has_an_immediate_operator_bounding_box() -> None:
    summaries = CameraDetectionService._summarize_detections(
        [
            {
                "x1": 10,
                "y1": 20,
                "x2": 110,
                "y2": 180,
                "confidence": 0.18,
                "label": "fire",
                "track_id": "fi-candidate",
                "provisional": True,
            }
        ]
    )

    assert len(summaries) == 1
    assert summaries[0].detection_type == "fire"
    assert summaries[0].bounding_box is not None
    assert summaries[0].bounding_box.model_dump() == {
        "x1": 10.0,
        "y1": 20.0,
        "x2": 110.0,
        "y2": 180.0,
        "label": "fire",
    }
    assert summaries[0].metadata["provisional"] is True


def test_overlapping_fire_predictions_use_one_enclosing_box_separate_from_smoke() -> None:
    summaries = CameraDetectionService._summarize_detections(
        [
            {
                "x1": 20,
                "y1": 20,
                "x2": 220,
                "y2": 220,
                "confidence": 0.91,
                "label": "smoke",
                "track_id": "sm-1",
            },
            {
                "x1": 80,
                "y1": 80,
                "x2": 180,
                "y2": 170,
                "confidence": 0.84,
                "label": "fire",
                "track_id": "fi-1",
            },
            {
                "x1": 60,
                "y1": 70,
                "x2": 200,
                "y2": 190,
                "confidence": 0.81,
                "label": "fire",
                "track_id": "fi-2",
            },
        ]
    )

    assert len(summaries) == 2
    fire = next(summary for summary in summaries if summary.detection_type == "fire")
    smoke = next(summary for summary in summaries if summary.detection_type == "smoke")
    assert fire.bounding_box is not None
    assert fire.bounding_box.model_dump() == {
        "x1": 60.0,
        "y1": 70.0,
        "x2": 200.0,
        "y2": 190.0,
        "label": "fire",
    }
    assert fire.confidence == 0.84
    assert fire.metadata["combined_hazard_labels"] == ["fire"]
    assert smoke.bounding_box is not None
    assert smoke.bounding_box.label == "smoke"
    assert smoke.confidence == 0.91


def test_separate_fire_and_smoke_keep_separate_operator_boxes() -> None:
    summaries = CameraDetectionService._summarize_detections(
        [
            {
                "x1": 10,
                "y1": 10,
                "x2": 80,
                "y2": 80,
                "confidence": 0.88,
                "label": "fire",
            },
            {
                "x1": 300,
                "y1": 300,
                "x2": 390,
                "y2": 390,
                "confidence": 0.82,
                "label": "smoke",
            },
        ]
    )

    assert len(summaries) == 2
    assert {summary.detection_type for summary in summaries} == {"fire", "smoke"}


def test_fragmented_fire_uses_smoke_context_and_discards_isolated_weak_speck() -> None:
    summaries = CameraDetectionService._summarize_detections(
        [
            {
                "x1": 432.1,
                "y1": 269.6,
                "x2": 495.2,
                "y2": 321.4,
                "confidence": 0.19,
                "label": "fire",
                "provisional": True,
            },
            {
                "x1": 390.5,
                "y1": 247.7,
                "x2": 422.4,
                "y2": 279.2,
                "confidence": 0.13,
                "label": "fire",
                "provisional": True,
            },
            {
                "x1": 594.6,
                "y1": 109.4,
                "x2": 720.0,
                "y2": 440.4,
                "confidence": 0.12,
                "label": "smoke",
                "provisional": True,
            },
            {
                "x1": 179.1,
                "y1": 1085.3,
                "x2": 200.4,
                "y2": 1100.3,
                "confidence": 0.09,
                "label": "fire",
                "provisional": True,
            },
            {
                "x1": 169.7,
                "y1": 0.4,
                "x2": 412.9,
                "y2": 255.5,
                "confidence": 0.09,
                "label": "smoke",
                "provisional": True,
            },
        ]
    )

    assert len(summaries) == 1
    fire = [summary for summary in summaries if summary.detection_type == "fire"]
    assert len(fire) == 1
    assert fire[0].bounding_box is not None
    assert fire[0].bounding_box.model_dump() == {
        "x1": 390.5,
        "y1": 109.4,
        "x2": 720.0,
        "y2": 440.4,
        "label": "fire",
    }
    assert fire[0].metadata["combined_hazard_labels"] == ["fire", "smoke"]


@pytest.mark.asyncio
async def test_confirmed_live_threat_stores_uploaded_frame_as_snapshot(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)
    camera = await cameras.create(
        CameraCreate(name="Evidence Cam", source_type=CameraSourceType.usb, source="0")
    )
    frame_bytes = b"confirmed-live-threat-frame"

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "live-evidence-test",
            "model_version": "1",
            "inference_fps": 5.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.82,
                    "label": "weapon",
                    "object_label": "knife",
                    "track_id": "we-confirmed",
                }
            ],
            "metadata": {"backend": "ultralytics"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    result = await service.run_scan(
        camera.id,
        CameraDetectionScanRequest(
            frame_content_base64=base64.b64encode(frame_bytes).decode("utf-8"),
            frame_content_type="image/jpeg",
            include_evidence=False,
            occurrence_hint="dashboard_live_scan",
        ),
    )

    incidents = await IncidentRepository(session).list()
    assert result.incident_count == 1
    assert len(incidents) == 1
    assert incidents[0].snapshot_path is not None
    assert (tmp_path / incidents[0].snapshot_path).read_bytes() == frame_bytes
    assert incidents[0].metadata_["snapshot_source"] == "confirmed_inference_frame_fallback"


@pytest.mark.asyncio
async def test_continuous_batch_scan_uses_batched_ai_request(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    service = CameraDetectionService(session)
    (tmp_path / "frame-1.jpg").write_bytes(b"frame-1")
    (tmp_path / "frame-2.jpg").write_bytes(b"frame-2")

    camera_one = await cameras.create(
        CameraCreate(name="Batch Cam 1", source_type=CameraSourceType.file, source="frame-1.jpg")
    )
    camera_two = await cameras.create(
        CameraCreate(name="Batch Cam 2", source_type=CameraSourceType.file, source="frame-2.jpg")
    )

    async def fake_run_inference_batch(payloads: list[dict]) -> list[dict]:
        assert len(payloads) == 2
        assert all(payload["metadata"]["manual_scan"] is False for payload in payloads)
        return [
            {
                "camera_id": str(camera_one.id),
                "model_name": "yolo11",
                "model_version": "batch-test",
                "inference_fps": 8.0,
                "detections": [
                    {
                        "x1": 10,
                        "y1": 20,
                        "x2": 110,
                        "y2": 180,
                        "confidence": 0.93,
                        "label": "weapon",
                        "track_id": "we-1",
                    }
                ],
                "metadata": {"backend": "ultralytics", "batched": True, "batch_size": 2},
            },
            {
                "camera_id": str(camera_two.id),
                "model_name": "yolo11",
                "model_version": "batch-test",
                "inference_fps": 8.0,
                "detections": [
                    {
                        "x1": 30,
                        "y1": 40,
                        "x2": 140,
                        "y2": 210,
                        "confidence": 0.88,
                        "label": "fire",
                        "track_id": "fi-1",
                    }
                ],
                "metadata": {"backend": "ultralytics", "batched": True, "batch_size": 2},
            },
        ]

    monkeypatch.setattr(service, "_run_inference_batch", fake_run_inference_batch)

    results = await service.run_continuous_batch(
        [camera_one.id, camera_two.id],
        CameraDetectionScanRequest(include_evidence=True),
    )

    incidents = await IncidentRepository(session).list()
    alerts = await AlertRepository(session).list()

    assert len(results) == 2
    assert all(result.success for result in results)
    assert len(incidents) == 2
    assert len(alerts) == 2
    assert all(incident.snapshot_path is not None for incident in incidents)
    assert {
        (tmp_path / str(incident.snapshot_path)).read_bytes()
        for incident in incidents
    } == {b"frame-1", b"frame-2"}


@pytest.mark.asyncio
async def test_confirmed_non_threat_incident_keeps_source_frame_when_ai_omits_evidence(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Person Evidence Cam", source_type=CameraSourceType.usb, source="0")
    )
    frame_bytes = b"confirmed-person-frame"
    service = CameraDetectionService(session)

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "person-evidence-test",
            "model_version": "1",
            "inference_fps": 5.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.82,
                    "label": "person",
                    "track_id": "person-confirmed",
                }
            ],
            "metadata": {"backend": "ultralytics"},
        }

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    result = await service.run_scan(
        camera.id,
        CameraDetectionScanRequest(
            frame_content_base64=base64.b64encode(frame_bytes).decode("utf-8"),
            frame_content_type="image/jpeg",
            include_evidence=False,
            occurrence_hint="dashboard_live_scan",
        ),
    )

    incidents = await IncidentRepository(session).list()
    assert result.incident_count == 1
    assert len(incidents) == 1
    assert incidents[0].snapshot_path is not None
    assert (tmp_path / str(incidents[0].snapshot_path)).read_bytes() == frame_bytes


@pytest.mark.asyncio
async def test_browser_usb_incident_stores_five_second_clip_from_transported_frame(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Browser USB Evidence", source_type=CameraSourceType.usb, source="0")
    )
    frame_bytes = b"browser-transported-frame"
    service = CameraDetectionService(session)

    async def fake_run_inference(*args, **kwargs):
        return {
            "camera_id": str(camera.id),
            "model_name": "browser-usb-evidence-test",
            "model_version": "1",
            "inference_fps": 5.0,
            "detections": [
                {
                    "x1": 10,
                    "y1": 20,
                    "x2": 110,
                    "y2": 180,
                    "confidence": 0.82,
                    "label": "person",
                    "track_id": "browser-person-confirmed",
                }
            ],
            "metadata": {"backend": "ultralytics"},
        }

    async def fail_direct_usb_capture(camera):
        raise RuntimeError("USB device is unavailable inside the API container")

    monkeypatch.setattr(service, "_run_inference", fake_run_inference)
    monkeypatch.setattr(service, "_load_frame_from_camera", fail_direct_usb_capture)
    monkeypatch.setattr(
        "app.services.ring_buffer_media.RingBufferMediaService._encode_mp4",
        staticmethod(lambda frames: b"five-second-browser-clip"),
    )

    result = await service.run_scan(
        camera.id,
        CameraDetectionScanRequest(
            frame_content_base64=base64.b64encode(frame_bytes).decode("utf-8"),
            frame_content_type="image/jpeg",
            include_evidence=True,
            occurrence_hint="dashboard_live_scan",
        ),
    )

    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    assert result.incident_count == 1
    assert len(incidents) == 1
    assert incidents[0].clip_path is not None
    assert (tmp_path / str(incidents[0].clip_path)).read_bytes() == b"five-second-browser-clip"
    event_clip = incidents[0].metadata_["event_clip"]
    assert event_clip["duration_seconds"] >= 5
    assert event_clip["minimum_duration_seconds"] == 5


@pytest.mark.asyncio
async def test_continuous_batch_keeps_recording_after_one_camera_ingest_fails(
    session: AsyncSession,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    service = CameraDetectionService(session)

    monkeypatch.setattr("app.services.camera_detection.settings.storage_root", tmp_path)
    (tmp_path / "bad.jpg").write_bytes(b"bad-frame")
    (tmp_path / "good.jpg").write_bytes(b"good-frame")
    bad_camera = await cameras.create(
        CameraCreate(name="Bad ingest", source_type=CameraSourceType.file, source="bad.jpg")
    )
    good_camera = await cameras.create(
        CameraCreate(name="Good ingest", source_type=CameraSourceType.file, source="good.jpg")
    )

    async def fake_run_inference_batch(payloads: list[dict]) -> list[dict]:
        return [
            {
                "camera_id": payload["camera_id"],
                "model_name": "yolo11",
                "model_version": "isolation-test",
                "detections": [
                    {
                        "x1": 1,
                        "y1": 2,
                        "x2": 30,
                        "y2": 40,
                        "confidence": 0.9,
                        "label": "weapon",
                        "track_id": f"track-{index}",
                    }
                ],
                "metadata": {},
            }
            for index, payload in enumerate(payloads)
        ]

    original_ingest = service.events.ingest

    async def fail_first_ingest(payload: DetectionEventIngest, **kwargs):
        if payload.camera_id == bad_camera.id:
            raise RuntimeError("simulated persistence failure")
        return await original_ingest(payload, **kwargs)

    monkeypatch.setattr(service, "_run_inference_batch", fake_run_inference_batch)
    monkeypatch.setattr(service.events, "ingest", fail_first_ingest)

    results = await service.run_continuous_batch(
        [bad_camera.id, good_camera.id],
        CameraDetectionScanRequest(include_evidence=False),
    )

    incidents = await IncidentRepository(session).list()
    result_by_camera = {result.camera_id: result for result in results}
    assert result_by_camera[bad_camera.id].success is False
    assert result_by_camera[good_camera.id].success is True
    assert [incident.camera_id for incident in incidents] == [good_camera.id]


@pytest.mark.asyncio
async def test_continuous_dispatcher_rechecks_queue_after_clearing_wake_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = ContinuousDetectionWorker()
    claimed_batches = iter(([], ["cam-queued-during-clear"]))
    dispatched = asyncio.Event()

    monkeypatch.setattr(worker, "_claim_batch", lambda: next(claimed_batches))

    async def observe_batch(camera_ids: list[object]) -> None:
        assert camera_ids == ["cam-queued-during-clear"]
        dispatched.set()
        raise asyncio.CancelledError

    monkeypatch.setattr(worker, "_run_batch", observe_batch)

    with pytest.raises(asyncio.CancelledError):
        await worker._dispatch_loop()
    assert dispatched.is_set()


def test_continuous_worker_claims_only_bounded_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.continuous_detection.settings.continuous_detection_batch_size", 2)
    worker = ContinuousDetectionWorker()
    worker._states = {
        "cam-a": CameraJobState(pending_runs=2, last_completed_at=1.0),
        "cam-b": CameraJobState(pending_runs=1, last_completed_at=2.0),
        "cam-c": CameraJobState(pending_runs=1, last_completed_at=3.0),
    }

    claimed = worker._claim_batch()

    assert claimed == ["cam-a", "cam-b"]
    assert worker._states["cam-a"].running is True
    assert worker._states["cam-a"].pending_runs == 1
    assert worker._states["cam-b"].running is True
    assert worker._states["cam-c"].running is False


def test_continuous_worker_uses_separate_hazard_and_recognition_lanes() -> None:
    worker = ContinuousDetectionWorker()
    worker._states = {
        "cam-a": CameraJobState(pending_hazards=False, pending_recognition=False),
        "cam-b": CameraJobState(pending_hazards=True, pending_recognition=True),
    }

    assert worker._requested_detectors_for_batch(["cam-a"]) == ["weapon", "person"]
    assert worker._recognition_enabled_for_batch(["cam-a"]) is False
    assert worker._requested_detectors_for_batch(["cam-a", "cam-b"]) == ["weapon", "person", "fire", "smoke"]
    assert worker._recognition_enabled_for_batch(["cam-a", "cam-b"]) is True


def test_continuous_worker_runs_all_safety_detectors_for_recorded_video() -> None:
    worker = ContinuousDetectionWorker()
    worker._states = {
        "recording": CameraJobState(recorded_file=True, pending_hazards=False),
    }

    assert worker._requested_detectors_for_batch(["recording"]) == [
        "weapon",
        "person",
        "fire",
        "smoke",
    ]


def test_continuous_worker_does_not_mix_expensive_lane_signatures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.services.continuous_detection.settings.continuous_detection_batch_size", 4)
    worker = ContinuousDetectionWorker()
    worker._states = {
        "fast-a": CameraJobState(pending_runs=1, last_completed_at=1.0),
        "hazard": CameraJobState(
            pending_runs=1,
            pending_hazards=True,
            last_completed_at=2.0,
        ),
        "fast-b": CameraJobState(pending_runs=1, last_completed_at=3.0),
    }

    claimed = worker._claim_batch()

    assert claimed == ["fast-a", "fast-b"]
    assert worker._states["hazard"].pending_runs == 1


@pytest.mark.parametrize(
    ("detection_type", "should_create_alert"),
    [
        (DetectionType.weapon, True),
        (DetectionType.fire, True),
        (DetectionType.smoke, True),
        (DetectionType.known_person, False),
    ],
    ids=["weapon", "fire", "smoke", "known-person"],
)
def test_should_create_alert_follows_emergency_policy(
    detection_type: DetectionType,
    should_create_alert: bool,
) -> None:
    assert DetectionEventService._should_create_alert(detection_type) is should_create_alert
