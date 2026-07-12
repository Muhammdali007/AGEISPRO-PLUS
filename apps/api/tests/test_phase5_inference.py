from collections.abc import AsyncIterator

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
from app.services.detection_events import DetectionEventService


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
async def test_detection_ingest_maps_smoke_to_high_priority(session: AsyncSession) -> None:
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
                    detection_type=DetectionType.smoke,
                    confidence=0.73,
                )
            ],
        )
    )

    assert response.results[0].priority is IncidentPriority.high


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
    assert response.detections[0].bounding_box is not None
    assert response.detections[0].bounding_box.label == "kitchen_knife"
    assert len(alerts) == 1
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
        service,
        "_load_frame_with_opencv",
        lambda source: ("encoded-frame", "image/jpeg"),
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
def test_only_weapon_detections_create_alerts() -> None:
    assert DetectionEventService._should_create_alert(DetectionType.weapon) is True
    assert DetectionEventService._should_create_alert(DetectionType.smoke) is False
    assert DetectionEventService._should_create_alert(DetectionType.known_person) is False
