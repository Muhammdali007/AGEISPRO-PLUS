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
from app.schemas.cameras import CameraCreate, CameraUpdate
from app.schemas.detections import DetectionBoundingBox, DetectionEventIngest, DetectionEventIngestItem
from app.db.session import get_db
from app.main import app
from httpx import ASGITransport, AsyncClient
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
