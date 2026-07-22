from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import require_roles
from app.core.security import verify_password
from app.db.metadata import Base
from app.models.alert import AlertStatus
from app.models.camera import CameraSourceType, CameraStatus
from app.models.incident import DetectionType, IncidentPriority, IncidentRetentionClass, IncidentStatus
from app.models.user import User, UserRole
from app.repositories.alerts import AlertRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.users import UserRepository
from app.schemas.alerts import AlertCreate
from app.schemas.cameras import CameraCreate, CameraUpdate
from app.schemas.incidents import IncidentCreate, IncidentUpdate
from app.schemas.users import UserCreate, UserUpdate
from app.services.camera_streams import CameraStreamingService


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as test_session:
        yield test_session

    await engine.dispose()


def test_phase_2_models_are_registered_in_metadata() -> None:
    assert {"users", "cameras", "incidents", "alerts"} <= set(Base.metadata.tables)


@pytest.mark.asyncio
async def test_user_management_create_update_and_delete(session: AsyncSession) -> None:
    users = UserRepository(session)

    user = await users.create(
        UserCreate(
            email="Operator@AegisPro.local",
            full_name="Operator One",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )

    assert user.email == "operator@aegispro.local"
    assert verify_password("ChangeMe123!", user.password_hash)

    updated = await users.update(user, UserUpdate(full_name="Operator Updated", is_active=False))

    assert updated.full_name == "Operator Updated"
    assert updated.is_active is False

    await users.delete(updated)

    assert await users.get_by_id(user.id) is None


@pytest.mark.asyncio
async def test_camera_crud_and_filters(session: AsyncSession) -> None:
    cameras = CameraRepository(session)

    camera = await cameras.create(
        CameraCreate(
            name="Lobby",
            source_type=CameraSourceType.rtsp,
            source="rtsp://camera/lobby",
            status=CameraStatus.online,
            group="hq",
            tags=["entrance"],
            metadata={"floor": 1},
        )
    )

    assert camera.metadata_ == {"floor": 1}
    assert [item.id for item in await cameras.list(status=CameraStatus.online)] == [camera.id]

    updated = await cameras.update(camera, CameraUpdate(detection_enabled=False, inference_fps=3))

    assert updated.detection_enabled is False
    assert updated.inference_fps == 3


@pytest.mark.asyncio
async def test_camera_repository_encrypts_network_sources_and_preserves_runtime_source(
    session: AsyncSession,
) -> None:
    cameras = CameraRepository(session)
    raw_source = "rtsp://operator:RotateMe123!@camera.local:554/live"

    camera = await cameras.create(
        CameraCreate(
            name="Secure Lobby",
            source_type=CameraSourceType.rtsp,
            source=raw_source,
        )
    )

    assert camera.source == "rtsp://camera.local:554/...."
    assert camera.source_redacted is True
    assert camera.credentials_rotation_required is True
    assert camera.secret is not None
    assert await cameras.get_runtime_source(camera) == raw_source


@pytest.mark.asyncio
async def test_camera_stream_service_supports_file_and_rtsp_sources(
    session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras = CameraRepository(session)
    file_path = tmp_path / "clip.mp4"
    file_path.write_bytes(b"fake video")

    file_camera = await cameras.create(
        CameraCreate(name="Evidence", source_type=CameraSourceType.file, source=str(file_path))
    )
    rtsp_camera = await cameras.create(
        CameraCreate(
            name="Relay",
            source_type=CameraSourceType.rtsp,
            source="rtsp://camera.local/live",
            metadata={"relay_url": "https://streams.aegispro.local/relay.m3u8"},
        )
    )

    service = CameraStreamingService(cameras)

    monkeypatch.setattr(service, "resolve_file_source", lambda camera: file_path)
    file_descriptor = await service.describe_stream(file_camera)
    rtsp_descriptor = await service.describe_stream(rtsp_camera)

    assert file_descriptor.stream_kind == "video"
    assert file_descriptor.browser_supported is True
    assert file_descriptor.stream_url is not None
    assert rtsp_descriptor.browser_supported is True
    assert rtsp_descriptor.requires_relay is True
    assert rtsp_descriptor.stream_kind == "hls"


@pytest.mark.asyncio
async def test_camera_stream_service_builds_live_monitor_summary(
    session: AsyncSession, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras = CameraRepository(session)
    file_path = tmp_path / "still.jpg"
    file_path.write_bytes(b"fake image")

    http_camera = await cameras.create(
        CameraCreate(
            name="North Gate",
            source_type=CameraSourceType.http,
            source="https://streams.aegispro.local/north.m3u8",
            status=CameraStatus.online,
            group="perimeter",
            detection_enabled=True,
        )
    )
    file_camera = await cameras.create(
        CameraCreate(
            name="Archive",
            source_type=CameraSourceType.file,
            source=str(file_path),
            status=CameraStatus.degraded,
            group="archive",
            detection_enabled=False,
        )
    )

    service = CameraStreamingService(cameras)
    monkeypatch.setattr(service, "resolve_file_source", lambda camera: file_path)

    response = await service.describe_live_monitor([http_camera, file_camera])

    assert response.summary.total == 2
    assert response.summary.online == 1
    assert response.summary.degraded == 1
    assert response.summary.live == 1
    assert response.summary.browser_ready == 2
    assert response.summary.detection_enabled == 1
    assert response.summary.groups == {"archive": 1, "perimeter": 1}
    assert {entry.camera.name for entry in response.entries} == {"North Gate", "Archive"}


@pytest.mark.asyncio
async def test_incident_filters_and_status_update(session: AsyncSession) -> None:
    users = UserRepository(session)
    cameras = CameraRepository(session)
    incidents = IncidentRepository(session)

    user = await users.create(
        UserCreate(
            email="supervisor@aegispro.local",
            full_name="Supervisor",
            role=UserRole.supervisor,
            password="ChangeMe123!",
        )
    )
    camera = await cameras.create(
        CameraCreate(
            name="Warehouse",
            source_type=CameraSourceType.usb,
            source="0",
            status=CameraStatus.online,
        )
    )

    incident = await incidents.create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.91,
            assigned_user_id=user.id,
            metadata={"model": "placeholder"},
        )
    )

    filtered = await incidents.list(
        camera_id=camera.id,
        detection_type=DetectionType.weapon,
        priority=IncidentPriority.critical,
        assigned_user_id=user.id,
    )
    assert [item.id for item in filtered] == [incident.id]
    assert incident.retention_class is IncidentRetentionClass.compliance
    assert incident.retention_expires_at is not None

    updated = await incidents.update(
        incident,
        IncidentUpdate(
            status=IncidentStatus.investigating,
            operator_notes="Reviewing evidence",
            retention_class=IncidentRetentionClass.manual,
            legal_hold=True,
            legal_hold_reason="Pending supervisor review",
        ),
    )
    assert updated.status is IncidentStatus.investigating
    assert updated.operator_notes == "Reviewing evidence"
    assert updated.retention_class is IncidentRetentionClass.manual
    assert updated.retention_expires_at is None
    assert updated.legal_hold is True
    assert updated.legal_hold_reason == "Pending supervisor review"


@pytest.mark.asyncio
async def test_alert_acknowledgement(session: AsyncSession) -> None:
    users = UserRepository(session)
    cameras = CameraRepository(session)
    incidents = IncidentRepository(session)
    alerts = AlertRepository(session)

    user = await users.create(
        UserCreate(
            email="operator@aegispro.local",
            full_name="Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    camera = await cameras.create(
        CameraCreate(name="Yard", source_type=CameraSourceType.http, source="http://camera/yard")
    )
    incident = await incidents.create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.fire,
            priority=IncidentPriority.high,
            confidence=0.82,
        )
    )
    alert = await alerts.create(
        AlertCreate(
            incident_id=incident.id,
            priority=IncidentPriority.high,
            title="Fire detected",
            message="Fire detected at Yard",
        )
    )

    acknowledged = await alerts.acknowledge(alert, user.id)

    assert acknowledged.acknowledged is True
    assert acknowledged.status is AlertStatus.acknowledged
    assert acknowledged.acknowledged_by_id == user.id
    assert acknowledged.acknowledged_at is not None


@pytest.mark.asyncio
async def test_rbac_dependency_allows_roles_and_rejects_others() -> None:
    admin = User(
        email="admin@aegispro.local",
        full_name="Admin",
        role=UserRole.administrator,
        password_hash="hash",
    )
    viewer = User(
        email="viewer@aegispro.local",
        full_name="Viewer",
        role=UserRole.viewer,
        password_hash="hash",
    )
    dependency = require_roles(UserRole.administrator)

    assert await dependency(admin) is admin
    with pytest.raises(HTTPException) as exc_info:
        await dependency(viewer)

    assert exc_info.value.status_code == 403
