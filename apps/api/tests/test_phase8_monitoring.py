from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.audit_log import AuditLog
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.alerts import AlertRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.users import UserRepository
from app.schemas.alerts import AlertCreate
from app.schemas.cameras import CameraCreate
from app.schemas.incidents import IncidentCreate, IncidentUpdate
from app.schemas.users import UserCreate
from app.services.audit_logs import AuditLogService


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as test_session:
        yield test_session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()


async def _set_current_user(user: User) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


@pytest.mark.asyncio
async def test_monitoring_overview_returns_operational_metrics(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase8-admin@aegispro.local",
            full_name="Phase8 Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase8 Dock", source_type="http", source="http://camera/dock", status="online")
    )
    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type="fire",
            priority="critical",
            confidence=0.92,
            occurred_at=datetime.now(UTC) - timedelta(hours=1),
        )
    )
    await AlertRepository(session).create(
        AlertCreate(
            incident_id=incident.id,
            priority="critical",
            title="Fire detected",
            message="Fire detected on camera Phase8 Dock.",
        )
    )

    async def stub_collect_system_health(_session):
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "api": {"status": "ok", "detail": None},
            "database": {"status": "ok", "detail": None},
            "redis": {"status": "ok", "detail": None},
            "ai": {
                "status": "ok",
                "inference_backend": "ultralytics",
                "fallback_backend": None,
                "recognition_backend": settings.recognition_backend,
                "recognition_providers": ["CPUExecutionProvider"],
                "model_device": None,
                "gpu_available": False,
                "gpu_name": None,
                "gpu_memory_total_mb": None,
                "gpu_memory_used_mb": None,
                "gpu_utilization_percent": None,
                "telemetry_supported": False,
                "detail": "CUDA is not available on this host.",
            },
        }

    monkeypatch.setattr("app.services.monitoring.collect_system_health", stub_collect_system_health)
    monkeypatch.setattr("app.api.v1.routes.monitoring.collect_system_health", stub_collect_system_health)

    response = await client.get("/api/v1/monitoring/overview")

    assert response.status_code == 200
    payload = response.json()
    assert payload["kpis"]["incident_volume"] == 1
    assert payload["kpis"]["active_alerts"] == 1
    assert payload["detection_mix"][0]["detection_type"] == "fire"
    assert payload["camera_health"]["online"] == 1


@pytest.mark.asyncio
async def test_monitoring_camera_health_reports_stale_cameras(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase8-health@aegispro.local",
            full_name="Phase8 Health",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Quiet Lobby", source_type="usb", source="0", status="offline")
    )
    camera.last_seen_at = datetime.now(UTC) - timedelta(minutes=15)
    await session.commit()

    response = await client.get("/api/v1/monitoring/camera-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["stale"] == 1
    assert payload["entries"][0]["stale"] is True


@pytest.mark.asyncio
async def test_system_health_degrades_when_ai_runtime_is_unavailable(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase8-system@aegispro.local",
            full_name="Phase8 System",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    async def stub_collect_system_health(_session):
        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "api": {"status": "ok", "detail": None},
            "database": {"status": "ok", "detail": None},
            "redis": {"status": "ok", "detail": None},
            "ai": {
                "status": "unavailable",
                "inference_backend": None,
                "fallback_backend": None,
                "recognition_backend": None,
                "recognition_providers": [],
                "model_device": None,
                "gpu_available": False,
                "gpu_name": None,
                "gpu_memory_total_mb": None,
                "gpu_memory_used_mb": None,
                "gpu_utilization_percent": None,
                "telemetry_supported": False,
                "detail": "connection refused",
            },
        }

    monkeypatch.setattr("app.api.v1.routes.monitoring.collect_system_health", stub_collect_system_health)

    response = await client.get("/api/v1/monitoring/system-health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ai"]["status"] == "unavailable"
    assert payload["database"]["status"] == "ok"


@pytest.mark.asyncio
async def test_audit_logs_capture_incident_and_alert_workflow_actions(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase8-audit@aegispro.local",
            full_name="Phase8 Audit",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase8 Audit Cam", source_type="http", source="http://camera/audit")
    )
    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type="smoke",
            priority="high",
            confidence=0.78,
        )
    )
    alert = await AlertRepository(session).create(
        AlertCreate(
            incident_id=incident.id,
            priority="high",
            title="Smoke detected",
            message="Smoke detected on camera Phase8 Audit Cam.",
        )
    )

    patch_response = await client.patch(
        f"/api/v1/incidents/{incident.id}",
        json={"status": IncidentUpdate(status="investigating").status},
    )
    assert patch_response.status_code == 200

    ack_response = await client.post(f"/api/v1/alerts/{alert.id}/acknowledge")
    assert ack_response.status_code == 200

    clear_response = await client.post(f"/api/v1/alerts/{alert.id}/clear")
    assert clear_response.status_code == 200

    list_response = await client.get("/api/v1/monitoring/audit-logs?resource_type=alert")
    assert list_response.status_code == 200
    payload = list_response.json()
    actions = [item["action"] for item in payload["items"]]
    assert "alerts.clear" in actions
    assert "alerts.acknowledge" in actions

    filtered = await client.get("/api/v1/monitoring/audit-logs?action=incidents.update")
    assert filtered.status_code == 200
    assert filtered.json()["total"] >= 1


@pytest.mark.asyncio
async def test_audit_log_endpoint_requires_admin_or_supervisor(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    operator = await UserRepository(session).create(
        UserCreate(
            email="phase8-operator@aegispro.local",
            full_name="Phase8 Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(operator)

    response = await client.get("/api/v1/monitoring/audit-logs")

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_audit_log_endpoint_redacts_legacy_sensitive_metadata(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase8-redaction@aegispro.local",
            full_name="Phase8 Redaction",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    session.add(
        AuditLog(
            actor_user_id=admin.id,
            actor_email=admin.email,
            actor_role=admin.role.value,
            action="users.update",
            resource_type="user",
            resource_id=str(admin.id),
            metadata_={
                "password": "legacy-plaintext",
                "profile": {"password_hash": "legacy-hash"},
                "tokens": [{"access_token": "legacy-token"}],
            },
        )
    )
    await session.commit()

    response = await client.get("/api/v1/monitoring/audit-logs")

    assert response.status_code == 200
    metadata = response.json()["items"][0]["metadata"]
    assert metadata["password"] == "[REDACTED]"
    assert metadata["profile"]["password_hash"] == "[REDACTED]"
    assert metadata["tokens"][0]["access_token"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_audit_log_service_redacts_sensitive_metadata_recursively(
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase8-service-redaction@aegispro.local",
            full_name="Phase8 Service Redaction",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )

    await AuditLogService(AuditLogRepository(session)).record(
        actor=admin,
        action="users.update",
        resource_type="user",
        resource_id=str(admin.id),
        metadata={
            "password": "plaintext",
            "profile": {"password_hash": "hash"},
            "credentials": [{"refresh_token": "token"}],
        },
    )

    items, _ = await AuditLogRepository(session).list(action="users.update")

    assert items[0].metadata_["password"] == "[REDACTED]"
    assert items[0].metadata_["profile"]["password_hash"] == "[REDACTED]"
    assert items[0].metadata_["credentials"][0]["refresh_token"] == "[REDACTED]"
