from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.user import User, UserRole
from app.repositories.alerts import AlertRepository
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.users import UserRepository
from app.schemas.alerts import AlertCreate
from app.schemas.cameras import CameraCreate
from app.schemas.incidents import IncidentCreate
from app.schemas.users import UserCreate


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
async def test_monitoring_overview_uses_windowed_database_aggregates(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase9-overview@aegispro.local",
            full_name="Phase9 Overview",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase9 Gate", source_type="http", source="http://camera/gate", status="online")
    )
    now = datetime.now(UTC)
    await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type="fire",
            priority="critical",
            confidence=0.95,
            occurred_at=now - timedelta(hours=2),
        )
    )
    await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type="smoke",
            priority="high",
            confidence=0.75,
            occurred_at=now - timedelta(days=2),
        )
    )

    async def stub_collect_system_health(_session):
        return {
            "generated_at": now.isoformat(),
            "api": {"status": "ok", "detail": None},
            "database": {"status": "ok", "detail": None},
            "redis": {"status": "ok", "detail": None},
            "ai": {
                "status": "ok",
                "inference_backend": "ultralytics",
                "fallback_backend": None,
                "recognition_backend": "hash",
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

    response = await client.get("/api/v1/monitoring/overview?window=24h")

    assert response.status_code == 200
    payload = response.json()
    assert payload["kpis"]["incident_volume"] == 1
    assert payload["kpis"]["average_confidence"] == 0.95
    assert payload["detection_mix"] == [{"detection_type": "fire", "count": 1}]


@pytest.mark.asyncio
async def test_optimization_report_surfaces_database_redis_and_runtime_telemetry(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="phase9-optimization@aegispro.local",
            full_name="Phase9 Optimization",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase9 Lobby", source_type="http", source="http://camera/lobby", status="online")
    )
    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type="unknown_person",
            priority="medium",
            confidence=0.67,
        )
    )
    await AlertRepository(session).create(
        AlertCreate(
            incident_id=incident.id,
            priority="medium",
            title="Unknown person detected",
            message="Unknown person detected on camera Phase9 Lobby.",
        )
    )
    await AuditLogRepository(session).create(
        actor_user_id=admin.id,
        actor_email=admin.email,
        actor_role=admin.role.value,
        action="alerts.create",
        resource_type="alert",
        resource_id="alert-1",
        metadata={"source": "phase9-test"},
    )

    async def stub_runtime():
        from app.schemas.monitoring import AiRuntimeHealth

        return AiRuntimeHealth(
            status="ok",
            inference_backend="ultralytics",
            fallback_backend=None,
            recognition_backend="hash",
            recognition_providers=["CPUExecutionProvider"],
            model_device=None,
            gpu_available=False,
            gpu_name=None,
            gpu_memory_total_mb=None,
            gpu_memory_used_mb=None,
            gpu_utilization_percent=None,
            telemetry_supported=False,
            detail="CUDA is not available on this host.",
        )

    class StubRedis:
        @classmethod
        def from_url(cls, *_args, **_kwargs):
            return cls()

        async def ping(self):
            return True

        async def info(self, section: str):
            if section == "memory":
                return {"used_memory_human": "2.00M"}
            if section == "clients":
                return {"connected_clients": 3}
            return {"pubsub_channels": 1}

        async def aclose(self):
            return None

    monkeypatch.setattr("app.services.optimization.fetch_ai_runtime_health", stub_runtime)
    monkeypatch.setattr("app.services.optimization.Redis", StubRedis)

    response = await client.get("/api/v1/monitoring/optimization")

    assert response.status_code == 200
    payload = response.json()
    assert payload["database"]["resources"]["incidents_total"] == 1
    assert payload["database"]["resources"]["active_alerts_total"] == 1
    assert payload["redis"]["status"] == "ok"
    assert payload["runtime"]["inference_backend"] == "ultralytics"
    assert payload["recommendations"][0]["title"] == "Database-side aggregation"
