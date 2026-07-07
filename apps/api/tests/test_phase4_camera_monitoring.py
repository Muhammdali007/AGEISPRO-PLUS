from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.camera import CameraSourceType
from app.models.user import UserRole
from app.repositories.cameras import CameraRepository
from app.repositories.users import UserRepository
from app.schemas.cameras import CameraCreate
from app.schemas.users import UserCreate
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


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_live_monitor_route_returns_multi_feed_summary(
    client: AsyncClient, session: AsyncSession, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.viewer)
    file_path = tmp_path / "archive.mp4"
    file_path.write_bytes(b"fake archive")

    cameras = CameraRepository(session)
    await cameras.create(
        CameraCreate(
            name="Parking",
            source_type=CameraSourceType.http,
            source="https://streams.aegispro.local/parking.m3u8",
            group="yard",
            metadata={"stream_format": "hls"},
        )
    )
    await cameras.create(
        CameraCreate(
            name="Archive",
            source_type=CameraSourceType.file,
            source=str(file_path),
            group="records",
        )
    )

    monkeypatch.setattr(CameraStreamingService, "resolve_file_source", lambda self, camera: file_path)

    response = await client.get(
        "/api/v1/cameras/live-monitor",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["browser_ready"] == 2
    assert payload["summary"]["groups"] == {"records": 1, "yard": 1}
    assert {entry["camera"]["name"] for entry in payload["entries"]} == {"Parking", "Archive"}


@pytest.mark.asyncio
async def test_live_monitor_batch_health_checks_can_be_filtered(
    client: AsyncClient, session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    cameras = CameraRepository(session)
    target_camera = await cameras.create(
        CameraCreate(
            name="Dock 1",
            source_type=CameraSourceType.usb,
            source="0",
            group="dock",
        )
    )
    await cameras.create(
        CameraCreate(
            name="HQ",
            source_type=CameraSourceType.usb,
            source="1",
            group="hq",
        )
    )

    async def probe(self: CameraStreamingService, camera):  # type: ignore[no-untyped-def]
        from datetime import UTC, datetime

        from app.services.camera_streams import HealthProbeResult
        from app.models.camera import CameraStatus

        checked_at = datetime.now(UTC)
        return HealthProbeResult(
            status=CameraStatus.online,
            message=f"{camera.name} healthy",
            checked_at=checked_at,
            latency_ms=7,
            last_seen_at=checked_at,
        )

    monkeypatch.setattr(CameraStreamingService, "_probe", probe)

    response = await client.post(
        "/api/v1/cameras/live-monitor/test-connections?group=dock",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["results"]) == 1
    assert payload["results"][0]["camera_id"] == str(target_camera.id)
    assert payload["results"][0]["message"] == "Dock 1 healthy"


@pytest.mark.asyncio
async def test_http_health_check_prefers_stream_url_metadata_and_private_https_tls_override(
    session: AsyncSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    cameras = CameraRepository(session)
    camera = await cameras.create(
        CameraCreate(
            name="Phone Cam",
            source_type=CameraSourceType.http,
            source="https://192.168.1.20:8443/",
            metadata={
                "stream_url": "https://192.168.1.20:8443/video",
                "stream_format": "mjpeg",
            },
        )
    )

    captured: dict[str, object] = {}

    def stub_probe(source: str, skip_tls_verification: bool = False) -> tuple[int, str]:
        captured["source"] = source
        captured["skip_tls_verification"] = skip_tls_verification
        return 200, "multipart/x-mixed-replace"

    monkeypatch.setattr(CameraStreamingService, "_probe_http_source_with_headers", staticmethod(stub_probe))

    result = await CameraStreamingService(cameras).test_connection(camera)

    assert result.status.value == "online"
    assert captured == {
        "source": "https://192.168.1.20:8443/video",
        "skip_tls_verification": True,
    }


@pytest.mark.asyncio
async def test_http_stream_descriptor_guesses_ip_webcam_video_endpoint(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    camera = await cameras.create(
        CameraCreate(
            name="Phone Cam Root",
            source_type=CameraSourceType.http,
            source="https://192.168.0.109:8080",
        )
    )

    descriptor = await CameraStreamingService(cameras).describe_stream(camera)

    assert descriptor.stream_url == "https://192.168.0.109:8080/video"
    assert descriptor.stream_kind == "image"
    assert any("/video endpoint" in note for note in descriptor.notes)


async def _create_user_and_login(client: AsyncClient, session: AsyncSession, role: UserRole) -> str:
    email = f"{role.value}@aegispro.local"
    await UserRepository(session).create(
        UserCreate(
            email=email,
            full_name=f"{role.value.title()} User",
            role=role,
            password="ChangeMe123!",
        )
    )

    response = await client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": "ChangeMe123!"},
    )

    assert response.status_code == 200
    return response.json()["access_token"]
