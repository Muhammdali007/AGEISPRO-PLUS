import asyncio
import base64
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi import WebSocketDisconnect
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.api.v1.routes.ws import event_stream
from app.core.config import settings
from app.core.security import TokenType, create_token
from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.camera import CameraSourceType
from app.models.incident import DetectionType, IncidentStatus
from app.models.user import User, UserRole
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.users import UserRepository
from app.schemas.cameras import CameraCreate
from app.schemas.detections import (
    DetectionEventIngest,
    DetectionEventIngestItem,
    InlineEvidencePayload,
    RecognitionStatus,
)
from app.schemas.incidents import IncidentCreate
from app.schemas.users import UserCreate
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
async def test_detection_ingest_persists_inline_evidence(
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase7 Dock", source_type=CameraSourceType.http, source="http://camera/dock")
    )

    response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="phase7-sim",
            detections=[
                DetectionEventIngestItem(
                    detection_type=DetectionType.person,
                    confidence=0.91,
                    track_id="trk-inline",
                    recognition_status=RecognitionStatus.unknown,
                    identity_label="Unidentified person",
                    face_image_evidence=InlineEvidencePayload(
                        content_base64=base64.b64encode(b"face-bytes").decode("utf-8"),
                        content_type="image/jpeg",
                    ),
                )
            ],
            snapshot_evidence=InlineEvidencePayload(
                content_base64=base64.b64encode(b"snapshot-bytes").decode("utf-8"),
                content_type="image/jpeg",
            ),
            clip_evidence=InlineEvidencePayload(
                content_base64=base64.b64encode(b"clip-bytes").decode("utf-8"),
                content_type="video/mp4",
            ),
        )
    )

    incident = await IncidentRepository(session).get(response.results[0].incident_id)
    assert incident is not None
    assert incident.snapshot_path is not None
    assert incident.clip_path is not None
    assert incident.recognized_identity is not None
    assert incident.recognized_identity["face_image_path"].startswith("incidents/")
    assert (tmp_path / incident.snapshot_path).read_bytes() == b"snapshot-bytes"
    assert (tmp_path / incident.clip_path).read_bytes() == b"clip-bytes"
    assert (tmp_path / incident.recognized_identity["face_image_path"]).read_bytes() == b"face-bytes"


@pytest.mark.asyncio
async def test_detection_ingest_rejects_invalid_inline_evidence_without_persisting_incident(
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase7 Gate", source_type=CameraSourceType.http, source="http://camera/gate")
    )

    with pytest.raises(HTTPException) as exc_info:
        await DetectionEventService(session).ingest(
            DetectionEventIngest(
                camera_id=camera.id,
                model_name="phase7-invalid-inline",
                detections=[
                    DetectionEventIngestItem(
                        detection_type=DetectionType.person,
                        confidence=0.78,
                        recognition_status=RecognitionStatus.unknown,
                    )
                ],
                snapshot_evidence=InlineEvidencePayload(
                    content_base64="not-valid-base64",
                    content_type="image/jpeg",
                ),
            )
        )

    assert exc_info.value.status_code == 422
    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    assert incidents == []


@pytest.mark.asyncio
async def test_incident_routes_support_evidence_and_alert_workflow(
    client: AsyncClient,
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    admin = await UserRepository(session).create(
        UserCreate(
            email="phase7-admin@aegispro.local",
            full_name="Phase7 Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Lobby", source_type=CameraSourceType.http, source="http://camera/lobby")
    )
    incident_response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="phase7-sim",
            detections=[
                DetectionEventIngestItem(
                    detection_type=DetectionType.fire,
                    confidence=0.94,
                )
            ],
            snapshot_evidence=InlineEvidencePayload(
                content_base64=base64.b64encode(b"fire-snapshot").decode("utf-8"),
                content_type="image/jpeg",
            ),
        )
    )
    incident_id = response_id = str(incident_response.results[0].incident_id)
    alert_id = str(incident_response.results[0].alert_id)

    alerts_response = await client.get(f"/api/v1/incidents/{incident_id}/alerts")
    assert alerts_response.status_code == 200
    assert alerts_response.json()[0]["id"] == alert_id

    snapshot_response = await client.get(f"/api/v1/incidents/{incident_id}/snapshot")
    assert snapshot_response.status_code == 200
    assert snapshot_response.content == b"fire-snapshot"

    ack_response = await client.post(f"/api/v1/alerts/{alert_id}/acknowledge")
    assert ack_response.status_code == 200
    assert ack_response.json()["status"] == "acknowledged"

    clear_response = await client.post(f"/api/v1/alerts/{alert_id}/clear")
    assert clear_response.status_code == 200
    assert clear_response.json()["status"] == "cleared"

    patch_response = await client.patch(
        f"/api/v1/incidents/{response_id}",
        json={"status": IncidentStatus.investigating.value, "operator_notes": "Phase 7 validation"},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["status"] == "investigating"
    assert patch_response.json()["operator_notes"] == "Phase 7 validation"


@pytest.mark.asyncio
async def test_incident_snapshot_route_rejects_paths_outside_storage_root(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="safety-admin@aegispro.local",
            full_name="Safety Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Safe Cam", source_type=CameraSourceType.http, source="http://camera/safe")
    )
    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.smoke,
            confidence=0.66,
            snapshot_path="../escape.jpg",
        )
    )

    response = await client.get(f"/api/v1/incidents/{incident.id}/snapshot")
    assert response.status_code == 400
    assert response.json()["detail"] == "Evidence path is outside the configured storage root"


@pytest.mark.asyncio
async def test_websocket_broadcasts_detection_and_alert_events(session: AsyncSession) -> None:
    class RecordingWebSocket:
        def __init__(self) -> None:
            self.accepted = False
            self.sent_messages: list[dict[str, str]] = []
            self._disconnect = asyncio.Event()
            self._connected = asyncio.Event()

        async def close(self, code: int, reason: str) -> None:
            self._disconnect.set()

        async def accept(self) -> None:
            self.accepted = True

        async def send_json(self, payload: dict[str, str]) -> None:
            self.sent_messages.append(payload)
            if payload["type"] == "system.connected":
                self._connected.set()
            if len([message for message in self.sent_messages if message["type"] != "system.connected"]) >= 2:
                self._disconnect.set()

        async def receive_text(self) -> str:
            await self._disconnect.wait()
            raise WebSocketDisconnect()

    user = await UserRepository(session).create(
        UserCreate(
            email="ws-admin@aegispro.local",
            full_name="WS Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    camera = await CameraRepository(session).create(
        CameraCreate(name="WS Cam", source_type=CameraSourceType.http, source="http://camera/ws")
    )
    token = create_token(user.id, user.role.value, TokenType.access)
    websocket = RecordingWebSocket()

    stream_task = asyncio.create_task(event_stream(websocket, token=token, session=session))
    await asyncio.wait_for(websocket._connected.wait(), timeout=2)

    response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="phase7-ws",
            detections=[DetectionEventIngestItem(detection_type=DetectionType.weapon, confidence=0.99)],
        )
    )

    assert response.incident_count == 1
    assert response.alert_count == 1

    await asyncio.wait_for(stream_task, timeout=2)

    assert websocket.accepted is True
    event_types = {message["type"] for message in websocket.sent_messages}
    assert "system.connected" in event_types
    assert "incident.created" in event_types
    assert "alert.created" in event_types
