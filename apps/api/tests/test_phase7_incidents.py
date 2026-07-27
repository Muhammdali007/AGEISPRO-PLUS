import asyncio
import base64
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi import WebSocketDisconnect
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.api.v1.routes.ws import event_stream
from app.core.config import settings
from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.alert import AlertStatus
from app.models.camera import CameraSourceType
from app.models.incident import (
    DetectionType,
    Incident,
    IncidentPriority,
    IncidentRetentionClass,
    IncidentStatus,
)
from app.models.user import User, UserRole
from app.repositories.alerts import AlertRepository
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
from app.services.evidence_storage import EvidenceStorageService
from app.services.auth import AuthService
from app.services.incident_retention import IncidentRetentionService
from app.services.ring_buffer_media import BufferedFrame, RingBufferMediaService


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
async def test_ring_buffer_media_service_signs_and_checksums_event_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_id = uuid4()
    service = RingBufferMediaService()
    frame = base64.b64encode(b"jpeg-frame").decode("utf-8")
    monkeypatch.setattr(settings, "event_clip_after_seconds", 1)
    monkeypatch.setattr(settings, "event_clip_fps", 1)
    monkeypatch.setattr(RingBufferMediaService, "_encode_mp4", staticmethod(lambda frames: b"mp4-bytes"))

    service.add_frame(camera_id, content_base64=frame, content_type="image/jpeg")

    async def capture_after_frame() -> tuple[str, str]:
        return frame, "image/jpeg"

    clip = await service.build_event_clip(camera_id, capture_after_frame=capture_after_frame)

    assert clip is not None
    assert base64.b64decode(clip.content_base64) == b"mp4-bytes"
    event_clip = clip.metadata["event_clip"]
    assert event_clip["sha256"] == "225e2e71f6963695684cf5c2aef7d582fff76acb8c028ed8b79c9c52bc93495d"
    assert event_clip["signature"]
    assert event_clip["signature_algorithm"] == "hmac-sha256"
    assert event_clip["duration_seconds"] >= 5
    assert event_clip["minimum_duration_seconds"] == 5
    assert event_clip["frame_count"] == 5
    assert event_clip["padded_frame_count"] == 3


@pytest.mark.asyncio
async def test_ring_buffer_builds_five_second_clip_from_single_live_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    camera_id = uuid4()
    service = RingBufferMediaService()
    frame = base64.b64encode(b"jpeg-frame").decode("utf-8")
    monkeypatch.setattr(settings, "event_clip_fps", 2)
    monkeypatch.setattr(
        RingBufferMediaService,
        "_encode_mp4",
        staticmethod(lambda frames: b"five-second-mp4"),
    )
    service.add_frame(camera_id, content_base64=frame, content_type="image/jpeg")

    clip = await service.build_event_clip(camera_id)

    assert clip is not None
    event_clip = clip.metadata["event_clip"]
    assert event_clip["frame_count"] == 10
    assert event_clip["duration_seconds"] == 5
    assert event_clip["padded_frame_count"] == 9


def test_ring_buffer_resamples_real_preview_motion_across_five_seconds() -> None:
    frames = [
        BufferedFrame(
            captured_at=datetime.now(UTC),
            monotonic_at=float(second),
            content_base64=f"frame-{second}",
            content_type="image/jpeg",
        )
        for second in range(5)
    ]

    sampled, repeated = RingBufferMediaService._resample_clip_frames(
        frames,
        duration_seconds=5,
        fps=2,
    )

    assert len(sampled) == 10
    assert len({frame.content_base64 for frame in sampled}) == 5
    assert sampled[0].content_base64 == "frame-0"
    assert sampled[-1].content_base64 == "frame-4"
    assert repeated == 5


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
async def test_detection_ingest_validates_entire_batch_before_mutation(
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Phase7 Batch Gate", source_type=CameraSourceType.http, source="http://camera/batch")
    )

    with pytest.raises(HTTPException) as exc_info:
        await DetectionEventService(session).ingest(
            DetectionEventIngest(
                camera_id=camera.id,
                model_name="phase7-batch-inline",
                detections=[
                    DetectionEventIngestItem(
                        detection_type=DetectionType.weapon,
                        confidence=0.94,
                    ),
                    DetectionEventIngestItem(
                        detection_type=DetectionType.person,
                        confidence=0.81,
                        recognition_status=RecognitionStatus.unknown,
                        face_image_evidence=InlineEvidencePayload(
                            content_base64="still-not-valid-base64",
                            content_type="image/jpeg",
                        ),
                    ),
                ],
            )
        )

    assert exc_info.value.status_code == 422
    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    alerts = await AlertRepository(session).list()
    assert incidents == []
    assert alerts == []


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
                        detection_type=DetectionType.weapon,
                        confidence=0.94,
                    )
                ],
                snapshot_evidence=InlineEvidencePayload(
                    content_base64=base64.b64encode(b"weapon-snapshot").decode("utf-8"),
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
    assert snapshot_response.content == b"weapon-snapshot"

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
async def test_incident_delete_requires_admin_or_supervisor_and_archives_without_auto_deleting_open_critical_evidence(
    client: AsyncClient,
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    operator = await UserRepository(session).create(
        UserCreate(
            email="delete-operator@aegispro.local",
            full_name="Delete Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    supervisor = await UserRepository(session).create(
        UserCreate(
            email="delete-supervisor@aegispro.local",
            full_name="Delete Supervisor",
            role=UserRole.supervisor,
            password="ChangeMe123!",
        )
    )
    camera = await CameraRepository(session).create(
        CameraCreate(name="Delete Cam", source_type=CameraSourceType.http, source="http://camera/delete")
    )
    ingest_response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="phase7-delete",
            detections=[DetectionEventIngestItem(detection_type=DetectionType.weapon, confidence=0.93)],
            snapshot_evidence=InlineEvidencePayload(
                content_base64=base64.b64encode(b"delete-snapshot").decode("utf-8"),
                content_type="image/jpeg",
            ),
            clip_evidence=InlineEvidencePayload(
                content_base64=base64.b64encode(b"delete-clip").decode("utf-8"),
                content_type="video/mp4",
            ),
        )
    )
    incident_id = ingest_response.results[0].incident_id
    alert_id = ingest_response.results[0].alert_id
    assert alert_id is not None

    await _set_current_user(operator)
    forbidden_response = await client.delete(f"/api/v1/incidents/{incident_id}")
    assert forbidden_response.status_code == 403

    await _set_current_user(supervisor)
    delete_response = await client.delete(f"/api/v1/incidents/{incident_id}")
    assert delete_response.status_code == 204
    archived_incident = await IncidentRepository(session).get(incident_id, include_archived=True)
    assert archived_incident is not None
    assert archived_incident.archived_at is not None
    assert archived_incident.deletion_requested_at is not None
    assert await IncidentRepository(session).get(incident_id) is None
    alert = await AlertRepository(session).get(alert_id)
    assert alert is not None
    assert alert.status is AlertStatus.cleared
    incident_directory = tmp_path / "incidents" / str(camera.id) / str(incident_id)
    assert incident_directory.exists()

    @asynccontextmanager
    async def test_session_local() -> AsyncIterator[AsyncSession]:
        yield session

    monkeypatch.setattr("app.services.incident_retention.AsyncSessionLocal", test_session_local)
    retention = IncidentRetentionService(EvidenceStorageService(tmp_path))
    deleted_count = await retention.process_pending_deletions()

    assert deleted_count == 0
    archived_incident = await IncidentRepository(session).get(incident_id, include_archived=True)
    assert archived_incident is not None
    assert archived_incident.deletion_completed_at is None
    assert archived_incident.snapshot_path is not None
    assert archived_incident.clip_path is not None
    assert incident_directory.exists()


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
async def test_retention_worker_archives_resolved_incidents_and_skips_open_critical_and_held_records(
    client: AsyncClient,
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    admin = await UserRepository(session).create(
        UserCreate(
            email="retention-admin@aegispro.local",
            full_name="Retention Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    policies_response = await client.get("/api/v1/incidents/retention-policies")
    assert policies_response.status_code == 200
    policies = {item["retention_class"]: item for item in policies_response.json()}
    assert policies["standard"]["retention_hours"] == settings.incident_retention_hours
    assert policies["manual"]["retention_hours"] is None

    camera = await CameraRepository(session).create(
        CameraCreate(name="Retention Cam", source_type=CameraSourceType.http, source="http://camera/retention")
    )

    archived_incident_id = uuid4()
    archived_directory = tmp_path / "incidents" / str(camera.id) / str(archived_incident_id)
    archived_directory.mkdir(parents=True, exist_ok=True)
    (archived_directory / "snapshot.jpg").write_bytes(b"expired-snapshot")
    (archived_directory / "clip.mp4").write_bytes(b"expired-clip")
    (archived_directory / "face.jpg").write_bytes(b"expired-face")

    archived_incident = Incident(
        id=archived_incident_id,
        camera_id=camera.id,
        detection_type=DetectionType.person,
        priority=IncidentPriority.medium,
        status=IncidentStatus.resolved,
        retention_class=IncidentRetentionClass.standard,
        confidence=0.71,
        occurred_at=datetime.now(UTC) - timedelta(hours=25),
        retention_expires_at=datetime.now(UTC) - timedelta(hours=1),
        snapshot_path=f"incidents/{camera.id}/{archived_incident_id}/snapshot.jpg",
        clip_path=f"incidents/{camera.id}/{archived_incident_id}/clip.mp4",
        recognized_identity={
            "status": "unknown",
            "identity_id": None,
            "identity_label": "Unknown person",
            "face_image_path": f"incidents/{camera.id}/{archived_incident_id}/face.jpg",
        },
        metadata_={},
    )
    open_incident = Incident(
        camera_id=camera.id,
        detection_type=DetectionType.smoke,
        priority=IncidentPriority.medium,
        status=IncidentStatus.open,
        retention_class=IncidentRetentionClass.standard,
        confidence=0.65,
        occurred_at=datetime.now(UTC) - timedelta(hours=30),
        retention_expires_at=datetime.now(UTC) - timedelta(hours=6),
        metadata_={},
    )
    critical_incident = Incident(
        camera_id=camera.id,
        detection_type=DetectionType.weapon,
        priority=IncidentPriority.critical,
        status=IncidentStatus.resolved,
        retention_class=IncidentRetentionClass.compliance,
        confidence=0.97,
        occurred_at=datetime.now(UTC) - timedelta(days=8),
        retention_expires_at=datetime.now(UTC) - timedelta(hours=4),
        metadata_={},
    )
    held_incident = Incident(
        camera_id=camera.id,
        detection_type=DetectionType.fire,
        priority=IncidentPriority.high,
        status=IncidentStatus.dismissed,
        retention_class=IncidentRetentionClass.standard,
        confidence=0.88,
        occurred_at=datetime.now(UTC) - timedelta(hours=26),
        retention_expires_at=datetime.now(UTC) - timedelta(hours=2),
        legal_hold=True,
        legal_hold_reason="Pending external review",
        metadata_={},
    )
    session.add_all([archived_incident, open_incident, critical_incident, held_incident])
    await session.commit()

    list_response = await client.get("/api/v1/incidents")
    assert list_response.status_code == 200
    incident_ids = {item["id"] for item in list_response.json()}
    assert str(archived_incident_id) in incident_ids
    assert str(open_incident.id) in incident_ids
    assert str(critical_incident.id) in incident_ids
    assert str(held_incident.id) in incident_ids

    @asynccontextmanager
    async def test_session_local() -> AsyncIterator[AsyncSession]:
        yield session

    monkeypatch.setattr("app.services.incident_retention.AsyncSessionLocal", test_session_local)
    retention = IncidentRetentionService(EvidenceStorageService(tmp_path))
    archived_count = await retention.purge_expired()
    deleted_count = await retention.process_pending_deletions()

    assert archived_count == 1
    assert deleted_count == 1
    assert await IncidentRepository(session).get(archived_incident_id) is None
    archived_record = await IncidentRepository(session).get(archived_incident_id, include_archived=True)
    assert archived_record is not None
    assert archived_record.archived_at is not None
    assert archived_record.deletion_completed_at is not None
    assert not archived_directory.exists()
    assert await retention.process_pending_deletions() == 0

    assert await IncidentRepository(session).get(open_incident.id) is not None
    assert await IncidentRepository(session).get(critical_incident.id) is not None
    assert await IncidentRepository(session).get(held_incident.id) is not None

    post_retention_list = await client.get("/api/v1/incidents")
    assert post_retention_list.status_code == 200
    incident_ids = {item["id"] for item in post_retention_list.json()}
    assert str(archived_incident_id) not in incident_ids
    assert str(open_incident.id) in incident_ids
    assert str(critical_incident.id) in incident_ids
    assert str(held_incident.id) in incident_ids

    detail_response = await client.get(f"/api/v1/incidents/{archived_incident_id}")
    assert detail_response.status_code == 404


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
async def test_emergency_detections_broadcast_alert_lifecycle_and_acknowledgement(
    client: AsyncClient,
    session: AsyncSession,
    detection_type: DetectionType,
    expected_priority: IncidentPriority,
) -> None:
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
            if len([message for message in self.sent_messages if message["type"] != "system.connected"]) >= 4:
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
    await _set_current_user(user)

    camera = await CameraRepository(session).create(
        CameraCreate(name="WS Cam", source_type=CameraSourceType.http, source="http://camera/ws")
    )
    token = (await AuthService(UserRepository(session)).issue_tokens(user))["access_token"]
    await session.commit()
    websocket = RecordingWebSocket()

    stream_task = asyncio.create_task(event_stream(websocket, access_cookie=token, session=session))
    await asyncio.wait_for(websocket._connected.wait(), timeout=2)

    response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="phase7-ws",
            detections=[DetectionEventIngestItem(detection_type=detection_type, confidence=0.99)],
        )
    )

    assert response.incident_count == 1
    assert response.alert_count == 1
    assert response.results[0].priority is expected_priority
    assert response.results[0].alert_id is not None

    incident = await IncidentRepository(session).get(response.results[0].incident_id)
    alert = await AlertRepository(session).get(response.results[0].alert_id)

    assert incident is not None
    assert incident.priority is expected_priority
    assert alert is not None
    assert alert.priority is expected_priority

    ack_response = await client.post(f"/api/v1/alerts/{alert.id}/acknowledge")
    assert ack_response.status_code == 200
    assert ack_response.json()["status"] == "acknowledged"
    assert ack_response.json()["priority"] == expected_priority.value

    await asyncio.wait_for(stream_task, timeout=2)

    assert websocket.accepted is True
    event_types = {message["type"] for message in websocket.sent_messages}
    assert "system.connected" in event_types
    assert "incident.created" in event_types
    assert "alert.created" in event_types
    assert "sound.alert" in event_types
    assert "alert.acknowledged" in event_types

    incident_created_event = next(
        message for message in websocket.sent_messages if message["type"] == "incident.created"
    )
    alert_created_event = next(
        message for message in websocket.sent_messages if message["type"] == "alert.created"
    )
    sound_alert_event = next(
        message for message in websocket.sent_messages if message["type"] == "sound.alert"
    )

    assert incident_created_event["detection_type"] == detection_type.value
    assert incident_created_event["priority"] == expected_priority.value
    assert alert_created_event["priority"] == expected_priority.value
    assert sound_alert_event["detection_type"] == detection_type.value
    assert sound_alert_event["scan_count"] == 1


@pytest.mark.asyncio
async def test_event_stream_accepts_query_token(
    session: AsyncSession,
) -> None:
    class RecordingWebSocket:
        def __init__(self) -> None:
            self.accepted = False
            self.sent_messages: list[dict[str, str]] = []
            self._disconnect = asyncio.Event()

        async def close(self, code: int, reason: str) -> None:
            self._disconnect.set()

        async def accept(self) -> None:
            self.accepted = True

        async def send_json(self, payload: dict[str, str]) -> None:
            self.sent_messages.append(payload)
            self._disconnect.set()

        async def receive_text(self) -> str:
            await self._disconnect.wait()
            raise WebSocketDisconnect()

    user = await UserRepository(session).create(
        UserCreate(
            email="ws-query-admin@aegispro.local",
            full_name="WS Query Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    token = (await AuthService(UserRepository(session)).issue_tokens(user))["access_token"]
    await session.commit()

    websocket = RecordingWebSocket()
    await event_stream(websocket, access_cookie=None, token=token, session=session)

    assert websocket.accepted is True
    assert websocket.sent_messages[0]["type"] == "system.connected"
