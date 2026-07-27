from collections.abc import AsyncIterator
from email.message import Message
import ipaddress
from pathlib import Path
from urllib.error import HTTPError

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.camera import CameraSourceType, CameraStatus
from app.models.incident import DetectionType, IncidentPriority
from app.models.user import UserRole
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.users import UserRepository
from app.schemas.cameras import CameraCreate, CameraDetectionScanResponse, CameraDetectionScanSummary
from app.schemas.detections import DetectionBoundingBox
from app.schemas.incidents import IncidentCreate
from app.services.camera_detection import CameraDetectionService
from app.services.camera_network_policy import CameraNetworkPolicy
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
    from app.api.v1.routes.cameras import manual_camera_scan_rate_limiter
    from app.services.camera_overlays import camera_overlay_store

    manual_camera_scan_rate_limiter.reset()
    camera_overlay_store.reset()

    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()
    manual_camera_scan_rate_limiter.reset()
    camera_overlay_store.reset()


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
async def test_deleting_camera_preserves_incidents_and_hides_camera(
    client: AsyncClient, session: AsyncSession
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    cameras = CameraRepository(session)
    incidents = IncidentRepository(session)
    camera = await cameras.create(
        CameraCreate(
            name="Retired Gate",
            source_type=CameraSourceType.http,
            source="http://camera/retired",
        )
    )
    incident = await incidents.create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.person,
            priority=IncidentPriority.low,
            confidence=0.82,
        )
    )

    response = await client.delete(
        f"/api/v1/cameras/{camera.id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 204
    assert camera.deleted_at is not None
    assert camera.status == CameraStatus.disabled
    assert camera.detection_enabled is False
    assert await cameras.get(camera.id) is None
    assert all(item.id != camera.id for item in await cameras.list())
    assert (await incidents.get(incident.id)) is not None


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

    assert descriptor.stream_url == f"/api/v1/cameras/{camera.id}/stream/http"
    assert descriptor.stream_kind == "image"
    assert any("/video endpoint" in note for note in descriptor.notes)
    assert any("proxied through the API" in note for note in descriptor.notes)


@pytest.mark.asyncio
async def test_http_stream_descriptor_infers_mjpeg_for_direct_phone_video_path(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    camera = await cameras.create(
        CameraCreate(
            name="Phone Cam Direct",
            source_type=CameraSourceType.http,
            source="https://192.168.0.109:8080/video",
        )
    )

    descriptor = await CameraStreamingService(cameras).describe_stream(camera)

    assert descriptor.stream_url == f"/api/v1/cameras/{camera.id}/stream/http"
    assert descriptor.stream_kind == "image"
    assert any("MJPEG phone-camera feed" in note for note in descriptor.notes)


@pytest.mark.asyncio
async def test_camera_read_route_returns_redacted_network_source(
    client: AsyncClient, session: AsyncSession
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.viewer)
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="Secure Gate",
            source_type=CameraSourceType.rtsp,
            source="rtsp://guard:RotateMe123!@camera.secure:554/live",
            metadata={"stream_url": "https://guard:RotateMe123!@camera.secure/video"},
        )
    )

    response = await client.get(
        f"/api/v1/cameras/{camera.id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "rtsp://camera.secure:554/...."
    assert payload["source_redacted"] is True
    assert payload["credentials_rotation_required"] is True
    assert payload["metadata"]["stream_url"] == "https://camera.secure/...."


@pytest.mark.asyncio
async def test_camera_update_audit_log_keeps_raw_source_out_of_metadata(
    client: AsyncClient, session: AsyncSession
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="Audit Secure",
            source_type=CameraSourceType.rtsp,
            source="rtsp://camera.audit/live",
        )
    )

    response = await client.patch(
        f"/api/v1/cameras/{camera.id}",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"source": "rtsp://guard:RotateMe123!@camera.audit:554/live"},
    )

    assert response.status_code == 200
    audit_response = await client.get(
        "/api/v1/monitoring/audit-logs?action=cameras.update",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert audit_response.status_code == 200
    item = audit_response.json()["items"][0]
    assert item["metadata"]["source_updated"] is True
    assert item["metadata"]["source_descriptor"] == "rtsp://camera.audit:554/...."
    assert "RotateMe123!" not in str(item["metadata"])


@pytest.mark.asyncio
async def test_camera_overlays_return_recent_server_detection_boxes(
    client: AsyncClient, session: AsyncSession
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.viewer)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Overlay Cam", source_type=CameraSourceType.http, source="http://camera/overlay")
    )
    await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.known_person,
            priority=IncidentPriority.medium,
            confidence=0.87,
            bounding_boxes=[
                {"x1": 10, "y1": 20, "x2": 120, "y2": 220, "label": "person"},
                {"x1": 30, "y1": 40, "x2": 90, "y2": 120, "label": "face"},
            ],
            recognized_identity={
                "status": "known",
                "identity_label": "Dana Rivers",
                "match_confidence": 0.93,
                "person_type": "employee",
                "department": "Security",
                "reference_id": "EMP-1042",
                "title": "Shift Lead",
            },
            metadata={"track_id": "trk-42", "detection_metadata": {"department": "Security"}},
        )
    )

    response = await client.get(
        f"/api/v1/cameras/{camera.id}/overlays",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    overlay = response.json()["overlays"][0]
    assert overlay["detection_type"] == "known_person"
    assert overlay["track_id"] == "trk-42"
    assert overlay["identity_label"] == "Dana Rivers"
    assert overlay["match_confidence"] == 0.93
    assert overlay["person_type"] == "employee"
    assert overlay["department"] == "Security"
    assert overlay["reference_id"] == "EMP-1042"
    assert overlay["title"] == "Shift Lead"
    assert overlay["bounding_box"]["label"] == "person"
    assert overlay["face_bounding_box"]["label"] == "face"


def test_detection_summary_exposes_known_person_profile_and_match_confidence() -> None:
    summary = CameraDetectionService._summarize_detections(
        [
            {
                "label": "known_person",
                "confidence": 0.88,
                "x1": 10,
                "y1": 20,
                "x2": 110,
                "y2": 220,
                "recognition": {
                    "status": "known",
                    "identity_id": "f634caa1-513f-46bc-bf1c-4f563d75844d",
                    "identity_label": "Dana Rivers",
                    "match_confidence": 0.94,
                    "metadata": {
                        "person_type": "employee",
                        "department": "Security",
                        "reference_id": "EMP-1042",
                        "title": "Shift Lead",
                    },
                },
            }
        ]
    )[0]

    assert summary.recognition_status == "known"
    assert summary.identity_label == "Dana Rivers"
    assert summary.match_confidence == 0.94
    assert summary.person_type == "employee"
    assert summary.department == "Security"
    assert summary.reference_id == "EMP-1042"
    assert summary.title == "Shift Lead"


@pytest.mark.asyncio
async def test_latest_scan_overlay_clears_threats_but_bridges_one_missed_person_frame(
    client: AsyncClient, session: AsyncSession
) -> None:
    from app.services.camera_overlays import camera_overlay_store

    access_token = await _create_user_and_login(client, session, UserRole.viewer)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Current Overlay Cam", source_type=CameraSourceType.http, source="http://camera/current")
    )
    person = CameraDetectionScanSummary(
        detection_type="person",
        confidence=0.94,
        track_id="person-1",
        bounding_box=DetectionBoundingBox(x1=10, y1=20, x2=110, y2=220, label="person"),
    )
    weapon = CameraDetectionScanSummary(
        detection_type="weapon",
        confidence=0.91,
        track_id="weapon-1",
        bounding_box=DetectionBoundingBox(x1=40, y1=60, x2=80, y2=120, label="weapon"),
    )

    await camera_overlay_store.publish(camera.id, [person, weapon])
    await camera_overlay_store.publish(camera.id, [])
    response = await client.get(
        f"/api/v1/cameras/{camera.id}/overlays",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert response.status_code == 200
    overlays = response.json()["overlays"]
    assert [overlay["detection_type"] for overlay in overlays] == ["person"]
    assert overlays[0]["incident_id"] is None


@pytest.mark.asyncio
async def test_live_frame_transport_rejects_server_readable_cameras(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Worker Owned Cam", source_type=CameraSourceType.http, source="http://camera/worker")
    )

    async def fail_if_scanned(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("Server-readable camera must not enter the browser live-scan path")

    monkeypatch.setattr("app.services.camera_detection.CameraDetectionService.run_scan", fail_if_scanned)
    response = await client.post(
        f"/api/v1/cameras/{camera.id}/live-scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"frame_content_base64": "ZmFrZS1mcmFtZQ=="},
    )

    assert response.status_code == 400
    assert "continuous detection worker" in response.json()["detail"]


@pytest.mark.asyncio
async def test_operator_cannot_trigger_manual_camera_scan(
    client: AsyncClient, session: AsyncSession
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.operator)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Restricted Scan Cam", source_type=CameraSourceType.http, source="http://camera/scan")
    )

    response = await client.post(
        f"/api/v1/cameras/{camera.id}/scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"occurrence_hint": "privileged_manual_scan"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_supervisor_can_run_repeated_transient_live_scans(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Live Scan Cam", source_type=CameraSourceType.usb, source="browser-camera")
    )
    received_payloads = []

    async def fake_run_scan(self, camera_id, payload):  # type: ignore[no-untyped-def]
        received_payloads.append(payload)
        return CameraDetectionScanResponse(
            camera_id=camera_id,
            model_name="test-model",
            model_version="1",
            detection_count=0,
            incident_count=0,
            alert_count=0,
            ignored_count=0,
            detections=[],
            ignored_reasons=[],
            backend="test",
            callback_delivered=False,
        )

    monkeypatch.setattr("app.services.camera_detection.CameraDetectionService.run_scan", fake_run_scan)
    request_payload = {
        "frame_content_base64": "ZmFrZS1mcmFtZQ==",
        "include_evidence": True,
        "occurrence_hint": "privileged_manual_scan",
    }

    first = await client.post(
        f"/api/v1/cameras/{camera.id}/live-scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json=request_payload,
    )
    second = await client.post(
        f"/api/v1/cameras/{camera.id}/live-scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json=request_payload,
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len(received_payloads) == 2
    assert all(payload.include_evidence is True for payload in received_payloads)
    assert all(payload.occurrence_hint == "dashboard_live_scan" for payload in received_payloads)


@pytest.mark.asyncio
async def test_file_camera_can_scan_the_current_browser_playback_frame(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Recorded Preview Cam", source_type=CameraSourceType.file, source="uploads/camera.mp4")
    )
    received_payloads = []

    async def fake_run_scan(self, camera_id, payload):  # type: ignore[no-untyped-def]
        received_payloads.append(payload)
        return CameraDetectionScanResponse(
            camera_id=camera_id,
            model_name="test-model",
            model_version="1",
            detection_count=0,
            incident_count=0,
            alert_count=0,
            ignored_count=0,
            detections=[],
            ignored_reasons=[],
            backend="test",
            callback_delivered=False,
        )

    monkeypatch.setattr("app.services.camera_detection.CameraDetectionService.run_scan", fake_run_scan)
    response = await client.post(
        f"/api/v1/cameras/{camera.id}/live-scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"frame_content_base64": "ZmFrZS1mcmFtZQ==", "include_evidence": True},
    )

    assert response.status_code == 200
    assert len(received_payloads) == 1
    assert received_payloads[0].frame_content_base64 == "ZmFrZS1mcmFtZQ=="
    assert received_payloads[0].include_evidence is True
    assert received_payloads[0].occurrence_hint == "dashboard_live_scan"


@pytest.mark.asyncio
async def test_privileged_manual_camera_scan_is_rate_limited(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)
    camera = await CameraRepository(session).create(
        CameraCreate(name="Rate Limited Scan Cam", source_type=CameraSourceType.http, source="http://camera/rate")
    )

    async def fake_run_scan(self, camera_id, payload):  # type: ignore[no-untyped-def]
        return CameraDetectionScanResponse(
            camera_id=camera_id,
            model_name="test-model",
            model_version="1",
            detection_count=0,
            incident_count=0,
            alert_count=0,
            ignored_count=0,
            detections=[],
            ignored_reasons=[],
            backend="test",
            callback_delivered=False,
        )

    monkeypatch.setattr("app.services.camera_detection.CameraDetectionService.run_scan", fake_run_scan)

    first = await client.post(
        f"/api/v1/cameras/{camera.id}/scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"occurrence_hint": "privileged_manual_scan"},
    )
    second = await client.post(
        f"/api/v1/cameras/{camera.id}/scan",
        headers={"Authorization": f"Bearer {access_token}"},
        json={"occurrence_hint": "privileged_manual_scan"},
    )

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"]


@pytest.mark.asyncio
async def test_http_health_check_blocks_metadata_and_link_local_ranges(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    camera = await cameras.create(
        CameraCreate(
            name="Metadata Trap",
            source_type=CameraSourceType.http,
            source="http://169.254.169.254/latest/meta-data/",
        )
    )

    result = await CameraStreamingService(cameras).test_connection(camera)

    assert result.status.value == "offline"
    assert "blocked address" in result.message
    assert "169.254.169.254" in result.message


def test_camera_network_policy_revalidates_redirect_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    policy = CameraNetworkPolicy()
    headers = Message()
    headers["Location"] = "http://169.254.169.254/latest/meta-data/"

    class FakeOpener:
        def open(self, request, timeout=0):  # type: ignore[no-untyped-def]
            raise HTTPError(
                request.full_url,
                302,
                "Found",
                headers,
                None,
            )

    monkeypatch.setattr("app.services.camera_network_policy.build_opener", lambda *args, **kwargs: FakeOpener())
    monkeypatch.setattr(
        CameraNetworkPolicy,
        "_resolve_host",
        staticmethod(
            lambda hostname: tuple(
                ipaddress.ip_address(address)
                for address in {
                    "camera.local": ("192.168.0.25",),
                    "169.254.169.254": ("169.254.169.254",),
                }[hostname]
            )
        ),
    )

    with pytest.raises(ValueError, match="blocked address 169.254.169.254"):
        policy.open_http_url(
            "http://camera.local/live",
            method="GET",
            timeout=3,
            headers={"User-Agent": "AegisPro/1.0"},
        )


@pytest.mark.asyncio
async def test_camera_media_upload_is_stored_and_registers_as_a_readable_file_camera(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "storage_root", tmp_path)
    access_token = await _create_user_and_login(client, session, UserRole.supervisor)

    upload_response = await client.post(
        "/api/v1/cameras/media",
        headers={"Authorization": f"Bearer {access_token}"},
        files={"media": ("weapons.mp4", b"fake-video-data", "video/mp4")},
    )

    assert upload_response.status_code == 201
    upload = upload_response.json()
    assert upload["source"].startswith("uploads/cameras/")
    assert upload["source"].endswith(".mp4")
    assert (tmp_path / upload["source"]).read_bytes() == b"fake-video-data"

    create_response = await client.post(
        "/api/v1/cameras",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "name": "Uploaded Evidence",
            "source_type": "file",
            "source": upload["source"],
        },
    )

    assert create_response.status_code == 201
    camera_id = create_response.json()["id"]
    health_response = await client.post(
        f"/api/v1/cameras/{camera_id}/test-connection",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "online"


@pytest.mark.asyncio
async def test_file_camera_registration_rejects_a_client_machine_path(
    client: AsyncClient,
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "storage_root", tmp_path)
    access_token = await _create_user_and_login(client, session, UserRole.operator)
    response = await client.post(
        "/api/v1/cameras",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "name": "Invalid browser path",
            "source_type": "file",
            "source": 'storage\\"C:\\Users\\operator\\Downloads\\weapons.mp4"',
        },
    )

    assert response.status_code == 400
    assert "uploaded media path" in response.json()["detail"]


@pytest.mark.asyncio
async def test_http_proxy_falls_back_to_a_working_phone_camera_endpoint(
    session: AsyncSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cameras = CameraRepository(session)
    camera = await cameras.create(
        CameraCreate(
            name="Phone fallback",
            source_type=CameraSourceType.http,
            source="http://192.168.0.25:8080",
        )
    )
    attempts: list[str] = []

    def stub_probe(source: str, skip_tls_verification: bool = False) -> tuple[int, str]:
        del skip_tls_verification
        attempts.append(source)
        if source.endswith("/video"):
            return 404, "text/plain"
        return 200, "image/jpeg"

    monkeypatch.setattr(CameraStreamingService, "_probe_http_source_with_headers", staticmethod(stub_probe))
    proxy = CameraStreamingService(cameras).prepare_http_stream_proxy(
        camera,
        "http://192.168.0.25:8080",
    )

    assert attempts == ["http://192.168.0.25:8080/video", "http://192.168.0.25:8080/shot.jpg"]
    assert proxy.source == "http://192.168.0.25:8080/shot.jpg"
    assert proxy.content_type == "image/jpeg"


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
