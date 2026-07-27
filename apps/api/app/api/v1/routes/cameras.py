import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_roles, require_stream_roles
from app.core.audit import dump_audit_model
from app.core.config import settings
from app.db.transactions import transaction_scope
from app.db.session import get_db
from app.models.camera import CameraSourceType, CameraStatus
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.schemas.cameras import (
    CameraConnectionBatch,
    CameraConnectionTest,
    CameraCreate,
    CameraDetectionOverlayRead,
    CameraDetectionOverlayResponse,
    CameraDetectionScanRequest,
    CameraDetectionScanResponse,
    CameraLiveMonitorResponse,
    CameraMediaUploadRead,
    CameraRead,
    CameraStreamDescriptor,
    CameraUpdate,
)
from app.services.camera_detection import CameraDetectionService
from app.services.camera_overlays import camera_overlay_store
from app.services.camera_secrets import CameraSecretManager
from app.services.camera_streams import CameraStreamingService
from app.services.audit_logs import AuditLogService
from app.services.transactional_outbox import TransactionalOutboxService

router = APIRouter()

CAMERA_MEDIA_EXTENSIONS = {".mp4", ".webm", ".mov", ".m4v", ".ogv", ".jpg", ".jpeg", ".png", ".webp"}
CAMERA_MEDIA_UPLOAD_MAX_BYTES = 1024 * 1024 * 1024
CAMERA_MEDIA_UPLOAD_CHUNK_BYTES = 1024 * 1024


class ManualCameraScanRateLimiter:
    def __init__(self) -> None:
        self._last_scan_by_key: dict[tuple[str, str], float] = {}

    def check(self, *, user_id: UUID, camera_id: UUID) -> None:
        now = monotonic()
        key = (str(user_id), str(camera_id))
        last_scan_at = self._last_scan_by_key.get(key)
        cooldown = settings.manual_camera_scan_cooldown_seconds
        if last_scan_at is not None and now - last_scan_at < cooldown:
            retry_after = max(1, int(cooldown - (now - last_scan_at)))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Manual AI scans are rate-limited. Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        self._last_scan_by_key[key] = now

    def reset(self) -> None:
        self._last_scan_by_key.clear()


manual_camera_scan_rate_limiter = ManualCameraScanRateLimiter()


def get_camera_repository(session: AsyncSession = Depends(get_db)) -> CameraRepository:
    return CameraRepository(session)


def get_camera_streaming_service(
    cameras: CameraRepository = Depends(get_camera_repository),
) -> CameraStreamingService:
    return CameraStreamingService(cameras)


def get_camera_detection_service(
    session: AsyncSession = Depends(get_db),
) -> CameraDetectionService:
    return CameraDetectionService(session)


@router.get("", response_model=list[CameraRead], response_model_by_alias=False)
async def list_cameras(
    status_filter: CameraStatus | None = None,
    group: str | None = None,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> list[CameraRead]:
    return [
        CameraSecretManager.serialize_camera(camera)
        for camera in await cameras.list(status=status_filter, group=group)
    ]


@router.get("/live-monitor", response_model=CameraLiveMonitorResponse, response_model_by_alias=False)
async def get_live_monitor(
    status_filter: CameraStatus | None = None,
    group: str | None = None,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
    camera_streams: CameraStreamingService = Depends(get_camera_streaming_service),
) -> CameraLiveMonitorResponse:
    return await camera_streams.describe_live_monitor(await cameras.list(status=status_filter, group=group))


@router.post("", response_model=CameraRead, response_model_by_alias=False, status_code=status.HTTP_201_CREATED)
async def create_camera(
    payload: CameraCreate,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> CameraRead:
    if payload.source_type == CameraSourceType.file:
        source_path = Path(payload.source)
        if not source_path.is_absolute() and any(character in payload.source for character in {'"', ':'}):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File camera source is not a valid uploaded media path.",
            )
        resolved_source = (
            source_path if source_path.is_absolute() else settings.storage_root / source_path
        ).resolve()
        storage_root = settings.storage_root.resolve()
        if resolved_source != storage_root and storage_root not in resolved_source.parents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File camera source must be uploaded into the configured media storage.",
            )
    async with transaction_scope(cameras.session):
        camera = await cameras.create(payload)
        await AuditLogService(AuditLogRepository(cameras.session)).record(
            actor=current_user,
            action="cameras.create",
            resource_type="camera",
            resource_id=str(camera.id),
            metadata={
                "source_type": camera.source_type.value,
                "source_descriptor": camera.source,
                "group": camera.group,
                "source_redacted": camera.source_redacted,
                "credentials_rotation_required": camera.credentials_rotation_required,
            },
        )
    return CameraSecretManager.serialize_camera(camera)


@router.post("/media", response_model=CameraMediaUploadRead, status_code=status.HTTP_201_CREATED)
async def upload_camera_media(
    media: UploadFile = File(...),
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
) -> CameraMediaUploadRead:
    original_filename = Path(media.filename or "").name
    extension = Path(original_filename).suffix.lower()
    if extension not in CAMERA_MEDIA_EXTENSIONS:
        await media.close()
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported camera media format. Upload MP4, WebM, MOV, M4V, OGV, JPG, PNG, or WebP.",
        )

    relative_path = Path("uploads") / "cameras" / f"{uuid4().hex}{extension}"
    destination = (settings.storage_root / relative_path).resolve()
    storage_root = settings.storage_root.resolve()
    if storage_root not in destination.parents:
        await media.close()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Invalid media storage path")

    destination.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with destination.open("xb") as output:
            while chunk := await media.read(CAMERA_MEDIA_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > CAMERA_MEDIA_UPLOAD_MAX_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail="Camera media file exceeds the 1 GB upload limit.",
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await media.close()

    if size == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Camera media file is empty")

    return CameraMediaUploadRead(
        source=relative_path.as_posix(),
        filename=original_filename,
        content_type=media.content_type or "application/octet-stream",
        size=size,
    )


@router.get("/{camera_id}", response_model=CameraRead, response_model_by_alias=False)
async def get_camera(
    camera_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> CameraRead:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return CameraSecretManager.serialize_camera(camera)


@router.patch("/{camera_id}", response_model=CameraRead, response_model_by_alias=False)
async def update_camera(
    camera_id: UUID,
    payload: CameraUpdate,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> CameraRead:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    async with transaction_scope(cameras.session):
        try:
            updated = await cameras.update(camera, payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        await AuditLogService(AuditLogRepository(cameras.session)).record(
            actor=current_user,
            action="cameras.update",
            resource_type="camera",
            resource_id=str(updated.id),
            metadata=CameraSecretManager.sanitize_audit_payload(
                {
                    **dump_audit_model(payload),
                    "source_type": payload.source_type or camera.source_type,
                }
            ),
        )
    return CameraSecretManager.serialize_camera(updated)


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: UUID,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> None:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    async with transaction_scope(cameras.session):
        await AuditLogService(AuditLogRepository(cameras.session)).record(
            actor=current_user,
            action="cameras.delete",
            resource_type="camera",
            resource_id=str(camera.id),
            metadata={"name": camera.name},
        )
        await cameras.delete(camera)


@router.post("/{camera_id}/test-connection", response_model=CameraConnectionTest)
async def test_camera_connection(
    camera_id: UUID,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    cameras: CameraRepository = Depends(get_camera_repository),
    camera_streams: CameraStreamingService = Depends(get_camera_streaming_service),
) -> CameraConnectionTest:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    result = await camera_streams.test_connection(camera)
    async with transaction_scope(cameras.session):
        await AuditLogService(AuditLogRepository(cameras.session)).record(
            actor=current_user,
            action="cameras.health_check",
            resource_type="camera",
            resource_id=str(camera.id),
            metadata={"status": result.status.value, "latency_ms": result.latency_ms},
        )
    return result


@router.post("/{camera_id}/scan", response_model=CameraDetectionScanResponse)
async def run_camera_scan(
    camera_id: UUID,
    payload: CameraDetectionScanRequest,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    events: CameraDetectionService = Depends(get_camera_detection_service),
) -> CameraDetectionScanResponse:
    manual_camera_scan_rate_limiter.check(user_id=current_user.id, camera_id=camera_id)
    result = await events.run_scan(camera_id, payload)
    async with transaction_scope(events.session) as scope:
        await AuditLogService(AuditLogRepository(events.session)).record(
            actor=current_user,
            action="cameras.scan",
            resource_type="camera",
            resource_id=str(camera_id),
            metadata={
                "detection_count": result.detection_count,
                "incident_count": result.incident_count,
                "alert_count": result.alert_count,
                "backend": result.backend,
            },
        )
    if scope.owns_transaction:
        await TransactionalOutboxService(events.session).publish_pending()
    return result


@router.post("/{camera_id}/live-scan", response_model=CameraDetectionScanResponse)
async def run_camera_live_scan(
    camera_id: UUID,
    payload: CameraDetectionScanRequest,
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    events: CameraDetectionService = Depends(get_camera_detection_service),
) -> CameraDetectionScanResponse:
    camera = await events.cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    preview_transport_sources = {CameraSourceType.usb, CameraSourceType.file}
    if camera.source_type not in preview_transport_sources or not payload.frame_content_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Live frame transport is only available for browser/USB and file cameras. "
                "Server-readable cameras are scanned by the continuous detection worker."
            ),
        )

    # The browser transports the exact frame currently displayed for local
    # devices and recorded files. Inference remains server-owned. Evidence is
    # retained only when the scan produces a confirmed incident.
    live_payload = payload.model_copy(
        update={
            "include_evidence": True,
            "occurrence_hint": "dashboard_live_scan",
        }
    )
    async with transaction_scope(events.session) as scope:
        result = await events.run_scan(camera_id, live_payload)
    if scope.owns_transaction:
        await TransactionalOutboxService(events.session).publish_pending()
    return result


@router.get("/{camera_id}/overlays", response_model=CameraDetectionOverlayResponse)
async def get_camera_overlays(
    camera_id: UUID,
    seconds: int = 30,
    limit: int = 20,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
    session: AsyncSession = Depends(get_db),
) -> CameraDetectionOverlayResponse:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

    latest_overlay = await camera_overlay_store.get(camera_id)
    if latest_overlay is not None:
        return latest_overlay

    lookback_seconds = min(max(seconds, 1), 300)
    overlay_limit = min(max(limit, 1), 100)
    incidents = await IncidentRepository(session).recent_for_camera(
        camera_id=camera_id,
        since=datetime.now(UTC) - timedelta(seconds=lookback_seconds),
        limit=overlay_limit,
    )
    # Incidents are persisted for investigation, but an overlay represents one
    # inference frame. Returning every recent incident paints stale boxes from
    # several frames on top of the current video.
    if incidents:
        latest_frame_at = incidents[0].occurred_at
        incidents = [incident for incident in incidents if incident.occurred_at == latest_frame_at]
    return CameraDetectionOverlayResponse(
        camera_id=camera_id,
        generated_at=datetime.now(UTC),
        overlays=[
            overlay
            for incident in incidents
            if (overlay := _incident_to_overlay(incident)) is not None
        ],
    )


def _incident_to_overlay(incident) -> CameraDetectionOverlayRead | None:
    primary_box = None
    face_box = None
    for box in incident.bounding_boxes or []:
        label = str(box.get("label", "")).lower()
        if label == "face":
            face_box = box
        elif primary_box is None:
            primary_box = box

    if primary_box is None and face_box is None:
        return None

    recognized_identity = incident.recognized_identity or {}
    detection_metadata = incident.metadata_.get("detection_metadata", {})
    if not isinstance(detection_metadata, dict):
        detection_metadata = {}
    return CameraDetectionOverlayRead(
        incident_id=incident.id,
        occurred_at=incident.occurred_at,
        detection_type=incident.detection_type.value,
        confidence=incident.confidence,
        track_id=incident.metadata_.get("track_id"),
        recognition_status=recognized_identity.get("status"),
        identity_id=recognized_identity.get("identity_id"),
        identity_label=recognized_identity.get("identity_label"),
        match_confidence=recognized_identity.get("match_confidence"),
        person_type=recognized_identity.get("person_type") or detection_metadata.get("person_type"),
        department=recognized_identity.get("department") or detection_metadata.get("department"),
        reference_id=recognized_identity.get("reference_id") or detection_metadata.get("reference_id"),
        title=recognized_identity.get("title") or detection_metadata.get("title"),
        bounding_box=_box_to_summary(primary_box),
        face_bounding_box=_box_to_summary(face_box),
        metadata=detection_metadata,
    )


def _box_to_summary(box: dict | None):
    if not box:
        return None
    return {
        "x1": float(box.get("x1", 0)),
        "y1": float(box.get("y1", 0)),
        "x2": float(box.get("x2", 0)),
        "y2": float(box.get("y2", 0)),
        "label": box.get("label"),
    }


@router.post("/live-monitor/test-connections", response_model=CameraConnectionBatch)
async def test_live_monitor_connections(
    status_filter: CameraStatus | None = None,
    group: str | None = None,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    cameras: CameraRepository = Depends(get_camera_repository),
    camera_streams: CameraStreamingService = Depends(get_camera_streaming_service),
) -> CameraConnectionBatch:
    batch = await camera_streams.test_connections(await cameras.list(status=status_filter, group=group))
    audit = AuditLogService(AuditLogRepository(cameras.session))
    async with transaction_scope(cameras.session):
        for result in batch.results:
            await audit.record(
                actor=current_user,
                action="cameras.health_check",
                resource_type="camera",
                resource_id=str(result.camera_id),
                metadata={"status": result.status.value, "latency_ms": result.latency_ms, "group": group},
            )
    return batch


@router.get("/{camera_id}/stream", response_model=CameraStreamDescriptor)
async def get_camera_stream(
    camera_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
    camera_streams: CameraStreamingService = Depends(get_camera_streaming_service),
) -> CameraStreamDescriptor:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return await camera_streams.describe_stream(camera)


@router.get("/{camera_id}/stream/file")
async def get_camera_file_stream(
    camera_id: UUID,
    _: User = Depends(
        require_stream_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
    camera_streams: CameraStreamingService = Depends(get_camera_streaming_service),
) -> FileResponse:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    if camera.source_type.value != "file":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Camera is not file-backed")

    try:
        source_path = camera_streams.resolve_file_source(camera)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not source_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Media file not found")
    return FileResponse(source_path)


@router.get("/{camera_id}/stream/http")
async def get_camera_http_stream(
    camera_id: UUID,
    _: User = Depends(
        require_stream_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    cameras: CameraRepository = Depends(get_camera_repository),
    camera_streams: CameraStreamingService = Depends(get_camera_streaming_service),
) -> StreamingResponse:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    if camera.source_type.value != "http":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Camera is not HTTP-backed")

    try:
        source = await cameras.get_runtime_source(camera)
        proxy = await asyncio.to_thread(camera_streams.prepare_http_stream_proxy, camera, source)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc

    return StreamingResponse(
        proxy.iterator,
        media_type=proxy.content_type or "application/octet-stream",
        headers={"Cache-Control": "no-store"},
    )
