from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_roles
from app.db.session import get_db
from app.models.camera import Camera, CameraStatus
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.cameras import CameraRepository
from app.schemas.cameras import (
    CameraConnectionBatch,
    CameraConnectionTest,
    CameraCreate,
    CameraLiveMonitorResponse,
    CameraRead,
    CameraStreamDescriptor,
    CameraUpdate,
)
from app.services.camera_streams import CameraStreamingService
from app.services.audit_logs import AuditLogService

router = APIRouter()


def get_camera_repository(session: AsyncSession = Depends(get_db)) -> CameraRepository:
    return CameraRepository(session)


def get_camera_streaming_service(
    cameras: CameraRepository = Depends(get_camera_repository),
) -> CameraStreamingService:
    return CameraStreamingService(cameras)


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
) -> list[Camera]:
    return await cameras.list(status=status_filter, group=group)


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
) -> Camera:
    camera = await cameras.create(payload)
    await AuditLogService(AuditLogRepository(cameras.session)).record(
        actor=current_user,
        action="cameras.create",
        resource_type="camera",
        resource_id=str(camera.id),
        metadata={"source_type": camera.source_type.value, "group": camera.group},
    )
    return camera


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
) -> Camera:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    return camera


@router.patch("/{camera_id}", response_model=CameraRead, response_model_by_alias=False)
async def update_camera(
    camera_id: UUID,
    payload: CameraUpdate,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> Camera:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    updated = await cameras.update(camera, payload)
    await AuditLogService(AuditLogRepository(cameras.session)).record(
        actor=current_user,
        action="cameras.update",
        resource_type="camera",
        resource_id=str(updated.id),
        metadata=payload.model_dump(exclude_unset=True),
    )
    return updated


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(
    camera_id: UUID,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    cameras: CameraRepository = Depends(get_camera_repository),
) -> None:
    camera = await cameras.get(camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
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
    await AuditLogService(AuditLogRepository(cameras.session)).record(
        actor=current_user,
        action="cameras.health_check",
        resource_type="camera",
        resource_id=str(camera.id),
        metadata={"status": result.status.value, "latency_ms": result.latency_ms},
    )
    return result


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
        require_roles(
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
