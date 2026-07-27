from time import monotonic

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_roles
from app.core.config import settings
from app.db.session import get_db
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.schemas.video_rag import VideoRagQueryRequest, VideoRagQueryResponse, VideoRagStatusResponse
from app.services.audit_logs import AuditLogService
from app.services.ollama import OllamaUnavailableError
from app.services.video_rag_query import VideoRagQueryService, question_digest

router = APIRouter()
allowed_roles = require_roles(
    UserRole.administrator, UserRole.supervisor, UserRole.operator, UserRole.viewer
)


@router.get("/status", response_model=VideoRagStatusResponse)
async def get_video_rag_status(
    _: User = Depends(allowed_roles), session: AsyncSession = Depends(get_db)
) -> VideoRagStatusResponse:
    return await VideoRagQueryService(session).status()


@router.post("/query", response_model=VideoRagQueryResponse)
async def query_video_rag(
    payload: VideoRagQueryRequest,
    current_user: User = Depends(allowed_roles),
    session: AsyncSession = Depends(get_db),
) -> VideoRagQueryResponse:
    if not settings.video_rag_enabled:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Incident Video RAG is not enabled",
        )
    started = monotonic()
    try:
        response = await VideoRagQueryService(session).query(payload)
    except OllamaUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    await AuditLogService(AuditLogRepository(session)).record(
        actor=current_user,
        action="video_rag.query",
        resource_type="incident_search",
        metadata={
            "question_sha256": question_digest(payload.question),
            "camera_ids": [str(value) for value in payload.camera_ids],
            "start_at": payload.start_at,
            "end_at": payload.end_at,
            "result_incident_ids": [str(item.incident_id) for item in response.evidence],
            "duration_ms": round((monotonic() - started) * 1000),
        },
    )
    await session.commit()
    return response
