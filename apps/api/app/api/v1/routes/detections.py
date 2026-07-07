from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_detection_ingest_access
from app.db.session import get_db
from app.models.user import User
from app.schemas.detections import DetectionEventIngest, DetectionEventIngestResponse
from app.services.detection_events import DetectionEventService

router = APIRouter()


def get_detection_event_service(session: AsyncSession = Depends(get_db)) -> DetectionEventService:
    return DetectionEventService(session)


@router.post("/ingest", response_model=DetectionEventIngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_detection_events(
    payload: DetectionEventIngest,
    _: User | None = Depends(require_detection_ingest_access),
    events: DetectionEventService = Depends(get_detection_event_service),
) -> DetectionEventIngestResponse:
    return await events.ingest(payload)
