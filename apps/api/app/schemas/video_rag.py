from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from app.core.config import settings
from app.models.incident import DetectionType


class VideoRagQueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    camera_ids: list[UUID] = Field(default_factory=list, max_length=50)
    start_at: datetime | None = None
    end_at: datetime | None = None
    limit: int = Field(default_factory=lambda: settings.video_rag_default_limit, ge=1, le=20)

    @model_validator(mode="after")
    def validate_range(self) -> "VideoRagQueryRequest":
        if self.start_at and self.end_at and self.start_at > self.end_at:
            raise ValueError("start_at must be before end_at")
        return self


class VideoRagEvidence(BaseModel):
    incident_id: UUID
    camera_id: UUID
    camera_name: str
    occurred_at: datetime
    detection_type: DetectionType
    confidence: float
    matched_excerpt: str
    relevance_score: float
    clip_start_seconds: float | None = None
    clip_end_seconds: float | None = None
    snapshot_url: str | None = None
    clip_url: str | None = None


class VideoRagIndexFreshness(BaseModel):
    latest_indexed_at: datetime | None
    ready: int
    queued: int
    processing: int
    failed: int


class VideoRagQueryResponse(BaseModel):
    answer: str
    evidence: list[VideoRagEvidence]
    warnings: list[str] = Field(default_factory=list)
    freshness: VideoRagIndexFreshness


class VideoRagStatusResponse(VideoRagIndexFreshness):
    enabled: bool
