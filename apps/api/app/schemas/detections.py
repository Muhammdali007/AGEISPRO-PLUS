from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.incident import DetectionType, IncidentPriority


class DetectionBoundingBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)
    label: str | None = None


class RecognitionStatus(StrEnum):
    known = "known"
    unknown = "unknown"


class InlineEvidencePayload(BaseModel):
    content_base64: str = Field(min_length=1)
    content_type: str | None = Field(default=None, max_length=120)


class DetectionEventIngestItem(BaseModel):
    detection_type: DetectionType
    confidence: float = Field(ge=0, le=1)
    track_id: str | None = Field(default=None, max_length=64)
    bounding_box: DetectionBoundingBox | None = None
    identity_id: UUID | None = None
    identity_label: str | None = Field(default=None, max_length=160)
    match_confidence: float | None = Field(default=None, ge=0, le=1)
    recognition_status: RecognitionStatus | None = None
    face_bounding_box: DetectionBoundingBox | None = None
    face_image_path: str | None = None
    face_image_evidence: InlineEvidencePayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionEventIngest(BaseModel):
    camera_id: UUID
    occurred_at: datetime | None = None
    model_name: str = Field(min_length=1, max_length=120)
    model_version: str | None = Field(default=None, max_length=80)
    inference_fps: float | None = Field(default=None, ge=0.1, le=120)
    source_fps: float | None = Field(default=None, ge=0.1, le=240)
    snapshot_path: str | None = None
    clip_path: str | None = None
    snapshot_evidence: InlineEvidencePayload | None = None
    clip_evidence: InlineEvidencePayload | None = None
    detections: list[DetectionEventIngestItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectionEventResult(BaseModel):
    incident_id: UUID
    detection_type: DetectionType
    priority: IncidentPriority
    alert_id: UUID | None = None


class DetectionEventIngestResponse(BaseModel):
    camera_id: UUID
    incident_count: int
    alert_count: int
    ignored_count: int
    results: list[DetectionEventResult] = Field(default_factory=list)
    ignored_reasons: list[str] = Field(default_factory=list)
