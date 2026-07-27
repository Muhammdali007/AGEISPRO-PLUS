from datetime import datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.camera import CameraSourceType, CameraStatus
from app.schemas.detections import DetectionBoundingBox


class CameraWriteBase(BaseModel):
    name: str = Field(min_length=1, max_length=160)
    source_type: CameraSourceType
    source: str = Field(min_length=1)
    status: CameraStatus = CameraStatus.unknown
    location: str | None = Field(default=None, max_length=255)
    group: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list)
    detection_enabled: bool = True
    inference_fps: int = Field(default=5, ge=1, le=30)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_source_for_type(self) -> "CameraWriteBase":
        if self.source_type == CameraSourceType.http:
            parsed = urlparse(self.source.strip())
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
                raise ValueError("HTTP camera source must be a complete http:// or https:// URL.")
        return self


class CameraCreate(CameraWriteBase):
    pass


class CameraMediaUploadRead(BaseModel):
    source: str
    filename: str
    content_type: str
    size: int


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=160)
    source_type: CameraSourceType | None = None
    source: str | None = Field(default=None, min_length=1)
    status: CameraStatus | None = None
    location: str | None = Field(default=None, max_length=255)
    group: str | None = Field(default=None, max_length=120)
    tags: list[str] | None = None
    detection_enabled: bool | None = None
    inference_fps: int | None = Field(default=None, ge=1, le=30)
    metadata: dict[str, Any] | None = None


class CameraRead(BaseModel):
    id: UUID
    name: str
    source_type: CameraSourceType
    source: str = Field(min_length=1)
    source_redacted: bool = False
    credentials_rotation_required: bool = False
    status: CameraStatus = CameraStatus.unknown
    location: str | None = Field(default=None, max_length=255)
    group: str | None = Field(default=None, max_length=120)
    tags: list[str] = Field(default_factory=list)
    detection_enabled: bool = True
    inference_fps: int = Field(default=5, ge=1, le=30)
    last_seen_at: datetime | None
    health_checked_at: datetime | None
    created_at: datetime
    updated_at: datetime

    metadata: dict[str, Any] = Field(default_factory=dict, alias="metadata_")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, serialize_by_alias=False)


class CameraConnectionTest(BaseModel):
    camera_id: UUID
    status: CameraStatus
    message: str
    checked_at: datetime
    latency_ms: int | None = None


class CameraStreamDescriptor(BaseModel):
    camera_id: UUID
    stream_kind: str
    stream_url: str | None = None
    browser_supported: bool
    requires_relay: bool = False
    is_live: bool = True
    health_status: CameraStatus
    health_message: str
    checked_at: datetime | None = None
    controls: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    browser_device_id: str | None = None


class CameraLiveMonitorEntry(BaseModel):
    camera: CameraRead
    stream: CameraStreamDescriptor


class CameraLiveMonitorSummary(BaseModel):
    total: int
    online: int
    offline: int
    degraded: int
    disabled: int
    unknown: int
    live: int
    browser_ready: int
    relay_required: int
    detection_enabled: int
    groups: dict[str, int] = Field(default_factory=dict)


class CameraLiveMonitorResponse(BaseModel):
    summary: CameraLiveMonitorSummary
    entries: list[CameraLiveMonitorEntry]


class CameraConnectionBatch(BaseModel):
    results: list[CameraConnectionTest]


class CameraDetectionScanRequest(BaseModel):
    frame_content_base64: str | None = Field(default=None, min_length=1)
    frame_content_type: str | None = Field(default="image/jpeg", max_length=120)
    include_evidence: bool = True
    requested_detectors: list[str] = Field(
        default_factory=lambda: ["weapon", "person", "fire", "smoke"]
    )
    recognition_enabled: bool = False
    occurrence_hint: str | None = Field(default=None, max_length=64)


class CameraDetectionScanSummary(BaseModel):
    detection_type: str
    object_label: str | None = None
    confidence: float
    track_id: str | None = None
    recognition_status: str | None = None
    identity_id: UUID | None = None
    identity_label: str | None = None
    match_confidence: float | None = Field(default=None, ge=0, le=1)
    person_type: str | None = None
    department: str | None = None
    reference_id: str | None = None
    title: str | None = None
    bounding_box: DetectionBoundingBox | None = None
    face_bounding_box: DetectionBoundingBox | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CameraDetectionScanResponse(BaseModel):
    camera_id: UUID
    model_name: str
    model_version: str
    detection_count: int
    incident_count: int
    alert_count: int
    ignored_count: int
    detections: list[CameraDetectionScanSummary] = Field(default_factory=list)
    ignored_reasons: list[str] = Field(default_factory=list)
    backend: str | None = None
    callback_delivered: bool = False


class CameraDetectionOverlayRead(CameraDetectionScanSummary):
    # The latest-frame overlay is transient and may be produced even when the
    # corresponding incident is suppressed as a duplicate.
    incident_id: UUID | None = None
    occurred_at: datetime


class CameraDetectionOverlayResponse(BaseModel):
    camera_id: UUID
    generated_at: datetime
    overlays: list[CameraDetectionOverlayRead] = Field(default_factory=list)
