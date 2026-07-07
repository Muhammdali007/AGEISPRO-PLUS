from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.camera import CameraSourceType, CameraStatus


class CameraBase(BaseModel):
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


class CameraCreate(CameraBase):
    pass


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


class CameraRead(CameraBase):
    id: UUID
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
