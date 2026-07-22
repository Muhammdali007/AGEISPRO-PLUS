from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models.incident import (
    DetectionType,
    IncidentPriority,
    IncidentRetentionClass,
    IncidentStatus,
)
from app.schemas.persons import PersonType


class IncidentCreate(BaseModel):
    camera_id: UUID
    detection_type: DetectionType
    priority: IncidentPriority = IncidentPriority.medium
    status: IncidentStatus = IncidentStatus.open
    retention_class: IncidentRetentionClass | None = None
    legal_hold: bool = False
    legal_hold_reason: str | None = None
    confidence: float = Field(ge=0, le=1)
    occurred_at: datetime | None = None
    bounding_boxes: list[dict[str, Any]] = Field(default_factory=list)
    snapshot_path: str | None = None
    clip_path: str | None = None
    recognized_identity: dict[str, Any] | None = None
    operator_notes: str | None = None
    assigned_user_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IncidentUpdate(BaseModel):
    priority: IncidentPriority | None = None
    status: IncidentStatus | None = None
    retention_class: IncidentRetentionClass | None = None
    legal_hold: bool | None = None
    legal_hold_reason: str | None = None
    operator_notes: str | None = None
    assigned_user_id: UUID | None = None
    metadata: dict[str, Any] | None = None


class IncidentRead(BaseModel):
    id: UUID
    camera_id: UUID
    detection_type: DetectionType
    priority: IncidentPriority
    status: IncidentStatus
    retention_class: IncidentRetentionClass
    confidence: float
    occurred_at: datetime
    retention_expires_at: datetime | None
    legal_hold: bool
    legal_hold_reason: str | None
    bounding_boxes: list[dict[str, Any]]
    snapshot_path: str | None
    clip_path: str | None
    recognized_identity: dict[str, Any] | None
    operator_notes: str | None
    assigned_user_id: UUID | None
    metadata: dict[str, Any] = Field(alias="metadata_")
    archived_at: datetime | None
    deletion_requested_at: datetime | None
    deletion_started_at: datetime | None
    deletion_completed_at: datetime | None
    deletion_error: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, serialize_by_alias=False)


class IncidentRetentionPolicyRead(BaseModel):
    retention_class: IncidentRetentionClass
    retention_hours: int | None
    description: str


class IncidentSavePersonRequest(BaseModel):
    full_name: str = Field(min_length=1, max_length=160)
    person_type: PersonType = "visitor"
    department: str | None = Field(default=None, max_length=120)
    reference_id: str | None = Field(default=None, min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=120)
    is_active: bool = True
    is_primary: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
