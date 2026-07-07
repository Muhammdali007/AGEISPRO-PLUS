from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.models.alert import AlertStatus
from app.models.incident import IncidentPriority


class AlertCreate(BaseModel):
    incident_id: UUID
    priority: IncidentPriority
    title: str = Field(min_length=1, max_length=180)
    message: str = Field(min_length=1)


class AlertRead(BaseModel):
    id: UUID
    incident_id: UUID
    priority: IncidentPriority
    status: AlertStatus
    title: str
    message: str
    acknowledged: bool
    acknowledged_by_id: UUID | None
    acknowledged_at: datetime | None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
