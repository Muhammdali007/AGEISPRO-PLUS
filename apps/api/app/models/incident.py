from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Enum, Float, ForeignKey, JSON, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class DetectionType(StrEnum):
    weapon = "weapon"
    fire = "fire"
    smoke = "smoke"
    person = "person"
    known_person = "known_person"
    unknown_person = "unknown_person"
    system = "system"


class IncidentPriority(StrEnum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class IncidentStatus(StrEnum):
    open = "open"
    acknowledged = "acknowledged"
    investigating = "investigating"
    resolved = "resolved"
    dismissed = "dismissed"


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    camera_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("cameras.id", ondelete="RESTRICT"), index=True
    )
    detection_type: Mapped[DetectionType] = mapped_column(Enum(DetectionType), index=True)
    priority: Mapped[IncidentPriority] = mapped_column(Enum(IncidentPriority), index=True)
    status: Mapped[IncidentStatus] = mapped_column(
        Enum(IncidentStatus), default=IncidentStatus.open, index=True
    )
    confidence: Mapped[float] = mapped_column(Float)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    bounding_boxes: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    snapshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    clip_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    recognized_identity: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    operator_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    assigned_user_id: Mapped[UUID | None] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    camera = relationship("Camera")
    assigned_user = relationship("User")
