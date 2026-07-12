from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, Enum, Index, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CameraSourceType(StrEnum):
    usb = "usb"
    rtsp = "rtsp"
    http = "http"
    file = "file"


class CameraStatus(StrEnum):
    online = "online"
    offline = "offline"
    degraded = "degraded"
    disabled = "disabled"
    unknown = "unknown"


class Camera(Base):
    __tablename__ = "cameras"
    __table_args__ = (Index("ix_cameras_status_group", "status", "group"),)

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(160), index=True)
    source_type: Mapped[CameraSourceType] = mapped_column(Enum(CameraSourceType), index=True)
    source: Mapped[str] = mapped_column(Text)
    status: Mapped[CameraStatus] = mapped_column(
        Enum(CameraStatus), default=CameraStatus.unknown, index=True
    )
    location: Mapped[str | None] = mapped_column(String(255), nullable=True)
    group: Mapped[str | None] = mapped_column(String(120), nullable=True, index=True)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    detection_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    inference_fps: Mapped[int] = mapped_column(Integer, default=5)
    last_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    health_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
