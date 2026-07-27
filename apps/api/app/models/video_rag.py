from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, Float, ForeignKey, Index, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class VideoRagIndexStatus(StrEnum):
    queued = "queued"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class VideoRagIndex(Base):
    __tablename__ = "video_rag_indexes"
    __table_args__ = (
        Index("ix_video_rag_indexes_status_available", "status", "available_at"),
        Index("ix_video_rag_indexes_lease_expires", "lease_expires_at"),
    )

    incident_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("incidents.id", ondelete="CASCADE"), primary_key=True
    )
    status: Mapped[VideoRagIndexStatus] = mapped_column(
        Enum(VideoRagIndexStatus), default=VideoRagIndexStatus.queued, index=True
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    evidence_fingerprint: Mapped[str | None] = mapped_column(String(64), nullable=True)
    vision_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), onupdate=lambda: datetime.now(UTC)
    )

    incident = relationship("Incident")


class VideoRagChunk(Base):
    __tablename__ = "video_rag_chunks"
    __table_args__ = (
        Index("ix_video_rag_chunks_incident_kind", "incident_id", "kind"),
    )

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    incident_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("incidents.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    clip_start_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    clip_end_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    metadata_: Mapped[dict[str, object]] = mapped_column("metadata", JSON, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(768).with_variant(JSON(), "sqlite"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))

    incident = relationship("Incident")
