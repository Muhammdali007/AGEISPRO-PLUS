from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class PersonFaceEmbedding(Base):
    __tablename__ = "person_face_embeddings"

    id: Mapped[UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    person_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("persons.id", ondelete="CASCADE"), index=True
    )
    face_profile_id: Mapped[str] = mapped_column(String(120), index=True)
    label: Mapped[str] = mapped_column(String(160))
    image_path: Mapped[str] = mapped_column(Text)
    embedding_literal: Mapped[str] = mapped_column(Text)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, index=True)
    embedding_model: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    person = relationship("Person")
