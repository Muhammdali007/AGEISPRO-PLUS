from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class CameraSecret(Base):
    __tablename__ = "camera_secrets"

    camera_id: Mapped[UUID] = mapped_column(
        Uuid(as_uuid=True), ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True
    )
    encrypted_source: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )

    camera = relationship("Camera", back_populates="secret")
