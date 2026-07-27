"""add incident video RAG indexes and chunks

Revision ID: 20260722_0014
Revises: 20260717_0013
Create Date: 2026-07-22 22:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "20260722_0014"
down_revision: str | None = "20260717_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgres = bind.dialect.name == "postgresql"
    if is_postgres:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    inspector = sa.inspect(bind)
    if "video_rag_indexes" not in inspector.get_table_names():
        op.create_table(
            "video_rag_indexes",
            sa.Column("incident_id", sa.Uuid(), nullable=False),
            sa.Column(
                "status",
                sa.Enum("queued", "processing", "ready", "failed", name="videoragindexstatus"),
                nullable=False,
            ),
            sa.Column("attempts", sa.Integer(), nullable=False),
            sa.Column("available_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("evidence_fingerprint", sa.String(length=64), nullable=True),
            sa.Column("vision_model", sa.String(length=160), nullable=True),
            sa.Column("embedding_model", sa.String(length=160), nullable=True),
            sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("incident_id"),
        )
        op.create_index("ix_video_rag_indexes_status", "video_rag_indexes", ["status"])
        op.create_index("ix_video_rag_indexes_available_at", "video_rag_indexes", ["available_at"])
        op.create_index("ix_video_rag_indexes_status_available", "video_rag_indexes", ["status", "available_at"])
        op.create_index("ix_video_rag_indexes_lease_expires", "video_rag_indexes", ["lease_expires_at"])

    inspector = sa.inspect(bind)
    if "video_rag_chunks" not in inspector.get_table_names():
        embedding_type = Vector(768) if is_postgres else sa.JSON()
        op.create_table(
            "video_rag_chunks",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("incident_id", sa.Uuid(), nullable=False),
            sa.Column("kind", sa.String(length=32), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("clip_start_seconds", sa.Float(), nullable=True),
            sa.Column("clip_end_seconds", sa.Float(), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=False),
            sa.Column("embedding", embedding_type, nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index("ix_video_rag_chunks_incident_id", "video_rag_chunks", ["incident_id"])
        op.create_index("ix_video_rag_chunks_kind", "video_rag_chunks", ["kind"])
        op.create_index("ix_video_rag_chunks_incident_kind", "video_rag_chunks", ["incident_id", "kind"])
        if is_postgres:
            op.execute(
                "CREATE INDEX ix_video_rag_chunks_embedding_hnsw ON video_rag_chunks "
                "USING hnsw (embedding vector_cosine_ops)"
            )
            op.execute(
                "CREATE INDEX ix_video_rag_chunks_content_fts ON video_rag_chunks "
                "USING gin (to_tsvector('english', content))"
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "video_rag_chunks" in inspector.get_table_names():
        op.drop_table("video_rag_chunks")
    inspector = sa.inspect(bind)
    if "video_rag_indexes" in inspector.get_table_names():
        op.drop_table("video_rag_indexes")
    if bind.dialect.name == "postgresql":
        sa.Enum(name="videoragindexstatus").drop(bind, checkfirst=True)
