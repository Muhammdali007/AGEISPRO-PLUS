"""person face embeddings

Revision ID: 20260706_0005
Revises: 20260706_0004
Create Date: 2026-07-06
"""

from collections.abc import Sequence
from datetime import UTC, datetime
import json
from uuid import uuid4

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260706_0005"
down_revision: str | None = "20260706_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "person_face_embeddings" not in inspector.get_table_names():
        op.create_table(
            "person_face_embeddings",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("person_id", postgresql.UUID(as_uuid=True), nullable=False),
            sa.Column("face_profile_id", sa.String(length=120), nullable=False),
            sa.Column("label", sa.String(length=160), nullable=False),
            sa.Column("image_path", sa.Text(), nullable=False),
            sa.Column("embedding_literal", sa.Text(), nullable=False),
            sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
            sa.Column("embedding_model", sa.String(length=120), nullable=True),
            sa.Column("is_primary", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["person_id"], ["persons.id"], ondelete="CASCADE"),
        )
        inspector = sa.inspect(bind)

    existing_indexes = {index["name"] for index in inspector.get_indexes("person_face_embeddings")}
    if "ix_person_face_embeddings_person_id" not in existing_indexes:
        op.create_index("ix_person_face_embeddings_person_id", "person_face_embeddings", ["person_id"])
    if "ix_person_face_embeddings_face_profile_id" not in existing_indexes:
        op.create_index(
            "ix_person_face_embeddings_face_profile_id",
            "person_face_embeddings",
            ["face_profile_id"],
        )
    if "ix_person_face_embeddings_embedding_dimensions" not in existing_indexes:
        op.create_index(
            "ix_person_face_embeddings_embedding_dimensions",
            "person_face_embeddings",
            ["embedding_dimensions"],
        )
    if "ix_person_face_embeddings_is_primary" not in existing_indexes:
        op.create_index(
            "ix_person_face_embeddings_is_primary",
            "person_face_embeddings",
            ["is_primary"],
        )

    rows = bind.execute(sa.text("SELECT id, full_name, face_profiles FROM persons")).mappings()
    for row in rows:
        profiles = row["face_profiles"] or []
        if isinstance(profiles, str):
            try:
                profiles = json.loads(profiles)
            except json.JSONDecodeError:
                profiles = []
        if isinstance(profiles, dict):
            profiles = list(profiles.values())

        for profile in profiles:
            if isinstance(profile, str):
                try:
                    profile = json.loads(profile)
                except json.JSONDecodeError:
                    continue
            if not isinstance(profile, dict):
                continue
            vector = profile.get("embedding_vector") or []
            if not vector:
                continue
            bind.execute(
                sa.text(
                    """
                    INSERT INTO person_face_embeddings (
                        id,
                        person_id,
                        face_profile_id,
                        label,
                        image_path,
                        embedding_literal,
                        embedding_dimensions,
                        embedding_model,
                        is_primary,
                        metadata,
                        created_at
                    )
                    SELECT
                        :id,
                        :person_id,
                        :face_profile_id,
                        :label,
                        :image_path,
                        :embedding_literal,
                        :embedding_dimensions,
                        :embedding_model,
                        :is_primary,
                        CAST(:metadata AS JSON),
                        :created_at
                    WHERE NOT EXISTS (
                        SELECT 1 FROM person_face_embeddings WHERE face_profile_id = :face_profile_id
                    )
                    """
                ),
                {
                    "id": str(uuid4()),
                    "person_id": str(row["id"]),
                    "face_profile_id": profile.get("id"),
                    "label": profile.get("label") or row["full_name"],
                    "image_path": profile.get("image_path") or "",
                    "embedding_literal": "[" + ",".join(str(value) for value in vector) + "]",
                    "embedding_dimensions": len(vector),
                    "embedding_model": profile.get("embedding_model"),
                    "is_primary": bool(profile.get("is_primary")),
                    "metadata": json.dumps(profile.get("metadata") or {}),
                    "created_at": datetime.now(UTC),
                },
            )


def downgrade() -> None:
    op.drop_index("ix_person_face_embeddings_is_primary", table_name="person_face_embeddings")
    op.drop_index("ix_person_face_embeddings_embedding_dimensions", table_name="person_face_embeddings")
    op.drop_index("ix_person_face_embeddings_face_profile_id", table_name="person_face_embeddings")
    op.drop_index("ix_person_face_embeddings_person_id", table_name="person_face_embeddings")
    op.drop_table("person_face_embeddings")
