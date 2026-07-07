"""phase 6 persons

Revision ID: 20260703_0003
Revises: 20260701_0002
Create Date: 2026-07-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260703_0003"
down_revision: str | None = "20260701_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "persons" not in inspector.get_table_names():
        op.create_table(
            "persons",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("full_name", sa.String(length=160), nullable=False),
            sa.Column("department", sa.String(length=120), nullable=True),
            sa.Column("employee_id", sa.String(length=64), nullable=False),
            sa.Column("title", sa.String(length=120), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("face_profiles", sa.JSON(), nullable=False),
            sa.Column("face_image_count", sa.Integer(), nullable=False),
            sa.Column("embedding_count", sa.Integer(), nullable=False),
            sa.Column("visit_count", sa.Integer(), nullable=False),
            sa.Column("recognition_count", sa.Integer(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_recognized_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
        inspector = sa.inspect(bind)

    existing_indexes = {index["name"] for index in inspector.get_indexes("persons")}
    if "ix_persons_employee_id" not in existing_indexes:
        op.create_index("ix_persons_employee_id", "persons", ["employee_id"], unique=True)
    if "ix_persons_full_name" not in existing_indexes:
        op.create_index("ix_persons_full_name", "persons", ["full_name"])
    if "ix_persons_is_active" not in existing_indexes:
        op.create_index("ix_persons_is_active", "persons", ["is_active"])
    if "ix_persons_last_seen_at" not in existing_indexes:
        op.create_index("ix_persons_last_seen_at", "persons", ["last_seen_at"])
    if "ix_persons_last_recognized_at" not in existing_indexes:
        op.create_index("ix_persons_last_recognized_at", "persons", ["last_recognized_at"])


def downgrade() -> None:
    op.drop_index("ix_persons_last_recognized_at", table_name="persons")
    op.drop_index("ix_persons_last_seen_at", table_name="persons")
    op.drop_index("ix_persons_is_active", table_name="persons")
    op.drop_index("ix_persons_full_name", table_name="persons")
    op.drop_index("ix_persons_employee_id", table_name="persons")
    op.drop_table("persons")
