"""add transactional outbox events

Revision ID: 20260714_0009
Revises: 20260714_0008
Create Date: 2026-07-14 22:10:00.000000
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260714_0009"
down_revision = "20260714_0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "outbox_events" not in inspector.get_table_names():
        op.create_table(
            "outbox_events",
            sa.Column("id", sa.Uuid(), nullable=False),
            sa.Column("event_type", sa.String(length=120), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        inspector = sa.inspect(bind)

    existing_indexes = {index["name"] for index in inspector.get_indexes("outbox_events")}
    if "ix_outbox_events_created_at" not in existing_indexes:
        op.create_index(op.f("ix_outbox_events_created_at"), "outbox_events", ["created_at"], unique=False)
    if "ix_outbox_events_event_type" not in existing_indexes:
        op.create_index(op.f("ix_outbox_events_event_type"), "outbox_events", ["event_type"], unique=False)
    if "ix_outbox_events_published_at" not in existing_indexes:
        op.create_index(op.f("ix_outbox_events_published_at"), "outbox_events", ["published_at"], unique=False)
    if "ix_outbox_events_published_created" not in existing_indexes:
        op.create_index(
            "ix_outbox_events_published_created",
            "outbox_events",
            ["published_at", "created_at"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "outbox_events" not in inspector.get_table_names():
        return

    existing_indexes = {index["name"] for index in inspector.get_indexes("outbox_events")}
    if "ix_outbox_events_published_created" in existing_indexes:
        op.drop_index("ix_outbox_events_published_created", table_name="outbox_events")
    if "ix_outbox_events_published_at" in existing_indexes:
        op.drop_index(op.f("ix_outbox_events_published_at"), table_name="outbox_events")
    if "ix_outbox_events_event_type" in existing_indexes:
        op.drop_index(op.f("ix_outbox_events_event_type"), table_name="outbox_events")
    if "ix_outbox_events_created_at" in existing_indexes:
        op.drop_index(op.f("ix_outbox_events_created_at"), table_name="outbox_events")
    op.drop_table("outbox_events")
