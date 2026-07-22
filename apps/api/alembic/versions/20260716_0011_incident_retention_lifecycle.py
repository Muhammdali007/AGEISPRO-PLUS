"""add incident retention lifecycle controls

Revision ID: 20260716_0011
Revises: 20260714_0010
Create Date: 2026-07-16 09:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260716_0011"
down_revision: str | None = "20260714_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

incident_retention_class = postgresql.ENUM(
    "standard",
    "extended",
    "compliance",
    "manual",
    name="incidentretentionclass",
    create_type=False,
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "incidents" not in inspector.get_table_names():
        return

    incident_retention_class.create(bind, checkfirst=True)
    incident_columns = {column["name"] for column in inspector.get_columns("incidents")}
    existing_indexes = {index["name"] for index in inspector.get_indexes("incidents")}

    with op.batch_alter_table("incidents") as batch_op:
        if "retention_class" not in incident_columns:
            batch_op.add_column(
                sa.Column(
                    "retention_class",
                    incident_retention_class,
                    nullable=False,
                    server_default="standard",
                )
            )
        if "retention_expires_at" not in incident_columns:
            batch_op.add_column(sa.Column("retention_expires_at", sa.DateTime(timezone=True), nullable=True))
        if "legal_hold" not in incident_columns:
            batch_op.add_column(
                sa.Column("legal_hold", sa.Boolean(), nullable=False, server_default=sa.false())
            )
        if "legal_hold_reason" not in incident_columns:
            batch_op.add_column(sa.Column("legal_hold_reason", sa.Text(), nullable=True))
        if "archived_at" not in incident_columns:
            batch_op.add_column(sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True))
        if "deletion_requested_at" not in incident_columns:
            batch_op.add_column(
                sa.Column("deletion_requested_at", sa.DateTime(timezone=True), nullable=True)
            )
        if "deletion_started_at" not in incident_columns:
            batch_op.add_column(
                sa.Column("deletion_started_at", sa.DateTime(timezone=True), nullable=True)
            )
        if "deletion_completed_at" not in incident_columns:
            batch_op.add_column(
                sa.Column("deletion_completed_at", sa.DateTime(timezone=True), nullable=True)
            )
        if "deletion_error" not in incident_columns:
            batch_op.add_column(sa.Column("deletion_error", sa.Text(), nullable=True))

    if bind.dialect.name == "sqlite":
        bind.execute(
            sa.text(
                """
                UPDATE incidents
                SET retention_class = CASE
                    WHEN priority = 'critical' THEN 'compliance'
                    ELSE 'standard'
                END,
                retention_expires_at = CASE
                    WHEN priority = 'critical' THEN datetime(occurred_at, '+168 hours')
                    ELSE datetime(occurred_at, '+24 hours')
                END,
                legal_hold = COALESCE(legal_hold, 0)
                WHERE retention_expires_at IS NULL OR retention_class IS NULL
                """
            )
        )
    else:
        bind.execute(
            sa.text(
                """
                UPDATE incidents
                SET retention_class = CASE
                    WHEN priority = 'critical' THEN 'compliance'
                    ELSE 'standard'
                END::incidentretentionclass,
                retention_expires_at = CASE
                    WHEN priority = 'critical' THEN occurred_at + INTERVAL '168 hours'
                    ELSE occurred_at + INTERVAL '24 hours'
                END,
                legal_hold = COALESCE(legal_hold, FALSE)
                WHERE retention_expires_at IS NULL OR retention_class IS NULL
                """
            )
        )

    if "ix_incidents_retention_class" not in existing_indexes:
        op.create_index("ix_incidents_retention_class", "incidents", ["retention_class"])
    if "ix_incidents_legal_hold" not in existing_indexes:
        op.create_index("ix_incidents_legal_hold", "incidents", ["legal_hold"])
    if "ix_incidents_retention_expires_at" not in existing_indexes:
        op.create_index("ix_incidents_retention_expires_at", "incidents", ["retention_expires_at"])
    if "ix_incidents_archived_at_deletion_requested_at" not in existing_indexes:
        op.create_index(
            "ix_incidents_archived_at_deletion_requested_at",
            "incidents",
            ["archived_at", "deletion_requested_at"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "incidents" not in inspector.get_table_names():
        return

    existing_indexes = {index["name"] for index in inspector.get_indexes("incidents")}
    if "ix_incidents_archived_at_deletion_requested_at" in existing_indexes:
        op.drop_index("ix_incidents_archived_at_deletion_requested_at", table_name="incidents")
    if "ix_incidents_retention_expires_at" in existing_indexes:
        op.drop_index("ix_incidents_retention_expires_at", table_name="incidents")
    if "ix_incidents_legal_hold" in existing_indexes:
        op.drop_index("ix_incidents_legal_hold", table_name="incidents")
    if "ix_incidents_retention_class" in existing_indexes:
        op.drop_index("ix_incidents_retention_class", table_name="incidents")

    incident_columns = {column["name"] for column in inspector.get_columns("incidents")}
    with op.batch_alter_table("incidents") as batch_op:
        if "deletion_error" in incident_columns:
            batch_op.drop_column("deletion_error")
        if "deletion_completed_at" in incident_columns:
            batch_op.drop_column("deletion_completed_at")
        if "deletion_started_at" in incident_columns:
            batch_op.drop_column("deletion_started_at")
        if "deletion_requested_at" in incident_columns:
            batch_op.drop_column("deletion_requested_at")
        if "archived_at" in incident_columns:
            batch_op.drop_column("archived_at")
        if "legal_hold_reason" in incident_columns:
            batch_op.drop_column("legal_hold_reason")
        if "legal_hold" in incident_columns:
            batch_op.drop_column("legal_hold")
        if "retention_expires_at" in incident_columns:
            batch_op.drop_column("retention_expires_at")
        if "retention_class" in incident_columns:
            batch_op.drop_column("retention_class")

    incident_retention_class.drop(bind, checkfirst=True)
