"""phase 8 monitoring

Revision ID: 20260707_0006
Revises: 20260706_0005
Create Date: 2026-07-07
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260707_0006"
down_revision: str | None = "20260706_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if "audit_logs" not in inspector.get_table_names():
        op.create_table(
            "audit_logs",
            sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
            sa.Column("actor_user_id", postgresql.UUID(as_uuid=True), nullable=True),
            sa.Column("actor_email", sa.String(length=320), nullable=True),
            sa.Column("actor_role", sa.String(length=32), nullable=True),
            sa.Column("action", sa.String(length=80), nullable=False),
            sa.Column("resource_type", sa.String(length=80), nullable=False),
            sa.Column("resource_id", sa.Text(), nullable=True),
            sa.Column("metadata", sa.JSON(), nullable=False, server_default=sa.text("'{}'::json")),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.ForeignKeyConstraint(["actor_user_id"], ["users.id"], ondelete="SET NULL"),
        )
        inspector = sa.inspect(bind)

    existing_indexes = {index["name"] for index in inspector.get_indexes("audit_logs")}
    for name, column in [
        ("ix_audit_logs_actor_user_id", "actor_user_id"),
        ("ix_audit_logs_actor_email", "actor_email"),
        ("ix_audit_logs_actor_role", "actor_role"),
        ("ix_audit_logs_action", "action"),
        ("ix_audit_logs_resource_type", "resource_type"),
        ("ix_audit_logs_resource_id", "resource_id"),
        ("ix_audit_logs_created_at", "created_at"),
    ]:
        if name not in existing_indexes:
            op.create_index(name, "audit_logs", [column], unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_resource_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_resource_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_role", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_email", table_name="audit_logs")
    op.drop_index("ix_audit_logs_actor_user_id", table_name="audit_logs")
    op.drop_table("audit_logs")
