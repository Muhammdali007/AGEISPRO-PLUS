"""add user sessions

Revision ID: 20260717_0013
Revises: 20260716_0012
Create Date: 2026-07-17 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260717_0013"
down_revision: str | None = "20260716_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "user_sessions" in inspector.get_table_names():
        return

    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("access_jti_digest", sa.String(length=64), nullable=False),
        sa.Column("refresh_jti_digest", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_user_sessions_access_jti_digest", "user_sessions", ["access_jti_digest"], unique=True)
    op.create_index("ix_user_sessions_created_at", "user_sessions", ["created_at"])
    op.create_index("ix_user_sessions_expires_at", "user_sessions", ["expires_at"])
    op.create_index("ix_user_sessions_refresh_jti_digest", "user_sessions", ["refresh_jti_digest"], unique=True)
    op.create_index("ix_user_sessions_revoked_at", "user_sessions", ["revoked_at"])
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "user_sessions" not in inspector.get_table_names():
        return

    op.drop_index("ix_user_sessions_user_id", table_name="user_sessions")
    op.drop_index("ix_user_sessions_revoked_at", table_name="user_sessions")
    op.drop_index("ix_user_sessions_refresh_jti_digest", table_name="user_sessions")
    op.drop_index("ix_user_sessions_expires_at", table_name="user_sessions")
    op.drop_index("ix_user_sessions_created_at", table_name="user_sessions")
    op.drop_index("ix_user_sessions_access_jti_digest", table_name="user_sessions")
    op.drop_table("user_sessions")
