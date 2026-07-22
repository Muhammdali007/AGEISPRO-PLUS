"""add password reset tokens

Revision ID: 20260716_0012
Revises: 20260716_0011
Create Date: 2026-07-16 14:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260716_0012"
down_revision: str | None = "20260716_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "password_reset_tokens" in inspector.get_table_names():
        return

    op.create_table(
        "password_reset_tokens",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("token_digest", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_password_reset_tokens_created_at", "password_reset_tokens", ["created_at"])
    op.create_index("ix_password_reset_tokens_expires_at", "password_reset_tokens", ["expires_at"])
    op.create_index(
        "ix_password_reset_tokens_token_digest",
        "password_reset_tokens",
        ["token_digest"],
        unique=True,
    )
    op.create_index("ix_password_reset_tokens_used_at", "password_reset_tokens", ["used_at"])
    op.create_index("ix_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "password_reset_tokens" not in inspector.get_table_names():
        return

    op.drop_index("ix_password_reset_tokens_user_id", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_used_at", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_token_digest", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_expires_at", table_name="password_reset_tokens")
    op.drop_index("ix_password_reset_tokens_created_at", table_name="password_reset_tokens")
    op.drop_table("password_reset_tokens")
