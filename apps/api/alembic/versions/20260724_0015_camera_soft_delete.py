"""soft delete cameras while retaining incident history

Revision ID: 20260724_0015
Revises: 20260722_0014
Create Date: 2026-07-24 16:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260724_0015"
down_revision: str | None = "20260722_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True))
    op.create_index("ix_cameras_deleted_at", "cameras", ["deleted_at"])


def downgrade() -> None:
    op.drop_index("ix_cameras_deleted_at", table_name="cameras")
    op.drop_column("cameras", "deleted_at")
