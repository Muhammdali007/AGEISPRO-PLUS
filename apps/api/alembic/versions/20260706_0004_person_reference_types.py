"""person reference types

Revision ID: 20260706_0004
Revises: 20260703_0003
Create Date: 2026-07-06
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260706_0004"
down_revision: str | None = "20260703_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("persons")}
    indexes = {index["name"] for index in inspector.get_indexes("persons")}

    with op.batch_alter_table("persons") as batch_op:
        if "person_type" not in columns:
            batch_op.add_column(
                sa.Column("person_type", sa.String(length=32), nullable=False, server_default="visitor")
            )
        if "employee_id" in columns and "reference_id" not in columns:
            batch_op.alter_column("employee_id", new_column_name="reference_id")

    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("persons")}
    indexes = {index["name"] for index in inspector.get_indexes("persons")}

    if "ix_persons_employee_id" in indexes:
        op.drop_index("ix_persons_employee_id", table_name="persons")
    if "person_type" in columns and "ix_persons_person_type" not in indexes:
        op.create_index("ix_persons_person_type", "persons", ["person_type"], unique=False)
    if "reference_id" in columns and "ix_persons_reference_id" not in indexes:
        op.create_index("ix_persons_reference_id", "persons", ["reference_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_persons_reference_id", table_name="persons")
    op.drop_index("ix_persons_person_type", table_name="persons")

    with op.batch_alter_table("persons") as batch_op:
        batch_op.alter_column("reference_id", new_column_name="employee_id")
        batch_op.drop_column("person_type")

    op.create_index("ix_persons_employee_id", "persons", ["employee_id"], unique=True)
