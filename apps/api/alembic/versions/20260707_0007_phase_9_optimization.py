"""phase 9 optimization

Revision ID: 20260707_0007
Revises: 20260707_0006
Create Date: 2026-07-07 22:30:00.000000
"""

from alembic import op
from sqlalchemy import inspect


revision = "20260707_0007"
down_revision = "20260707_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)

    index_specs = {
        "cameras": [("ix_cameras_status_group", ["status", "group"])],
        "incidents": [
            ("ix_incidents_occurred_at_detection_type", ["occurred_at", "detection_type"]),
            ("ix_incidents_status_occurred_at", ["status", "occurred_at"]),
        ],
        "alerts": [("ix_alerts_status_created_at", ["status", "created_at"])],
        "audit_logs": [
            ("ix_audit_logs_action_created_at", ["action", "created_at"]),
            ("ix_audit_logs_resource_type_created_at", ["resource_type", "created_at"]),
        ],
    }

    for table_name, indexes in index_specs.items():
        existing = {index["name"] for index in inspector.get_indexes(table_name)}
        for name, columns in indexes:
            if name not in existing:
                op.create_index(name, table_name, columns, unique=False)


def downgrade() -> None:
    op.drop_index("ix_audit_logs_resource_type_created_at", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action_created_at", table_name="audit_logs")
    op.drop_index("ix_alerts_status_created_at", table_name="alerts")
    op.drop_index("ix_incidents_status_occurred_at", table_name="incidents")
    op.drop_index("ix_incidents_occurred_at_detection_type", table_name="incidents")
    op.drop_index("ix_cameras_status_group", table_name="cameras")
