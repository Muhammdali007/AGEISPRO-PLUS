"""phase 2 core backend

Revision ID: 20260701_0002
Revises: 20260701_0001
Create Date: 2026-07-01
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260701_0002"
down_revision: str | None = "20260701_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

camera_source_type = postgresql.ENUM(
    "usb", "rtsp", "http", "file", name="camerasourcetype", create_type=False
)
camera_status = postgresql.ENUM(
    "online", "offline", "degraded", "disabled", "unknown", name="camerastatus", create_type=False
)
detection_type = postgresql.ENUM(
    "weapon",
    "fire",
    "smoke",
    "person",
    "known_person",
    "unknown_person",
    "system",
    name="detectiontype",
    create_type=False,
)
incident_priority = postgresql.ENUM(
    "critical", "high", "medium", "low", name="incidentpriority", create_type=False
)
incident_status = postgresql.ENUM(
    "open",
    "acknowledged",
    "investigating",
    "resolved",
    "dismissed",
    name="incidentstatus",
    create_type=False,
)
alert_status = postgresql.ENUM(
    "active", "acknowledged", "cleared", name="alertstatus", create_type=False
)


def upgrade() -> None:
    bind = op.get_bind()
    camera_source_type.create(bind, checkfirst=True)
    camera_status.create(bind, checkfirst=True)
    detection_type.create(bind, checkfirst=True)
    incident_priority.create(bind, checkfirst=True)
    incident_status.create(bind, checkfirst=True)
    alert_status.create(bind, checkfirst=True)

    op.create_table(
        "cameras",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("source_type", camera_source_type, nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("status", camera_status, nullable=False),
        sa.Column("location", sa.String(length=255), nullable=True),
        sa.Column("group", sa.String(length=120), nullable=True),
        sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("detection_enabled", sa.Boolean(), nullable=False),
        sa.Column("inference_fps", sa.Integer(), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("health_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_cameras_group", "cameras", ["group"])
    op.create_index("ix_cameras_name", "cameras", ["name"])
    op.create_index("ix_cameras_source_type", "cameras", ["source_type"])
    op.create_index("ix_cameras_status", "cameras", ["status"])

    op.create_table(
        "incidents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("camera_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("detection_type", detection_type, nullable=False),
        sa.Column("priority", incident_priority, nullable=False),
        sa.Column("status", incident_status, nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bounding_boxes", sa.JSON(), nullable=False),
        sa.Column("snapshot_path", sa.Text(), nullable=True),
        sa.Column("clip_path", sa.Text(), nullable=True),
        sa.Column("recognized_identity", sa.JSON(), nullable=True),
        sa.Column("operator_notes", sa.Text(), nullable=True),
        sa.Column("assigned_user_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("metadata", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["assigned_user_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["camera_id"], ["cameras.id"], ondelete="RESTRICT"),
    )
    op.create_index("ix_incidents_assigned_user_id", "incidents", ["assigned_user_id"])
    op.create_index("ix_incidents_camera_id", "incidents", ["camera_id"])
    op.create_index("ix_incidents_detection_type", "incidents", ["detection_type"])
    op.create_index("ix_incidents_occurred_at", "incidents", ["occurred_at"])
    op.create_index("ix_incidents_priority", "incidents", ["priority"])
    op.create_index("ix_incidents_status", "incidents", ["status"])

    op.create_table(
        "alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("incident_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("priority", incident_priority, nullable=False),
        sa.Column("status", alert_status, nullable=False),
        sa.Column("title", sa.String(length=180), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("acknowledged", sa.Boolean(), nullable=False),
        sa.Column("acknowledged_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["acknowledged_by_id"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["incident_id"], ["incidents.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_alerts_acknowledged", "alerts", ["acknowledged"])
    op.create_index("ix_alerts_created_at", "alerts", ["created_at"])
    op.create_index("ix_alerts_incident_id", "alerts", ["incident_id"])
    op.create_index("ix_alerts_priority", "alerts", ["priority"])
    op.create_index("ix_alerts_status", "alerts", ["status"])


def downgrade() -> None:
    op.drop_index("ix_alerts_status", table_name="alerts")
    op.drop_index("ix_alerts_priority", table_name="alerts")
    op.drop_index("ix_alerts_incident_id", table_name="alerts")
    op.drop_index("ix_alerts_created_at", table_name="alerts")
    op.drop_index("ix_alerts_acknowledged", table_name="alerts")
    op.drop_table("alerts")

    op.drop_index("ix_incidents_status", table_name="incidents")
    op.drop_index("ix_incidents_priority", table_name="incidents")
    op.drop_index("ix_incidents_occurred_at", table_name="incidents")
    op.drop_index("ix_incidents_detection_type", table_name="incidents")
    op.drop_index("ix_incidents_camera_id", table_name="incidents")
    op.drop_index("ix_incidents_assigned_user_id", table_name="incidents")
    op.drop_table("incidents")

    op.drop_index("ix_cameras_status", table_name="cameras")
    op.drop_index("ix_cameras_source_type", table_name="cameras")
    op.drop_index("ix_cameras_name", table_name="cameras")
    op.drop_index("ix_cameras_group", table_name="cameras")
    op.drop_table("cameras")

    bind = op.get_bind()
    alert_status.drop(bind, checkfirst=True)
    incident_status.drop(bind, checkfirst=True)
    incident_priority.drop(bind, checkfirst=True)
    detection_type.drop(bind, checkfirst=True)
    camera_status.drop(bind, checkfirst=True)
    camera_source_type.drop(bind, checkfirst=True)
