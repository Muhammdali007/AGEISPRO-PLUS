"""move camera network sources into encrypted secret storage

Revision ID: 20260714_0010
Revises: 20260714_0009
Create Date: 2026-07-14 23:10:00.000000
"""

from __future__ import annotations

import base64
import hashlib
import os
from datetime import UTC, datetime
from urllib.parse import urlparse

import sqlalchemy as sa
from alembic import op
from cryptography.fernet import Fernet, MultiFernet

revision = "20260714_0010"
down_revision = "20260714_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "cameras" not in inspector.get_table_names():
        return

    camera_columns = {column["name"] for column in inspector.get_columns("cameras")}
    with op.batch_alter_table("cameras") as batch_op:
        if "source_redacted" not in camera_columns:
            batch_op.add_column(
                sa.Column("source_redacted", sa.Boolean(), nullable=False, server_default=sa.false())
            )
        if "credentials_rotation_required" not in camera_columns:
            batch_op.add_column(
                sa.Column(
                    "credentials_rotation_required",
                    sa.Boolean(),
                    nullable=False,
                    server_default=sa.false(),
                )
            )

    if "camera_secrets" not in inspector.get_table_names():
        op.create_table(
            "camera_secrets",
            sa.Column("camera_id", sa.Uuid(), sa.ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("encrypted_source", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )

    cameras = sa.table(
        "cameras",
        sa.column("id", sa.Uuid()),
        sa.column("source_type", sa.String()),
        sa.column("source", sa.Text()),
        sa.column("source_redacted", sa.Boolean()),
        sa.column("credentials_rotation_required", sa.Boolean()),
    )
    camera_secrets = sa.table(
        "camera_secrets",
        sa.column("camera_id", sa.Uuid()),
        sa.column("encrypted_source", sa.Text()),
        sa.column("created_at", sa.DateTime(timezone=True)),
        sa.column("updated_at", sa.DateTime(timezone=True)),
    )

    now = datetime.now(UTC)
    encrypt = _build_encryptor()
    rows = bind.execute(
        sa.select(cameras.c.id, cameras.c.source_type, cameras.c.source)
    ).mappings()
    for row in rows:
        source_type = str(row["source_type"] or "").lower()
        source = str(row["source"] or "")
        if source_type not in {"http", "rtsp"} or not source:
            continue

        bind.execute(
            sa.insert(camera_secrets).values(
                camera_id=row["id"],
                encrypted_source=encrypt(source),
                created_at=now,
                updated_at=now,
            )
        )
        bind.execute(
            sa.update(cameras)
            .where(cameras.c.id == row["id"])
            .values(
                source=_build_descriptor(source_type, source),
                source_redacted=True,
                credentials_rotation_required=_requires_rotation(source),
            )
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "cameras" not in inspector.get_table_names() or "camera_secrets" not in inspector.get_table_names():
        return

    cameras = sa.table(
        "cameras",
        sa.column("id", sa.Uuid()),
        sa.column("source", sa.Text()),
    )
    camera_secrets = sa.table(
        "camera_secrets",
        sa.column("camera_id", sa.Uuid()),
        sa.column("encrypted_source", sa.Text()),
    )

    decrypt = _build_decryptor()
    rows = bind.execute(
        sa.select(camera_secrets.c.camera_id, camera_secrets.c.encrypted_source)
    ).mappings()
    for row in rows:
        bind.execute(
            sa.update(cameras)
            .where(cameras.c.id == row["camera_id"])
            .values(source=decrypt(str(row["encrypted_source"])))
        )

    op.drop_table("camera_secrets")
    with op.batch_alter_table("cameras") as batch_op:
        batch_op.drop_column("credentials_rotation_required")
        batch_op.drop_column("source_redacted")


def _build_descriptor(source_type: str, source: str) -> str:
    parsed = urlparse(source)
    if not parsed.scheme or not parsed.hostname:
        return f"{source_type}://[redacted]/...."
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}/...."


def _requires_rotation(source: str) -> bool:
    parsed = urlparse(source)
    return bool(parsed.username or parsed.password)


def _build_encryptor():
    primary_key, *_ = _camera_secret_keys()
    fernet = Fernet(_coerce_fernet_key(primary_key))

    def encrypt(value: str) -> str:
        return fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    return encrypt


def _build_decryptor():
    fernets = [Fernet(_coerce_fernet_key(key)) for key in _camera_secret_keys()]
    multi = MultiFernet(fernets)

    def decrypt(value: str) -> str:
        return multi.decrypt(value.encode("utf-8")).decode("utf-8")

    return decrypt


def _camera_secret_keys() -> list[str]:
    configured = (
        os.getenv("CAMERA_SECRET_KEYS")
        or os.getenv("API_CAMERA_SECRET_KEYS")
        or os.getenv("SECRET_KEY")
        or "replace-with-a-long-random-secret"
    )
    return [item.strip() for item in configured.split(",") if item.strip()]


def _coerce_fernet_key(value: str) -> bytes:
    stripped = value.strip().encode("utf-8")
    try:
        decoded = base64.urlsafe_b64decode(stripped)
    except Exception:
        decoded = b""
    if len(decoded) == 32:
        return stripped
    return base64.urlsafe_b64encode(hashlib.sha256(stripped).digest())
