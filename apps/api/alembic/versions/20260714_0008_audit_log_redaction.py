"""redact sensitive audit log metadata

Revision ID: 20260714_0008
Revises: 20260707_0007
Create Date: 2026-07-14 20:30:00.000000
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

import sqlalchemy as sa
from alembic import op

revision = "20260714_0008"
down_revision = "20260707_0007"
branch_labels = None
depends_on = None

REDACTED_AUDIT_VALUE = "[REDACTED]"
_SENSITIVE_AUDIT_KEYS = frozenset(
    {
        "password",
        "passwordhash",
        "currentpassword",
        "newpassword",
        "confirmpassword",
        "accesstoken",
        "refreshtoken",
        "token",
        "authorization",
        "secret",
        "apikey",
        "clientsecret",
    }
)
_KEY_SANITIZER = re.compile(r"[^a-z0-9]+")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "audit_logs" not in inspector.get_table_names():
        return

    audit_logs = sa.table(
        "audit_logs",
        sa.column("id", sa.String()),
        sa.column("metadata", sa.JSON()),
    )

    rows = bind.execute(sa.select(audit_logs.c.id, audit_logs.c.metadata)).mappings()
    for row in rows:
        metadata = row["metadata"] or {}
        redacted = _redact_audit_metadata(metadata)
        if redacted == metadata:
            continue
        bind.execute(
            sa.update(audit_logs).where(audit_logs.c.id == row["id"]).values(metadata=redacted)
        )


def downgrade() -> None:
    return None


def _redact_audit_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            serialized_key = str(key)
            if _is_sensitive_audit_key(serialized_key):
                redacted[serialized_key] = REDACTED_AUDIT_VALUE
                continue
            redacted[serialized_key] = _redact_audit_metadata(item)
        return redacted

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_audit_metadata(item) for item in value]

    return value


def _is_sensitive_audit_key(key: str) -> bool:
    normalized = _KEY_SANITIZER.sub("", key.lower())
    return normalized in _SENSITIVE_AUDIT_KEYS or normalized.endswith(
        ("password", "token", "secret", "apikey")
    )
