from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

REDACTED_AUDIT_VALUE = "[REDACTED]"
DEFAULT_AUDIT_EXCLUDE_FIELDS = frozenset({"password", "password_hash"})
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


def dump_audit_model(model: BaseModel, *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded_fields = set(DEFAULT_AUDIT_EXCLUDE_FIELDS)
    if exclude:
        excluded_fields.update(exclude)
    return model.model_dump(exclude=excluded_fields, exclude_unset=True)


def redact_audit_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            serialized_key = str(key)
            if _is_sensitive_audit_key(serialized_key):
                redacted[serialized_key] = REDACTED_AUDIT_VALUE
                continue
            redacted[serialized_key] = redact_audit_metadata(item)
        return redacted

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_audit_metadata(item) for item in value]

    return value


def _is_sensitive_audit_key(key: str) -> bool:
    normalized = _KEY_SANITIZER.sub("", key.lower())
    return normalized in _SENSITIVE_AUDIT_KEYS or normalized.endswith(
        ("password", "token", "secret", "apikey")
    )
