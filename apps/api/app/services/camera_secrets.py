from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlparse

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from app.core.config import settings
from app.models.camera import Camera, CameraSourceType
from app.models.camera_secret import CameraSecret
from app.schemas.cameras import CameraRead

NETWORK_CAMERA_SOURCE_TYPES = frozenset({CameraSourceType.http, CameraSourceType.rtsp})


class CameraSecretManager:
    def __init__(self) -> None:
        primary_key, *secondary_keys = settings.camera_secret_keys or [settings.secret_key]
        self._primary = Fernet(_coerce_fernet_key(primary_key))
        self._multi = MultiFernet(
            [self._primary, *(Fernet(_coerce_fernet_key(key)) for key in secondary_keys)]
        )

    def encrypt_source(self, source: str) -> str:
        return self._primary.encrypt(source.encode("utf-8")).decode("utf-8")

    def decrypt_source(self, encrypted_source: str) -> str:
        try:
            return self._multi.decrypt(encrypted_source.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise ValueError("Camera source secret could not be decrypted.") from exc

    @staticmethod
    def source_requires_secret_storage(source_type: CameraSourceType) -> bool:
        return source_type in NETWORK_CAMERA_SOURCE_TYPES

    @staticmethod
    def requires_device_credential_rotation(source_type: CameraSourceType, source: str) -> bool:
        if source_type not in NETWORK_CAMERA_SOURCE_TYPES:
            return False
        parsed = urlparse(source)
        return bool(parsed.username or parsed.password)

    @staticmethod
    def build_source_descriptor(source_type: CameraSourceType, source: str) -> str:
        if source_type not in NETWORK_CAMERA_SOURCE_TYPES:
            return source

        parsed = urlparse(source)
        if not parsed.scheme or not parsed.hostname:
            return f"{source_type.value}://[redacted]/...."

        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{parsed.hostname}{port}/...."

    @classmethod
    def build_candidate_descriptor(cls, source_type: CameraSourceType, source: str) -> str:
        return cls.build_source_descriptor(source_type, source)

    @classmethod
    def sanitize_metadata(cls, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}
        return {
            str(key): cls._sanitize_metadata_value(str(key), value) for key, value in metadata.items()
        }

    @classmethod
    def serialize_camera(cls, camera: Camera) -> CameraRead:
        return CameraRead.model_validate(
            {
                "id": camera.id,
                "name": camera.name,
                "source_type": camera.source_type,
                "source": camera.source,
                "source_redacted": camera.source_redacted,
                "credentials_rotation_required": camera.credentials_rotation_required,
                "status": camera.status,
                "location": camera.location,
                "group": camera.group,
                "tags": list(camera.tags),
                "detection_enabled": camera.detection_enabled,
                "inference_fps": camera.inference_fps,
                "metadata_": cls.sanitize_metadata(camera.metadata_),
                "last_seen_at": camera.last_seen_at,
                "health_checked_at": camera.health_checked_at,
                "created_at": camera.created_at,
                "updated_at": camera.updated_at,
            }
        )

    @classmethod
    def sanitize_audit_payload(cls, payload: Mapping[str, Any]) -> dict[str, Any]:
        sanitized = dict(payload)
        source_type_raw = sanitized.get("source_type")
        source_type = (
            source_type_raw
            if isinstance(source_type_raw, CameraSourceType)
            else CameraSourceType(str(source_type_raw))
            if source_type_raw
            else None
        )
        if "source" in sanitized:
            source = sanitized.pop("source")
            if isinstance(source, str) and source and source_type:
                sanitized["source_descriptor"] = cls.build_source_descriptor(source_type, source)
            sanitized["source_updated"] = True
        if "metadata" in sanitized and isinstance(sanitized["metadata"], Mapping):
            sanitized["metadata"] = cls.sanitize_metadata(sanitized["metadata"])
        return sanitized

    @classmethod
    def _sanitize_metadata_value(cls, key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                str(child_key): cls._sanitize_metadata_value(str(child_key), child_value)
                for child_key, child_value in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return [cls._sanitize_metadata_value(key, item) for item in value]
        if isinstance(value, str) and _looks_like_network_uri_key(key):
            return cls._sanitize_uri_string(value)
        return value

    @classmethod
    def _sanitize_uri_string(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme == "rtsp":
            return cls.build_source_descriptor(CameraSourceType.rtsp, value)
        if parsed.scheme in {"http", "https"}:
            return cls.build_source_descriptor(CameraSourceType.http, value)
        return value


def build_camera_secret(camera_id, encrypted_source: str) -> CameraSecret:
    return CameraSecret(camera_id=camera_id, encrypted_source=encrypted_source)


def _coerce_fernet_key(value: str) -> bytes:
    stripped = value.strip().encode("utf-8")
    try:
        decoded = base64.urlsafe_b64decode(stripped)
    except Exception:
        decoded = b""
    if len(decoded) == 32:
        return stripped
    return base64.urlsafe_b64encode(hashlib.sha256(stripped).digest())


def _looks_like_network_uri_key(key: str) -> bool:
    normalized = "".join(character for character in key.lower() if character.isalnum() or character == "_")
    return normalized.endswith("url") or normalized.endswith("uri") or normalized == "source"
