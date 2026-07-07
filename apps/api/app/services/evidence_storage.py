import base64
import binascii
import mimetypes
from pathlib import Path
from uuid import UUID

from fastapi import HTTPException, status

from app.core.config import settings


class EvidenceStorageService:
    def __init__(self, storage_root: Path | None = None) -> None:
        self.storage_root = (storage_root or settings.storage_root).resolve()

    @staticmethod
    def validate_base64_payload(content_base64: str) -> None:
        try:
            base64.b64decode(content_base64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid base64 evidence payload",
            ) from exc

    def resolve_relative_path(self, relative_path: str) -> Path:
        candidate = (self.storage_root / relative_path).resolve()
        if not self._is_within_root(candidate):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Evidence path is outside the configured storage root",
            )
        return candidate

    def store_incident_blob(
        self,
        *,
        camera_id: UUID,
        incident_id: UUID,
        stem: str,
        content_base64: str,
        content_type: str | None,
    ) -> str:
        self.validate_base64_payload(content_base64)
        blob = base64.b64decode(content_base64, validate=True)

        extension = self._extension_for_content_type(content_type)
        relative_path = Path("incidents") / str(camera_id) / str(incident_id) / f"{stem}{extension}"
        destination = self.resolve_relative_path(relative_path.as_posix())
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(blob)
        return relative_path.as_posix()

    def ensure_exists(self, relative_path: str) -> Path:
        path = self.resolve_relative_path(relative_path)
        if not path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Evidence file not found")
        return path

    def _is_within_root(self, candidate: Path) -> bool:
        try:
            candidate.relative_to(self.storage_root)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extension_for_content_type(content_type: str | None) -> str:
        if not content_type:
            return ".bin"

        guessed = mimetypes.guess_extension(content_type, strict=False)
        if guessed == ".jpe":
            return ".jpg"
        return guessed or ".bin"
