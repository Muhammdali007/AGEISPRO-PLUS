from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, UploadFile, status

from app.core.config import settings
from app.models.person import Person
from app.schemas.persons import PersonFaceEnrollment
from app.services.face_embeddings import FaceEmbeddingError, HashFaceEmbeddingBackend, build_face_embedding_backend

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}


class FaceEnrollmentService:
    def __init__(self) -> None:
        self._embedding_backend = self._build_backend()

    @staticmethod
    def _build_backend():
        try:
            return build_face_embedding_backend()
        except FaceEmbeddingError:
            if not settings.recognition_allow_fallback:
                raise
            return HashFaceEmbeddingBackend()

    async def build_enrollments_from_uploads(
        self,
        person: Person,
        files: list[UploadFile],
        *,
        is_primary: bool = False,
    ) -> list[PersonFaceEnrollment]:
        if not files:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one image is required",
            )

        enrollments: list[PersonFaceEnrollment] = []
        for index, upload in enumerate(files):
            contents = await upload.read()
            if not contents:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{upload.filename or 'Image'} is empty",
                )

            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in ALLOWED_IMAGE_EXTENSIONS:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"{upload.filename or 'Image'} is not a supported image format",
                )

            embedding_vector, embedding_model, embedding_metadata = self._extract_embedding(contents)
            relative_path = self._relative_storage_path(person, suffix)
            destination = (settings.storage_root / relative_path).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)

            enrollments.append(
                PersonFaceEnrollment(
                    image_path=relative_path.as_posix(),
                    label=self._label_for_upload(person.full_name, upload.filename, index),
                    embedding_vector=embedding_vector,
                    embedding_model=embedding_model,
                    is_primary=is_primary and index == 0,
                    metadata={
                        "source_filename": upload.filename or "",
                        "content_type": upload.content_type or "application/octet-stream",
                        "uploaded": True,
                        "file_size_bytes": len(contents),
                        **embedding_metadata,
                    },
                )
            )

        return enrollments

    def build_enrollment_from_stored_image(
        self,
        person: Person,
        image_path: str,
        *,
        label: str | None = None,
        is_primary: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> PersonFaceEnrollment:
        source = self.resolve_storage_image_path(image_path)
        suffix = source.suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Captured face image is not a supported image format",
            )

        contents = source.read_bytes()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Captured face image is empty",
            )

        embedding_vector, embedding_model, embedding_metadata = self._extract_embedding(contents)
        relative_path = self._relative_storage_path(person, suffix)
        destination = (settings.storage_root / relative_path).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(contents)

        return PersonFaceEnrollment(
            image_path=relative_path.as_posix(),
            label=label or self._label_for_upload(person.full_name, source.name, 0),
            embedding_vector=embedding_vector,
            embedding_model=embedding_model,
            is_primary=is_primary,
            metadata={
                "uploaded": False,
                "copied_from_path": image_path,
                "file_size_bytes": len(contents),
                **embedding_metadata,
                **(metadata or {}),
            },
        )

    def resolve_face_image(self, person: Person, face_id: str) -> Path:
        for profile in person.face_profiles:
            if profile.get("id") != face_id:
                continue

            image_path = profile.get("image_path")
            if not image_path:
                break

            resolved = (settings.storage_root / image_path).resolve()
            storage_root = settings.storage_root.resolve()
            if resolved != storage_root and storage_root not in resolved.parents:
                break
            return resolved

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Face image not found")

    def resolve_storage_image_path(self, image_path: str) -> Path:
        candidate = (settings.storage_root / image_path).resolve()
        storage_root = settings.storage_root.resolve()
        if candidate != storage_root and storage_root not in candidate.parents:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image path must remain inside storage")
        if not candidate.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Captured face image not found")
        return candidate

    @staticmethod
    def _relative_storage_path(person: Person, suffix: str) -> Path:
        person_segment = "".join(
            char.lower() if char.isalnum() else "-" for char in person.reference_id
        ).strip("-") or str(person.id)
        return Path("faces") / "persons" / person_segment / f"{uuid4()}{suffix}"

    @staticmethod
    def _label_for_upload(full_name: str, filename: str | None, index: int) -> str:
        stem = Path(filename or "").stem.strip()
        return stem or f"{full_name} #{index + 1}"

    def _extract_embedding(self, contents: bytes) -> tuple[list[float], str | None, dict[str, object]]:
        try:
            result = self._embedding_backend.extract_embedding(contents)
            return result.vector, result.model_name, result.metadata
        except FaceEmbeddingError as exc:
            if not settings.recognition_allow_fallback:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail=str(exc),
                ) from exc

        fallback = HashFaceEmbeddingBackend().extract_embedding(contents)
        fallback.metadata = {
            **fallback.metadata,
            "fallback_used": True,
            "requested_backend": settings.recognition_backend,
        }
        return fallback.vector, fallback.model_name, fallback.metadata
