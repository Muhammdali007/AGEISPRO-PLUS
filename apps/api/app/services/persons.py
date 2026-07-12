from __future__ import annotations

from pathlib import Path
from datetime import datetime
from uuid import UUID

from fastapi import HTTPException, UploadFile, status

from app.models.person import Person
from app.repositories.persons import PersonRepository
from app.schemas.persons import (
    PersonCreate,
    PersonEmbeddingMatchRequest,
    PersonEmbeddingMatchResponse,
    PersonFaceEnrollment,
    PersonUpdate,
)
from app.services.face_enrollment import FaceEnrollmentService


class PersonService:
    def __init__(
        self,
        repository: PersonRepository,
        face_enrollment: FaceEnrollmentService | None = None,
    ) -> None:
        self.repository = repository
        self._face_enrollment = face_enrollment

    @property
    def face_enrollment(self) -> FaceEnrollmentService:
        if self._face_enrollment is None:
            self._face_enrollment = FaceEnrollmentService()
        return self._face_enrollment

    async def list(self) -> list[Person]:
        return await self.repository.list()

    async def get_or_404(self, person_id: UUID) -> Person:
        person = await self.repository.get(person_id)
        if not person:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Person not found")
        return person

    async def create(self, payload: PersonCreate) -> Person:
        return await self.repository.create(payload)

    async def delete(self, person_id: UUID) -> None:
        person = await self.get_or_404(person_id)
        await self.repository.delete(person)

    async def match_embeddings(
        self, payload: PersonEmbeddingMatchRequest
    ) -> PersonEmbeddingMatchResponse:
        return await self.repository.match_embeddings(payload)

    async def update(self, person_id: UUID, payload: PersonUpdate) -> Person:
        person = await self.get_or_404(person_id)
        return await self.repository.update(person, payload)

    async def enroll_face(self, person_id: UUID, payload: PersonFaceEnrollment) -> Person:
        person = await self.get_or_404(person_id)
        return await self.repository.add_face_profile(person, payload)

    async def enroll_face_images(
        self, person_id: UUID, files: list[UploadFile], *, is_primary: bool = False
    ) -> Person:
        person = await self.get_or_404(person_id)
        enrollments = await self.face_enrollment.build_enrollments_from_uploads(
            person,
            files,
            is_primary=is_primary,
        )
        return await self.repository.add_face_profiles(person, enrollments)

    async def enroll_face_image_from_storage(
        self,
        person_id: UUID,
        image_path: str,
        *,
        label: str | None = None,
        is_primary: bool = False,
        metadata: dict[str, object] | None = None,
    ) -> Person:
        person = await self.get_or_404(person_id)
        enrollment = self.face_enrollment.build_enrollment_from_stored_image(
            person,
            image_path,
            label=label,
            is_primary=is_primary,
            metadata=metadata,
        )
        return await self.repository.add_face_profile(person, enrollment)

    async def get_face_image(self, person_id: UUID, face_id: str) -> Path:
        person = await self.get_or_404(person_id)
        return self.face_enrollment.resolve_face_image(person, face_id)

    async def record_recognition(self, person_id: UUID, occurred_at: datetime) -> Person | None:
        person = await self.repository.get(person_id)
        if not person:
            return None
        return await self.repository.record_recognition(person, occurred_at)
