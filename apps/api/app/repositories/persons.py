from __future__ import annotations

import re
from math import sqrt
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.person import Person
from app.models.person_face_embedding import PersonFaceEmbedding
from app.schemas.persons import (
    PersonCreate,
    PersonEmbeddingMatchRequest,
    PersonEmbeddingMatchResponse,
    PersonEmbeddingMatchResult,
    PersonFaceEnrollment,
    PersonUpdate,
)


class PersonRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self) -> list[Person]:
        result = await self.session.scalars(select(Person).order_by(Person.created_at.desc()))
        return list(result)

    async def get(self, person_id: UUID) -> Person | None:
        return await self.session.get(Person, person_id)

    async def get_by_reference_id(self, reference_id: str) -> Person | None:
        return await self.session.scalar(select(Person).where(Person.reference_id == reference_id))

    async def delete(self, person: Person) -> None:
        await self.session.delete(person)
        await self.session.commit()

    async def create(self, payload: PersonCreate) -> Person:
        person = Person(
            full_name=payload.full_name,
            person_type=payload.person_type,
            department=payload.department,
            reference_id=self._resolve_reference_id(payload.reference_id, payload.person_type, payload.full_name),
            title=payload.title,
            is_active=payload.is_active,
            metadata_=payload.metadata,
        )
        self.session.add(person)
        await self.session.commit()
        await self.session.refresh(person)
        return person

    async def update(self, person: Person, payload: PersonUpdate) -> Person:
        updates = payload.model_dump(exclude_unset=True, exclude={"metadata"})
        if "reference_id" in updates:
            updates["reference_id"] = self._resolve_reference_id(
                updates["reference_id"],
                updates.get("person_type", person.person_type),
                updates.get("full_name", person.full_name),
            )
        for key, value in updates.items():
            setattr(person, key, value)
        if payload.metadata is not None:
            person.metadata_ = payload.metadata
        await self.session.commit()
        await self.session.refresh(person)
        return person

    @staticmethod
    def _resolve_reference_id(
        reference_id: str | None, person_type: str, full_name: str
    ) -> str:
        if reference_id and reference_id.strip():
            return reference_id.strip()

        prefix = {
            "employee": "EMP",
            "student": "STU",
            "visitor": "VIS",
            "contractor": "CTR",
            "other": "PRS",
        }.get(person_type, "PRS")
        slug = re.sub(r"[^a-z0-9]+", "-", full_name.lower()).strip("-") or "known-person"
        return f"{prefix}-{slug[:24]}-{uuid4().hex[:6].upper()}"

    async def add_face_profile(self, person: Person, payload: PersonFaceEnrollment) -> Person:
        return await self.add_face_profiles(person, [payload])

    async def add_face_profiles(
        self, person: Person, payloads: list[PersonFaceEnrollment]
    ) -> Person:
        if not payloads:
            return person

        existing_profiles = list(person.face_profiles)
        embedding_records: list[PersonFaceEmbedding] = []
        for payload in payloads:
            captured_at = payload.captured_at or datetime.now(UTC)
            face_profile_id = str(uuid4())
            face_profile = {
                "id": face_profile_id,
                "label": payload.label or person.full_name,
                "image_path": payload.image_path,
                "embedding_vector": payload.embedding_vector,
                "embedding_model": payload.embedding_model,
                "embedding_dimensions": len(payload.embedding_vector),
                "is_primary": payload.is_primary,
                "captured_at": captured_at.isoformat(),
                "metadata": payload.metadata,
            }
            existing_profiles.append(face_profile)
            if payload.embedding_vector:
                embedding_records.append(
                    PersonFaceEmbedding(
                        person_id=person.id,
                        face_profile_id=face_profile_id,
                        label=face_profile["label"],
                        image_path=payload.image_path,
                        embedding_literal=self._serialize_embedding(payload.embedding_vector),
                        embedding_dimensions=len(payload.embedding_vector),
                        embedding_model=payload.embedding_model,
                        is_primary=payload.is_primary,
                        metadata_=payload.metadata,
                    )
                )

        person.face_profiles = existing_profiles
        person.face_image_count = len(person.face_profiles)
        person.embedding_count = sum(
            1 for profile in person.face_profiles if profile.get("embedding_vector")
        )
        self.session.add_all(embedding_records)
        await self.session.commit()
        await self.session.refresh(person)
        return person

    async def match_embeddings(
        self, payload: PersonEmbeddingMatchRequest
    ) -> PersonEmbeddingMatchResponse:
        if await self._pgvector_available():
            return await self._match_embeddings_with_pgvector(payload)
        return await self._match_embeddings_in_python(payload)

    async def record_recognition(self, person: Person, occurred_at: datetime) -> Person:
        person.visit_count += 1
        person.recognition_count += 1
        person.last_seen_at = occurred_at
        person.last_recognized_at = occurred_at
        await self.session.commit()
        await self.session.refresh(person)
        return person

    async def _match_embeddings_with_pgvector(
        self, payload: PersonEmbeddingMatchRequest
    ) -> PersonEmbeddingMatchResponse:
        query_vector = self._serialize_embedding(payload.embedding_vector)
        sql = text(
            """
            SELECT
                p.id AS person_id,
                p.full_name,
                p.reference_id,
                p.person_type,
                e.face_profile_id,
                e.image_path,
                e.embedding_model,
                1 - (CAST(e.embedding_literal AS vector) <=> CAST(:query_vector AS vector)) AS similarity
            FROM person_face_embeddings e
            JOIN persons p ON p.id = e.person_id
            WHERE p.is_active = true
              AND e.embedding_dimensions = :dimensions
              AND (1 - (CAST(e.embedding_literal AS vector) <=> CAST(:query_vector AS vector))) >= :min_similarity
            ORDER BY CAST(e.embedding_literal AS vector) <=> CAST(:query_vector AS vector)
            LIMIT :top_k
            """
        )
        rows = (
            await self.session.execute(
                sql,
                {
                    "query_vector": query_vector,
                    "dimensions": len(payload.embedding_vector),
                    "min_similarity": payload.min_similarity,
                    "top_k": payload.top_k,
                },
            )
        ).mappings()

        results = [
            PersonEmbeddingMatchResult(
                person_id=row["person_id"],
                full_name=row["full_name"],
                reference_id=row["reference_id"],
                person_type=row["person_type"],
                face_profile_id=row["face_profile_id"],
                image_path=row["image_path"],
                embedding_model=row["embedding_model"],
                similarity=round(float(row["similarity"]), 4),
            )
            for row in rows
        ]
        return PersonEmbeddingMatchResponse(
            query_dimensions=len(payload.embedding_vector),
            matched_count=len(results),
            backend="pgvector",
            results=results,
        )

    async def _match_embeddings_in_python(
        self, payload: PersonEmbeddingMatchRequest
    ) -> PersonEmbeddingMatchResponse:
        query = (
            select(PersonFaceEmbedding, Person)
            .join(Person, Person.id == PersonFaceEmbedding.person_id)
            .where(Person.is_active.is_(True))
            .where(PersonFaceEmbedding.embedding_dimensions == len(payload.embedding_vector))
        )
        rows = (await self.session.execute(query)).all()

        matches: list[PersonEmbeddingMatchResult] = []
        for embedding, person in rows:
            similarity = self._cosine_similarity(
                payload.embedding_vector,
                self._parse_embedding_literal(embedding.embedding_literal),
            )
            if similarity < payload.min_similarity:
                continue
            matches.append(
                PersonEmbeddingMatchResult(
                    person_id=person.id,
                    full_name=person.full_name,
                    reference_id=person.reference_id,
                    person_type=person.person_type,
                    face_profile_id=embedding.face_profile_id,
                    image_path=embedding.image_path,
                    embedding_model=embedding.embedding_model,
                    similarity=round(similarity, 4),
                )
            )

        matches.sort(key=lambda item: item.similarity, reverse=True)
        limited = matches[: payload.top_k]
        return PersonEmbeddingMatchResponse(
            query_dimensions=len(payload.embedding_vector),
            matched_count=len(limited),
            backend="python-fallback",
            results=limited,
        )

    async def _pgvector_available(self) -> bool:
        bind = self.session.get_bind()
        if bind is None or bind.dialect.name != "postgresql":
            return False
        result = await self.session.execute(
            text("SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')")
        )
        return bool(result.scalar())

    @staticmethod
    def _serialize_embedding(values: list[float]) -> str:
        return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"

    @staticmethod
    def _parse_embedding_literal(value: str) -> list[float]:
        normalized = value.strip().strip("[]")
        if not normalized:
            return []
        return [float(item) for item in normalized.split(",") if item.strip()]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return -1.0
        size = min(len(left), len(right))
        numerator = sum(left[index] * right[index] for index in range(size))
        left_magnitude = sqrt(sum(value * value for value in left[:size]))
        right_magnitude = sqrt(sum(value * value for value in right[:size]))
        if left_magnitude == 0 or right_magnitude == 0:
            return -1.0
        return numerator / (left_magnitude * right_magnitude)
