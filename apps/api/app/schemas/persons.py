from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

PersonType = Literal["employee", "student", "visitor", "contractor", "other"]


class PersonFaceProfileRead(BaseModel):
    id: str
    label: str
    image_path: str
    embedding_vector: list[float] = Field(default_factory=list)
    embedding_model: str | None = None
    embedding_dimensions: int = 0
    is_primary: bool = False
    captured_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonRead(BaseModel):
    id: UUID
    full_name: str
    person_type: PersonType
    department: str | None
    reference_id: str
    title: str | None
    is_active: bool
    face_profiles: list[PersonFaceProfileRead] = Field(default_factory=list)
    face_image_count: int
    embedding_count: int
    visit_count: int
    recognition_count: int
    last_seen_at: datetime | None
    last_recognized_at: datetime | None
    metadata: dict[str, Any] = Field(alias="metadata_")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, serialize_by_alias=False)


class PersonCreate(BaseModel):
    full_name: str = Field(min_length=1, max_length=160)
    person_type: PersonType = "visitor"
    department: str | None = Field(default=None, max_length=120)
    reference_id: str | None = Field(default=None, min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=120)
    is_active: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=160)
    person_type: PersonType | None = None
    department: str | None = Field(default=None, max_length=120)
    reference_id: str | None = Field(default=None, min_length=1, max_length=64)
    title: str | None = Field(default=None, max_length=120)
    is_active: bool | None = None
    metadata: dict[str, Any] | None = None


class PersonFaceEnrollment(BaseModel):
    image_path: str = Field(min_length=1)
    label: str | None = Field(default=None, max_length=160)
    embedding_vector: list[float] = Field(default_factory=list)
    embedding_model: str | None = Field(default=None, max_length=120)
    is_primary: bool = False
    captured_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PersonEmbeddingMatchRequest(BaseModel):
    embedding_vector: list[float] = Field(min_length=1)
    embedding_model: str | None = Field(default=None, max_length=120)
    top_k: int = Field(default=5, ge=1, le=25)
    min_similarity: float = Field(default=0.82, ge=0, le=1)


class PersonEmbeddingMatchResult(BaseModel):
    person_id: UUID
    full_name: str
    reference_id: str
    person_type: PersonType
    face_profile_id: str
    image_path: str
    embedding_model: str | None = None
    similarity: float = Field(ge=0, le=1)


class PersonEmbeddingMatchResponse(BaseModel):
    query_dimensions: int
    matched_count: int
    backend: str
    results: list[PersonEmbeddingMatchResult] = Field(default_factory=list)
