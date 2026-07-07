from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class KnownPersonFaceProfile(BaseModel):
    face_id: str | None = Field(default=None, max_length=120)
    image_path: str | None = None
    embedding_vector: list[float] = Field(default_factory=list)
    embedding_model: str | None = Field(default=None, max_length=120)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnownPersonProfile(BaseModel):
    person_id: UUID
    full_name: str = Field(min_length=1, max_length=160)
    person_type: str | None = Field(default=None, max_length=32)
    department: str | None = Field(default=None, max_length=120)
    reference_id: str | None = Field(default=None, max_length=64)
    title: str | None = Field(default=None, max_length=120)
    face_profiles: list[KnownPersonFaceProfile] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FaceRegion(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    image_path: str | None = None


class InlineEvidencePayload(BaseModel):
    content_base64: str = Field(min_length=1)
    content_type: str | None = Field(default=None, max_length=120)


class InferenceRecognition(BaseModel):
    status: str = Field(pattern="^(known|unknown)$")
    identity_id: UUID | None = None
    identity_label: str | None = Field(default=None, max_length=160)
    match_confidence: float | None = Field(default=None, ge=0, le=1)
    embedding_model: str | None = Field(default=None, max_length=120)
    deduplicated: bool = False
    face_region: FaceRegion | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceBox(BaseModel):
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    label: str = Field(min_length=1, max_length=64)
    track_id: str | None = Field(default=None, max_length=64)
    face_region: FaceRegion | None = None
    recognition: InferenceRecognition | None = None
    face_image_evidence: InlineEvidencePayload | None = None


class InferenceRequest(BaseModel):
    camera_id: UUID
    frame_reference: str = Field(min_length=1)
    source_type: str = Field(min_length=1, max_length=32)
    frame_content_base64: str | None = None
    frame_content_type: str | None = Field(default=None, max_length=120)
    include_evidence: bool = False
    requested_detectors: list[str] = Field(default_factory=lambda: ["weapon", "fire", "smoke", "person"])
    recognition_enabled: bool = False
    known_persons: list[KnownPersonProfile] = Field(default_factory=list)
    occurrence_hint: str | None = Field(default=None, max_length=64)
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceResult(BaseModel):
    camera_id: UUID
    model_name: str
    model_version: str
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    inference_fps: float
    source_fps: float | None = None
    detections: list[InferenceBox] = Field(default_factory=list)
    snapshot_evidence: InlineEvidencePayload | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class InferenceEventDispatchResult(BaseModel):
    delivered: bool
    callback_url: str | None = None
    status_code: int | None = None
    message: str
