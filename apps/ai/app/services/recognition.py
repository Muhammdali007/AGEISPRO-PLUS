import base64
import hashlib
import math
from io import BytesIO
from typing import Iterable

from app.core.config import settings
from app.schemas.inference import (
    FaceRegion,
    InferenceBox,
    InferenceRecognition,
    InferenceRequest,
    KnownPersonProfile,
)
from app.services.face_embeddings import (
    FaceEmbeddingError,
    HashFaceEmbeddingBackend,
    build_face_embedding_backend,
)


RecognitionCacheEntry = tuple[str, InferenceRecognition]


class FaceRecognitionService:
    def __init__(self) -> None:
        self._track_cache: dict[str, RecognitionCacheEntry] = {}
        self._embedding_backend = self._build_backend()

    @staticmethod
    def _build_backend():
        try:
            return build_face_embedding_backend()
        except FaceEmbeddingError:
            if not settings.recognition_allow_fallback:
                raise
            return HashFaceEmbeddingBackend()

    def enrich_detection(
        self, payload: InferenceRequest, detection: InferenceBox
    ) -> tuple[FaceRegion, InferenceRecognition]:
        cache_key = (
            f"{payload.camera_id}:{detection.track_id}"
            if detection.track_id
            else f"{payload.camera_id}:{payload.frame_reference}:{detection.label}"
        )
        face_region = self._extract_face_region(detection)
        candidates_signature = self._known_persons_signature(payload.known_persons)
        # Real camera frames must be compared again. Tracker IDs can be recycled
        # when one person leaves and another enters the same stream.
        cached = self._track_cache.get(cache_key) if not payload.frame_content_base64 else None
        if cached is not None and cached[0] == candidates_signature:
            reused = cached[1].model_copy(deep=True)
            reused.deduplicated = True
            return face_region, reused

        try:
            embedding, embedding_model, embedding_metadata = self._extract_embedding(
                payload, detection, face_region
            )
        except FaceEmbeddingError as exc:
            return face_region, InferenceRecognition(
                status="unknown",
                identity_label="Unknown visitor",
                embedding_model=None,
                deduplicated=False,
                face_region=face_region,
                metadata={
                    "backend": settings.recognition_backend,
                    "recognition_error": str(exc),
                },
            )
        face_region = self._refine_face_region(face_region, detection, embedding_metadata)
        recognition = self._match_known_person(face_region, embedding, payload.known_persons)
        recognition.embedding_model = embedding_model
        recognition.metadata = {
            **recognition.metadata,
            **embedding_metadata,
        }
        self._track_cache[cache_key] = (candidates_signature, recognition.model_copy(deep=True))
        return face_region, recognition

    @staticmethod
    def _extract_face_region(detection: InferenceBox) -> FaceRegion:
        width = detection.x2 - detection.x1
        height = detection.y2 - detection.y1
        face_width = max(width * 0.42, 24.0)
        face_height = max(height * 0.32, 24.0)
        face_x1 = detection.x1 + ((width - face_width) / 2)
        face_y1 = detection.y1 + max(height * 0.08, 6.0)
        return FaceRegion(
            x1=round(face_x1, 2),
            y1=round(face_y1, 2),
            x2=round(face_x1 + face_width, 2),
            y2=round(face_y1 + face_height, 2),
            confidence=round(min(detection.confidence + 0.03, 0.99), 2),
            image_path=f"storage/faces/{detection.track_id or 'frame'}.jpg",
        )

    def _match_known_person(
        self,
        face_region: FaceRegion,
        embedding: list[float],
        known_persons: list[KnownPersonProfile],
    ) -> InferenceRecognition:
        ranked_matches: list[tuple[float, KnownPersonProfile]] = []
        for person in known_persons:
            score = max(
                (
                    self._cosine_similarity(embedding, profile_embedding)
                    for profile_embedding in self._profile_embeddings(person)
                    if len(profile_embedding) == len(embedding)
                ),
                default=-1.0,
            )
            if score >= 0:
                ranked_matches.append((score, person))

        ranked_matches.sort(key=lambda match: match[0], reverse=True)
        best_score, best_person = ranked_matches[0] if ranked_matches else (-1.0, None)
        second_best_score = ranked_matches[1][0] if len(ranked_matches) > 1 else -1.0
        match_margin = best_score - second_best_score if second_best_score >= 0 else 1.0

        if (
            best_person
            and best_score >= settings.recognition_match_threshold
            and match_margin >= settings.recognition_min_margin
        ):
            return InferenceRecognition(
                status="known",
                identity_id=best_person.person_id,
                identity_label=best_person.full_name,
                match_confidence=round(min(best_score, 0.99), 2),
                embedding_model=None,
                deduplicated=False,
                face_region=face_region,
                metadata={
                    "backend": settings.recognition_backend,
                    "person_type": best_person.person_type,
                    "department": best_person.department,
                    "reference_id": best_person.reference_id,
                    "title": best_person.title,
                    "person_metadata": best_person.metadata,
                    "match_margin": round(match_margin, 4),
                },
            )

        return InferenceRecognition(
            status="unknown",
            identity_label="Unknown visitor",
            match_confidence=round(max(best_score, 0.0), 2) if best_score >= 0 else None,
            embedding_model=None,
            deduplicated=False,
            face_region=face_region,
            metadata={
                "backend": settings.recognition_backend,
                "match_margin": round(match_margin, 4) if best_score >= 0 else None,
            },
        )

    def _profile_embeddings(self, person: KnownPersonProfile) -> Iterable[list[float]]:
        profiles = person.face_profiles or []
        for profile in profiles:
            if profile.embedding_vector:
                yield profile.embedding_vector

    @staticmethod
    def _known_persons_signature(known_persons: list[KnownPersonProfile]) -> str:
        normalized: list[tuple[str, str, tuple[tuple[str, str, tuple[float, ...]], ...]]] = []
        for person in known_persons:
            profiles = tuple(
                sorted(
                    (
                        profile.face_id or "",
                        profile.image_path or "",
                        tuple(round(value, 6) for value in profile.embedding_vector),
                    )
                    for profile in person.face_profiles
                )
            )
            normalized.append((str(person.person_id), person.full_name, profiles))

        digest = hashlib.sha256(repr(sorted(normalized)).encode("utf-8")).hexdigest()
        return digest

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return -1.0
        size = min(len(left), len(right))
        numerator = sum(left[index] * right[index] for index in range(size))
        left_magnitude = math.sqrt(sum(value * value for value in left[:size]))
        right_magnitude = math.sqrt(sum(value * value for value in right[:size]))
        if left_magnitude == 0 or right_magnitude == 0:
            return -1.0
        return numerator / (left_magnitude * right_magnitude)

    def _extract_embedding(
        self,
        payload: InferenceRequest,
        detection: InferenceBox,
        face_region: FaceRegion,
    ) -> tuple[list[float], str | None, dict[str, object]]:
        if payload.frame_content_base64:
            try:
                person_region = FaceRegion(
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    confidence=detection.confidence,
                )
                image_bytes = self._crop_face_image(
                    payload.frame_content_base64, payload.frame_content_type, person_region
                )
                result = self._embedding_backend.extract_embedding(image_bytes)
                return (
                    result.vector,
                    result.model_name,
                    {
                        **result.metadata,
                        "crop_origin_x": detection.x1,
                        "crop_origin_y": detection.y1,
                    },
                )
            except (FaceEmbeddingError, OSError, ValueError):
                if not settings.recognition_allow_fallback:
                    raise

        fallback_bytes = (
            f"{payload.camera_id}:{payload.frame_reference}:{detection.track_id or detection.label}"
        ).encode("utf-8")
        fallback = HashFaceEmbeddingBackend().extract_embedding(fallback_bytes)
        fallback.metadata = {
            **fallback.metadata,
            "fallback_used": True,
            "requested_backend": settings.recognition_backend,
        }
        return fallback.vector, fallback.model_name, fallback.metadata

    @staticmethod
    def _refine_face_region(
        face_region: FaceRegion,
        detection: InferenceBox,
        embedding_metadata: dict[str, object],
    ) -> FaceRegion:
        bbox = embedding_metadata.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return face_region

        try:
            crop_x1, crop_y1, crop_x2, crop_y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return face_region

        origin_x = float(embedding_metadata.get("crop_origin_x", face_region.x1))
        origin_y = float(embedding_metadata.get("crop_origin_y", face_region.y1))
        refined_x1 = max(detection.x1, origin_x + crop_x1)
        refined_y1 = max(detection.y1, origin_y + crop_y1)
        refined_x2 = min(detection.x2, origin_x + crop_x2)
        refined_y2 = min(detection.y2, origin_y + crop_y2)

        if refined_x2 <= refined_x1 or refined_y2 <= refined_y1:
            return face_region

        return FaceRegion(
            x1=round(refined_x1, 2),
            y1=round(refined_y1, 2),
            x2=round(refined_x2, 2),
            y2=round(refined_y2, 2),
            confidence=face_region.confidence,
            image_path=face_region.image_path,
        )

    @staticmethod
    def _crop_face_image(
        frame_content_base64: str,
        frame_content_type: str | None,
        face_region: FaceRegion,
    ) -> bytes:
        from PIL import Image

        frame_bytes = base64.b64decode(frame_content_base64)
        image = Image.open(BytesIO(frame_bytes)).convert("RGB")
        left = max(int(face_region.x1), 0)
        top = max(int(face_region.y1), 0)
        right = min(int(face_region.x2), image.width)
        bottom = min(int(face_region.y2), image.height)
        if right <= left or bottom <= top:
            raise ValueError("Face crop is outside the frame bounds.")

        cropped = image.crop((left, top, right, bottom))
        buffer = BytesIO()
        image_format = "PNG" if (frame_content_type or "").lower() == "image/png" else "JPEG"
        cropped.save(buffer, format=image_format)
        return buffer.getvalue()
