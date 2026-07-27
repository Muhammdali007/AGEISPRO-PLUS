import base64
import hashlib
import math
from dataclasses import dataclass
from io import BytesIO
from time import monotonic
from typing import Iterable

from app.core.config import PROJECT_ROOT, settings
from app.schemas.inference import (
    FaceRegion,
    InferenceBox,
    InferenceRecognition,
    InferenceRequest,
    KnownPersonProfile,
)
from app.services.face_embeddings import (
    FaceEmbeddingError,
    FaceEmbeddingResult,
    HashFaceEmbeddingBackend,
    build_face_embedding_backend,
)


RecognitionCacheEntry = tuple[str, InferenceRecognition]


@dataclass(slots=True)
class LiveIdentityState:
    detection: InferenceBox
    recognition: InferenceRecognition
    confirmed_at: float
    unknown_frames: int = 0


@dataclass(slots=True)
class FrameIdentityState:
    detection: InferenceBox
    face_region: FaceRegion
    recognition: InferenceRecognition
    candidates_signature: str
    refreshed_at: float


class FaceRecognitionService:
    def __init__(self) -> None:
        self._track_cache: dict[str, RecognitionCacheEntry] = {}
        self._live_identity_cache: dict[str, list[LiveIdentityState]] = {}
        self._frame_identity_cache: dict[str, list[FrameIdentityState]] = {}
        self._embedding_backend = self._build_backend()

    def warmup(self) -> None:
        """Initialize the face detector before the first live camera request."""
        if not hasattr(self._embedding_backend, "extract_embeddings"):
            return

        from PIL import Image

        warmup_bytes: bytes | None = None
        faces_root = PROJECT_ROOT / "storage" / "faces"
        if faces_root.is_dir():
            for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                candidate = next(faces_root.rglob(pattern), None)
                if candidate is not None:
                    try:
                        warmup_bytes = candidate.read_bytes()
                    except OSError:
                        pass
                    if warmup_bytes:
                        break

        if warmup_bytes is None:
            width, height = settings.recognition_insightface_det_size
            buffer = BytesIO()
            Image.new("RGB", (width, height), color=(0, 0, 0)).save(buffer, format="JPEG")
            warmup_bytes = buffer.getvalue()
        try:
            # Prefer an enrolled sample when available so both detection and
            # recognition graphs are initialized. A blank fallback still warms
            # the detector on a fresh installation with no enrolled people.
            self._embedding_backend.extract_embeddings(warmup_bytes)
        except (FaceEmbeddingError, OSError):
            pass

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
        recognition = self._stabilize_live_recognition(payload, detection, recognition)
        self._track_cache[cache_key] = (candidates_signature, recognition.model_copy(deep=True))
        return face_region, recognition

    def enrich_detections(
        self,
        payload: InferenceRequest,
        detections: list[InferenceBox],
    ) -> list[InferenceBox]:
        """Recognize all people with one full-frame face-analysis invocation.

        InsightFace detection is substantially more expensive than matching a
        512-value embedding. Running it once per frame also prevents one face
        from being assigned independently to multiple overlapping person boxes.
        """
        person_indices = [
            index for index, detection in enumerate(detections) if detection.label == "person"
        ]

        if (
            not payload.frame_content_base64
            or not hasattr(self._embedding_backend, "extract_embeddings")
        ):
            return [self._enrich_single_box(payload, detection) for detection in detections]

        should_find_unboxed_faces = self._should_run_face_only_detection(payload)
        if not person_indices and not should_find_unboxed_faces:
            return detections

        now = monotonic()
        camera_key = str(payload.camera_id)
        candidates_signature = self._known_persons_signature(payload.known_persons)
        previous = [
            state
            for state in self._frame_identity_cache.get(camera_key, [])
            if (
                state.candidates_signature == candidates_signature
                and now - state.refreshed_at <= self._refresh_window(state)
            )
        ]
        resolved: dict[int, tuple[FaceRegion, InferenceRecognition]] = {}
        next_states: list[FrameIdentityState] = []
        used_previous: set[int] = set()

        if max(
            settings.recognition_refresh_seconds,
            settings.recognition_known_refresh_seconds,
        ) > 0:
            for detection_index in person_indices:
                detection = detections[detection_index]
                match_index = self._best_spatial_state_index(detection, previous, used_previous)
                if match_index is None:
                    continue
                used_previous.add(match_index)
                cached = previous[match_index]
                recognition = cached.recognition.model_copy(deep=True)
                recognition.deduplicated = True
                recognition.metadata = {
                    **recognition.metadata,
                    "frame_recognition_cache_hit": True,
                    "recognition_age_ms": round((now - cached.refreshed_at) * 1000, 1),
                }
                face_region = self._project_face_region(
                    cached.face_region,
                    cached.detection,
                    detection,
                )
                recognition.face_region = face_region
                resolved[detection_index] = (face_region, recognition)
                next_states.append(
                    FrameIdentityState(
                        detection=detection.model_copy(deep=True),
                        face_region=face_region.model_copy(deep=True),
                        recognition=recognition.model_copy(deep=True),
                        candidates_signature=candidates_signature,
                        refreshed_at=cached.refreshed_at,
                    )
                )

        unresolved = [index for index in person_indices if index not in resolved]
        face_results: list[FaceEmbeddingResult] = []
        recognition_error: str | None = None
        if unresolved or (should_find_unboxed_faces and not person_indices):
            try:
                frame_bytes = base64.b64decode(payload.frame_content_base64, validate=True)
                face_results = self._embedding_backend.extract_embeddings(frame_bytes)
            except (FaceEmbeddingError, OSError, ValueError) as exc:
                recognition_error = str(exc)

        used_faces: set[int] = set()
        for detection_index in unresolved:
            detection = detections[detection_index]
            face_index = self._best_face_index(detection, face_results, used_faces)
            if face_index is None:
                face_region = self._extract_face_region(detection)
                recognition = self._unknown_recognition(
                    face_region,
                    recognition_error or "No face was associated with this person detection.",
                )
            else:
                used_faces.add(face_index)
                face_result = face_results[face_index]
                face_region = self._face_region_from_frame_result(detection, face_result)
                recognition = self._match_known_person(
                    face_region,
                    face_result.vector,
                    payload.known_persons,
                )
                recognition.embedding_model = face_result.model_name
                recognition.metadata = {
                    **recognition.metadata,
                    **face_result.metadata,
                    "frame_recognition_cache_hit": False,
                    "frame_face_analysis_count": 1,
                }

            recognition = self._stabilize_live_recognition(payload, detection, recognition)
            resolved[detection_index] = (face_region, recognition)
            next_states.append(
                FrameIdentityState(
                    detection=detection.model_copy(deep=True),
                    face_region=face_region.model_copy(deep=True),
                    recognition=recognition.model_copy(deep=True),
                    candidates_signature=candidates_signature,
                    # Cache age starts after the expensive face analysis. On a
                    # cold CPU call, measuring from frame arrival can consume
                    # the entire refresh window before this response is sent.
                    refreshed_at=monotonic(),
                )
            )

        face_only_detections: list[InferenceBox] = []
        if should_find_unboxed_faces and face_results:
            for face_index, face_result in enumerate(face_results):
                if face_index in used_faces:
                    continue
                face_region = self._face_region_from_result(face_result)
                if face_region is None:
                    continue
                detection = InferenceBox(
                    x1=face_region.x1,
                    y1=face_region.y1,
                    x2=face_region.x2,
                    y2=face_region.y2,
                    confidence=face_region.confidence,
                    label="person",
                    object_label="face",
                    track_id=self._face_only_track_id(payload, face_result, face_index),
                    face_region=face_region,
                )
                recognition = self._match_known_person(
                    face_region,
                    face_result.vector,
                    payload.known_persons,
                )
                recognition.embedding_model = face_result.model_name
                recognition.metadata = {
                    **recognition.metadata,
                    **face_result.metadata,
                    "frame_recognition_cache_hit": False,
                    "frame_face_analysis_count": 1,
                    "face_only_detection": True,
                }
                recognition = self._stabilize_live_recognition(payload, detection, recognition)
                enriched = self._build_enriched_box(detection, face_region, recognition)
                face_only_detections.append(enriched)
                next_states.append(
                    FrameIdentityState(
                        detection=enriched.model_copy(deep=True),
                        face_region=face_region.model_copy(deep=True),
                        recognition=recognition.model_copy(deep=True),
                        candidates_signature=candidates_signature,
                        refreshed_at=monotonic(),
                    )
                )

        current_track_ids = {
            state.detection.track_id for state in next_states if state.detection.track_id
        }
        next_states.extend(
            state
            for index, state in enumerate(previous)
            if index not in used_previous
            and (
                not state.detection.track_id
                or state.detection.track_id not in current_track_ids
            )
        )
        self._frame_identity_cache[camera_key] = next_states
        enriched_detections = [
            self._build_enriched_box(detection, *resolved[index])
            if index in resolved
            else detection
            for index, detection in enumerate(detections)
        ]
        return enriched_detections + face_only_detections

    def _enrich_single_box(
        self,
        payload: InferenceRequest,
        detection: InferenceBox,
    ) -> InferenceBox:
        if detection.label != "person":
            return detection
        face_region, recognition = self.enrich_detection(payload, detection)
        return self._build_enriched_box(detection, face_region, recognition)

    @staticmethod
    def _build_enriched_box(
        detection: InferenceBox,
        face_region: FaceRegion,
        recognition: InferenceRecognition,
    ) -> InferenceBox:
        return InferenceBox(
            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,
            confidence=detection.confidence,
            label="known_person" if recognition.status == "known" else "unknown_person",
            object_label=detection.object_label,
            track_id=detection.track_id,
            face_region=face_region,
            recognition=recognition,
        )

    @staticmethod
    def _unknown_recognition(
        face_region: FaceRegion,
        error_message: str,
    ) -> InferenceRecognition:
        return InferenceRecognition(
            status="unknown",
            identity_label="Unknown visitor",
            embedding_model=None,
            deduplicated=False,
            face_region=face_region,
            metadata={
                "backend": settings.recognition_backend,
                "recognition_error": error_message,
            },
        )

    def _best_spatial_state_index(
        self,
        detection: InferenceBox,
        states: list[FrameIdentityState],
        used: set[int],
    ) -> int | None:
        if detection.track_id:
            for index, state in enumerate(states):
                if index not in used and state.detection.track_id == detection.track_id:
                    return index
        matches = [
            (self._intersection_over_union(state.detection, detection), index)
            for index, state in enumerate(states)
            if index not in used
        ]
        score, index = max(matches, default=(0.0, -1))
        return index if score >= settings.recognition_cache_iou_threshold else None

    @staticmethod
    def _refresh_window(state: FrameIdentityState) -> float:
        if state.recognition.status == "known":
            return max(
                settings.recognition_refresh_seconds,
                settings.recognition_known_refresh_seconds,
            )
        return settings.recognition_refresh_seconds

    @staticmethod
    def _best_face_index(
        detection: InferenceBox,
        faces: list[FaceEmbeddingResult],
        used: set[int],
    ) -> int | None:
        candidates: list[tuple[float, int]] = []
        for index, face in enumerate(faces):
            if index in used:
                continue
            bbox = face.metadata.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            if not (
                detection.x1 <= center_x <= detection.x2
                and detection.y1 <= center_y <= detection.y2
            ):
                continue
            face_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            det_score = float(face.metadata.get("det_score", 0.0))
            if det_score < settings.recognition_min_face_detection_score:
                continue
            minimum_side = min(max(0.0, x2 - x1), max(0.0, y2 - y1))
            if minimum_side < settings.recognition_min_face_size:
                continue
            candidates.append((det_score + min(face_area / 100_000, 0.2), index))
        return max(candidates, default=(0.0, -1))[1] if candidates else None

    @staticmethod
    def _face_region_from_frame_result(
        detection: InferenceBox,
        result: FaceEmbeddingResult,
    ) -> FaceRegion:
        bbox = result.metadata.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return FaceRecognitionService._extract_face_region(detection)
        x1, y1, x2, y2 = [float(value) for value in bbox]
        return FaceRegion(
            x1=round(max(detection.x1, x1), 2),
            y1=round(max(detection.y1, y1), 2),
            x2=round(min(detection.x2, x2), 2),
            y2=round(min(detection.y2, y2), 2),
            confidence=round(float(result.metadata.get("det_score", detection.confidence)), 2),
            image_path=f"storage/faces/{detection.track_id or 'frame'}.jpg",
        )

    @staticmethod
    def _face_region_from_result(result: FaceEmbeddingResult) -> FaceRegion | None:
        bbox = result.metadata.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return None

        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
            confidence = float(result.metadata.get("det_score", 0.0))
        except (TypeError, ValueError):
            return None

        minimum_side = min(max(0.0, x2 - x1), max(0.0, y2 - y1))
        if (
            confidence < settings.recognition_min_face_detection_score
            or minimum_side < settings.recognition_min_face_size
            or x2 <= x1
            or y2 <= y1
        ):
            return None

        return FaceRegion(
            x1=round(x1, 2),
            y1=round(y1, 2),
            x2=round(x2, 2),
            y2=round(y2, 2),
            confidence=round(min(confidence, 0.99), 2),
            image_path="storage/faces/frame.jpg",
        )

    @staticmethod
    def _should_run_face_only_detection(payload: InferenceRequest) -> bool:
        requested = {detector.strip().lower() for detector in payload.requested_detectors}
        return bool(requested & {"person", "known_person", "unknown_person", "face"})

    @staticmethod
    def _face_only_track_id(
        payload: InferenceRequest,
        result: FaceEmbeddingResult,
        face_index: int,
    ) -> str:
        bbox = result.metadata.get("bbox")
        bbox_key = (
            ",".join(str(round(float(value), 1)) for value in bbox)
            if isinstance(bbox, list)
            else ""
        )
        digest = hashlib.sha1(
            f"{payload.camera_id}:{payload.frame_reference}:{face_index}:{bbox_key}".encode("utf-8")
        ).hexdigest()
        return f"fa-{digest[:10]}"

    @staticmethod
    def _project_face_region(
        face_region: FaceRegion,
        previous_detection: InferenceBox,
        current_detection: InferenceBox,
    ) -> FaceRegion:
        previous_width = max(previous_detection.x2 - previous_detection.x1, 1.0)
        previous_height = max(previous_detection.y2 - previous_detection.y1, 1.0)
        current_width = max(current_detection.x2 - current_detection.x1, 1.0)
        current_height = max(current_detection.y2 - current_detection.y1, 1.0)

        def map_x(value: float) -> float:
            return current_detection.x1 + (
                (value - previous_detection.x1) / previous_width
            ) * current_width

        def map_y(value: float) -> float:
            return current_detection.y1 + (
                (value - previous_detection.y1) / previous_height
            ) * current_height

        return FaceRegion(
            x1=round(max(current_detection.x1, map_x(face_region.x1)), 2),
            y1=round(max(current_detection.y1, map_y(face_region.y1)), 2),
            x2=round(min(current_detection.x2, map_x(face_region.x2)), 2),
            y2=round(min(current_detection.y2, map_y(face_region.y2)), 2),
            confidence=face_region.confidence,
            image_path=face_region.image_path,
        )

    def _stabilize_live_recognition(
        self,
        payload: InferenceRequest,
        detection: InferenceBox,
        recognition: InferenceRecognition,
    ) -> InferenceRecognition:
        if payload.occurrence_hint != "dashboard_live_scan":
            return recognition

        now = monotonic()
        camera_key = str(payload.camera_id)
        previous = [
            state
            for state in self._live_identity_cache.get(camera_key, [])
            if now - state.confirmed_at <= settings.recognition_live_hold_seconds
        ]
        match = next(
            (
                state
                for state in previous
                if detection.track_id and state.detection.track_id == detection.track_id
            ),
            None,
        )
        if match is None:
            match = max(
                (
                    state
                    for state in previous
                    if self._intersection_over_union(state.detection, detection) >= 0.35
                ),
                key=lambda state: self._intersection_over_union(state.detection, detection),
                default=None,
            )

        if recognition.status == "known":
            current = LiveIdentityState(
                detection=detection.model_copy(deep=True),
                recognition=recognition.model_copy(deep=True),
                confirmed_at=now,
            )
            self._live_identity_cache[camera_key] = [
                state for state in previous if state is not match
            ] + [current]
            return recognition

        candidate_identity_id = recognition.metadata.get("candidate_identity_id")
        raw_margin = recognition.metadata.get("match_margin")
        candidate_matches = (
            match is not None
            and match.recognition.identity_id is not None
            and candidate_identity_id == str(match.recognition.identity_id)
            and recognition.match_confidence is not None
            and recognition.match_confidence >= max(settings.recognition_match_threshold - 0.08, 0.0)
            and isinstance(raw_margin, (int, float))
            and raw_margin >= settings.recognition_min_margin
        )
        if (
            candidate_matches
            and match is not None
            and match.unknown_frames < settings.recognition_live_max_unknown_frames
        ):
            match.detection = detection.model_copy(deep=True)
            match.unknown_frames += 1
            self._live_identity_cache[camera_key] = previous
            stabilized = match.recognition.model_copy(deep=True)
            stabilized.deduplicated = True
            stabilized.face_region = recognition.face_region
            stabilized.metadata = {
                **stabilized.metadata,
                "live_identity_stabilized": True,
                "raw_recognition_status": recognition.status,
            }
            return stabilized

        self._live_identity_cache[camera_key] = [
            state for state in previous if state is not match
        ]
        return recognition

    @staticmethod
    def _intersection_over_union(first: InferenceBox, second: InferenceBox) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
        second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

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
        ranked_matches: list[tuple[float, KnownPersonProfile, dict[str, object]]] = []
        known_person_count = len(known_persons)
        source_template_count = 0
        eligible_template_count = 0
        compatible_template_count = 0
        skipped_dimension_count = 0
        skipped_dimensions: set[int] = set()
        for person in known_persons:
            person_profile_embeddings = list(self._profile_embeddings(person))
            source_template_count += len(person.face_profiles or [])
            eligible_template_count += len(person_profile_embeddings)
            profile_embeddings: list[list[float]] = []
            for profile_embedding in person_profile_embeddings:
                if len(profile_embedding) != len(embedding):
                    skipped_dimension_count += 1
                    skipped_dimensions.add(len(profile_embedding))
                    continue
                profile_embeddings.append(profile_embedding)
            compatible_template_count += len(profile_embeddings)
            template_scores = sorted(
                (
                    self._cosine_similarity(embedding, profile_embedding)
                    for profile_embedding in profile_embeddings
                ),
                reverse=True,
            )
            template_scores = [score for score in template_scores if score >= 0]
            if not template_scores:
                continue

            top_scores = template_scores[: settings.recognition_template_top_k]
            max_score = top_scores[0]
            mean_top_score = sum(top_scores) / len(top_scores)
            centroid = [
                sum(profile[index] for profile in profile_embeddings) / len(profile_embeddings)
                for index in range(len(embedding))
            ]
            centroid_score = max(self._cosine_similarity(embedding, centroid), 0.0)
            score = (
                max_score
                if len(top_scores) == 1
                else (0.75 * max_score) + (0.15 * mean_top_score) + (0.10 * centroid_score)
            )
            ranked_matches.append(
                (
                    score,
                    person,
                    {
                        "source_template_count": len(person.face_profiles),
                        "template_count": len(profile_embeddings),
                        "top_template_count": len(top_scores),
                        "max_template_score": round(max_score, 4),
                        "mean_top_template_score": round(mean_top_score, 4),
                        "centroid_score": round(centroid_score, 4),
                    },
                )
            )

        ranked_matches.sort(key=lambda match: match[0], reverse=True)
        best_score, best_person, score_metadata = (
            ranked_matches[0] if ranked_matches else (-1.0, None, {})
        )
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
                    **score_metadata,
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
                "candidate_identity_id": str(best_person.person_id) if best_person else None,
                "candidate_identity_label": best_person.full_name if best_person else None,
                "known_person_count": known_person_count,
                "source_template_count": source_template_count,
                "eligible_template_count": eligible_template_count,
                "compatible_template_count": compatible_template_count,
                "query_embedding_dimensions": len(embedding),
                "skipped_template_dimension_count": skipped_dimension_count,
                "skipped_template_dimensions": sorted(skipped_dimensions),
                **score_metadata,
            },
        )

    def _profile_embeddings(self, person: KnownPersonProfile) -> Iterable[list[float]]:
        selected: list[list[float]] = []
        for profile in person.face_profiles or []:
            if not profile.embedding_vector:
                continue
            raw_score = profile.metadata.get("det_score")
            try:
                detection_score = float(raw_score) if raw_score is not None else None
            except (TypeError, ValueError):
                detection_score = None
            if (
                detection_score is not None
                and detection_score < settings.recognition_template_min_det_score
            ):
                continue
            if any(
                len(profile.embedding_vector) == len(existing)
                and self._cosine_similarity(profile.embedding_vector, existing)
                    >= settings.recognition_template_duplicate_similarity
                for existing in selected
            ):
                continue
            selected.append(profile.embedding_vector)
        return selected

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
