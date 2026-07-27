import base64
import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from urllib import error, request

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.camera import Camera, CameraSourceType
from app.models.person import Person
from app.repositories.cameras import CameraRepository
from app.repositories.persons import PersonRepository
from app.schemas.cameras import (
    CameraDetectionScanRequest,
    CameraDetectionScanResponse,
    CameraDetectionScanSummary,
)
from app.schemas.detections import (
    DetectionBoundingBox,
    DetectionEventIngest,
    DetectionEventIngestItem,
    InlineEvidencePayload,
)
from app.services.camera_overlays import camera_overlay_store
from app.services.camera_secrets import CameraSecretManager
from app.services.camera_streams import CameraStreamingService
from app.services.detection_events import DetectionEventService
from app.services.face_embeddings import FaceEmbeddingBackend, FaceEmbeddingError, build_face_embedding_backend
from app.services.media_agent import LocalSubprocessMediaAgent
from app.services.ring_buffer_media import ring_buffer_media_service

logger = logging.getLogger(__name__)
INCIDENT_DETECTION_TYPES = {"weapon", "fire", "smoke", "person", "known_person", "unknown_person"}


@dataclass(slots=True)
class ContinuousScanResult:
    camera_id: object
    success: bool
    detection_count: int = 0
    incident_count: int = 0
    alert_count: int = 0
    ignored_count: int = 0
    error: str | None = None


class CameraDetectionService:
    _runtime_refresh_retry_at: dict[str, float] = {}
    _video_file_positions: dict[str, tuple[tuple[str, int, int], float]] = {}

    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.cameras = CameraRepository(session)
        self.persons = PersonRepository(session)
        self.events = DetectionEventService(session)
        self.streams = CameraStreamingService(self.cameras)
        self.media_agent = LocalSubprocessMediaAgent(self.streams.network_policy)
        self.media_buffer = ring_buffer_media_service
        self._face_embedding_backend: FaceEmbeddingBackend | None = None

    async def run_scan(
        self,
        camera_id,
        payload: CameraDetectionScanRequest,
    ) -> CameraDetectionScanResponse:
        camera = await self.cameras.get(camera_id)
        if not camera:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
        if camera.status.value == "disabled" and not payload.frame_content_base64:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Start the camera or provide a live preview frame before running a manual AI scan.",
            )

        frame_content_base64, frame_content_type = payload.frame_content_base64, payload.frame_content_type
        if not frame_content_base64:
            frame_content_base64, frame_content_type = await self._load_frame_from_camera(camera)
        else:
            # Browser-owned USB/file previews send every analyzed frame in the
            # request. Retain those transient frames in memory before inference
            # so a later incident contains real pre-event motion.
            self.media_buffer.add_frame(
                camera.id,
                content_base64=frame_content_base64,
                content_type=frame_content_type or "image/jpeg",
            )

        known_persons = await self._prepare_known_persons_for_recognition(
            [person for person in await self.persons.list() if person.is_active]
        )
        inference_result = await self._run_inference(
            camera=camera,
            payload=payload,
            frame_content_base64=frame_content_base64,
            frame_content_type=frame_content_type,
            known_persons=known_persons,
        )
        ingest_payload = await self._build_ingest_payload(
            camera,
            inference_result,
            payload,
            frame_content_base64=frame_content_base64,
            frame_content_type=frame_content_type,
        )
        ingest_result = await self.events.ingest(ingest_payload, allow_disabled_camera=True)

        detections = inference_result.get("detections", [])
        metadata = inference_result.get("metadata", {})
        logger.info(
            "camera_scan camera_id=%s detections=%s active_models=%s inference_fps=%s",
            camera_id,
            [
                {
                    "label": detection.get("label"),
                    "object_label": detection.get("object_label"),
                    "confidence": detection.get("confidence"),
                    "provisional": detection.get("provisional", False),
                }
                for detection in detections
            ],
            metadata.get("active_models"),
            inference_result.get("inference_fps"),
        )
        ignored_reasons = list(ingest_result.ignored_reasons)
        unsupported = metadata.get("unsupported_requested_detectors") or []
        if unsupported:
            ignored_reasons.append(
                "The configured model has no classes for: " + ", ".join(map(str, unsupported)) + "."
            )
        if metadata.get("backend_warning"):
            ignored_reasons.append(str(metadata["backend_warning"]))
        response = CameraDetectionScanResponse(
            camera_id=camera_id,
            model_name=inference_result["model_name"],
            model_version=inference_result.get("model_version") or "",
            detection_count=len(detections),
            incident_count=ingest_result.incident_count,
            alert_count=ingest_result.alert_count,
            ignored_count=ingest_result.ignored_count,
            detections=self._summarize_detections(detections),
            ignored_reasons=ignored_reasons,
            backend=metadata.get("backend"),
            callback_delivered=False,
        )
        await camera_overlay_store.publish(camera.id, response.detections)
        return response

    async def run_continuous_batch(
        self,
        camera_ids: list[object],
        payload: CameraDetectionScanRequest,
    ) -> list[ContinuousScanResult]:
        cameras: list[Camera] = []
        for camera_id in camera_ids:
            camera = await self.cameras.get(camera_id)
            if camera is not None:
                cameras.append(camera)

        if not cameras:
            return []

        known_persons = (
            await self._prepare_known_persons_for_recognition(
                [person for person in await self.persons.list() if person.is_active]
            )
            if payload.recognition_enabled
            else []
        )
        frame_results = await asyncio.gather(
            *(self._load_frame_from_camera(camera) for camera in cameras),
            return_exceptions=True,
        )

        inference_requests: list[dict] = []
        ready_cameras: list[Camera] = []
        ready_frames: list[tuple[str, str]] = []
        results: list[ContinuousScanResult] = []
        for camera, frame_result in zip(cameras, frame_results):
            if isinstance(frame_result, Exception):
                results.append(
                    ContinuousScanResult(
                        camera_id=camera.id,
                        success=False,
                        error=str(getattr(frame_result, "detail", frame_result)),
                    )
                )
                continue

            frame_content_base64, frame_content_type = frame_result
            inference_requests.append(
                self._build_inference_request_payload(
                    camera=camera,
                    payload=payload,
                    frame_content_base64=frame_content_base64,
                    frame_content_type=frame_content_type,
                    known_persons=known_persons,
                    manual_scan=False,
                )
            )
            ready_cameras.append(camera)
            ready_frames.append(frame_result)

        if not inference_requests:
            return results

        inference_results = await self._run_inference_batch(inference_requests)
        for camera, inference_result, frame in zip(
            ready_cameras,
            inference_results,
            ready_frames,
        ):
            try:
                detections = inference_result.get("detections", [])
                await camera_overlay_store.publish(
                    camera.id,
                    self._summarize_detections(detections),
                )
                ingest_payload = await self._build_ingest_payload(
                    camera,
                    inference_result,
                    payload,
                    frame_content_base64=frame[0],
                    frame_content_type=frame[1],
                )
                ingest_result = await self.events.ingest(ingest_payload, allow_disabled_camera=True)
                logger.info(
                    "camera_batch_scan camera_id=%s detections=%s inference_fps=%s batch_size=%s",
                    camera.id,
                    [
                        {
                            "label": detection.get("label"),
                            "object_label": detection.get("object_label"),
                            "confidence": detection.get("confidence"),
                        }
                        for detection in detections
                    ],
                    inference_result.get("inference_fps"),
                    len(inference_requests),
                )
                results.append(
                    ContinuousScanResult(
                        camera_id=camera.id,
                        success=True,
                        detection_count=len(detections),
                        incident_count=ingest_result.incident_count,
                        alert_count=ingest_result.alert_count,
                        ignored_count=ingest_result.ignored_count,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Continuous incident persistence failed for camera %s", camera.id)
                results.append(
                    ContinuousScanResult(
                        camera_id=camera.id,
                        success=False,
                        error=str(getattr(exc, "detail", exc)),
                    )
                )

        return results

    async def _load_frame_from_camera(self, camera: Camera) -> tuple[str, str]:
        if camera.source_type is CameraSourceType.file:
            try:
                source_path = self.streams.resolve_file_source(camera)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            return self._remember_frame(
                camera,
                await asyncio.to_thread(self._load_frame_from_file, camera, source_path),
            )

        if camera.source_type is CameraSourceType.http:
            errors: list[str] = []
            runtime_source = await self.streams.cameras.get_runtime_source(camera)
            for source in self.streams._http_source_candidates(camera, runtime_source):
                try:
                    return self._remember_frame(
                        camera,
                        await asyncio.to_thread(self._load_frame_from_http, camera, source),
                    )
                except HTTPException as exc:
                    errors.append(str(exc.detail))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "No frame could be read from this HTTP camera. Configure metadata.stream_url "
                    "with a snapshot or media URL. " + (" ".join(errors) if errors else "")
                ).strip(),
            )

        if camera.source_type is CameraSourceType.rtsp:
            runtime_source = await self.streams.cameras.get_runtime_source(camera)
            frame = await asyncio.to_thread(
                self.media_agent.capture_opencv_frame,
                runtime_source,
                display_source=camera.source,
            )
            return self._remember_frame(camera, (frame.content_base64, frame.content_type))

        if camera.source_type is CameraSourceType.usb:
            source = int(camera.source) if camera.source.isdigit() else camera.source
            frame = await asyncio.to_thread(self.media_agent.capture_opencv_frame, source)
            return self._remember_frame(camera, (frame.content_base64, frame.content_type))

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This camera source needs a browser-captured frame. Open the camera page and run the AI scan from the live preview.",
        )

    def _remember_frame(self, camera: Camera, frame: tuple[str, str]) -> tuple[str, str]:
        content_base64, content_type = frame
        self.media_buffer.add_frame(
            camera.id,
            content_base64=content_base64,
            content_type=content_type,
        )
        return frame

    def _load_frame_from_file(self, camera: Camera, source_path: Path) -> tuple[str, str]:
        if not source_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera media file not found")
        content_type = CameraDetectionService._guess_content_type(source_path.suffix.lower())
        if not content_type.startswith("image/"):
            stat = source_path.stat()
            signature = (str(source_path.resolve()), stat.st_size, stat.st_mtime_ns)
            key = str(camera.id)
            previous_signature, position = self._video_file_positions.get(
                key,
                (signature, 0.0),
            )
            if previous_signature != signature:
                position = 0.0

            try:
                frame = self.media_agent.capture_opencv_frame(
                    str(source_path),
                    display_source=str(source_path),
                    position_seconds=position,
                )
            except HTTPException:
                if position <= 0:
                    raise
                # Recorded cameras loop in the browser. Mirror that behavior in
                # unattended detection when the cursor reaches the final frame.
                position = 0.0
                frame = self.media_agent.capture_opencv_frame(
                    str(source_path),
                    display_source=str(source_path),
                    position_seconds=position,
                )

            step = max(
                settings.file_video_scan_step_seconds,
                1.0 / max(1, camera.inference_fps),
            )
            next_position = position + step
            if (
                frame.source_duration_seconds is not None
                and next_position >= frame.source_duration_seconds
            ):
                next_position = 0.0
            self._video_file_positions[key] = (signature, next_position)
            return frame.content_base64, frame.content_type
        return base64.b64encode(source_path.read_bytes()).decode("utf-8"), content_type

    def _load_frame_from_http(self, camera: Camera, source: str) -> tuple[str, str]:
        source_descriptor = CameraSecretManager.build_candidate_descriptor(CameraSourceType.http, source)
        frame = self.media_agent.capture_http_frame(
            source=source,
            source_descriptor=source_descriptor,
            skip_tls_verification=self.streams._should_skip_tls_verification(camera, source),
        )
        return frame.content_base64, frame.content_type

    async def _run_inference(
        self,
        *,
        camera: Camera,
        payload: CameraDetectionScanRequest,
        frame_content_base64: str,
        frame_content_type: str | None,
        known_persons: list[Person],
    ) -> dict:
        body = json.dumps(
            self._build_inference_request_payload(
                camera=camera,
                payload=payload,
                frame_content_base64=frame_content_base64,
                frame_content_type=frame_content_type,
                known_persons=known_persons,
                manual_scan=True,
            )
        ).encode("utf-8")
        req = request.Request(
            f"{settings.ai_service_url.rstrip('/')}/v1/inference/run",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            return await asyncio.to_thread(self._send_inference_request, req)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI service rejected the scan request: {detail or exc.reason}",
            ) from exc
        except error.URLError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI service is unavailable at {settings.ai_service_url}: {exc.reason}",
            ) from exc

    async def _run_inference_batch(self, payloads: list[dict]) -> list[dict]:
        body = json.dumps({"requests": payloads}).encode("utf-8")
        req = request.Request(
            f"{settings.ai_service_url.rstrip('/')}/v1/inference/run-batch",
            data=body,
            headers={"content-type": "application/json"},
            method="POST",
        )
        try:
            response = await asyncio.to_thread(self._send_inference_request, req)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI service rejected the batch scan request: {detail or exc.reason}",
            ) from exc
        except error.URLError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"AI service is unavailable at {settings.ai_service_url}: {exc.reason}",
            ) from exc
        return response.get("results", [])

    @staticmethod
    def _send_inference_request(req: request.Request) -> dict:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    def _build_inference_request_payload(
        self,
        *,
        camera: Camera,
        payload: CameraDetectionScanRequest,
        frame_content_base64: str,
        frame_content_type: str | None,
        known_persons: list[Person],
        manual_scan: bool,
    ) -> dict:
        return {
            "camera_id": str(camera.id),
            "frame_reference": f"camera-scan:{camera.id}",
            "source_type": camera.source_type.value,
            "frame_content_base64": frame_content_base64,
            "frame_content_type": frame_content_type,
            # Continuous requests already retain the exact source frame in the
            # API ring buffer and attach it only when a confirmed event exists.
            # Do not make the AI service echo a large base64 image on every
            # empty scan.
            "include_evidence": payload.include_evidence and manual_scan,
            "requested_detectors": payload.requested_detectors,
            "recognition_enabled": payload.recognition_enabled,
            "known_persons": [self._serialize_known_person(person) for person in known_persons],
            "occurrence_hint": payload.occurrence_hint or ("manual_scan" if manual_scan else "continuous_monitoring"),
            "metadata": {
                "camera_name": camera.name,
                "camera_location": camera.location,
                "camera_group": camera.group,
                "manual_scan": manual_scan,
            },
        }

    @staticmethod
    def _serialize_known_person(person: Person) -> dict:
        profiles = CameraDetectionService._curate_face_profiles(person.face_profiles)
        return {
            "person_id": str(person.id),
            "full_name": person.full_name,
            "person_type": person.person_type,
            "department": person.department,
            "reference_id": person.reference_id,
            "title": person.title,
            "metadata": person.metadata_,
            "face_profiles": [
                {
                    "face_id": profile.get("id"),
                    "image_path": profile.get("image_path"),
                    "embedding_vector": profile.get("embedding_vector", []),
                    "embedding_model": profile.get("embedding_model"),
                    "metadata": profile.get("metadata", {}),
                }
                for profile in profiles
            ],
        }

    @staticmethod
    def _curate_face_profiles(profiles: list[dict]) -> list[dict]:
        eligible: list[tuple[float, dict]] = []
        for profile in profiles:
            vector = profile.get("embedding_vector") or []
            if not vector:
                continue
            if not CameraDetectionService._profile_matches_runtime_embedding_model(profile):
                continue
            raw_score = (profile.get("metadata") or {}).get("det_score")
            try:
                score = float(raw_score) if raw_score is not None else 1.0
            except (TypeError, ValueError):
                score = 1.0
            if score < settings.recognition_runtime_template_min_det_score:
                continue
            eligible.append((score, profile))

        selected: list[dict] = []
        for _, profile in sorted(eligible, key=lambda item: item[0], reverse=True):
            vector = profile.get("embedding_vector") or []
            if any(
                len(vector) == len(existing.get("embedding_vector") or [])
                and CameraDetectionService._cosine_similarity(
                    vector,
                    existing.get("embedding_vector") or [],
                )
                >= settings.recognition_runtime_template_duplicate_similarity
                for existing in selected
            ):
                continue
            selected.append(profile)
            if len(selected) >= settings.recognition_runtime_max_templates_per_person:
                break
        return selected

    async def _prepare_known_persons_for_recognition(self, persons: list[Person]) -> list[Person]:
        for person in persons:
            if self._curate_face_profiles(person.face_profiles):
                continue
            await self._refresh_stale_face_embeddings(person)
        return persons

    async def _refresh_stale_face_embeddings(self, person: Person) -> None:
        refreshed = 0
        for profile in person.face_profiles:
            if refreshed >= settings.recognition_runtime_max_templates_per_person:
                break
            if self._profile_matches_runtime_embedding_model(profile):
                continue

            image_path = profile.get("image_path")
            face_profile_id = profile.get("id")
            if not image_path or not face_profile_id:
                continue
            retry_key = str(face_profile_id)
            if self._runtime_refresh_retry_at.get(retry_key, 0) > monotonic():
                continue

            source = (settings.storage_root / image_path).resolve()
            storage_root = settings.storage_root.resolve()
            if source != storage_root and storage_root not in source.parents:
                continue
            if not source.is_file():
                continue

            try:
                result = self._runtime_face_embedding_backend().extract_embedding(
                    source.read_bytes()
                )
            except (FaceEmbeddingError, OSError):
                self._runtime_refresh_retry_at[retry_key] = monotonic() + 300
                logger.warning(
                    "runtime_face_embedding_refresh_failed person_id=%s face_profile_id=%s",
                    person.id,
                    face_profile_id,
                    exc_info=True,
                )
                continue

            self._runtime_refresh_retry_at.pop(retry_key, None)
            await self.persons.update_face_profile_embedding(
                person,
                str(face_profile_id),
                embedding_vector=result.vector,
                embedding_model=result.model_name,
                metadata={
                    **result.metadata,
                    "runtime_embedding_refreshed": True,
                    "runtime_embedding_previous_model": profile.get("embedding_model"),
                },
            )
            refreshed += 1

    def _runtime_face_embedding_backend(self) -> FaceEmbeddingBackend:
        if self._face_embedding_backend is None:
            self._face_embedding_backend = build_face_embedding_backend()
        return self._face_embedding_backend

    @staticmethod
    def _profile_matches_runtime_embedding_model(profile: dict) -> bool:
        expected_model = CameraDetectionService._runtime_embedding_model_name()
        if profile.get("embedding_model") != expected_model:
            return False

        vector = profile.get("embedding_vector") or []
        if not vector:
            return False
        if settings.recognition_backend == "hash":
            return len(vector) == settings.recognition_embedding_dimensions
        return True

    @staticmethod
    def _runtime_embedding_model_name() -> str:
        if settings.recognition_backend == "insightface":
            return f"insightface-{settings.recognition_insightface_model}"
        return settings.recognition_embedding_model

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        numerator = sum(first * second for first, second in zip(left, right))
        left_magnitude = math.sqrt(sum(value * value for value in left))
        right_magnitude = math.sqrt(sum(value * value for value in right))
        if not left_magnitude or not right_magnitude:
            return -1.0
        return numerator / (left_magnitude * right_magnitude)

    async def _build_ingest_payload(
        self,
        camera: Camera,
        inference_result: dict,
        scan_request: CameraDetectionScanRequest,
        *,
        frame_content_base64: str | None = None,
        frame_content_type: str | None = None,
    ) -> DetectionEventIngest:
        clip = None
        metadata = dict(inference_result.get("metadata", {}))
        metadata["requested_detectors"] = list(scan_request.requested_detectors)
        incident_detections = [
            detection
            for detection in inference_result.get("detections", [])
            if str(detection.get("label", "")).lower() in INCIDENT_DETECTION_TYPES
        ]
        confirmed_detections = [
            detection
            for detection in incident_detections
            if not detection.get("provisional", False)
        ]
        snapshot_evidence = CameraDetectionService._inline_evidence(
            inference_result.get("snapshot_evidence")
        )
        if (
            snapshot_evidence is None
            and confirmed_detections
            and frame_content_base64
        ):
            # The API already has the exact frame used for inference. Keep it for
            # every confirmed incident even if the AI response omitted its echoed
            # evidence payload (or live-preview transport disabled that echo).
            # Frames without confirmed detections remain transient.
            snapshot_evidence = InlineEvidencePayload(
                content_base64=frame_content_base64,
                content_type=frame_content_type or "image/jpeg",
            )
            metadata["snapshot_source"] = "confirmed_inference_frame_fallback"
        if scan_request.include_evidence and confirmed_detections:
            clip = await self.media_buffer.build_event_clip(
                camera.id,
                capture_after_frame=lambda: self._load_frame_from_camera(camera),
            )
            if clip is not None:
                metadata = {**metadata, **clip.metadata}

        return DetectionEventIngest(
            camera_id=camera.id,
            occurred_at=inference_result.get("occurred_at"),
            model_name=inference_result["model_name"],
            model_version=inference_result.get("model_version"),
            inference_fps=CameraDetectionService._normalize_inference_fps(
                inference_result.get("inference_fps")
            ),
            source_fps=inference_result.get("source_fps"),
            snapshot_evidence=snapshot_evidence,
            clip_evidence=InlineEvidencePayload(
                content_base64=clip.content_base64,
                content_type=clip.content_type,
            )
            if clip is not None
            else None,
            detections=[
                DetectionEventIngestItem(
                    detection_type=detection["label"],
                    confidence=detection["confidence"],
                    track_id=detection.get("track_id"),
                    bounding_box=DetectionBoundingBox(
                        x1=detection["x1"],
                        y1=detection["y1"],
                        x2=detection["x2"],
                        y2=detection["y2"],
                        label=detection.get("object_label") or detection.get("label"),
                    ),
                    identity_id=(detection.get("recognition") or {}).get("identity_id"),
                    identity_label=(detection.get("recognition") or {}).get("identity_label"),
                    match_confidence=(detection.get("recognition") or {}).get("match_confidence"),
                    recognition_status=(detection.get("recognition") or {}).get("status"),
                    face_bounding_box=CameraDetectionService._face_box(detection.get("face_region")),
                    face_image_path=(detection.get("face_region") or {}).get("image_path"),
                    face_image_evidence=CameraDetectionService._inline_evidence(
                        detection.get("face_image_evidence")
                    ),
                    metadata={
                        **(detection.get("recognition") or {}).get("metadata", {}),
                        "provisional": bool(detection.get("provisional", False)),
                    },
                )
                for detection in incident_detections
            ],
            metadata=metadata,
        )

    @staticmethod
    def _normalize_inference_fps(value: float | None) -> float | None:
        if value is None:
            return None
        return min(max(float(value), 0.1), 120.0)

    @staticmethod
    def _inline_evidence(payload: dict | None) -> InlineEvidencePayload | None:
        if not payload:
            return None
        return InlineEvidencePayload(
            content_base64=payload["content_base64"],
            content_type=payload.get("content_type"),
        )

    @staticmethod
    def _face_box(face_region: dict | None) -> DetectionBoundingBox | None:
        if not face_region:
            return None
        return DetectionBoundingBox(
            x1=face_region["x1"],
            y1=face_region["y1"],
            x2=face_region["x2"],
            y2=face_region["y2"],
            label="face",
        )

    @staticmethod
    def _summarize_detections(detections: list[dict]) -> list[CameraDetectionScanSummary]:
        summaries: list[CameraDetectionScanSummary] = []
        for detection in detections:
            if CameraDetectionService._hide_from_operator_overlay(detection):
                continue
            recognition = detection.get("recognition") or {}
            recognition_metadata = recognition.get("metadata") or {}
            summaries.append(
                CameraDetectionScanSummary(
                    detection_type=str(detection.get("label", "unknown")),
                    object_label=detection.get("object_label"),
                    confidence=float(detection.get("confidence", 0)),
                    track_id=detection.get("track_id"),
                    recognition_status=recognition.get("status"),
                    identity_id=recognition.get("identity_id"),
                    identity_label=recognition.get("identity_label"),
                    match_confidence=recognition.get("match_confidence"),
                    person_type=recognition_metadata.get("person_type"),
                    department=recognition_metadata.get("department"),
                    reference_id=recognition_metadata.get("reference_id"),
                    title=recognition_metadata.get("title"),
                    bounding_box=DetectionBoundingBox(
                        x1=float(detection.get("x1", 0)),
                        y1=float(detection.get("y1", 0)),
                        x2=float(detection.get("x2", 0)),
                        y2=float(detection.get("y2", 0)),
                        label=str(detection.get("object_label") or detection.get("label", "unknown")),
                    ),
                    face_bounding_box=CameraDetectionService._face_box(detection.get("face_region")),
                    metadata={
                        **recognition_metadata,
                        "provisional": bool(detection.get("provisional", False)),
                    },
                )
            )
        return summaries

    @staticmethod
    def _hide_from_operator_overlay(detection: dict) -> bool:
        return False

    @staticmethod
    def _guess_content_type(extension: str) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
        }.get(extension, "application/octet-stream")
