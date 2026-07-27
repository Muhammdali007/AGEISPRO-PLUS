import json
import base64
from io import BytesIO
from time import perf_counter
from urllib import error, request

from app.core.config import settings
from app.schemas.inference import (
    FaceRegion,
    InferenceBox,
    InlineEvidencePayload,
    InferenceEventDispatchResult,
    InferenceRequest,
    InferenceResult,
)
from app.services.backends import (
    InferenceBackendError,
    build_inference_backend,
)
from app.services.recognition import FaceRecognitionService
from app.services.temporal_confirmation import TemporalDetectionConfirmation
from app.services.temporal_tracking import TemporalBoxTracker


class InferencePipeline:
    def __init__(self) -> None:
        self.primary_backend = build_inference_backend(settings.model_backend)
        self.fallback_backend = (
            build_inference_backend(settings.model_fallback_backend)
            if settings.allow_backend_fallback
            else None
        )
        self.recognition = FaceRecognitionService()
        self.box_tracker = TemporalBoxTracker()
        self.temporal_confirmation = TemporalDetectionConfirmation()

    def warmup(self) -> None:
        self.primary_backend.warmup()
        if self.fallback_backend and self.fallback_backend.backend_name != self.primary_backend.backend_name:
            self.fallback_backend.warmup()
        self.recognition.warmup()

    def run(self, payload: InferenceRequest) -> InferenceResult:
        backend_result, backend_metadata = self._run_backend(payload)
        return self._build_result(payload, backend_result, backend_metadata)

    def run_batch(self, payloads: list[InferenceRequest]) -> list[InferenceResult]:
        backend_results, backend_metadata = self._run_backend_batch(payloads)
        return [
            self._build_result(payload, backend_result, backend_metadata)
            for payload, backend_result in zip(payloads, backend_results)
        ]

    def _build_result(
        self,
        payload: InferenceRequest,
        backend_result,
        backend_metadata: dict[str, object],
    ) -> InferenceResult:
        postprocess_started_at = perf_counter()
        detections = self.box_tracker.update(payload, backend_result.detections)
        recognition_started_at = perf_counter()
        detections = self._apply_recognition(payload, detections)
        recognition_ms = round((perf_counter() - recognition_started_at) * 1000, 2)
        detections, suppressed_candidates = self.temporal_confirmation.filter(payload, detections)
        postprocess_ms = round((perf_counter() - postprocess_started_at) * 1000, 2)
        return InferenceResult(
            camera_id=payload.camera_id,
            model_name=backend_result.model_name,
            model_version=backend_result.model_version,
            inference_fps=backend_result.inference_fps,
            source_fps=payload.metadata.get("source_fps"),
            detections=detections,
            snapshot_evidence=self._snapshot_evidence(payload),
            metadata={
                **payload.metadata,
                **backend_result.metadata,
                **backend_metadata,
                "backend": backend_result.backend_name,
                "requested_backend": settings.model_backend,
                "frame_reference": payload.frame_reference,
                "source_type": payload.source_type,
                "recognition_enabled": payload.recognition_enabled,
                "snapshot_tracking_enabled": (
                    settings.model_enable_tracking
                    and payload.occurrence_hint
                    in {"continuous_monitoring", "dashboard_live_scan"}
                ),
                "recognition_ms": recognition_ms,
                "postprocess_ms": postprocess_ms,
                "temporal_candidates_suppressed": suppressed_candidates,
            },
        )

    def dispatch_events(self, result: InferenceResult) -> InferenceEventDispatchResult:
        if not settings.enable_event_callback or not settings.api_event_callback_url:
            return InferenceEventDispatchResult(
                delivered=False,
                callback_url=settings.api_event_callback_url,
                message="Event callback is disabled.",
            )

        body = json.dumps(
            {
                "camera_id": str(result.camera_id),
                "occurred_at": result.occurred_at.isoformat(),
                "model_name": result.model_name,
                "model_version": result.model_version,
                "inference_fps": result.inference_fps,
                "source_fps": result.source_fps,
                "snapshot_evidence": result.snapshot_evidence.model_dump(exclude_none=True)
                if result.snapshot_evidence
                else None,
                "detections": [
                    {
                        "detection_type": detection.label,
                        "confidence": detection.confidence,
                        "track_id": detection.track_id,
                        "bounding_box": {
                            "x1": detection.x1,
                            "y1": detection.y1,
                            "x2": detection.x2,
                            "y2": detection.y2,
                            "label": detection.object_label or detection.label,
                        }
                        if detection.label
                        else None,
                        "identity_id": str(detection.recognition.identity_id)
                        if detection.recognition and detection.recognition.identity_id
                        else None,
                        "identity_label": detection.recognition.identity_label
                        if detection.recognition
                        else None,
                        "match_confidence": detection.recognition.match_confidence
                        if detection.recognition
                        else None,
                        "recognition_status": detection.recognition.status
                        if detection.recognition
                        else None,
                        "face_bounding_box": {
                            "x1": detection.face_region.x1,
                            "y1": detection.face_region.y1,
                            "x2": detection.face_region.x2,
                            "y2": detection.face_region.y2,
                            "label": "face",
                        }
                        if detection.face_region
                        else None,
                        "face_image_path": detection.face_region.image_path
                        if detection.face_region
                        else None,
                        "face_image_evidence": detection.face_image_evidence.model_dump(exclude_none=True)
                        if detection.face_image_evidence
                        else None,
                        "metadata": {
                            **(detection.recognition.metadata if detection.recognition else {}),
                            "deduplicated": detection.recognition.deduplicated
                            if detection.recognition
                            else False,
                            "embedding_model": detection.recognition.embedding_model
                            if detection.recognition
                            else None,
                        },
                    }
                    for detection in result.detections
                    if not detection.provisional
                ],
                "metadata": result.metadata,
            }
        ).encode("utf-8")
        req = request.Request(
            settings.api_event_callback_url,
            data=body,
            headers={
                "content-type": "application/json",
                **(
                    {"x-service-token": settings.api_event_callback_token}
                    if settings.api_event_callback_token
                    else {}
                ),
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=5) as response:
                return InferenceEventDispatchResult(
                    delivered=True,
                    callback_url=settings.api_event_callback_url,
                    status_code=response.status,
                    message="Detection batch delivered to the API.",
                )
        except error.URLError as exc:
            return InferenceEventDispatchResult(
                delivered=False,
                callback_url=settings.api_event_callback_url,
                message=str(exc),
            )

    def _run_backend(self, payload: InferenceRequest):
        try:
            return self.primary_backend.infer(payload), {}
        except InferenceBackendError as exc:
            if not self.fallback_backend or self.fallback_backend.backend_name == self.primary_backend.backend_name:
                raise
            fallback_result = self.fallback_backend.infer(payload)
            return (
                fallback_result,
                {
                    "backend_fallback": True,
                    "backend_warning": str(exc),
                    "fallback_backend": fallback_result.backend_name,
                },
            )

    def _run_backend_batch(self, payloads: list[InferenceRequest]):
        try:
            return self.primary_backend.infer_batch(payloads), {}
        except InferenceBackendError as exc:
            if not self.fallback_backend or self.fallback_backend.backend_name == self.primary_backend.backend_name:
                raise
            fallback_results = self.fallback_backend.infer_batch(payloads)
            return (
                fallback_results,
                {
                    "backend_fallback": True,
                    "backend_warning": str(exc),
                    "fallback_backend": self.fallback_backend.backend_name,
                },
            )

    def _apply_recognition(
        self, payload: InferenceRequest, detections: list[InferenceBox]
    ) -> list[InferenceBox]:
        if not payload.recognition_enabled:
            return detections

        enriched = self.recognition.enrich_detections(payload, detections)
        return [
            detection.model_copy(
                update={
                    "face_image_evidence": self._face_evidence(payload, detection.face_region)
                }
            )
            if detection.face_region is not None
            else detection
            for detection in enriched
        ]

    @staticmethod
    def _snapshot_evidence(payload: InferenceRequest) -> InlineEvidencePayload | None:
        if not payload.include_evidence or not payload.frame_content_base64:
            return None
        return InlineEvidencePayload(
            content_base64=payload.frame_content_base64,
            content_type=payload.frame_content_type or "image/jpeg",
        )

    @staticmethod
    def _face_evidence(
        payload: InferenceRequest,
        face_region: FaceRegion,
    ) -> InlineEvidencePayload | None:
        if not payload.include_evidence or not payload.frame_content_base64:
            return None
        try:
            from PIL import Image

            frame_bytes = base64.b64decode(payload.frame_content_base64, validate=True)
            image = Image.open(BytesIO(frame_bytes)).convert("RGB")
            left = max(int(face_region.x1), 0)
            top = max(int(face_region.y1), 0)
            right = min(int(face_region.x2), image.width)
            bottom = min(int(face_region.y2), image.height)
            if right <= left or bottom <= top:
                return None

            buffer = BytesIO()
            image.crop((left, top, right, bottom)).save(buffer, format="JPEG", quality=90)
            return InlineEvidencePayload(
                content_base64=base64.b64encode(buffer.getvalue()).decode("utf-8"),
                content_type="image/jpeg",
            )
        except (OSError, ValueError):
            return None
