import json
from urllib import error, request

from app.core.config import settings
from app.schemas.inference import (
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


class InferencePipeline:
    def __init__(self) -> None:
        self.primary_backend = build_inference_backend(settings.model_backend)
        self.fallback_backend = (
            build_inference_backend(settings.model_fallback_backend)
            if settings.allow_backend_fallback
            else None
        )
        self.recognition = FaceRecognitionService()

    def run(self, payload: InferenceRequest) -> InferenceResult:
        backend_result, backend_metadata = self._run_backend(payload)
        detections = backend_result.detections
        detections = self._apply_recognition(payload, detections)
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

    def _apply_recognition(
        self, payload: InferenceRequest, detections: list[InferenceBox]
    ) -> list[InferenceBox]:
        if not payload.recognition_enabled:
            return detections

        enriched: list[InferenceBox] = []
        for detection in detections:
            if detection.label != "person":
                enriched.append(detection)
                continue

            face_region, recognition = self.recognition.enrich_detection(payload, detection)
            effective_label = "known_person" if recognition.status == "known" else "unknown_person"
            enriched.append(
                InferenceBox(
                    x1=detection.x1,
                    y1=detection.y1,
                    x2=detection.x2,
                    y2=detection.y2,
                    confidence=detection.confidence,
                    label=effective_label,
                    track_id=detection.track_id,
                    face_region=face_region,
                    recognition=recognition,
                    face_image_evidence=self._face_evidence(payload, detection),
                )
            )
        return enriched

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
        detection: InferenceBox,
    ) -> InlineEvidencePayload | None:
        if not payload.include_evidence or not payload.frame_content_base64 or detection.label != "person":
            return None
        return InlineEvidencePayload(
            content_base64=payload.frame_content_base64,
            content_type=payload.frame_content_type or "image/jpeg",
        )
