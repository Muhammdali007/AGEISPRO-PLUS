import base64
import asyncio
import json
import logging
from pathlib import Path
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
from app.services.camera_streams import CameraStreamingService
from app.services.detection_events import DetectionEventService

logger = logging.getLogger(__name__)


class CameraDetectionService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.cameras = CameraRepository(session)
        self.persons = PersonRepository(session)
        self.events = DetectionEventService(session)
        self.streams = CameraStreamingService(self.cameras)

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

        known_persons = [person for person in await self.persons.list() if person.is_active]
        inference_result = await self._run_inference(
            camera=camera,
            payload=payload,
            frame_content_base64=frame_content_base64,
            frame_content_type=frame_content_type,
            known_persons=known_persons,
        )
        ingest_payload = self._build_ingest_payload(camera_id, inference_result)
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
        return CameraDetectionScanResponse(
            camera_id=camera_id,
            model_name=inference_result["model_name"],
            model_version=inference_result.get("model_version") or "",
            detection_count=len(detections),
            incident_count=ingest_result.incident_count,
            alert_count=ingest_result.alert_count,
            ignored_count=ingest_result.ignored_count,
            detections=[
                CameraDetectionScanSummary(
                    detection_type=str(detection.get("label", "unknown")),
                    confidence=float(detection.get("confidence", 0)),
                    track_id=detection.get("track_id"),
                    recognition_status=(detection.get("recognition") or {}).get("status"),
                    identity_label=(detection.get("recognition") or {}).get("identity_label"),
                    bounding_box=DetectionBoundingBox(
                        x1=float(detection.get("x1", 0)),
                        y1=float(detection.get("y1", 0)),
                        x2=float(detection.get("x2", 0)),
                        y2=float(detection.get("y2", 0)),
                        label=str(detection.get("object_label") or detection.get("label", "unknown")),
                    ),
                    face_bounding_box=self._face_box(detection.get("face_region")),
                    metadata=(detection.get("recognition") or {}).get("metadata", {}),
                )
                for detection in detections
            ],
            ignored_reasons=ignored_reasons,
            backend=metadata.get("backend"),
            callback_delivered=False,
        )

    async def _load_frame_from_camera(self, camera: Camera) -> tuple[str, str]:
        if camera.source_type is CameraSourceType.file:
            try:
                source_path = self.streams.resolve_file_source(camera)
            except ValueError as exc:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
            return await asyncio.to_thread(self._load_frame_from_file, source_path)

        if camera.source_type is CameraSourceType.http:
            errors: list[str] = []
            for source in self.streams._http_source_candidates(camera):
                if Path(source).suffix.lower() in {".m3u8", ".mp4", ".webm", ".mov"}:
                    continue
                try:
                    return await asyncio.to_thread(self._load_frame_from_http, source)
                except HTTPException as exc:
                    errors.append(str(exc.detail))
            try:
                return await asyncio.to_thread(self._load_frame_with_opencv, camera.source)
            except HTTPException:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=(
                        "No frame could be read from this HTTP camera. Configure metadata.stream_url "
                        "with a snapshot or media URL. " + (" ".join(errors) if errors else "")
                    ).strip(),
                )

        if camera.source_type is CameraSourceType.rtsp:
            return await asyncio.to_thread(self._load_frame_with_opencv, camera.source)

        if camera.source_type is CameraSourceType.usb:
            source = int(camera.source) if camera.source.isdigit() else camera.source
            return await asyncio.to_thread(self._load_frame_with_opencv, source)

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This camera source needs a browser-captured frame. Open the camera page and run the AI scan from the live preview.",
        )

    @staticmethod
    def _load_frame_from_file(source_path: Path) -> tuple[str, str]:
        if not source_path.is_file():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera media file not found")
        content_type = CameraDetectionService._guess_content_type(source_path.suffix.lower())
        if not content_type.startswith("image/"):
            return CameraDetectionService._load_frame_with_opencv(str(source_path))
        return base64.b64encode(source_path.read_bytes()).decode("utf-8"), content_type

    @staticmethod
    def _load_frame_with_opencv(source: str | int) -> tuple[str, str]:
        try:
            import cv2
        except ImportError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="OpenCV is required to read USB, RTSP, and video camera frames.",
            ) from exc
        capture = cv2.VideoCapture(source)
        try:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Unable to decode a frame from camera source {source}.",
                )
            encoded, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not encoded:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Unable to encode the captured camera frame.",
                )
            return base64.b64encode(buffer.tobytes()).decode("utf-8"), "image/jpeg"
        finally:
            capture.release()

    @staticmethod
    def _load_frame_from_http(source: str) -> tuple[str, str]:
        req = request.Request(source, headers={"User-Agent": "AegisPro/1.0"})
        try:
            with request.urlopen(req, timeout=5) as response:
                content_type = response.headers.get_content_type()
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="This HTTP source is not serving a single image frame. Open the camera page and run the AI scan from the live preview.",
                    )
                return base64.b64encode(response.read()).decode("utf-8"), content_type
        except HTTPException:
            raise
        except error.URLError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Unable to read the camera source: {exc}",
            ) from exc

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
            {
                "camera_id": str(camera.id),
                "frame_reference": f"camera-scan:{camera.id}",
                "source_type": camera.source_type.value,
                "frame_content_base64": frame_content_base64,
                "frame_content_type": frame_content_type,
                "include_evidence": payload.include_evidence,
                "requested_detectors": payload.requested_detectors,
                "recognition_enabled": payload.recognition_enabled,
                "known_persons": [self._serialize_known_person(person) for person in known_persons],
                "occurrence_hint": payload.occurrence_hint or "manual_scan",
                "metadata": {
                    "camera_name": camera.name,
                    "camera_location": camera.location,
                    "camera_group": camera.group,
                    "manual_scan": True,
                },
            }
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

    @staticmethod
    def _send_inference_request(req: request.Request) -> dict:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _serialize_known_person(person: Person) -> dict:
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
                for profile in person.face_profiles
                if profile.get("embedding_vector")
            ],
        }

    @staticmethod
    def _build_ingest_payload(camera_id, inference_result: dict) -> DetectionEventIngest:
        return DetectionEventIngest(
            camera_id=camera_id,
            occurred_at=inference_result.get("occurred_at"),
            model_name=inference_result["model_name"],
            model_version=inference_result.get("model_version"),
            inference_fps=CameraDetectionService._normalize_inference_fps(
                inference_result.get("inference_fps")
            ),
            source_fps=inference_result.get("source_fps"),
            snapshot_evidence=CameraDetectionService._inline_evidence(
                inference_result.get("snapshot_evidence")
            ),
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
                    metadata=(detection.get("recognition") or {}).get("metadata", {}),
                )
                for detection in inference_result.get("detections", [])
            ],
            metadata=inference_result.get("metadata", {}),
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
    def _guess_content_type(extension: str) -> str:
        return {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
        }.get(extension, "application/octet-stream")
