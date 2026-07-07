from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.incident import DetectionType, IncidentPriority
from app.repositories.alerts import AlertRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.persons import PersonRepository
from app.schemas.alerts import AlertCreate
from app.schemas.detections import (
    DetectionEventIngest,
    DetectionEventIngestItem,
    DetectionEventIngestResponse,
    DetectionEventResult,
    RecognitionStatus,
)
from app.schemas.incidents import IncidentCreate
from app.services.evidence_storage import EvidenceStorageService
from app.services.event_broadcaster import event_broadcaster
from app.services.persons import PersonService


class DetectionEventService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.cameras = CameraRepository(session)
        self.incidents = IncidentRepository(session)
        self.alerts = AlertRepository(session)
        self.persons = PersonService(PersonRepository(session))
        self.evidence = EvidenceStorageService()

    async def ingest(self, payload: DetectionEventIngest) -> DetectionEventIngestResponse:
        camera = await self.cameras.get(payload.camera_id)
        if not camera:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

        if not camera.detection_enabled:
            return DetectionEventIngestResponse(
                camera_id=payload.camera_id,
                incident_count=0,
                alert_count=0,
                ignored_count=len(payload.detections),
                ignored_reasons=["Camera detection is disabled."],
            )

        occurred_at = payload.occurred_at or datetime.now(UTC)
        results: list[DetectionEventResult] = []
        ignored_reasons: list[str] = []

        for detection in payload.detections:
            detection_type = self._resolve_detection_type(detection)
            if not self._supports_event_generation(detection_type):
                ignored_reasons.append(
                    f"Detection type {detection_type.value} is not handled in phase 7."
                )
                continue

            self._validate_inline_evidence(payload, detection)
            priority = self._priority_for_detection(detection_type)
            recognized_identity = self._recognized_identity_for_detection(detection)
            incident = await self.incidents.create(
                IncidentCreate(
                    camera_id=payload.camera_id,
                    detection_type=detection_type,
                    priority=priority,
                    confidence=detection.confidence,
                    occurred_at=occurred_at,
                    bounding_boxes=self._serialize_bounding_boxes(detection),
                    snapshot_path=payload.snapshot_path,
                    clip_path=payload.clip_path,
                    recognized_identity=recognized_identity,
                    metadata={
                        **payload.metadata,
                        "model_name": payload.model_name,
                        "model_version": payload.model_version,
                        "source_fps": payload.source_fps,
                        "inference_fps": payload.inference_fps,
                        "original_detection_type": detection.detection_type.value,
                        "track_id": detection.track_id,
                        "recognition_status": detection.recognition_status.value
                        if detection.recognition_status
                        else None,
                        "face_image_path": detection.face_image_path,
                        "detection_metadata": detection.metadata,
                    },
                )
            )
            incident = await self._persist_inline_evidence(
                incident=incident,
                payload=payload,
                detection=detection,
            )

            alert_id = None
            if self._should_create_alert(detection_type):
                alert = await self.alerts.create(
                    AlertCreate(
                        incident_id=incident.id,
                        priority=priority,
                        title=self._alert_title(detection_type),
                        message=(
                            f"{detection_type.value.replace('_', ' ').title()} detected "
                            f"on camera {camera.name} with confidence {detection.confidence:.2f}."
                        ),
                    )
                )
                alert_id = alert.id
                await event_broadcaster.publish(
                    {
                        "type": "alert.created",
                        "alert_id": str(alert.id),
                        "incident_id": str(alert.incident_id),
                        "priority": alert.priority.value,
                    }
                )

            if detection_type is DetectionType.known_person and detection.identity_id:
                await self.persons.record_recognition(detection.identity_id, occurred_at)

            await event_broadcaster.publish(
                {
                    "type": "incident.created",
                    "incident_id": str(incident.id),
                    "camera_id": str(incident.camera_id),
                    "detection_type": incident.detection_type.value,
                    "priority": incident.priority.value,
                }
            )

            results.append(
                DetectionEventResult(
                    incident_id=incident.id,
                    detection_type=detection_type,
                    priority=priority,
                    alert_id=alert_id,
                )
            )

        return DetectionEventIngestResponse(
            camera_id=payload.camera_id,
            incident_count=len(results),
            alert_count=sum(1 for item in results if item.alert_id is not None),
            ignored_count=len(payload.detections) - len(results),
            results=results,
            ignored_reasons=ignored_reasons,
        )

    def _validate_inline_evidence(
        self,
        payload: DetectionEventIngest,
        detection: DetectionEventIngestItem,
    ) -> None:
        if payload.snapshot_evidence:
            self.evidence.validate_base64_payload(payload.snapshot_evidence.content_base64)
        if payload.clip_evidence:
            self.evidence.validate_base64_payload(payload.clip_evidence.content_base64)
        if detection.face_image_evidence:
            self.evidence.validate_base64_payload(detection.face_image_evidence.content_base64)

    @staticmethod
    def _supports_event_generation(detection_type: DetectionType) -> bool:
        return detection_type in {
            DetectionType.weapon,
            DetectionType.fire,
            DetectionType.smoke,
            DetectionType.person,
            DetectionType.known_person,
            DetectionType.unknown_person,
        }

    @staticmethod
    def _resolve_detection_type(detection: DetectionEventIngestItem) -> DetectionType:
        if detection.detection_type is DetectionType.person:
            if detection.recognition_status is RecognitionStatus.known:
                return DetectionType.known_person
            if detection.recognition_status is RecognitionStatus.unknown:
                return DetectionType.unknown_person
        return detection.detection_type

    @staticmethod
    def _priority_for_detection(detection_type: DetectionType) -> IncidentPriority:
        return {
            DetectionType.weapon: IncidentPriority.critical,
            DetectionType.fire: IncidentPriority.critical,
            DetectionType.smoke: IncidentPriority.high,
            DetectionType.person: IncidentPriority.medium,
            DetectionType.known_person: IncidentPriority.medium,
            DetectionType.unknown_person: IncidentPriority.medium,
        }[detection_type]

    @staticmethod
    def _should_create_alert(detection_type: DetectionType) -> bool:
        return detection_type in {
            DetectionType.weapon,
            DetectionType.fire,
            DetectionType.smoke,
        }

    @staticmethod
    def _alert_title(detection_type: DetectionType) -> str:
        return f"{detection_type.value.replace('_', ' ').title()} detected"

    @staticmethod
    def _serialize_bounding_boxes(detection: DetectionEventIngestItem) -> list[dict[str, float | str]]:
        boxes: list[dict[str, float | str]] = []
        if detection.bounding_box is not None:
            data = detection.bounding_box.model_dump(exclude_none=True)
            if detection.track_id is not None:
                data["track_id"] = detection.track_id
            boxes.append(data)
        if detection.face_bounding_box is not None:
            face_box = detection.face_bounding_box.model_dump(exclude_none=True)
            face_box["label"] = face_box.get("label") or "face"
            boxes.append(face_box)
        return boxes

    @staticmethod
    def _recognized_identity_for_detection(
        detection: DetectionEventIngestItem,
    ) -> dict[str, object] | None:
        if not detection.recognition_status:
            return None
        return {
            "status": detection.recognition_status.value,
            "identity_id": str(detection.identity_id) if detection.identity_id else None,
            "identity_label": detection.identity_label,
            "match_confidence": detection.match_confidence,
            "face_image_path": detection.face_image_path,
            "face_bounding_box": detection.face_bounding_box.model_dump(exclude_none=True)
            if detection.face_bounding_box
            else None,
        }

    async def _persist_inline_evidence(
        self,
        *,
        incident: Any,
        payload: DetectionEventIngest,
        detection: DetectionEventIngestItem,
    ):
        updated = False

        if payload.snapshot_evidence:
            incident.snapshot_path = self.evidence.store_incident_blob(
                camera_id=incident.camera_id,
                incident_id=incident.id,
                stem="snapshot",
                content_base64=payload.snapshot_evidence.content_base64,
                content_type=payload.snapshot_evidence.content_type,
            )
            updated = True

        if payload.clip_evidence:
            incident.clip_path = self.evidence.store_incident_blob(
                camera_id=incident.camera_id,
                incident_id=incident.id,
                stem="clip",
                content_base64=payload.clip_evidence.content_base64,
                content_type=payload.clip_evidence.content_type,
            )
            updated = True

        if detection.face_image_evidence:
            recognized_identity = dict(incident.recognized_identity or {})
            recognized_identity["face_image_path"] = self.evidence.store_incident_blob(
                camera_id=incident.camera_id,
                incident_id=incident.id,
                stem=f"face-{detection.track_id or 'detection'}",
                content_base64=detection.face_image_evidence.content_base64,
                content_type=detection.face_image_evidence.content_type,
            )
            incident.recognized_identity = recognized_identity
            updated = True

        if updated:
            await self.session.commit()
            await self.session.refresh(incident)
        return incident
