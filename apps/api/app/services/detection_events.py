from datetime import UTC, datetime, timedelta
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.transactions import transaction_scope
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
from app.services.persons import PersonService
from app.services.sound_alerts import sound_alert_service
from app.services.transactional_outbox import TransactionalOutboxService

EMERGENCY_ALERT_PRIORITY_POLICY: dict[DetectionType, IncidentPriority] = {
    DetectionType.weapon: IncidentPriority.critical,
    DetectionType.fire: IncidentPriority.critical,
    DetectionType.smoke: IncidentPriority.high,
}

INCIDENT_PRIORITY_POLICY: dict[DetectionType, IncidentPriority] = {
    **EMERGENCY_ALERT_PRIORITY_POLICY,
    DetectionType.person: IncidentPriority.medium,
    DetectionType.known_person: IncidentPriority.medium,
    DetectionType.unknown_person: IncidentPriority.medium,
}


@dataclass(slots=True)
class PlannedDetection:
    detection: DetectionEventIngestItem
    detection_type: DetectionType
    priority: IncidentPriority
    recognized_identity: dict[str, object] | None
    bounding_boxes: list[dict[str, float | str]]


class DetectionEventService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.cameras = CameraRepository(session)
        self.incidents = IncidentRepository(session)
        self.alerts = AlertRepository(session)
        self.persons = PersonService(PersonRepository(session))
        self.evidence = EvidenceStorageService()
        self.outbox = TransactionalOutboxService(session)

    async def ingest(
        self,
        payload: DetectionEventIngest,
        *,
        allow_disabled_camera: bool = False,
    ) -> DetectionEventIngestResponse:
        async with transaction_scope(self.session) as scope:
            camera = await self.cameras.get(payload.camera_id)
            if not camera:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")

            if not camera.detection_enabled and not allow_disabled_camera:
                return DetectionEventIngestResponse(
                    camera_id=payload.camera_id,
                    incident_count=0,
                    alert_count=0,
                    ignored_count=len(payload.detections),
                    ignored_reasons=["Camera detection is disabled."],
                )

            occurred_at = payload.occurred_at or datetime.now(UTC)
            planned_detections, ignored_reasons = await self._plan_detections(payload, occurred_at)
            requested_detectors = self._requested_detectors(payload.metadata)
            sound_events = await sound_alert_service.observe_scan(
                camera_id=payload.camera_id,
                camera_name=camera.name,
                detections=payload.detections,
                requested_detectors=requested_detectors,
            )
            results: list[DetectionEventResult] = []

            for planned in planned_detections:
                detection = planned.detection
                incident = await self.incidents.create(
                    IncidentCreate(
                        camera_id=payload.camera_id,
                        detection_type=planned.detection_type,
                        priority=planned.priority,
                        confidence=detection.confidence,
                        occurred_at=occurred_at,
                        bounding_boxes=planned.bounding_boxes,
                        snapshot_path=payload.snapshot_path,
                        clip_path=payload.clip_path,
                        recognized_identity=planned.recognized_identity,
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
                await self._persist_inline_evidence(
                    incident=incident,
                    payload=payload,
                    detection=detection,
                )

                alert_id = None
                await self.outbox.enqueue(
                    {
                        "type": "incident.created",
                        "incident_id": str(incident.id),
                        "camera_id": str(incident.camera_id),
                        "detection_type": incident.detection_type.value,
                        "priority": incident.priority.value,
                    }
                )
                if self._should_create_alert(planned.detection_type):
                    alert = await self.alerts.create(
                        AlertCreate(
                            incident_id=incident.id,
                            priority=planned.priority,
                            title=self._alert_title(planned.detection_type, detection),
                            message=(
                                f"{self._detection_display_name(planned.detection_type, detection)} detected "
                                f"on camera {camera.name} with confidence {detection.confidence:.2f}."
                            ),
                        )
                    )
                    alert_id = alert.id
                    await self.outbox.enqueue(
                        {
                            "type": "alert.created",
                            "alert_id": str(alert.id),
                            "incident_id": str(alert.incident_id),
                            "priority": alert.priority.value,
                        }
                    )

                if planned.detection_type is DetectionType.known_person and detection.identity_id:
                    await self.persons.record_recognition(detection.identity_id, occurred_at)

                results.append(
                    DetectionEventResult(
                        incident_id=incident.id,
                        detection_type=planned.detection_type,
                        priority=planned.priority,
                        alert_id=alert_id,
                    )
                )

            for sound_event in sound_events:
                await self.outbox.enqueue(sound_event)

        if scope.owns_transaction:
            await self.outbox.publish_pending()

        return DetectionEventIngestResponse(
            camera_id=payload.camera_id,
            incident_count=len(results),
            alert_count=sum(1 for item in results if item.alert_id is not None),
            ignored_count=len(payload.detections) - len(results),
            results=results,
            ignored_reasons=ignored_reasons,
        )

    @staticmethod
    def _requested_detectors(metadata: dict[str, Any]) -> set[str] | None:
        raw_detectors = metadata.get("requested_detectors")
        if not isinstance(raw_detectors, list):
            return None
        return {
            str(detector).strip().lower()
            for detector in raw_detectors
            if str(detector).strip()
        }

    async def _plan_detections(
        self,
        payload: DetectionEventIngest,
        occurred_at: datetime,
    ) -> tuple[list[PlannedDetection], list[str]]:
        self._validate_request_evidence(payload)
        planned: list[PlannedDetection] = []
        ignored_reasons: list[str] = []

        for detection in payload.detections:
            detection_type = self._resolve_detection_type(detection)
            if not self._supports_event_generation(detection_type):
                ignored_reasons.append(
                    f"Detection type {detection_type.value} is not handled in phase 7."
                )
                continue

            planned_detection = PlannedDetection(
                detection=detection,
                detection_type=detection_type,
                priority=self._priority_for_detection(detection_type),
                recognized_identity=self._recognized_identity_for_detection(detection),
                bounding_boxes=self._serialize_bounding_boxes(detection),
            )
            if await self._is_duplicate_detection(
                camera_id=payload.camera_id,
                planned_detection=planned_detection,
                occurred_at=occurred_at,
                planned_batch=planned,
            ):
                ignored_reasons.append(
                    f"Duplicate {detection_type.value} detection suppressed within the cooldown window."
                )
                continue
            planned.append(planned_detection)

        return planned, ignored_reasons

    def _validate_request_evidence(self, payload: DetectionEventIngest) -> None:
        if payload.snapshot_evidence:
            self.evidence.validate_base64_payload(payload.snapshot_evidence.content_base64)
        if payload.clip_evidence:
            self.evidence.validate_base64_payload(payload.clip_evidence.content_base64)
        for detection in payload.detections:
            if detection.face_image_evidence:
                self.evidence.validate_base64_payload(detection.face_image_evidence.content_base64)

    async def _is_duplicate_detection(
        self,
        *,
        camera_id,
        planned_detection: PlannedDetection,
        occurred_at: datetime,
        planned_batch: list[PlannedDetection],
    ) -> bool:
        recent = await self.incidents.recent_for_detection(
            camera_id=camera_id,
            detection_type=planned_detection.detection_type,
            since=occurred_at - timedelta(seconds=settings.detection_duplicate_window_seconds),
        )
        if not recent:
            return self._matches_planned_duplicate(planned_detection, planned_batch)

        for incident in recent:
            if self._matches_incident_duplicate(planned_detection, incident):
                return True
        return self._matches_planned_duplicate(planned_detection, planned_batch)

    def _matches_incident_duplicate(self, planned_detection: PlannedDetection, incident: Any) -> bool:
        detection = planned_detection.detection
        if detection.track_id and incident.metadata_.get("track_id") == detection.track_id:
            return True
        if detection.identity_id and str(detection.identity_id) == str(
            (incident.recognized_identity or {}).get("identity_id")
        ):
            return True
        if detection.bounding_box and incident.bounding_boxes:
            return self._box_iou(detection.bounding_box.model_dump(), incident.bounding_boxes[0]) >= 0.7
        return False

    def _matches_planned_duplicate(
        self,
        planned_detection: PlannedDetection,
        planned_batch: list[PlannedDetection],
    ) -> bool:
        detection = planned_detection.detection
        for existing in planned_batch:
            if existing.detection_type is not planned_detection.detection_type:
                continue
            if detection.track_id and existing.detection.track_id == detection.track_id:
                return True
            if detection.identity_id and existing.detection.identity_id == detection.identity_id:
                return True
            if detection.bounding_box and existing.bounding_boxes:
                if self._box_iou(detection.bounding_box.model_dump(), existing.bounding_boxes[0]) >= 0.7:
                    return True
        return False

    @staticmethod
    def _box_iou(first: dict, second: dict) -> float:
        left = max(float(first.get("x1", 0)), float(second.get("x1", 0)))
        top = max(float(first.get("y1", 0)), float(second.get("y1", 0)))
        right = min(float(first.get("x2", 0)), float(second.get("x2", 0)))
        bottom = min(float(first.get("y2", 0)), float(second.get("y2", 0)))
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, float(first.get("x2", 0)) - float(first.get("x1", 0))) * max(
            0.0, float(first.get("y2", 0)) - float(first.get("y1", 0))
        )
        second_area = max(0.0, float(second.get("x2", 0)) - float(second.get("x1", 0))) * max(
            0.0, float(second.get("y2", 0)) - float(second.get("y1", 0))
        )
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

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
        return INCIDENT_PRIORITY_POLICY[detection_type]

    @staticmethod
    def _should_create_alert(detection_type: DetectionType) -> bool:
        return detection_type in EMERGENCY_ALERT_PRIORITY_POLICY

    @staticmethod
    def _alert_title(
        detection_type: DetectionType,
        detection: DetectionEventIngestItem | None = None,
    ) -> str:
        return f"{DetectionEventService._detection_display_name(detection_type, detection)} detected"

    @staticmethod
    def _detection_display_name(
        detection_type: DetectionType,
        detection: DetectionEventIngestItem | None,
    ) -> str:
        if detection_type is DetectionType.weapon and detection and detection.bounding_box:
            object_label = (detection.bounding_box.label or "").strip().lower()
            if object_label and object_label not in {"weapon", "face"}:
                return object_label.replace("_", " ").title()
        return detection_type.value.replace("_", " ").title()

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
    ) -> Any:
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
            await self.session.flush()
        return incident
