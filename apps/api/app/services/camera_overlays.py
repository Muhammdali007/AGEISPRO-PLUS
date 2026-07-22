import asyncio
from datetime import UTC, datetime
from uuid import UUID

from app.core.config import settings
from app.schemas.cameras import (
    CameraDetectionOverlayRead,
    CameraDetectionOverlayResponse,
    CameraDetectionScanSummary,
)


class CameraOverlayStore:
    def __init__(self) -> None:
        self._items: dict[UUID, CameraDetectionOverlayResponse] = {}
        self._lock = asyncio.Lock()

    async def publish(
        self,
        camera_id: UUID,
        detections: list[CameraDetectionScanSummary],
    ) -> CameraDetectionOverlayResponse:
        now = datetime.now(UTC)
        camera_overlays = [
            CameraDetectionOverlayRead(
                **detection.model_dump(),
                incident_id=None,
                occurred_at=now,
            )
            for detection in detections
        ]

        async with self._lock:
            previous = self._items.get(camera_id)
            overlays = self._with_person_grace(previous, camera_overlays, now)
            response = CameraDetectionOverlayResponse(
                camera_id=camera_id,
                generated_at=now,
                overlays=overlays,
            )
            self._items[camera_id] = response
            return response

    async def get(self, camera_id: UUID) -> CameraDetectionOverlayResponse | None:
        async with self._lock:
            response = self._items.get(camera_id)
            if response is None:
                return None
            if self._is_expired(response.generated_at, settings.camera_overlay_ttl_seconds):
                self._items.pop(camera_id, None)
                return None
            return response

    def reset(self) -> None:
        self._items.clear()

    def _with_person_grace(
        self,
        previous: CameraDetectionOverlayResponse | None,
        current: list[CameraDetectionOverlayRead],
        now: datetime,
    ) -> list[CameraDetectionOverlayRead]:
        if previous is None:
            return current

        current_keys = {self._overlay_key(overlay) for overlay in current}
        bridged_people = [
            overlay
            for overlay in previous.overlays
            if self._is_person_overlay(overlay)
            and self._overlay_key(overlay) not in current_keys
            and not self._is_expired(overlay.occurred_at, settings.camera_overlay_person_grace_seconds)
        ]
        return current + bridged_people

    @staticmethod
    def _is_person_overlay(overlay: CameraDetectionOverlayRead) -> bool:
        labels = {
            overlay.detection_type.lower(),
            (overlay.bounding_box.label or "").lower() if overlay.bounding_box else "",
        }
        return bool(labels & {"person", "known_person", "unknown_person"})

    @staticmethod
    def _overlay_key(overlay: CameraDetectionOverlayRead) -> tuple[str, str | None]:
        return overlay.detection_type, overlay.track_id

    @staticmethod
    def _is_expired(occurred_at: datetime, ttl_seconds: float) -> bool:
        return (datetime.now(UTC) - occurred_at).total_seconds() > ttl_seconds


camera_overlay_store = CameraOverlayStore()
