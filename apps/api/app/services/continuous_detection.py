import asyncio
import logging
from contextlib import suppress
from time import monotonic

from app.db.session import AsyncSessionLocal
from app.models.camera import CameraSourceType, CameraStatus
from app.repositories.cameras import CameraRepository
from app.schemas.cameras import CameraDetectionScanRequest
from app.services.camera_detection import CameraDetectionService


logger = logging.getLogger(__name__)


class ContinuousDetectionWorker:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._last_scan: dict[str, float] = {}

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run(), name="continuous-camera-detection")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            try:
                await self._scan_due_cameras()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Continuous detection cycle failed")
            await asyncio.sleep(0.25)

    async def _scan_due_cameras(self) -> None:
        async with AsyncSessionLocal() as session:
            cameras = await CameraRepository(session).list()
            camera_schedule = [
                (camera.id, max(1, camera.inference_fps))
                for camera in cameras
                if camera.detection_enabled and camera.status != CameraStatus.disabled
                and camera.source_type != CameraSourceType.usb
            ]

        for camera_id, inference_fps in camera_schedule:
            key = str(camera_id)
            now = monotonic()
            if now - self._last_scan.get(key, 0.0) < 1.0 / inference_fps:
                continue
            self._last_scan[key] = now
            try:
                async with AsyncSessionLocal() as session:
                    await CameraDetectionService(session).run_scan(
                        camera_id,
                        CameraDetectionScanRequest(
                            include_evidence=True,
                            requested_detectors=["weapon", "person", "fire", "smoke"],
                            recognition_enabled=True,
                            occurrence_hint="continuous_monitoring",
                        ),
                    )
            except Exception as exc:
                logger.warning("Continuous scan failed for camera %s: %s", camera_id, exc)


continuous_detection_worker = ContinuousDetectionWorker()
