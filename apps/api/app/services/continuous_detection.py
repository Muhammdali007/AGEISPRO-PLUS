import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import Any
from uuid import UUID

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.camera import CameraSourceType, CameraStatus
from app.repositories.cameras import CameraRepository
from app.schemas.cameras import CameraDetectionScanRequest
from app.services.camera_detection import CameraDetectionService, ContinuousScanResult


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraJobState:
    inference_fps: int = 1
    pending_runs: int = 0
    running: bool = False
    last_enqueued_at: float = 0.0
    last_completed_at: float = 0.0
    last_hazard_enqueued_at: float = 0.0
    last_recognition_enqueued_at: float = 0.0
    pending_hazards: bool = False
    pending_recognition: bool = False
    dropped_runs: int = 0
    failures: int = 0


class ContinuousDetectionWorker:
    def __init__(self) -> None:
        self._scheduler_task: asyncio.Task | None = None
        self._dispatcher_task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()
        self._states: dict[str, CameraJobState] = {}

    def start(self) -> None:
        if not self._scheduler_task or self._scheduler_task.done():
            self._scheduler_task = asyncio.create_task(
                self._schedule_loop(),
                name="continuous-camera-detection-scheduler",
            )
        if not self._dispatcher_task or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(
                self._dispatch_loop(),
                name="continuous-camera-detection-dispatcher",
            )

    async def stop(self) -> None:
        tasks = [task for task in (self._scheduler_task, self._dispatcher_task) if task]
        for task in tasks:
            task.cancel()
        self._wake_event.set()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        self._scheduler_task = None
        self._dispatcher_task = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "tracked_cameras": len(self._states),
            "running_cameras": sum(state.running for state in self._states.values()),
            "queued_jobs": sum(state.pending_runs for state in self._states.values()),
            "dropped_jobs": sum(state.dropped_runs for state in self._states.values()),
            "failures": sum(state.failures for state in self._states.values()),
            "batch_size": settings.continuous_detection_batch_size,
            "max_pending_per_camera": settings.continuous_detection_max_pending_per_camera,
            "hazard_interval_seconds": settings.continuous_detection_hazard_interval_seconds,
            "recognition_interval_seconds": settings.continuous_detection_recognition_interval_seconds,
        }

    async def _schedule_loop(self) -> None:
        interval_seconds = settings.continuous_detection_scheduler_interval_ms / 1000
        while True:
            try:
                await self._refresh_due_cameras()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Continuous detection scheduling failed")
            await asyncio.sleep(interval_seconds)

    async def _dispatch_loop(self) -> None:
        while True:
            batch_camera_ids = self._claim_batch()
            if not batch_camera_ids:
                # Clear before checking the queue again. A scheduler tick may
                # have queued work between the first check and this clear; if
                # we waited immediately, that wake-up would be lost and a full
                # per-camera queue would never signal the dispatcher again.
                self._wake_event.clear()
                batch_camera_ids = self._claim_batch()
                if not batch_camera_ids:
                    await self._wake_event.wait()
                    continue

            try:
                await self._run_batch(batch_camera_ids)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Continuous detection batch failed for cameras %s", batch_camera_ids)
                self._mark_batch_complete(batch_camera_ids, error="batch execution failed")

    async def _refresh_due_cameras(self) -> None:
        async with AsyncSessionLocal() as session:
            cameras = await CameraRepository(session).list()

        eligible_camera_ids: set[str] = set()
        now = monotonic()
        for camera in cameras:
            if (
                not camera.detection_enabled
                or camera.status == CameraStatus.disabled
                or camera.source_type == CameraSourceType.usb
            ):
                continue

            key = str(camera.id)
            eligible_camera_ids.add(key)
            state = self._states.setdefault(key, CameraJobState())
            state.inference_fps = max(1, camera.inference_fps)
            if now - state.last_enqueued_at < 1.0 / state.inference_fps:
                continue

            state.last_enqueued_at = now
            hazards_due = (
                now - state.last_hazard_enqueued_at
                >= settings.continuous_detection_hazard_interval_seconds
            )
            if hazards_due:
                state.pending_hazards = True
                state.last_hazard_enqueued_at = now
            recognition_due = (
                now - state.last_recognition_enqueued_at
                >= settings.continuous_detection_recognition_interval_seconds
            )
            if recognition_due:
                state.pending_recognition = True
                state.last_recognition_enqueued_at = now
            if state.pending_runs < settings.continuous_detection_max_pending_per_camera:
                state.pending_runs += 1
                self._wake_event.set()
            else:
                state.dropped_runs += 1

        for key in list(self._states):
            if key not in eligible_camera_ids and not self._states[key].running:
                del self._states[key]

    def _claim_batch(self) -> list[object]:
        ready_items = [
            (camera_id, state)
            for camera_id, state in self._states.items()
            if state.pending_runs > 0 and not state.running
        ]
        if not ready_items:
            return []

        # Prefer backlogged work, then the camera that has waited longest.
        # The previous reverse tuple ordering repeatedly favored the most
        # recently completed camera and could starve a slower feed.
        ready_items.sort(
            key=lambda item: (-item[1].pending_runs, item[1].last_completed_at),
        )
        lead_signature = self._lane_signature(ready_items[0][1])
        claimed: list[object] = []
        for camera_id, state in ready_items:
            if self._lane_signature(state) != lead_signature:
                continue
            state.pending_runs -= 1
            state.running = True
            claimed.append(self._parse_camera_id(camera_id))
            if len(claimed) >= settings.continuous_detection_batch_size:
                break
        return claimed

    @staticmethod
    def _lane_signature(state: CameraJobState) -> tuple[bool, bool]:
        """Batch only cameras requiring the same expensive specialist lanes."""
        return state.pending_hazards, state.pending_recognition

    @staticmethod
    def _parse_camera_id(camera_id: str) -> object:
        try:
            return UUID(camera_id)
        except ValueError:
            return camera_id

    async def _run_batch(self, batch_camera_ids: list[object]) -> None:
        requested_detectors = self._requested_detectors_for_batch(batch_camera_ids)
        recognition_enabled = self._recognition_enabled_for_batch(batch_camera_ids)
        for camera_id in batch_camera_ids:
            state = self._states.get(str(camera_id))
            if state is not None:
                state.pending_hazards = False
                state.pending_recognition = False

        async with AsyncSessionLocal() as session:
            results = await CameraDetectionService(session).run_continuous_batch(
                batch_camera_ids,
                CameraDetectionScanRequest(
                    include_evidence=True,
                    requested_detectors=requested_detectors,
                    recognition_enabled=recognition_enabled,
                    occurrence_hint="continuous_monitoring",
                ),
            )
        self._mark_batch_complete(batch_camera_ids, results=results)

    def _requested_detectors_for_batch(self, batch_camera_ids: list[object]) -> list[str]:
        requested = ["weapon", "person"]
        if any(
            self._states.get(str(camera_id), CameraJobState()).pending_hazards
            for camera_id in batch_camera_ids
        ):
            requested.extend(["fire", "smoke"])
        return requested

    def _recognition_enabled_for_batch(self, batch_camera_ids: list[object]) -> bool:
        return any(
            self._states.get(str(camera_id), CameraJobState()).pending_recognition
            for camera_id in batch_camera_ids
        )

    def _mark_batch_complete(
        self,
        batch_camera_ids: list[object],
        *,
        results: list[ContinuousScanResult] | None = None,
        error: str | None = None,
    ) -> None:
        completed_at = monotonic()
        result_by_camera = {
            str(result.camera_id): result
            for result in (results or [])
        }
        for camera_id in batch_camera_ids:
            key = str(camera_id)
            state = self._states.get(key)
            if state is None:
                continue
            state.running = False
            state.last_completed_at = completed_at

            result = result_by_camera.get(key)
            if error or result is None or not result.success:
                state.failures += 1
                logger.warning(
                    "Continuous scan failed for camera %s: %s",
                    camera_id,
                    error or (result.error if result is not None else "no result returned"),
                )

            if state.pending_runs > 0:
                self._wake_event.set()


continuous_detection_worker = ContinuousDetectionWorker()
