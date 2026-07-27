from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from uuid import UUID

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.camera import Camera
from app.models.incident import Incident
from app.models.video_rag import VideoRagChunk, VideoRagIndex, VideoRagIndexStatus
from app.services.evidence_storage import EvidenceStorageService
from app.services.ollama import OllamaClient, OllamaUnavailableError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SampledEvidence:
    images_base64: list[str]
    offsets: list[float]


class VideoRagIndexer:
    def __init__(self, ollama: OllamaClient | None = None) -> None:
        self.ollama = ollama or OllamaClient()
        self.storage = EvidenceStorageService()

    async def reconcile(self, session: AsyncSession) -> int:
        rows = (
            await session.execute(
                select(Incident, Camera)
                .join(Camera, Camera.id == Incident.camera_id)
                .where(
                    Incident.archived_at.is_(None),
                    or_(Incident.clip_path.is_not(None), Incident.snapshot_path.is_not(None)),
                )
            )
        ).all()
        changed = 0
        visible_ids: set[UUID] = set()
        existing_states = {
            state.incident_id: state
            for state in await session.scalars(select(VideoRagIndex))
        }
        for incident, camera in rows:
            visible_ids.add(incident.id)
            fingerprint = self.fingerprint(incident, camera)
            state = existing_states.get(incident.id)
            if state is None:
                session.add(
                    VideoRagIndex(
                        incident_id=incident.id,
                        status=VideoRagIndexStatus.queued,
                        evidence_fingerprint=fingerprint,
                    )
                )
                changed += 1
            elif state.evidence_fingerprint != fingerprint:
                state.status = VideoRagIndexStatus.queued
                state.attempts = 0
                state.available_at = datetime.now(UTC)
                state.lease_expires_at = None
                state.last_error = None
                state.evidence_fingerprint = fingerprint
                changed += 1

        for state in existing_states.values():
            if state.incident_id not in visible_ids:
                await session.execute(
                    delete(VideoRagChunk).where(VideoRagChunk.incident_id == state.incident_id)
                )
                await session.delete(state)
                changed += 1
        await session.commit()
        return changed

    async def claim_next(self, session: AsyncSession) -> UUID | None:
        now = datetime.now(UTC)
        query = (
            select(VideoRagIndex)
            .join(Incident, Incident.id == VideoRagIndex.incident_id)
            .where(
                VideoRagIndex.attempts < settings.video_rag_max_attempts,
                VideoRagIndex.available_at <= now,
                or_(
                    VideoRagIndex.status.in_(
                        (VideoRagIndexStatus.queued, VideoRagIndexStatus.failed)
                    ),
                    (
                        (VideoRagIndex.status == VideoRagIndexStatus.processing)
                        & (VideoRagIndex.lease_expires_at < now)
                    ),
                ),
            )
            # Keep live monitoring useful while a historical backfill is draining.
            # Newly-created incidents should not wait behind the entire old queue.
            .order_by(Incident.occurred_at.desc(), VideoRagIndex.available_at.asc())
            .limit(1)
        )
        if session.bind and session.bind.dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        state = await session.scalar(query)
        if state is None:
            return None
        state.status = VideoRagIndexStatus.processing
        state.lease_expires_at = now + timedelta(seconds=settings.video_rag_lease_seconds)
        state.last_error = None
        await session.commit()
        return state.incident_id

    async def process(self, incident_id: UUID) -> None:
        try:
            async with AsyncSessionLocal() as session:
                row = (
                    await session.execute(
                        select(Incident, Camera)
                        .join(Camera, Camera.id == Incident.camera_id)
                        .where(Incident.id == incident_id, Incident.archived_at.is_(None))
                    )
                ).one_or_none()
                if row is None:
                    await self._discard(session, incident_id)
                    return
                incident, camera = row
                sampled = await asyncio.to_thread(self.sample_evidence, incident)
                description = await self._describe_evidence(sampled, incident_id)
                chunks = self.build_chunks(incident, camera, description, sampled.offsets)
                embeddings = await self.ollama.embed([chunk[1] for chunk in chunks])

                current_incident = await session.scalar(
                    select(Incident)
                    .where(Incident.id == incident_id)
                    .with_for_update()
                    .execution_options(populate_existing=True)
                )
                state = await session.get(VideoRagIndex, incident_id)
                if current_incident is None or current_incident.archived_at is not None or state is None:
                    await self._discard(session, incident_id)
                    return
                await session.execute(
                    delete(VideoRagChunk).where(VideoRagChunk.incident_id == incident_id)
                )
                for (kind, content, start, end, metadata), embedding in zip(
                    chunks, embeddings, strict=True
                ):
                    session.add(
                        VideoRagChunk(
                            incident_id=incident_id,
                            kind=kind,
                            content=content,
                            clip_start_seconds=start,
                            clip_end_seconds=end,
                            metadata_=metadata,
                            embedding=embedding,
                        )
                    )
                state.status = VideoRagIndexStatus.ready
                state.attempts = 0
                state.lease_expires_at = None
                state.last_error = None
                state.vision_model = settings.video_rag_vision_model
                state.embedding_model = settings.video_rag_embedding_model
                state.indexed_at = datetime.now(UTC)
                state.evidence_fingerprint = self.fingerprint(current_incident, camera)
                await session.commit()
        except asyncio.CancelledError:
            await asyncio.shield(self._requeue_interrupted(incident_id))
            raise
        except OllamaUnavailableError as exc:
            logger.warning("Video RAG models unavailable while indexing incident %s", incident_id)
            await self._mark_model_unavailable(incident_id, exc)
        except Exception as exc:
            logger.exception("Video RAG indexing failed for incident %s", incident_id)
            await self._mark_failed(incident_id, exc)

    async def _describe_evidence(
        self,
        sampled: SampledEvidence,
        incident_id: UUID,
    ) -> dict[str, object]:
        if not settings.video_rag_visual_indexing_enabled:
            return self._metadata_only_description()
        try:
            description = await self.ollama.describe_frames(
                sampled.images_base64, sampled.offsets
            )
            return {**description, "visual_available": True}
        except OllamaUnavailableError:
            # Snapshot retrieval and authoritative detector metadata remain
            # useful evidence when a CPU-only vision model is too slow. Let the
            # fast embedding model index that evidence instead of leaving the
            # incident queued forever.
            logger.warning(
                "Vision description unavailable for incident %s; indexing authoritative metadata",
                incident_id,
            )
            return self._metadata_only_description()

    @staticmethod
    def _metadata_only_description() -> dict[str, object]:
        return {
            "summary": (
                "Visual description is unavailable; the retained snapshot or clip "
                "is available for operator review."
            ),
            "observations": [],
            "visual_available": False,
        }

    def sample_evidence(self, incident: Incident) -> SampledEvidence:
        if incident.clip_path:
            clip = self.storage.ensure_exists(incident.clip_path)
            sampled = self._sample_clip(clip)
            if sampled.images_base64:
                return sampled
        if incident.snapshot_path:
            snapshot = self.storage.ensure_exists(incident.snapshot_path)
            normalized = self._normalize_image_bytes(self._read_file_bounded(snapshot))
            return SampledEvidence(
                images_base64=[base64.b64encode(normalized).decode("ascii")],
                offsets=[0.0],
            )
        raise FileNotFoundError("Incident has no readable clip or snapshot evidence")

    @staticmethod
    def _read_file_bounded(path: Path) -> bytes:
        script = "import pathlib,sys; sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())"
        try:
            result = subprocess.run(
                [sys.executable, "-c", script, str(path)],
                check=True,
                capture_output=True,
                timeout=settings.video_rag_evidence_read_timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                "Evidence file could not be read before the timeout; it may be an online-only placeholder"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
            raise OSError(detail or "Evidence file could not be read") from exc
        return result.stdout

    @staticmethod
    def _normalize_image_bytes(image_bytes: bytes, max_dimension: int = 768) -> bytes:
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("Video RAG image indexing requires OpenCV and NumPy") from exc

        frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Incident snapshot is not a readable image")
        encoded = VideoRagIndexer._encode_frame(frame, max_dimension=max_dimension)
        if encoded is None:
            raise ValueError("Incident snapshot could not be normalized")
        return encoded

    @staticmethod
    def _encode_frame(frame: object, *, max_dimension: int = 768) -> bytes | None:
        import cv2

        height, width = frame.shape[:2]  # type: ignore[union-attr]
        longest = max(height, width)
        if longest > max_dimension:
            scale = max_dimension / longest
            frame = cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return encoded.tobytes() if encoded_ok else None

    @staticmethod
    def _sample_clip(path: Path) -> SampledEvidence:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video RAG clip indexing requires OpenCV") from exc

        capture = cv2.VideoCapture(str(path))
        try:
            frame_count = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
            if frame_count <= 0:
                return SampledEvidence([], [])
            sample_count = min(settings.video_rag_max_frames, frame_count)
            positions = sorted(
                {round(index * (frame_count - 1) / max(1, sample_count - 1)) for index in range(sample_count)}
            )
            images: list[str] = []
            offsets: list[float] = []
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok:
                    continue
                encoded = VideoRagIndexer._encode_frame(frame)
                if encoded is not None:
                    images.append(base64.b64encode(encoded).decode("ascii"))
                    offsets.append(position / fps)
            return SampledEvidence(images, offsets)
        finally:
            capture.release()

    @staticmethod
    def build_chunks(
        incident: Incident,
        camera: Camera,
        description: dict[str, object],
        offsets: list[float],
    ) -> list[tuple[str, str, float | None, float | None, dict[str, object]]]:
        subtype = next(
            (
                str(box.get("label"))
                for box in incident.bounding_boxes
                if box.get("label") and box.get("label") != "face"
            ),
            incident.detection_type.value,
        )
        identity = (incident.recognized_identity or {}).get("identity_label")
        authoritative = (
            f"Incident {incident.id}; detector event {incident.detection_type.value}; subtype {subtype}; "
            f"camera {camera.name}; location {camera.location or 'unspecified'}; "
            f"occurred {incident.occurred_at.isoformat()}; confidence {incident.confidence:.3f}; "
            f"status {incident.status.value}; recognized identity {identity or 'none'}; "
            f"operator notes {incident.operator_notes or 'none'}."
        )
        visual_available = description.get("visual_available") is not False
        summary = str(description.get("summary") or "No visual summary was produced.")
        chunks = [
            (
                "summary",
                (
                    f"{authoritative} Model-generated visual summary: {summary}"
                    if visual_available
                    else f"{authoritative} Evidence availability: {summary}"
                ),
                None,
                None,
                {
                    "authoritative": True,
                    "visual_observation": visual_available,
                    "visual_description_available": visual_available,
                },
            )
        ]
        observations = description.get("observations")
        if isinstance(observations, list):
            for observation in observations:
                if not isinstance(observation, dict):
                    continue
                try:
                    frame_index = int(observation.get("frame_index", 0)) - 1
                except (TypeError, ValueError):
                    continue
                if frame_index < 0 or frame_index >= len(offsets):
                    continue
                text = str(observation.get("description") or "").strip()
                if not text:
                    continue
                offset = offsets[frame_index]
                chunks.append(
                    (
                        "observation",
                        f"{authoritative} Model-generated observation at {offset:.2f}s: {text}",
                        offset,
                        offset,
                        {"authoritative": False, "visual_observation": True, "frame_index": frame_index + 1},
                    )
                )
        return chunks

    def fingerprint(self, incident: Incident, camera: Camera) -> str:
        digest = hashlib.sha256()
        fields = {
            "incident_updated_at": incident.updated_at.isoformat(),
            "camera_updated_at": camera.updated_at.isoformat(),
            "clip_path": incident.clip_path,
            "snapshot_path": incident.snapshot_path,
            "notes": incident.operator_notes,
            "identity": incident.recognized_identity,
            "boxes": incident.bounding_boxes,
        }
        digest.update(json.dumps(fields, sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()

    async def _mark_failed(self, incident_id: UUID, exc: Exception) -> None:
        async with AsyncSessionLocal() as session:
            state = await session.get(VideoRagIndex, incident_id)
            if state:
                state.status = VideoRagIndexStatus.failed
                state.attempts += 1
                state.lease_expires_at = None
                state.last_error = str(exc)[:1000]
                state.available_at = datetime.now(UTC) + timedelta(
                    seconds=min(300, 2 ** state.attempts * 5)
                )
                await session.commit()

    async def _mark_model_unavailable(self, incident_id: UUID, exc: Exception) -> None:
        async with AsyncSessionLocal() as session:
            state = await session.get(VideoRagIndex, incident_id)
            if state:
                state.status = VideoRagIndexStatus.queued
                state.lease_expires_at = None
                state.last_error = str(exc)[:1000]
                state.available_at = datetime.now(UTC) + timedelta(
                    seconds=max(60, settings.video_rag_worker_interval_seconds)
                )
                await session.commit()

    async def _requeue_interrupted(self, incident_id: UUID) -> None:
        async with AsyncSessionLocal() as session:
            state = await session.get(VideoRagIndex, incident_id)
            if state and state.status == VideoRagIndexStatus.processing:
                state.status = VideoRagIndexStatus.queued
                state.lease_expires_at = None
                state.available_at = datetime.now(UTC)
                state.last_error = "Indexing was interrupted and has been requeued"
                await session.commit()

    @staticmethod
    async def _discard(session: AsyncSession, incident_id: UUID) -> None:
        await session.execute(delete(VideoRagChunk).where(VideoRagChunk.incident_id == incident_id))
        state = await session.get(VideoRagIndex, incident_id)
        if state:
            await session.delete(state)
        await session.commit()


async def run_video_rag_worker() -> None:
    indexer = VideoRagIndexer()
    next_reconciliation = 0.0
    logger.info("Video RAG worker started")
    while True:
        try:
            if settings.video_rag_enabled:
                async with AsyncSessionLocal() as session:
                    now = monotonic()
                    if now >= next_reconciliation:
                        await indexer.reconcile(session)
                        next_reconciliation = now + settings.video_rag_worker_interval_seconds
                    incident_id = await indexer.claim_next(session)
                if incident_id:
                    await indexer.process(incident_id)
                    # Drain queued work continuously. The interval is an idle
                    # polling/reconciliation interval, not a per-incident delay.
                    continue
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Video RAG worker iteration failed")
        await asyncio.sleep(settings.video_rag_worker_interval_seconds)
