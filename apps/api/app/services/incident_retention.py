from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from datetime import UTC, datetime
from uuid import UUID

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.incident import IncidentPriority, IncidentStatus
from app.repositories.incidents import IncidentRepository
from app.services.evidence_storage import EvidenceStorageService

logger = logging.getLogger(__name__)


class IncidentRetentionService:
    """Archives expired incident records and removes evidence asynchronously.

    Retention is intentionally two-phase: the database record is soft-deleted by
    setting archive/deletion timestamps, then evidence files are removed by the
    deletion worker. Open, critical, legal-hold, and manual-retention incidents
    are never selected by automatic retention cleanup.
    """

    def __init__(self, storage: EvidenceStorageService | None = None) -> None:
        self.storage = storage or EvidenceStorageService()

    async def purge_expired(self) -> int:
        reference_time = datetime.now(UTC)
        async with AsyncSessionLocal() as session:
            archived_incidents = await IncidentRepository(session).archive_expired(
                reference_time=reference_time
            )
            await session.commit()
        return len(archived_incidents)

    async def process_pending_deletions(self) -> int:
        async with AsyncSessionLocal() as session:
            pending_incidents = await IncidentRepository(session).list_pending_evidence_deletions()

        deleted_count = 0
        for incident in pending_incidents:
            deleted_count += await self._delete_evidence_for_incident(incident.id)
        return deleted_count

    async def _delete_evidence_for_incident(self, incident_id: UUID) -> int:
        async with AsyncSessionLocal() as session:
            incidents = IncidentRepository(session)
            incident = await incidents.get(incident_id, include_archived=True)
            if (
                not incident
                or incident.deletion_completed_at is not None
                or incident.legal_hold
                or incident.priority is IncidentPriority.critical
                or incident.status not in (IncidentStatus.resolved, IncidentStatus.dismissed)
                or incident.deletion_requested_at is None
                or incident.archived_at is None
            ):
                return 0
            await incidents.mark_deletion_started(incident)
            await session.commit()
            camera_id = incident.camera_id

        try:
            self.storage.delete_incident_directory(camera_id=camera_id, incident_id=incident_id)
        except Exception as exc:
            logger.warning("Failed to delete evidence for archived incident %s", incident_id, exc_info=True)
            async with AsyncSessionLocal() as session:
                incidents = IncidentRepository(session)
                pending_incident = await incidents.get(incident_id, include_archived=True)
                if pending_incident:
                    await incidents.mark_deletion_failed(pending_incident, error=str(exc))
                    await session.commit()
            return 0

        async with AsyncSessionLocal() as session:
            incidents = IncidentRepository(session)
            archived_incident = await incidents.get(incident_id, include_archived=True)
            if not archived_incident or archived_incident.deletion_completed_at is not None:
                return 0
            await incidents.mark_deletion_completed(archived_incident)
            await session.commit()
        return 1


class IncidentCleanupWorker:
    def __init__(self, retention_service: IncidentRetentionService | None = None) -> None:
        self.retention_service = retention_service or IncidentRetentionService()
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if not self._task or self._task.done():
            self._task = asyncio.create_task(self._run(), name="incident-retention-cleanup")

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
                await self.retention_service.purge_expired()
                await self.retention_service.process_pending_deletions()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Incident retention cleanup failed")
            await asyncio.sleep(settings.incident_cleanup_interval_seconds)


incident_cleanup_worker = IncidentCleanupWorker()
