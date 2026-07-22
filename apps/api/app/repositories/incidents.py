from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertStatus
from app.models.incident import (
    DetectionType,
    Incident,
    IncidentPriority,
    IncidentRetentionClass,
    IncidentStatus,
)
from app.schemas.incidents import IncidentCreate, IncidentUpdate
from app.services.incident_retention_policy import (
    compute_incident_retention_expiry,
    resolve_incident_retention_class,
)


class IncidentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(
        self,
        *,
        camera_id: UUID | None = None,
        status: IncidentStatus | None = None,
        detection_type: DetectionType | None = None,
        priority: IncidentPriority | None = None,
        assigned_user_id: UUID | None = None,
        include_archived: bool = False,
    ) -> list[Incident]:
        query = select(Incident).order_by(Incident.occurred_at.desc())
        if camera_id:
            query = query.where(Incident.camera_id == camera_id)
        if status:
            query = query.where(Incident.status == status)
        if detection_type:
            query = query.where(Incident.detection_type == detection_type)
        if priority:
            query = query.where(Incident.priority == priority)
        if assigned_user_id:
            query = query.where(Incident.assigned_user_id == assigned_user_id)
        query = self._apply_visibility(query, include_archived=include_archived)
        result = await self.session.scalars(query)
        return list(result)

    async def get(
        self,
        incident_id: UUID,
        *,
        include_archived: bool = False,
    ) -> Incident | None:
        query = select(Incident).where(Incident.id == incident_id)
        query = self._apply_visibility(query, include_archived=include_archived)
        return await self.session.scalar(query)

    async def recent_for_detection(
        self,
        *,
        camera_id: UUID,
        detection_type: DetectionType,
        since: datetime,
    ) -> list[Incident]:
        result = await self.session.scalars(
            select(Incident)
            .where(
                Incident.camera_id == camera_id,
                Incident.detection_type == detection_type,
                Incident.occurred_at >= since,
                Incident.archived_at.is_(None),
            )
            .order_by(Incident.occurred_at.desc())
            .limit(20)
        )
        return list(result)

    async def recent_for_camera(
        self,
        *,
        camera_id: UUID,
        since: datetime,
        limit: int = 20,
    ) -> list[Incident]:
        result = await self.session.scalars(
            select(Incident)
            .where(
                Incident.camera_id == camera_id,
                Incident.occurred_at >= since,
                Incident.archived_at.is_(None),
            )
            .order_by(Incident.occurred_at.desc())
            .limit(limit)
        )
        return list(result)

    async def summary_since(self, started_at: datetime) -> tuple[int, float]:
        result = await self.session.execute(
            select(func.count(Incident.id), func.avg(Incident.confidence)).where(
                Incident.occurred_at >= started_at
            )
        )
        total, average = result.one()
        return int(total or 0), float(average or 0.0)

    async def timestamps_since(self, started_at: datetime) -> list[datetime]:
        result = await self.session.scalars(
            select(Incident.occurred_at)
            .where(Incident.occurred_at >= started_at)
            .order_by(Incident.occurred_at.asc())
        )
        return list(result)

    async def detection_mix_since(self, started_at: datetime) -> list[tuple[str, int]]:
        rows = await self.session.execute(
            select(Incident.detection_type, func.count(Incident.id))
            .where(Incident.occurred_at >= started_at)
            .group_by(Incident.detection_type)
            .order_by(func.count(Incident.id).desc(), Incident.detection_type.asc())
        )
        return [(detection_type.value, int(count)) for detection_type, count in rows.all()]

    async def count_all(self) -> int:
        return int((await self.session.execute(select(func.count()).select_from(Incident))).scalar_one())

    async def create(self, payload: IncidentCreate) -> Incident:
        data = payload.model_dump(exclude={"metadata"})
        data["occurred_at"] = data["occurred_at"] or datetime.now(UTC)
        data["retention_class"] = resolve_incident_retention_class(
            requested=data.get("retention_class"),
            priority=data["priority"],
        )
        data["retention_expires_at"] = compute_incident_retention_expiry(
            retention_class=data["retention_class"],
            occurred_at=data["occurred_at"],
        )
        incident = Incident(**data, metadata_=payload.metadata)
        self.session.add(incident)
        await self.session.flush()
        await self.session.refresh(incident)
        return incident

    async def update(self, incident: Incident, payload: IncidentUpdate) -> Incident:
        updates = payload.model_dump(exclude_unset=True)
        if "metadata" in updates:
            incident.metadata_ = updates.pop("metadata")

        if "retention_class" in updates:
            requested_retention_class = updates["retention_class"]
            incident.retention_class = resolve_incident_retention_class(
                requested=requested_retention_class,
                priority=incident.priority,
            )
            incident.retention_expires_at = compute_incident_retention_expiry(
                retention_class=incident.retention_class,
                occurred_at=incident.occurred_at,
            )
            updates.pop("retention_class")

        for key, value in updates.items():
            setattr(incident, key, value)

        if payload.legal_hold is False and "legal_hold_reason" not in updates:
            incident.legal_hold_reason = None

        await self.session.flush()
        await self.session.refresh(incident)
        return incident

    async def archive(self, incident: Incident, *, reference_time: datetime | None = None) -> Incident:
        timestamp = reference_time or datetime.now(UTC)
        if incident.archived_at is None:
            incident.archived_at = timestamp
        if incident.deletion_requested_at is None:
            incident.deletion_requested_at = timestamp
        incident.deletion_error = None
        incident.deletion_started_at = None
        await self.session.execute(
            update(Alert)
            .where(Alert.incident_id == incident.id)
            .values(status=AlertStatus.cleared, updated_at=timestamp)
        )
        await self.session.flush()
        await self.session.refresh(incident)
        return incident

    async def archive_expired(
        self,
        *,
        reference_time: datetime,
        limit: int = 200,
    ) -> list[Incident]:
        incidents = await self.list_retention_candidates(reference_time=reference_time, limit=limit)
        for incident in incidents:
            await self.archive(incident, reference_time=reference_time)
        return incidents

    async def list_retention_candidates(
        self,
        *,
        reference_time: datetime,
        limit: int = 200,
    ) -> list[Incident]:
        result = await self.session.scalars(
            select(Incident)
            .where(
                Incident.archived_at.is_(None),
                Incident.legal_hold.is_(False),
                Incident.retention_class != IncidentRetentionClass.manual,
                Incident.retention_expires_at.is_not(None),
                Incident.retention_expires_at <= reference_time,
                Incident.priority != IncidentPriority.critical,
                Incident.status.in_((IncidentStatus.resolved, IncidentStatus.dismissed)),
            )
            .order_by(Incident.retention_expires_at.asc(), Incident.occurred_at.asc())
            .limit(limit)
        )
        return list(result)

    async def list_pending_evidence_deletions(self, *, limit: int = 200) -> list[Incident]:
        result = await self.session.scalars(
            select(Incident)
            .where(
                Incident.archived_at.is_not(None),
                Incident.legal_hold.is_(False),
                Incident.deletion_requested_at.is_not(None),
                Incident.deletion_completed_at.is_(None),
                Incident.priority != IncidentPriority.critical,
                Incident.status.in_((IncidentStatus.resolved, IncidentStatus.dismissed)),
            )
            .order_by(Incident.deletion_requested_at.asc(), Incident.occurred_at.asc())
            .limit(limit)
        )
        return list(result)

    async def mark_deletion_started(
        self,
        incident: Incident,
        *,
        reference_time: datetime | None = None,
    ) -> Incident:
        incident.deletion_started_at = reference_time or datetime.now(UTC)
        incident.deletion_error = None
        await self.session.flush()
        await self.session.refresh(incident)
        return incident

    async def mark_deletion_completed(
        self,
        incident: Incident,
        *,
        reference_time: datetime | None = None,
    ) -> Incident:
        timestamp = reference_time or datetime.now(UTC)
        recognized_identity = dict(incident.recognized_identity or {})
        if "face_image_path" in recognized_identity:
            recognized_identity["face_image_path"] = None
        if recognized_identity:
            incident.recognized_identity = recognized_identity
        incident.snapshot_path = None
        incident.clip_path = None
        incident.deletion_started_at = incident.deletion_started_at or timestamp
        incident.deletion_completed_at = timestamp
        incident.deletion_error = None
        await self.session.flush()
        await self.session.refresh(incident)
        return incident

    async def mark_deletion_failed(self, incident: Incident, *, error: str) -> Incident:
        incident.deletion_error = error[:500]
        await self.session.flush()
        await self.session.refresh(incident)
        return incident

    def _apply_visibility(self, query, *, include_archived: bool):
        if include_archived:
            return query
        return query.where(Incident.archived_at.is_(None))
