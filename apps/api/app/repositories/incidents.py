from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.incident import DetectionType, Incident, IncidentPriority, IncidentStatus
from app.schemas.incidents import IncidentCreate, IncidentUpdate


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
        result = await self.session.scalars(query)
        return list(result)

    async def get(self, incident_id: UUID) -> Incident | None:
        return await self.session.get(Incident, incident_id)

    async def create(self, payload: IncidentCreate) -> Incident:
        data = payload.model_dump(exclude={"metadata"})
        data["occurred_at"] = data["occurred_at"] or datetime.now(UTC)
        incident = Incident(**data, metadata_=payload.metadata)
        self.session.add(incident)
        await self.session.commit()
        await self.session.refresh(incident)
        return incident

    async def update(self, incident: Incident, payload: IncidentUpdate) -> Incident:
        updates = payload.model_dump(exclude_unset=True)
        if "metadata" in updates:
            incident.metadata_ = updates.pop("metadata")
        for key, value in updates.items():
            setattr(incident, key, value)
        await self.session.commit()
        await self.session.refresh(incident)
        return incident
