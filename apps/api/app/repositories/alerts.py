from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.alert import Alert, AlertStatus
from app.models.incident import IncidentPriority
from app.schemas.alerts import AlertCreate


class AlertRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(
        self,
        *,
        status: AlertStatus | None = None,
        priority: IncidentPriority | None = None,
        incident_id: UUID | None = None,
    ) -> list[Alert]:
        query = select(Alert).order_by(Alert.created_at.desc())
        if status:
            query = query.where(Alert.status == status)
        if priority:
            query = query.where(Alert.priority == priority)
        if incident_id:
            query = query.where(Alert.incident_id == incident_id)
        result = await self.session.scalars(query)
        return list(result)

    async def get(self, alert_id: UUID) -> Alert | None:
        return await self.session.get(Alert, alert_id)

    async def count_since(self, *, started_at: datetime, status: AlertStatus | None = None) -> int:
        query = select(func.count(Alert.id)).where(Alert.created_at >= started_at)
        if status:
            query = query.where(Alert.status == status)
        return int((await self.session.execute(query)).scalar_one() or 0)

    async def count_all(self) -> int:
        return int((await self.session.execute(select(func.count()).select_from(Alert))).scalar_one())

    async def create(self, payload: AlertCreate) -> Alert:
        alert = Alert(**payload.model_dump())
        self.session.add(alert)
        await self.session.commit()
        await self.session.refresh(alert)
        return alert

    async def acknowledge(self, alert: Alert, user_id: UUID) -> Alert:
        alert.acknowledged = True
        alert.status = AlertStatus.acknowledged
        alert.acknowledged_by_id = user_id
        alert.acknowledged_at = datetime.now(UTC)
        await self.session.commit()
        await self.session.refresh(alert)
        return alert

    async def clear(self, alert: Alert) -> Alert:
        alert.status = AlertStatus.cleared
        await self.session.commit()
        await self.session.refresh(alert)
        return alert
