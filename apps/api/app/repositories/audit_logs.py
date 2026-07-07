from datetime import datetime
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.audit_log import AuditLog


class AuditLogRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        actor_user_id: UUID | None,
        actor_email: str | None,
        actor_role: str | None,
        action: str,
        resource_type: str,
        resource_id: str | None,
        metadata: dict[str, object] | None = None,
    ) -> AuditLog:
        entry = AuditLog(
            actor_user_id=actor_user_id,
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata_=metadata or {},
        )
        self.session.add(entry)
        await self.session.commit()
        await self.session.refresh(entry)
        return entry

    async def list(
        self,
        *,
        action: str | None = None,
        actor_email: str | None = None,
        resource_type: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[AuditLog], int]:
        query = select(AuditLog)
        total_query = select(func.count()).select_from(AuditLog)

        if action:
            query = query.where(AuditLog.action == action)
            total_query = total_query.where(AuditLog.action == action)
        if actor_email:
            normalized = actor_email.lower()
            query = query.where(AuditLog.actor_email == normalized)
            total_query = total_query.where(AuditLog.actor_email == normalized)
        if resource_type:
            query = query.where(AuditLog.resource_type == resource_type)
            total_query = total_query.where(AuditLog.resource_type == resource_type)
        if date_from:
            query = query.where(AuditLog.created_at >= date_from)
            total_query = total_query.where(AuditLog.created_at >= date_from)
        if date_to:
            query = query.where(AuditLog.created_at <= date_to)
            total_query = total_query.where(AuditLog.created_at <= date_to)

        query = query.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit)
        items = list(await self.session.scalars(query))
        total = int((await self.session.execute(total_query)).scalar_one())
        return items, total
