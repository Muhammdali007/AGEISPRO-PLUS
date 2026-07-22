from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.outbox_event import OutboxEvent
from app.services.event_broadcaster import event_broadcaster


class TransactionalOutboxService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def enqueue(self, payload: dict[str, Any]) -> OutboxEvent:
        event_type = str(payload.get("type") or "unknown")
        entry = OutboxEvent(event_type=event_type, payload=payload)
        self.session.add(entry)
        await self.session.flush()
        return entry

    async def publish_pending(self, *, limit: int | None = None) -> int:
        query = select(OutboxEvent).where(OutboxEvent.published_at.is_(None)).order_by(OutboxEvent.created_at.asc())
        if limit is not None:
            query = query.limit(limit)

        pending = list(await self.session.scalars(query))
        published = 0
        for entry in pending:
            await event_broadcaster.publish(entry.payload)
            entry.published_at = datetime.now(UTC)
            await self.session.commit()
            published += 1
        return published
