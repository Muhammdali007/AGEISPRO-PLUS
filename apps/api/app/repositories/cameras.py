from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera import Camera, CameraStatus
from app.schemas.cameras import CameraCreate, CameraUpdate


class CameraRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(
        self,
        *,
        status: CameraStatus | None = None,
        group: str | None = None,
    ) -> list[Camera]:
        query = select(Camera).order_by(Camera.created_at.desc())
        if status:
            query = query.where(Camera.status == status)
        if group:
            query = query.where(Camera.group == group)
        result = await self.session.scalars(query)
        return list(result)

    async def get(self, camera_id: UUID) -> Camera | None:
        return await self.session.get(Camera, camera_id)

    async def create(self, payload: CameraCreate) -> Camera:
        camera = Camera(
            **payload.model_dump(exclude={"metadata"}),
            metadata_=payload.metadata,
        )
        self.session.add(camera)
        await self.session.commit()
        await self.session.refresh(camera)
        return camera

    async def update(self, camera: Camera, payload: CameraUpdate) -> Camera:
        updates = payload.model_dump(exclude_unset=True)
        if "metadata" in updates:
            camera.metadata_ = updates.pop("metadata")
        for key, value in updates.items():
            setattr(camera, key, value)
        await self.session.commit()
        await self.session.refresh(camera)
        return camera

    async def apply_health(
        self,
        camera: Camera,
        *,
        status: CameraStatus,
        checked_at: datetime,
        last_seen_at: datetime | None,
    ) -> Camera:
        camera.status = status
        camera.health_checked_at = checked_at
        if last_seen_at is not None:
            camera.last_seen_at = last_seen_at
        await self.session.commit()
        await self.session.refresh(camera)
        return camera

    async def delete(self, camera: Camera) -> None:
        await self.session.delete(camera)
        await self.session.commit()
