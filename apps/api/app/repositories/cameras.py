from datetime import datetime
from uuid import UUID

from sqlalchemy.orm import selectinload

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera import Camera, CameraStatus
from app.models.camera_secret import CameraSecret
from app.schemas.cameras import CameraCreate, CameraUpdate
from app.services.camera_secrets import CameraSecretManager


class CameraRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.secret_manager = CameraSecretManager()

    async def list(
        self,
        *,
        status: CameraStatus | None = None,
        group: str | None = None,
    ) -> list[Camera]:
        query = select(Camera).options(selectinload(Camera.secret)).order_by(Camera.created_at.desc())
        if status:
            query = query.where(Camera.status == status)
        if group:
            query = query.where(Camera.group == group)
        result = await self.session.scalars(query)
        return list(result)

    async def get(self, camera_id: UUID) -> Camera | None:
        query = select(Camera).options(selectinload(Camera.secret)).where(Camera.id == camera_id)
        return await self.session.scalar(query)

    async def create(self, payload: CameraCreate) -> Camera:
        source_descriptor, source_redacted, rotation_required, secret = self._prepare_source_fields(
            payload.source_type,
            payload.source,
        )
        camera = Camera(
            **payload.model_dump(exclude={"metadata", "source"}),
            source=source_descriptor,
            source_redacted=source_redacted,
            credentials_rotation_required=rotation_required,
            metadata_=payload.metadata,
        )
        if secret:
            camera.secret = CameraSecret(encrypted_source=secret)
        self.session.add(camera)
        await self.session.flush()
        await self.session.refresh(camera)
        await self._load_secret_relationship(camera)
        return camera

    async def update(self, camera: Camera, payload: CameraUpdate) -> Camera:
        updates = payload.model_dump(exclude_unset=True)
        if "metadata" in updates:
            camera.metadata_ = updates.pop("metadata")
        next_source_type = updates.get("source_type", camera.source_type)
        if "source_type" in updates and "source" not in updates:
            raise ValueError("Updating the camera source type requires a replacement source value.")
        if "source" in updates:
            source_descriptor, source_redacted, rotation_required, secret = self._prepare_source_fields(
                next_source_type,
                updates.pop("source"),
            )
            camera.source = source_descriptor
            camera.source_redacted = source_redacted
            camera.credentials_rotation_required = rotation_required
            if secret:
                if camera.secret:
                    camera.secret.encrypted_source = secret
                else:
                    camera.secret = CameraSecret(camera_id=camera.id, encrypted_source=secret)
            else:
                camera.secret = None
        for key, value in updates.items():
            setattr(camera, key, value)
        await self.session.flush()
        await self.session.refresh(camera)
        await self._load_secret_relationship(camera)
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
        await self.session.flush()
        await self.session.refresh(camera)
        return camera

    async def delete(self, camera: Camera) -> None:
        await self.session.delete(camera)
        await self.session.flush()

    async def get_runtime_source(self, camera: Camera) -> str:
        if not camera.source_redacted:
            return camera.source
        await self._load_secret_relationship(camera)
        if not camera.secret:
            raise ValueError("Camera source secret is missing.")
        return self.secret_manager.decrypt_source(camera.secret.encrypted_source)

    def _prepare_source_fields(
        self,
        source_type,
        source: str,
    ) -> tuple[str, bool, bool, str | None]:
        if self.secret_manager.source_requires_secret_storage(source_type):
            return (
                self.secret_manager.build_source_descriptor(source_type, source),
                True,
                self.secret_manager.requires_device_credential_rotation(source_type, source),
                self.secret_manager.encrypt_source(source),
            )
        return source, False, False, None

    async def _load_secret_relationship(self, camera: Camera) -> None:
        await self.session.refresh(camera, attribute_names=["secret"])
