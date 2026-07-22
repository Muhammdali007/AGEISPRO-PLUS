from fastapi.encoders import jsonable_encoder

from app.core.audit import redact_audit_metadata
from app.models.user import User
from app.repositories.audit_logs import AuditLogRepository


class AuditLogService:
    def __init__(self, repository: AuditLogRepository) -> None:
        self.repository = repository

    async def record(
        self,
        *,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        actor: User | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        await self.repository.create(
            actor_user_id=actor.id if actor else None,
            actor_email=actor.email.lower() if actor else None,
            actor_role=actor.role.value if actor else None,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            metadata=redact_audit_metadata(jsonable_encoder(metadata or {})),
        )
