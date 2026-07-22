from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user_session import UserSession


class UserSessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        *,
        user_id: UUID,
        access_jti_digest: str,
        refresh_jti_digest: str,
        expires_at: datetime,
    ) -> UserSession:
        user_session = UserSession(
            user_id=user_id,
            access_jti_digest=access_jti_digest,
            refresh_jti_digest=refresh_jti_digest,
            expires_at=expires_at,
        )
        self.session.add(user_session)
        await self.session.flush()
        await self.session.refresh(user_session)
        return user_session

    async def get_active_by_access_jti(self, digest: str) -> UserSession | None:
        return await self._get_active_by_digest(UserSession.access_jti_digest, digest)

    async def get_active_by_refresh_jti(self, digest: str) -> UserSession | None:
        return await self._get_active_by_digest(UserSession.refresh_jti_digest, digest)

    async def rotate(
        self,
        user_session: UserSession,
        *,
        access_jti_digest: str,
        refresh_jti_digest: str,
        expires_at: datetime,
    ) -> UserSession:
        user_session.access_jti_digest = access_jti_digest
        user_session.refresh_jti_digest = refresh_jti_digest
        user_session.expires_at = expires_at
        user_session.updated_at = datetime.now(UTC)
        await self.session.flush()
        await self.session.refresh(user_session)
        return user_session

    async def revoke(self, user_session: UserSession) -> None:
        if user_session.revoked_at:
            return
        user_session.revoked_at = datetime.now(UTC)
        user_session.updated_at = datetime.now(UTC)
        await self.session.flush()

    async def _get_active_by_digest(self, column, digest: str) -> UserSession | None:
        now = datetime.now(UTC)
        return await self.session.scalar(
            select(UserSession).where(
                column == digest,
                UserSession.revoked_at.is_(None),
                UserSession.expires_at > now,
            )
        )
