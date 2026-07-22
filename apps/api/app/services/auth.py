from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from app.core.config import settings
from app.core.security import TokenType, create_token, decode_token, hash_token_identifier, verify_password
from app.models.user import User, UserRole
from app.models.user_session import UserSession
from app.repositories.user_sessions import UserSessionRepository
from app.repositories.users import UserRepository
from app.schemas.auth import SignupRequest
from app.schemas.users import UserCreate


class AuthError(ValueError):
    pass


class AuthService:
    def __init__(self, users: UserRepository, sessions: UserSessionRepository | None = None) -> None:
        self.users = users
        self.sessions = sessions or UserSessionRepository(users.session)

    async def register(self, payload: SignupRequest) -> User:
        return await self.users.create(
            UserCreate(
                email=payload.email,
                full_name=payload.full_name,
                password=payload.password,
                role=UserRole.viewer,
                is_active=False,
            )
        )

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.users.get_by_email(email)
        if not user or not user.is_active or not verify_password(password, user.password_hash):
            raise AuthError("Invalid email or password")
        return user

    async def issue_tokens(self, user: User) -> dict[str, str]:
        expires_at = datetime.now(UTC) + timedelta(days=settings.refresh_token_days)
        user_session = await self.sessions.create(
            user_id=user.id,
            access_jti_digest=hash_token_identifier(str(uuid4())),
            refresh_jti_digest=hash_token_identifier(str(uuid4())),
            expires_at=expires_at,
        )
        return await self._rotate_session_tokens(user, user_session, expires_at=expires_at)

    async def refresh(self, refresh_token: str) -> dict[str, str]:
        try:
            payload = decode_token(refresh_token, TokenType.refresh)
            user_id = UUID(payload["sub"])
            refresh_jti_digest = hash_token_identifier(payload["jti"])
        except (KeyError, ValueError) as exc:
            raise AuthError("Invalid refresh token") from exc

        user_session = await self.sessions.get_active_by_refresh_jti(refresh_jti_digest)
        if not user_session or user_session.user_id != user_id:
            raise AuthError("Invalid refresh token")

        user = await self.users.get_by_id(user_id)
        if not user or not user.is_active:
            raise AuthError("Invalid refresh token")
        return await self._rotate_session_tokens(user, user_session, expires_at=user_session.expires_at)

    async def revoke_refresh_token(self, refresh_token: str) -> None:
        try:
            payload = decode_token(refresh_token, TokenType.refresh)
            refresh_jti_digest = hash_token_identifier(payload["jti"])
        except (KeyError, ValueError) as exc:
            raise AuthError("Invalid refresh token") from exc

        user_session = await self.sessions.get_active_by_refresh_jti(refresh_jti_digest)
        if user_session:
            await self.sessions.revoke(user_session)

    async def revoke_session(self, user_session: UserSession) -> None:
        await self.sessions.revoke(user_session)

    async def _rotate_session_tokens(
        self,
        user: User,
        user_session: UserSession,
        *,
        expires_at: datetime,
    ) -> dict[str, str]:
        access_token = create_token(user.id, user.role.value, TokenType.access, session_id=user_session.id)
        refresh_token = create_token(user.id, user.role.value, TokenType.refresh, session_id=user_session.id)
        access_payload = decode_token(access_token, TokenType.access)
        refresh_payload = decode_token(refresh_token, TokenType.refresh)
        await self.sessions.rotate(
            user_session,
            access_jti_digest=hash_token_identifier(access_payload["jti"]),
            refresh_jti_digest=hash_token_identifier(refresh_payload["jti"]),
            expires_at=expires_at,
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
        }
