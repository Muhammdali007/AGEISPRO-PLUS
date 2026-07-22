import secrets
from uuid import UUID

from fastapi import Cookie, Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import TokenType, decode_token, hash_token_identifier
from app.db.session import get_db
from app.models.user import User
from app.models.user import UserRole
from app.models.user_session import UserSession
from app.repositories.user_sessions import UserSessionRepository
from app.repositories.users import UserRepository

bearer_scheme = HTTPBearer(auto_error=False)
AegisAccessCookie = "aegispro_access"
AegisRefreshCookie = "aegispro_refresh"


async def get_optional_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    access_cookie: str | None = Cookie(default=None, alias=AegisAccessCookie),
    session: AsyncSession = Depends(get_db),
) -> User | None:
    auth_session = await get_optional_current_session(credentials, access_cookie, session)
    if not auth_session:
        return None
    user = await UserRepository(session).get_by_id(auth_session.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")
    return user


async def get_optional_current_session(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    access_cookie: str | None = Cookie(default=None, alias=AegisAccessCookie),
    session: AsyncSession = Depends(get_db),
) -> UserSession | None:
    raw_token = credentials.credentials if credentials else access_cookie
    if not raw_token:
        return None
    try:
        payload = decode_token(raw_token, TokenType.access)
        UUID(payload["sub"])
        access_jti_digest = hash_token_identifier(payload["jti"])
    except (KeyError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    auth_session = await UserSessionRepository(session).get_active_by_access_jti(access_jti_digest)
    if not auth_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Revoked session")
    if payload.get("sid") != str(auth_session.id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return auth_session


async def get_optional_stream_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    access_cookie: str | None = Cookie(default=None, alias=AegisAccessCookie),
    session: AsyncSession = Depends(get_db),
) -> User | None:
    auth_session = await get_optional_current_session(credentials, access_cookie, session)
    if not auth_session:
        return None
    user = await UserRepository(session).get_by_id(auth_session.user_id)
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive user")
    return user


async def get_current_user(
    current_user: User | None = Depends(get_optional_current_user),
) -> User:
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    return current_user


async def get_current_session(
    current_session: UserSession | None = Depends(get_optional_current_session),
) -> UserSession:
    if not current_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    return current_session


async def get_stream_user(
    current_user: User | None = Depends(get_optional_stream_user),
) -> User:
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    return current_user


def require_roles(*roles: UserRole):
    async def dependency(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return current_user

    return dependency


def require_stream_roles(*roles: UserRole):
    async def dependency(current_user: User = Depends(get_stream_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return current_user

    return dependency


async def require_detection_ingest_access(
    current_user: User | None = Depends(get_optional_current_user),
    service_token: str | None = Header(default=None, alias="X-Service-Token"),
) -> User | None:
    configured_service_token = settings.service_callback_token
    if service_token and configured_service_token:
        if secrets.compare_digest(service_token, configured_service_token):
            return None
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token")

    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing credentials")
    if current_user.role not in {UserRole.administrator, UserRole.supervisor, UserRole.operator}:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    return current_user
