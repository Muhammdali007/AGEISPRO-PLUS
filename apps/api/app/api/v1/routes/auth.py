from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models.user import User
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.users import UserRepository
from app.schemas.auth import LoginRequest, RefreshRequest, SignupRequest, TokenPair
from app.schemas.users import UserRead
from app.services.audit_logs import AuditLogService
from app.services.auth import AuthError, AuthService

router = APIRouter()


def get_auth_service(session: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(UserRepository(session))


@router.post("/login", response_model=TokenPair)
async def login(
    payload: LoginRequest,
    auth: AuthService = Depends(get_auth_service),
    session: AsyncSession = Depends(get_db),
) -> TokenPair:
    try:
        user = await auth.authenticate(payload.email, payload.password)
    except AuthError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    await AuditLogService(AuditLogRepository(session)).record(
        actor=user,
        action="auth.login",
        resource_type="session",
        resource_id=str(user.id),
        metadata={"email": user.email.lower()},
    )
    return TokenPair(**auth.issue_tokens(user))


@router.post("/signup", response_model=TokenPair, status_code=status.HTTP_201_CREATED)
async def signup(payload: SignupRequest, auth: AuthService = Depends(get_auth_service)) -> TokenPair:
    try:
        user = await auth.register(payload)
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
    return TokenPair(**auth.issue_tokens(user))


@router.post("/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest, auth: AuthService = Depends(get_auth_service)) -> TokenPair:
    try:
        return TokenPair(**await auth.refresh(payload.refresh_token))
    except AuthError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")


@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
