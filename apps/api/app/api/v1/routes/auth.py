from fastapi import APIRouter, Body, Cookie, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import AegisAccessCookie, AegisRefreshCookie, get_current_session, get_current_user
from app.core.config import settings
from app.db.transactions import transaction_scope
from app.db.session import get_db
from app.models.user import User
from app.models.user_session import UserSession
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.user_sessions import UserSessionRepository
from app.repositories.users import UserRepository
from app.schemas.auth import (
    ForgotPasswordRequest,
    LoginRequest,
    PasswordResetRequest,
    PasswordResetResponse,
    RefreshRequest,
    SignupRequest,
    TokenPair,
)
from app.schemas.users import UserRead
from app.services.audit_logs import AuditLogService
from app.services.auth import AuthError, AuthService
from app.services.password_reset import (
    PASSWORD_RESET_INVALID_DETAIL,
    PASSWORD_RESET_RESPONSE_DETAIL,
    PasswordResetEmailSender,
    PasswordResetError,
    PasswordResetService,
    get_password_reset_email_sender,
)

router = APIRouter()


def get_auth_service(session: AsyncSession = Depends(get_db)) -> AuthService:
    return AuthService(UserRepository(session), UserSessionRepository(session))


def set_auth_cookies(response: Response, tokens: TokenPair) -> None:
    response.set_cookie(
        AegisAccessCookie,
        tokens.access_token,
        max_age=settings.access_token_minutes * 60,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )
    response.set_cookie(
        AegisRefreshCookie,
        tokens.refresh_token,
        max_age=settings.refresh_token_days * 24 * 60 * 60,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite=settings.auth_cookie_samesite,
        path="/",
    )


def clear_auth_cookies(response: Response) -> None:
    for cookie_name in (AegisAccessCookie, AegisRefreshCookie):
        response.delete_cookie(
            cookie_name,
            httponly=True,
            secure=settings.auth_cookie_secure,
            samesite=settings.auth_cookie_samesite,
            path="/",
        )


def get_password_reset_service(
    session: AsyncSession = Depends(get_db),
    email_sender: PasswordResetEmailSender = Depends(get_password_reset_email_sender),
) -> PasswordResetService:
    return PasswordResetService(
        session,
        users=UserRepository(session),
        email_sender=email_sender,
    )


@router.post("/login", response_model=TokenPair)
async def login(
    payload: LoginRequest,
    response: Response,
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
    async with transaction_scope(session):
        await AuditLogService(AuditLogRepository(session)).record(
            actor=user,
            action="auth.login",
            resource_type="session",
            resource_id=str(user.id),
            metadata={"email": user.email.lower()},
        )
    async with transaction_scope(session):
        tokens = TokenPair(**await auth.issue_tokens(user))
    set_auth_cookies(response, tokens)
    return tokens


@router.post("/signup", status_code=status.HTTP_403_FORBIDDEN)
async def signup(_: SignupRequest) -> None:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Self-service signup is disabled. Contact an administrator for access.",
    )


@router.post("/refresh", response_model=TokenPair)
async def refresh(
    response: Response,
    payload: RefreshRequest | None = Body(default=None),
    refresh_cookie: str | None = Cookie(default=None, alias=AegisRefreshCookie),
    auth: AuthService = Depends(get_auth_service),
    session: AsyncSession = Depends(get_db),
) -> TokenPair:
    refresh_token = payload.refresh_token if payload else refresh_cookie
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")
    try:
        async with transaction_scope(session):
            tokens = TokenPair(**await auth.refresh(refresh_token))
    except AuthError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    set_auth_cookies(response, tokens)
    return tokens


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    current_session: UserSession = Depends(get_current_session),
    auth: AuthService = Depends(get_auth_service),
    session: AsyncSession = Depends(get_db),
) -> None:
    async with transaction_scope(session):
        await auth.revoke_session(current_session)
    clear_auth_cookies(response)


@router.post("/forgot-password", response_model=PasswordResetResponse)
async def forgot_password(
    payload: ForgotPasswordRequest,
    password_reset: PasswordResetService = Depends(get_password_reset_service),
    session: AsyncSession = Depends(get_db),
) -> PasswordResetResponse:
    async with transaction_scope(session):
        await password_reset.request_admin_password_reset(payload.email)
    return PasswordResetResponse(detail=PASSWORD_RESET_RESPONSE_DETAIL)


@router.post("/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    payload: PasswordResetRequest,
    password_reset: PasswordResetService = Depends(get_password_reset_service),
    session: AsyncSession = Depends(get_db),
) -> PasswordResetResponse:
    try:
        async with transaction_scope(session):
            await password_reset.reset_admin_password(payload.token, payload.password)
    except PasswordResetError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=PASSWORD_RESET_INVALID_DETAIL,
        )
    return PasswordResetResponse(detail="Password reset complete. You can now sign in.")


@router.get("/me", response_model=UserRead)
async def me(current_user: User = Depends(get_current_user)) -> User:
    return current_user
