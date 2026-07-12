from uuid import UUID

from app.core.security import TokenType, create_token, decode_token, verify_password
from app.models.user import User
from app.repositories.users import UserRepository
from app.schemas.auth import SignupRequest
from app.schemas.users import UserCreate


class AuthError(ValueError):
    pass


class AuthService:
    def __init__(self, users: UserRepository) -> None:
        self.users = users

    async def register(self, payload: SignupRequest) -> User:
        return await self.users.create(
            UserCreate(
                email=payload.email,
                full_name=payload.full_name,
                password=payload.password,
                role=payload.role,
                is_active=True,
            )
        )

    async def authenticate(self, email: str, password: str) -> User:
        user = await self.users.get_by_email(email)
        if not user or not user.is_active or not verify_password(password, user.password_hash):
            raise AuthError("Invalid email or password")
        return user

    def issue_tokens(self, user: User) -> dict[str, str]:
        return {
            "access_token": create_token(user.id, user.role.value, TokenType.access),
            "refresh_token": create_token(user.id, user.role.value, TokenType.refresh),
        }

    async def refresh(self, refresh_token: str) -> dict[str, str]:
        try:
            payload = decode_token(refresh_token, TokenType.refresh)
            user_id = UUID(payload["sub"])
        except (KeyError, ValueError) as exc:
            raise AuthError("Invalid refresh token") from exc

        user = await self.users.get_by_id(user_id)
        if not user or not user.is_active:
            raise AuthError("Invalid refresh token")
        return self.issue_tokens(user)
