from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.user import User, UserRole


class AdminRecoveryError(ValueError):
    pass


@dataclass(frozen=True)
class AdminRecoveryResult:
    user_id: UUID
    email: str
    created: bool
    reactivated: bool


async def reset_admin_password(
    session: AsyncSession,
    *,
    email: str,
    password: str,
    full_name: str = "AegisPro Administrator",
) -> AdminRecoveryResult:
    normalized_email = email.strip().lower()
    if not normalized_email:
        raise AdminRecoveryError("Admin email is required")
    if len(password) < 8 or len(password) > 128:
        raise AdminRecoveryError("Password must be between 8 and 128 characters")

    user = await session.scalar(select(User).where(User.email == normalized_email))
    if user is None:
        user = User(
            email=normalized_email,
            full_name=full_name,
            role=UserRole.administrator,
            password_hash=hash_password(password),
            is_active=True,
        )
        session.add(user)
        await session.flush()
        await session.refresh(user)
        return AdminRecoveryResult(
            user_id=user.id,
            email=user.email,
            created=True,
            reactivated=False,
        )

    if user.role is not UserRole.administrator:
        raise AdminRecoveryError("Refusing to promote a non-administrator account")

    was_inactive = not user.is_active
    user.password_hash = hash_password(password)
    user.is_active = True
    await session.flush()
    await session.refresh(user)
    return AdminRecoveryResult(
        user_id=user.id,
        email=user.email,
        created=False,
        reactivated=was_inactive,
    )
