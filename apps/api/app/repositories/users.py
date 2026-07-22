from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.models.user import User, UserRole
from app.schemas.users import UserCreate, UserSelfUpdate, UserUpdate


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get_by_email(self, email: str) -> User | None:
        return await self.session.scalar(select(User).where(User.email == email.lower()))

    async def get_by_id(self, user_id: UUID) -> User | None:
        return await self.session.get(User, user_id)

    async def list(self) -> list[User]:
        result = await self.session.scalars(select(User).order_by(User.created_at.desc()))
        return list(result)

    async def count_active_administrators(self) -> int:
        return (
            await self.session.scalar(
                select(func.count())
                .select_from(User)
                .where(User.role == UserRole.administrator, User.is_active.is_(True))
            )
            or 0
        )

    async def create(self, payload: UserCreate) -> User:
        user = User(
            email=payload.email.lower(),
            full_name=payload.full_name,
            role=payload.role,
            password_hash=hash_password(payload.password),
            is_active=payload.is_active,
        )
        self.session.add(user)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def update(self, user: User, payload: UserUpdate | UserSelfUpdate) -> User:
        updates = payload.model_dump(exclude_unset=True)
        password = updates.pop("password", None)
        if "email" in updates and updates["email"] is not None:
            updates["email"] = updates["email"].lower()
        for key, value in updates.items():
            setattr(user, key, value)
        if password:
            user.password_hash = hash_password(password)
        await self.session.flush()
        await self.session.refresh(user)
        return user

    async def delete(self, user: User) -> None:
        await self.session.delete(user)
        await self.session.flush()
