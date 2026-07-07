from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_roles
from app.db.session import get_db
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.users import UserRepository
from app.schemas.users import UserCreate, UserRead, UserSelfUpdate, UserUpdate
from app.services.audit_logs import AuditLogService

router = APIRouter()


def get_user_repository(session: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(session)


@router.get("", response_model=list[UserRead])
async def list_users(
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    users: UserRepository = Depends(get_user_repository),
) -> list[User]:
    return await users.list()


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    current_user: User = Depends(require_roles(UserRole.administrator)),
    users: UserRepository = Depends(get_user_repository),
) -> User:
    try:
        user = await users.create(payload)
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
    await AuditLogService(AuditLogRepository(users.session)).record(
        actor=current_user,
        action="users.create",
        resource_type="user",
        resource_id=str(user.id),
        metadata={"email": user.email.lower(), "role": user.role.value},
    )
    return user


@router.get("/{user_id}", response_model=UserRead)
async def get_user(
    user_id: UUID,
    current_user: User = Depends(get_current_user),
    users: UserRepository = Depends(get_user_repository),
) -> User:
    if current_user.role not in {UserRole.administrator, UserRole.supervisor} and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    user = await users.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user


@router.patch("/{user_id}", response_model=UserRead)
async def update_user(
    user_id: UUID,
    payload: UserUpdate,
    current_user: User = Depends(get_current_user),
    users: UserRepository = Depends(get_user_repository),
) -> User:
    user = await users.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if current_user.role is UserRole.administrator:
        try:
            return await users.update(user, payload)
        except IntegrityError:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")

    if current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")

    attempted = payload.model_dump(exclude_unset=True)
    forbidden = {"email", "role", "is_active"} & attempted.keys()
    if forbidden:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot update protected fields")
    self_payload = UserSelfUpdate(**attempted)
    return await users.update(user, self_payload)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def deactivate_user(
    user_id: UUID,
    _: User = Depends(require_roles(UserRole.administrator)),
    users: UserRepository = Depends(get_user_repository),
) -> None:
    user = await users.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    await users.update(user, UserUpdate(is_active=False))
