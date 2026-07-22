from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_roles
from app.core.audit import dump_audit_model
from app.db.transactions import transaction_scope
from app.db.session import get_db
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.users import UserRepository
from app.schemas.users import UserCreate, UserRead, UserSelfUpdate, UserUpdate
from app.services.audit_logs import AuditLogService

router = APIRouter()


def get_user_repository(session: AsyncSession = Depends(get_db)) -> UserRepository:
    return UserRepository(session)


def _ensure_supervisor_can_manage_user(
    current_user: User,
    *,
    target_user: User | None = None,
    target_role: UserRole | None = None,
) -> None:
    if current_user.role is UserRole.administrator:
        return
    if current_user.role is not UserRole.supervisor:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    if target_user and target_user.role is UserRole.administrator:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Supervisors cannot manage administrator accounts",
        )
    if target_role is UserRole.administrator:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Supervisors cannot assign the administrator role",
        )


async def _ensure_not_removing_last_active_admin(
    users: UserRepository,
    target_user: User,
    *,
    target_role: UserRole | None = None,
    target_is_active: bool | None = None,
) -> None:
    if target_user.role is not UserRole.administrator or not target_user.is_active:
        return
    removes_admin_role = target_role is not None and target_role is not UserRole.administrator
    deactivates_admin = target_is_active is False
    if not removes_admin_role and not deactivates_admin:
        return
    if await users.count_active_administrators() <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot remove the last active administrator",
        )


@router.get("", response_model=list[UserRead])
async def list_users(
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    users: UserRepository = Depends(get_user_repository),
) -> list[User]:
    return await users.list()


@router.post("", response_model=UserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    users: UserRepository = Depends(get_user_repository),
) -> User:
    _ensure_supervisor_can_manage_user(current_user, target_role=payload.role)
    try:
        async with transaction_scope(users.session):
            user = await users.create(payload)
            await AuditLogService(AuditLogRepository(users.session)).record(
                actor=current_user,
                action="users.create",
                resource_type="user",
                resource_id=str(user.id),
                metadata={"email": user.email.lower(), "role": user.role.value},
            )
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
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

    if current_user.role in {UserRole.administrator, UserRole.supervisor}:
        _ensure_supervisor_can_manage_user(current_user, target_user=user, target_role=payload.role)
        await _ensure_not_removing_last_active_admin(
            users,
            user,
            target_role=payload.role,
            target_is_active=payload.is_active,
        )
        try:
            async with transaction_scope(users.session):
                updated = await users.update(user, payload)
                await AuditLogService(AuditLogRepository(users.session)).record(
                    actor=current_user,
                    action="users.update",
                    resource_type="user",
                    resource_id=str(updated.id),
                    metadata=dump_audit_model(payload),
                )
        except IntegrityError:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already exists")
        return updated

    if current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")

    attempted = payload.model_dump(exclude_unset=True)
    forbidden = {"email", "role", "is_active"} & attempted.keys()
    if forbidden:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot update protected fields")
    self_payload = UserSelfUpdate(**attempted)
    async with transaction_scope(users.session):
        updated = await users.update(user, self_payload)
        await AuditLogService(AuditLogRepository(users.session)).record(
            actor=current_user,
            action="users.self_update",
            resource_type="user",
            resource_id=str(updated.id),
            metadata=dump_audit_model(self_payload),
        )
    return updated


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: UUID,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    users: UserRepository = Depends(get_user_repository),
) -> None:
    user = await users.get_by_id(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    _ensure_supervisor_can_manage_user(current_user, target_user=user)
    await _ensure_not_removing_last_active_admin(
        users,
        user,
        target_role=UserRole.viewer,
        target_is_active=False,
    )
    async with transaction_scope(users.session):
        await AuditLogService(AuditLogRepository(users.session)).record(
            actor=current_user,
            action="users.delete",
            resource_type="user",
            resource_id=str(user.id),
            metadata={"email": user.email.lower(), "role": user.role.value, "was_active": user.is_active},
        )
        await users.delete(user)
