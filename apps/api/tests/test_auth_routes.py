from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlparse

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.api.v1.routes.auth import get_password_reset_email_sender
from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.password_reset_token import PasswordResetToken
from app.models.user import UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.users import UserRepository
from app.schemas.auth import SignupRequest
from app.schemas.users import UserCreate
from app.services.admin_recovery import AdminRecoveryError, reset_admin_password
from app.services.auth import AuthService
from app.services.password_reset import PASSWORD_RESET_RESPONSE_DETAIL


class FakePasswordResetEmailSender:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def send_password_reset_email(self, *, email: str, reset_url: str) -> None:
        self.messages.append((email, reset_url))


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as test_session:
        yield test_session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()


async def _set_current_user(user) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


def _extract_token(reset_url: str) -> str:
    token = parse_qs(urlparse(reset_url).query).get("token")
    assert token
    return token[0]


@pytest.mark.asyncio
async def test_public_signup_is_disabled(
    client: AsyncClient, session: AsyncSession
) -> None:
    response = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "new.user@aegispro.local",
            "full_name": "New User",
            "password": "ChangeMe123!",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "Self-service signup is disabled. Contact an administrator for access."

    user = await UserRepository(session).get_by_email("new.user@aegispro.local")
    assert user is None


@pytest.mark.asyncio
async def test_auth_service_register_creates_inactive_viewer_account(session: AsyncSession) -> None:
    auth = AuthService(UserRepository(session))

    user = await auth.register(
        SignupRequest(
            email="pending.user@aegispro.local",
            full_name="Pending User",
            password="ChangeMe123!",
        )
    )

    assert user.role is UserRole.viewer
    assert user.is_active is False


@pytest.mark.asyncio
async def test_login_and_me_return_authenticated_profile(
    client: AsyncClient, session: AsyncSession
) -> None:
    user = await UserRepository(session).create(
        UserCreate(
            email="operator@aegispro.local",
            full_name="Operator One",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )

    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "ChangeMe123!"},
    )

    assert login_response.status_code == 200
    access_token = login_response.json()["access_token"]

    me_response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )

    assert me_response.status_code == 200
    assert me_response.json()["email"] == user.email
    assert me_response.json()["role"] == UserRole.operator.value


@pytest.mark.asyncio
async def test_refresh_rotates_refresh_token_and_rejects_reuse(
    client: AsyncClient, session: AsyncSession
) -> None:
    user = await UserRepository(session).create(
        UserCreate(
            email="refresh@aegispro.local",
            full_name="Refresh User",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "ChangeMe123!"},
    )
    old_refresh_token = login_response.json()["refresh_token"]

    refresh_response = await client.post("/api/v1/auth/refresh")
    assert refresh_response.status_code == 200
    assert refresh_response.json()["refresh_token"] != old_refresh_token

    reused_response = await client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": old_refresh_token},
    )
    assert reused_response.status_code == 401


@pytest.mark.asyncio
async def test_logout_revokes_current_session(
    client: AsyncClient, session: AsyncSession
) -> None:
    user = await UserRepository(session).create(
        UserCreate(
            email="logout@aegispro.local",
            full_name="Logout User",
            role=UserRole.viewer,
            password="ChangeMe123!",
        )
    )
    login_response = await client.post(
        "/api/v1/auth/login",
        json={"email": user.email, "password": "ChangeMe123!"},
    )
    access_token = login_response.json()["access_token"]

    logout_response = await client.post("/api/v1/auth/logout")
    assert logout_response.status_code == 204

    me_response = await client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_response.status_code == 401


@pytest.mark.asyncio
async def test_forgot_password_is_neutral_and_only_emails_active_admins(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    sender = FakePasswordResetEmailSender()
    app.dependency_overrides[get_password_reset_email_sender] = lambda: sender
    await UserRepository(session).create(
        UserCreate(
            email="admin@aegispro.local",
            full_name="Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await UserRepository(session).create(
        UserCreate(
            email="inactive-admin@aegispro.local",
            full_name="Inactive Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
            is_active=False,
        )
    )
    await UserRepository(session).create(
        UserCreate(
            email="operator@aegispro.local",
            full_name="Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    await session.commit()

    for email in [
        "missing@aegispro.local",
        "operator@aegispro.local",
        "inactive-admin@aegispro.local",
        "admin@aegispro.local",
    ]:
        response = await client.post("/api/v1/auth/forgot-password", json={"email": email})
        assert response.status_code == 200
        assert response.json() == {"detail": PASSWORD_RESET_RESPONSE_DETAIL}

    tokens = list(await session.scalars(select(PasswordResetToken)))
    assert len(tokens) == 1
    assert len(sender.messages) == 1
    assert sender.messages[0][0] == "admin@aegispro.local"


@pytest.mark.asyncio
async def test_reset_password_with_valid_token_updates_admin_password(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    sender = FakePasswordResetEmailSender()
    app.dependency_overrides[get_password_reset_email_sender] = lambda: sender
    admin_email = "reset-admin@aegispro.local"
    await UserRepository(session).create(
        UserCreate(
            email=admin_email,
            full_name="Reset Admin",
            role=UserRole.administrator,
            password="OldPassword123!",
        )
    )
    await session.commit()

    forgot_response = await client.post(
        "/api/v1/auth/forgot-password",
        json={"email": admin_email},
    )
    assert forgot_response.status_code == 200
    raw_token = _extract_token(sender.messages[0][1])

    reset_response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": raw_token, "password": "NewPassword123!"},
    )

    assert reset_response.status_code == 200
    auth = AuthService(UserRepository(session))
    assert await auth.authenticate(admin_email, "NewPassword123!")


@pytest.mark.asyncio
async def test_reset_password_rejects_invalid_expired_and_used_tokens(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    sender = FakePasswordResetEmailSender()
    app.dependency_overrides[get_password_reset_email_sender] = lambda: sender
    admin_email = "token-admin@aegispro.local"
    await UserRepository(session).create(
        UserCreate(
            email=admin_email,
            full_name="Token Admin",
            role=UserRole.administrator,
            password="OldPassword123!",
        )
    )
    await session.commit()

    unknown_response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": "x" * 48, "password": "NewPassword123!"},
    )
    assert unknown_response.status_code == 400

    await client.post("/api/v1/auth/forgot-password", json={"email": admin_email})
    expired_token = _extract_token(sender.messages[-1][1])
    token_record = await session.scalar(select(PasswordResetToken))
    assert token_record is not None
    token_record.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    await session.commit()
    expired_response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": expired_token, "password": "NewPassword123!"},
    )
    assert expired_response.status_code == 400

    await client.post("/api/v1/auth/forgot-password", json={"email": admin_email})
    valid_token = _extract_token(sender.messages[-1][1])
    first_use = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": valid_token, "password": "NewPassword123!"},
    )
    assert first_use.status_code == 200
    second_use = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": valid_token, "password": "AnotherPassword123!"},
    )
    assert second_use.status_code == 400


@pytest.mark.asyncio
async def test_new_forgot_password_request_invalidates_previous_token(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    sender = FakePasswordResetEmailSender()
    app.dependency_overrides[get_password_reset_email_sender] = lambda: sender
    admin_email = "rotation-admin@aegispro.local"
    await UserRepository(session).create(
        UserCreate(
            email=admin_email,
            full_name="Rotation Admin",
            role=UserRole.administrator,
            password="OldPassword123!",
        )
    )
    await session.commit()

    await client.post("/api/v1/auth/forgot-password", json={"email": admin_email})
    first_token = _extract_token(sender.messages[-1][1])
    await client.post("/api/v1/auth/forgot-password", json={"email": admin_email})
    second_token = _extract_token(sender.messages[-1][1])

    old_response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": first_token, "password": "OldLinkPassword123!"},
    )
    assert old_response.status_code == 400

    new_response = await client.post(
        "/api/v1/auth/reset-password",
        json={"token": second_token, "password": "LatestPassword123!"},
    )
    assert new_response.status_code == 200


@pytest.mark.asyncio
async def test_admin_recovery_resets_and_reactivates_admin(session: AsyncSession) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="locked-admin@aegispro.local",
            full_name="Locked Admin",
            role=UserRole.administrator,
            password="OldPassword123!",
            is_active=False,
        )
    )

    result = await reset_admin_password(
        session,
        email="LOCKED-ADMIN@aegispro.local",
        password="NewPassword123!",
    )
    await session.commit()

    refreshed = await UserRepository(session).get_by_email("locked-admin@aegispro.local")
    assert result.user_id == admin.id
    assert result.created is False
    assert result.reactivated is True
    assert refreshed is not None
    assert refreshed.is_active is True

    auth = AuthService(UserRepository(session))
    assert await auth.authenticate("locked-admin@aegispro.local", "NewPassword123!")


@pytest.mark.asyncio
async def test_admin_recovery_recreates_missing_admin(session: AsyncSession) -> None:
    result = await reset_admin_password(
        session,
        email="admin@aegispro.local",
        password="Recovered123!",
    )
    await session.commit()

    recovered = await UserRepository(session).get_by_email("admin@aegispro.local")
    assert result.created is True
    assert recovered is not None
    assert recovered.role is UserRole.administrator
    assert recovered.is_active is True


@pytest.mark.asyncio
async def test_admin_recovery_refuses_to_promote_non_admin(session: AsyncSession) -> None:
    await UserRepository(session).create(
        UserCreate(
            email="operator@aegispro.local",
            full_name="Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )

    with pytest.raises(AdminRecoveryError, match="Refusing to promote"):
        await reset_admin_password(
            session,
            email="operator@aegispro.local",
            password="Recovered123!",
        )


@pytest.mark.asyncio
async def test_supervisor_can_manage_non_admin_users_but_not_administrators(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    supervisor = await UserRepository(session).create(
        UserCreate(
            email="supervisor@aegispro.local",
            full_name="Supervisor",
            role=UserRole.supervisor,
            password="ChangeMe123!",
        )
    )
    admin = await UserRepository(session).create(
        UserCreate(
            email="admin@aegispro.local",
            full_name="Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    operator = await UserRepository(session).create(
        UserCreate(
            email="operator@aegispro.local",
            full_name="Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(supervisor)

    create_response = await client.post(
        "/api/v1/users",
        json={
            "email": "viewer@aegispro.local",
            "full_name": "Viewer",
            "role": UserRole.viewer.value,
            "password": "ChangeMe123!",
            "is_active": True,
        },
    )
    assert create_response.status_code == 201

    create_admin_response = await client.post(
        "/api/v1/users",
        json={
            "email": "new-admin@aegispro.local",
            "full_name": "New Admin",
            "role": UserRole.administrator.value,
            "password": "ChangeMe123!",
            "is_active": True,
        },
    )
    assert create_admin_response.status_code == 403

    update_response = await client.patch(
        f"/api/v1/users/{operator.id}",
        json={"full_name": "Operator Updated", "role": UserRole.viewer.value},
    )
    assert update_response.status_code == 200
    assert update_response.json()["role"] == UserRole.viewer.value

    promote_response = await client.patch(
        f"/api/v1/users/{operator.id}",
        json={"role": UserRole.administrator.value},
    )
    assert promote_response.status_code == 403

    admin_update_response = await client.patch(
        f"/api/v1/users/{admin.id}",
        json={"full_name": "Blocked Admin Edit"},
    )
    assert admin_update_response.status_code == 403

    delete_response = await client.delete(f"/api/v1/users/{operator.id}")
    assert delete_response.status_code == 204
    assert await UserRepository(session).get_by_id(operator.id) is None

    delete_admin_response = await client.delete(f"/api/v1/users/{admin.id}")
    assert delete_admin_response.status_code == 403


@pytest.mark.asyncio
async def test_administrator_can_create_and_delete_another_administrator(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="root-admin@aegispro.local",
            full_name="Root Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    create_response = await client.post(
        "/api/v1/users",
        json={
            "email": "peer-admin@aegispro.local",
            "full_name": "Peer Admin",
            "role": UserRole.administrator.value,
            "password": "ChangeMe123!",
            "is_active": True,
        },
    )
    assert create_response.status_code == 201
    assert create_response.json()["role"] == UserRole.administrator.value

    delete_response = await client.delete(f"/api/v1/users/{create_response.json()['id']}")
    assert delete_response.status_code == 204


@pytest.mark.asyncio
async def test_cannot_remove_last_active_administrator(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="last-admin@aegispro.local",
            full_name="Last Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    demote_response = await client.patch(
        f"/api/v1/users/{admin.id}",
        json={"role": UserRole.supervisor.value},
    )
    assert demote_response.status_code == 400
    assert demote_response.json()["detail"] == "Cannot remove the last active administrator"

    deactivate_response = await client.patch(
        f"/api/v1/users/{admin.id}",
        json={"is_active": False},
    )
    assert deactivate_response.status_code == 400
    assert deactivate_response.json()["detail"] == "Cannot remove the last active administrator"

    delete_response = await client.delete(f"/api/v1/users/{admin.id}")
    assert delete_response.status_code == 400
    assert delete_response.json()["detail"] == "Cannot remove the last active administrator"


@pytest.mark.asyncio
async def test_user_password_updates_are_excluded_from_audit_metadata(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="audit-admin@aegispro.local",
            full_name="Audit Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    operator = await UserRepository(session).create(
        UserCreate(
            email="audit-operator@aegispro.local",
            full_name="Audit Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    response = await client.patch(
        f"/api/v1/users/{operator.id}",
        json={"full_name": "Rotated Operator", "password": "EvenStronger123!"},
    )

    assert response.status_code == 200

    items, _ = await AuditLogRepository(session).list(action="users.update")
    assert items[0].metadata_ == {"full_name": "Rotated Operator"}

    audit_response = await client.get("/api/v1/monitoring/audit-logs?action=users.update")
    assert audit_response.status_code == 200
    assert audit_response.json()["items"][0]["metadata"] == {"full_name": "Rotated Operator"}
