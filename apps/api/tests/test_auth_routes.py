from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.user import UserRole
from app.repositories.users import UserRepository
from app.schemas.users import UserCreate


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


@pytest.mark.asyncio
async def test_signup_creates_account_with_selected_role_and_returns_tokens(
    client: AsyncClient, session: AsyncSession
) -> None:
    response = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "new.user@aegispro.local",
            "full_name": "New User",
            "role": UserRole.operator.value,
            "password": "ChangeMe123!",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["access_token"]
    assert payload["refresh_token"]
    assert payload["token_type"] == "bearer"

    user = await UserRepository(session).get_by_email("new.user@aegispro.local")
    assert user is not None
    assert user.role is UserRole.operator
    assert user.is_active is True


@pytest.mark.asyncio
async def test_signup_rejects_duplicate_email(client: AsyncClient, session: AsyncSession) -> None:
    await UserRepository(session).create(
        UserCreate(
            email="existing@aegispro.local",
            full_name="Existing User",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )

    response = await client.post(
        "/api/v1/auth/signup",
        json={
            "email": "existing@aegispro.local",
            "full_name": "Duplicate User",
            "role": UserRole.viewer.value,
            "password": "ChangeMe123!",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Email already exists"


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
