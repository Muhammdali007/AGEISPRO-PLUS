import asyncio
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.config import PROJECT_ROOT, settings
from app.core.security import hash_password
from app.db.metadata import Base
from app.db.session import AsyncSessionLocal, engine
from app.models.user import User, UserRole


async def bootstrap_database() -> None:
    if settings.database_url.startswith("sqlite+aiosqlite://"):
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    else:
        await asyncio.to_thread(_run_migrations)

    async with AsyncSessionLocal() as session:
        existing_admin = await session.scalar(
            select(User).where(User.email == settings.bootstrap_admin_email)
        )
        if existing_admin:
            return
        session.add(
            User(
                email=settings.bootstrap_admin_email,
                full_name="AegisPro Administrator",
                role=UserRole.administrator,
                password_hash=hash_password(settings.bootstrap_admin_password),
                is_active=True,
            )
        )
        await session.commit()


def _run_migrations() -> None:
    api_root = Path(PROJECT_ROOT) / "apps" / "api"
    alembic_config = Config(str(api_root / "alembic.ini"))
    alembic_config.set_main_option("sqlalchemy.url", settings.database_url)
    alembic_config.set_main_option("script_location", str(api_root / "alembic"))
    alembic_config.set_main_option("prepend_sys_path", str(api_root))
    command.upgrade(alembic_config, "head")
