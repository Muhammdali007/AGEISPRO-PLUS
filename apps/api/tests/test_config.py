from pathlib import Path

import pytest

from app.core.config import PROJECT_ROOT, Settings


def test_storage_root_defaults_to_project_storage() -> None:
    settings = Settings()

    assert settings.storage_root == PROJECT_ROOT / "storage"


def test_storage_root_relative_values_are_resolved_from_project_root() -> None:
    settings = Settings(storage_root=Path("storage"))

    assert settings.storage_root == PROJECT_ROOT / "storage"


def test_production_requires_insightface_and_non_default_secrets() -> None:
    with pytest.raises(ValueError):
        Settings(
            environment="production",
            database_url="postgresql+psycopg://aegispro:secret@postgres:5432/aegispro",
            secret_key="replace-with-a-long-random-secret",
            service_callback_token="service-token",
            bootstrap_admin_password="ChangedPassword!42",
            recognition_backend="hash",
            recognition_allow_fallback=False,
        )
