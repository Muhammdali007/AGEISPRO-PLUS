from pathlib import Path

from app.core.config import PROJECT_ROOT, Settings


def test_storage_root_defaults_to_project_storage() -> None:
    settings = Settings()

    assert settings.storage_root == PROJECT_ROOT / "storage"


def test_storage_root_relative_values_are_resolved_from_project_root() -> None:
    settings = Settings(storage_root=Path("storage"))

    assert settings.storage_root == PROJECT_ROOT / "storage"

