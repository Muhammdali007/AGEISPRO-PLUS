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


def test_camera_network_settings_parse_csv_values() -> None:
    settings = Settings(
        CAMERA_ALLOWED_PROTOCOLS="http, https",
        CAMERA_ALLOWED_PORTS="80, 8080",
        CAMERA_ALLOWED_NETWORKS="192.168.0.0/16,10.0.0.0/8",
        CAMERA_ALLOWED_HOSTNAMES="cam-1.local,*.camera.example",
        CAMERA_BLOCKED_NETWORKS="169.254.0.0/16,::1/128",
    )

    assert settings.camera_allowed_protocols == ["http", "https"]
    assert settings.camera_allowed_ports == [80, 8080]
    assert settings.camera_allowed_networks == ["192.168.0.0/16", "10.0.0.0/8"]
    assert settings.camera_allowed_hostnames == ["cam-1.local", "*.camera.example"]
    assert settings.camera_blocked_networks == ["169.254.0.0/16", "::1/128"]


def test_sound_alert_defaults_require_three_unknown_person_scans() -> None:
    settings = Settings()

    assert settings.sound_alert_unknown_scan_threshold == 3
    assert settings.sound_alert_unknown_cooldown_seconds == 30
    assert settings.sound_alert_hazard_cooldown_seconds == 10
