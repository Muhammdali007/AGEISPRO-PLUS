import pytest

from app.core.config import Settings


def test_relative_model_paths_resolve_to_workspace_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AI_MODEL_WEIGHTS_PATH", raising=False)
    monkeypatch.delenv("AI_MODEL_PERSON_WEAPON_WEIGHTS_PATH", raising=False)
    monkeypatch.setenv("AI_MODEL_WEAPON_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_MODEL_FIRE_SMOKE_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_MODEL_BACKEND", "ultralytics")
    monkeypatch.setenv("AI_MODEL_FALLBACK_BACKEND", "simulated")
    monkeypatch.setenv("AI_ALLOW_BACKEND_FALLBACK", "true")
    monkeypatch.setenv("AI_RECOGNITION_BACKEND", "hash")
    monkeypatch.setenv("AI_RECOGNITION_ALLOW_FALLBACK", "true")
    settings = Settings()
    assert settings.model_weights_path.replace("\\", "/").endswith("storage/models/yolo11n.pt")
    assert settings.model_person_weapon_weights_path.replace("\\", "/").endswith(
        "storage/models/yolo11n.pt"
    )
    assert settings.model_weapon_weights_path is None
    assert settings.model_fire_smoke_weights_path is None


def test_production_requires_real_backends_and_weapon_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ENVIRONMENT", "production")
    monkeypatch.setenv("AI_MODEL_BACKEND", "simulated")
    monkeypatch.setenv("AI_MODEL_FALLBACK_BACKEND", "ultralytics")
    monkeypatch.setenv("AI_ALLOW_BACKEND_FALLBACK", "false")
    monkeypatch.setenv("AI_RECOGNITION_BACKEND", "insightface")
    monkeypatch.setenv("AI_RECOGNITION_ALLOW_FALLBACK", "false")
    monkeypatch.setenv("AI_MODEL_WEIGHTS_PATH", "C:/detector.pt")
    monkeypatch.setenv("AI_MODEL_WEAPON_WEIGHTS_PATH", "C:/weapon.pt")
    monkeypatch.setenv("AI_API_EVENT_CALLBACK_URL", "http://api:8000/api/v1/detections/ingest")
    monkeypatch.setenv("AI_API_EVENT_CALLBACK_TOKEN", "service-token")
    with pytest.raises(ValueError):
        Settings()
