import hashlib
import json
from pathlib import Path

import pytest

from app.core.config import Settings


def test_relative_model_paths_resolve_to_workspace_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_MODEL_WEIGHTS_PATH", "storage/models/yolo11n.pt")
    monkeypatch.setenv("AI_MODEL_PERSON_WEAPON_WEIGHTS_PATH", "storage/models/yolo11n.pt")
    monkeypatch.setenv("AI_MODEL_WEAPON_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_MODEL_FIRE_SMOKE_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_MODEL_FIRE_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_MODEL_SMOKE_WEIGHTS_PATH", "")
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
    assert settings.model_fire_weights_path is None
    assert settings.model_smoke_weights_path is None


def test_production_requires_real_backends_and_weapon_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AI_ENVIRONMENT", "production")
    monkeypatch.setenv("AI_MODEL_BACKEND", "simulated")
    monkeypatch.setenv("AI_MODEL_FALLBACK_BACKEND", "ultralytics")
    monkeypatch.setenv("AI_ALLOW_BACKEND_FALLBACK", "false")
    monkeypatch.setenv("AI_MODEL_RUNTIME_AUTOINSTALL", "true")
    monkeypatch.setenv("AI_RECOGNITION_BACKEND", "insightface")
    monkeypatch.setenv("AI_RECOGNITION_ALLOW_FALLBACK", "false")
    monkeypatch.setenv("AI_MODEL_WEIGHTS_PATH", "C:/detector.pt")
    monkeypatch.setenv("AI_MODEL_WEAPON_WEIGHTS_PATH", "C:/weapon.pt")
    monkeypatch.setenv("AI_API_EVENT_CALLBACK_URL", "http://api:8000/api/v1/detections/ingest")
    monkeypatch.setenv("AI_API_EVENT_CALLBACK_TOKEN", "service-token")
    with pytest.raises(ValueError):
        Settings()


def test_production_accepts_signed_weapon_and_fire_promotions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    general = tmp_path / "detector.pt"
    weapon = tmp_path / "weapon.pt"
    fire_smoke = tmp_path / "fire-smoke.pt"
    general.write_bytes(b"general")
    weapon.write_bytes(b"weapon")
    fire_smoke.write_bytes(b"fire-smoke")

    weapon_manifest = weapon.with_suffix(".pt.promotion.json")
    weapon_manifest.write_text(
        json.dumps(
            {
                "model_id": "weapon-2026-07-14",
                "checkpoint_sha256": sha256(weapon),
                "selected_operating_threshold": 0.25,
                "datasets": [{"name": "Licensed weapon set", "license": "Internal"}],
                "independent_holdout": {
                    "name": "weapon-holdout",
                    "license": "Internal",
                    "independent_from_training": True,
                },
                "per_class_metrics": {
                    "weapon": {"precision": 0.92, "recall": 0.91},
                },
                "gates": [
                    {
                        "name": "weapon_recall_at_selected_threshold",
                        "minimum": 0.90,
                        "actual": 0.91,
                        "passed": True,
                    }
                ],
                "signatures": [
                    {"name": "Owner", "role": "Model owner", "signed_at": "2026-07-14T10:00:00Z"},
                    {"name": "Reviewer", "role": "Independent reviewer", "signed_at": "2026-07-14T11:00:00Z"},
                ],
            }
        ),
        encoding="utf-8",
    )

    fire_smoke_manifest = fire_smoke.with_suffix(".pt.promotion.json")
    fire_smoke_manifest.write_text(
        json.dumps(
            {
                "model_id": "fire-smoke-2026-07-14",
                "checkpoint_sha256": sha256(fire_smoke),
                "selected_operating_threshold": 0.30,
                "datasets": [{"name": "Licensed fire set", "license": "Internal"}],
                "independent_holdout": {
                    "name": "fire-smoke-holdout",
                    "license": "Internal",
                    "independent_from_training": True,
                },
                "per_class_metrics": {
                    "fire": {"precision": 0.93, "recall": 0.90},
                    "smoke": {"precision": 0.91, "recall": 0.84},
                },
                "gates": [
                    {
                        "name": "fire_recall_at_selected_threshold",
                        "minimum": 0.85,
                        "actual": 0.90,
                        "passed": True,
                    },
                    {
                        "name": "smoke_recall_at_selected_threshold",
                        "minimum": 0.80,
                        "actual": 0.84,
                        "passed": True,
                    },
                ],
                "signatures": [
                    {"name": "Owner", "role": "Model owner", "signed_at": "2026-07-14T10:00:00Z"},
                    {"name": "Reviewer", "role": "Independent reviewer", "signed_at": "2026-07-14T11:00:00Z"},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("AI_ENVIRONMENT", "production")
    monkeypatch.setenv("AI_MODEL_BACKEND", "ultralytics")
    monkeypatch.setenv("AI_MODEL_FALLBACK_BACKEND", "ultralytics")
    monkeypatch.setenv("AI_ALLOW_BACKEND_FALLBACK", "false")
    monkeypatch.setenv("AI_RECOGNITION_BACKEND", "insightface")
    monkeypatch.setenv("AI_RECOGNITION_ALLOW_FALLBACK", "false")
    monkeypatch.setenv("AI_MODEL_WEIGHTS_PATH", str(general))
    monkeypatch.setenv("AI_MODEL_PERSON_WEAPON_WEIGHTS_PATH", str(general))
    monkeypatch.setenv("AI_MODEL_WEAPON_WEIGHTS_PATH", str(weapon))
    monkeypatch.setenv("AI_MODEL_FIRE_SMOKE_WEIGHTS_PATH", str(fire_smoke))
    monkeypatch.setenv("AI_MODEL_FIRE_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_MODEL_SMOKE_WEIGHTS_PATH", "")
    monkeypatch.setenv("AI_API_EVENT_CALLBACK_URL", "http://api:8000/api/v1/detections/ingest")
    monkeypatch.setenv("AI_API_EVENT_CALLBACK_TOKEN", "service-token")

    settings = Settings()

    assert settings.model_weapon_weights_path == str(weapon)
    assert settings.model_fire_smoke_weights_path == str(fire_smoke)
