from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.core.model_promotion import collect_promotion_status, validate_promotion_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_weapon_promotion_manifest_accepts_signed_holdout_with_gate(tmp_path: Path) -> None:
    weights_path = tmp_path / "weapon.pt"
    weights_path.write_bytes(b"weapon-checkpoint")
    manifest_path = weights_path.with_suffix(".pt.promotion.json")
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "weapon-2026-07-14",
                "checkpoint_sha256": _sha256(weights_path),
                "selected_operating_threshold": 0.27,
                "datasets": [
                    {"name": "Licensed CCTV weapon corpus", "license": "Internal signed collection"},
                    {"name": "Open Images weapon subset", "license": "Open Images terms"},
                ],
                "independent_holdout": {
                    "name": "weapon-holdout-jul2026",
                    "license": "Internal signed collection",
                    "independent_from_training": True,
                },
                "per_class_metrics": {
                    "weapon": {"precision": 0.94, "recall": 0.91},
                    "knife": {"precision": 0.93, "recall": 0.90},
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
                    {
                        "name": "Analyst A",
                        "role": "Model owner",
                        "signed_at": "2026-07-14T10:00:00Z",
                    },
                    {
                        "name": "Reviewer B",
                        "role": "Independent reviewer",
                        "signed_at": "2026-07-14T11:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_promotion_manifest(detector="weapon", model_path=str(weights_path))

    assert payload["valid"] is True
    assert payload["model_id"] == "weapon-2026-07-14"
    assert payload["selected_operating_threshold"] == 0.27
    assert len(payload["signatures"]) == 2


def test_validate_weapon_promotion_manifest_rejects_weak_required_gate(tmp_path: Path) -> None:
    weights_path = tmp_path / "weapon.pt"
    weights_path.write_bytes(b"weapon-checkpoint")
    manifest_path = weights_path.with_suffix(".pt.promotion.json")
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "weapon-weak",
                "checkpoint_sha256": _sha256(weights_path),
                "selected_operating_threshold": 0.25,
                "datasets": [{"name": "Licensed set", "license": "Internal"}],
                "independent_holdout": {
                    "name": "holdout",
                    "license": "Internal",
                    "independent_from_training": True,
                },
                "per_class_metrics": {
                    "weapon": {"precision": 0.95, "recall": 0.88},
                },
                "gates": [
                    {
                        "name": "weapon_recall_at_selected_threshold",
                        "minimum": 0.85,
                        "actual": 0.88,
                        "passed": True,
                    }
                ],
                "signatures": [
                    {
                        "name": "Analyst A",
                        "role": "Model owner",
                        "signed_at": "2026-07-14T10:00:00Z",
                    },
                    {
                        "name": "Reviewer B",
                        "role": "Independent reviewer",
                        "signed_at": "2026-07-14T11:00:00Z",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="weapon_recall_at_selected_threshold >= 0.90"):
        validate_promotion_manifest(detector="weapon", model_path=str(weights_path))


def test_collect_promotion_status_reports_missing_manifest_without_raising(tmp_path: Path) -> None:
    weights_path = tmp_path / "fire-smoke.pt"
    weights_path.write_bytes(b"fire-smoke-checkpoint")

    payload = collect_promotion_status(
        detector="fire_smoke",
        model_path=str(weights_path),
        required=True,
        min_signatures=2,
    )

    assert payload["required"] is True
    assert payload["exists"] is False
    assert payload["valid"] is False
    assert payload["detail"] == "Promotion manifest missing."


def test_validate_promotion_manifest_hashes_exported_model_directory(tmp_path: Path) -> None:
    model_path = tmp_path / "weapon_openvino_model"
    model_path.mkdir()
    (model_path / "weapon.xml").write_text("model-graph", encoding="utf-8")
    (model_path / "weapon.bin").write_bytes(b"model-weights")

    digest = hashlib.sha256()
    for item in sorted(model_path.iterdir()):
        relative = item.relative_to(model_path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        digest.update(item.read_bytes())
    manifest_path = model_path.with_suffix(".promotion.json")
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": "weapon-openvino-2026-07-18",
                "checkpoint_sha256": digest.hexdigest(),
                "selected_operating_threshold": 0.30,
                "datasets": [{"name": "Licensed set", "license": "Internal"}],
                "independent_holdout": {
                    "name": "holdout",
                    "license": "Internal",
                    "independent_from_training": True,
                },
                "per_class_metrics": {"weapon": {"precision": 0.93, "recall": 0.91}},
                "gates": [
                    {
                        "name": "weapon_recall_at_selected_threshold",
                        "minimum": 0.90,
                        "actual": 0.91,
                        "passed": True,
                    }
                ],
                "signatures": [
                    {
                        "name": "Analyst",
                        "role": "Model owner",
                        "signed_at": "2026-07-18T10:00:00Z",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = validate_promotion_manifest(detector="weapon", model_path=str(model_path))

    assert payload["valid"] is True
    assert payload["checkpoint_sha256"] == digest.hexdigest()
