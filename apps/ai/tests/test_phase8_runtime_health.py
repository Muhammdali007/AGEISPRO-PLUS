import json

import pytest

from app.services.runtime_health import _model_status, collect_runtime_health


def test_runtime_health_returns_nullable_gpu_metrics_when_unavailable() -> None:
    payload = collect_runtime_health()

    assert payload["status"] == "ok"
    assert "inference_backend" in payload
    assert "recognition_backend" in payload
    assert "recognition_providers" in payload
    assert "gpu_available" in payload
    assert "telemetry_supported" in payload
    assert "promotion" in payload["models"]["weapon"]
    assert "promotion" in payload["models"]["fire_smoke"]
    assert "runtime" in payload
    assert "capacity" in payload
    assert "validation_gates" in payload
    assert payload["temporal_confirmation"]["required_frames"]["weapon"] >= 1
    assert payload["temporal_confirmation"]["required_frames"]["fire"] >= 1
    if not payload["gpu_available"]:
        assert payload["gpu_name"] is None or isinstance(payload["gpu_name"], str)
        assert payload["gpu_memory_total_mb"] is None or isinstance(payload["gpu_memory_total_mb"], int)


def test_runtime_health_reads_validation_gates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    gate_report = tmp_path / "runtime-gates.json"
    gate_report.write_text(
        json.dumps(
            {
                "generated_at": "2026-07-14T10:00:00Z",
                "suite_id": "prod-candidate",
                "gates": {
                    "load": {"status": "pass", "duration_hours": 1, "detail": "Load gate passed."},
                    "soak_8h": {"status": "pass", "duration_hours": 8},
                    "soak_24h": {"status": "pass", "duration_hours": 24},
                    "soak_72h": {"status": "pass", "duration_hours": 72},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("app.services.runtime_health.settings.runtime_gate_report_path", str(gate_report))

    payload = collect_runtime_health({"runtime": {"loaded_model_count": 2}, "capacity": {"max_batch_size": 8}})

    assert payload["runtime"]["loaded_model_count"] == 2
    assert payload["capacity"]["max_batch_size"] == 8
    assert payload["validation_gates"]["status"] == "pass"
    assert payload["validation_gates"]["gates"]["soak_72h"]["status"] == "pass"


def test_runtime_health_accepts_exported_model_directory(tmp_path) -> None:
    model_path = tmp_path / "weapon_openvino_model"
    model_path.mkdir()
    (model_path / "weapon.xml").write_bytes(b"graph")
    (model_path / "weapon.bin").write_bytes(b"weights")

    payload = _model_status(str(model_path))

    assert payload["exists"] is True
    assert payload["artifact_type"] == "directory"
    assert payload["size_bytes"] == 12
