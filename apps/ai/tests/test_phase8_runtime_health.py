from app.services.runtime_health import collect_runtime_health


def test_runtime_health_returns_nullable_gpu_metrics_when_unavailable() -> None:
    payload = collect_runtime_health()

    assert payload["status"] == "ok"
    assert "inference_backend" in payload
    assert "recognition_backend" in payload
    assert "recognition_providers" in payload
    assert "gpu_available" in payload
    assert "telemetry_supported" in payload
    if not payload["gpu_available"]:
        assert payload["gpu_name"] is None or isinstance(payload["gpu_name"], str)
        assert payload["gpu_memory_total_mb"] is None or isinstance(payload["gpu_memory_total_mb"], int)
