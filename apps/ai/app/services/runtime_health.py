import csv
import subprocess
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.core.model_promotion import collect_promotion_status
from app.core.runtime_gates import load_runtime_gate_report


def collect_runtime_health(backend_state: dict[str, Any] | None = None) -> dict[str, object]:
    gpu_available = False
    gpu_name: str | None = None
    gpu_memory_total_mb: int | None = None
    gpu_memory_used_mb: int | None = None
    gpu_utilization_percent: float | None = None
    telemetry_supported = False
    detail: str | None = None

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            telemetry_supported = True
            gpu_available = True
            row = next(csv.reader([result.stdout.splitlines()[0]]))
            gpu_name = row[0].strip() if len(row) > 0 else None
            gpu_memory_total_mb = int(row[1].strip()) if len(row) > 1 and row[1].strip() else None
            gpu_memory_used_mb = int(row[2].strip()) if len(row) > 2 and row[2].strip() else None
            gpu_utilization_percent = float(row[3].strip()) if len(row) > 3 and row[3].strip() else None
        else:
            detail = "GPU telemetry is unavailable on this host."
    except FileNotFoundError:
        detail = "GPU telemetry is unavailable on this host."
    except Exception as exc:
        detail = str(exc)

    backend_state = backend_state or {}
    runtime_state = backend_state.get("runtime", {}) if isinstance(backend_state, dict) else {}
    capacity_state = backend_state.get("capacity", {}) if isinstance(backend_state, dict) else {}
    validation_gates = load_runtime_gate_report(settings.runtime_gate_report_path)

    return {
        "status": "ok",
        "inference_backend": settings.model_backend,
        "fallback_backend": settings.model_fallback_backend,
        "recognition_backend": settings.recognition_backend,
        "recognition_providers": list(settings.recognition_insightface_providers),
        "model_device": settings.model_device,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "gpu_memory_total_mb": gpu_memory_total_mb,
        "gpu_memory_used_mb": gpu_memory_used_mb,
        "gpu_utilization_percent": gpu_utilization_percent,
        "telemetry_supported": telemetry_supported,
        "detail": detail,
        "runtime": runtime_state if isinstance(runtime_state, dict) else {},
        "capacity": capacity_state if isinstance(capacity_state, dict) else {},
        "validation_gates": validation_gates,
        "models": {
            "person": _model_status(
                settings.model_person_weapon_weights_path or settings.model_weights_path
            ),
            "weapon": _model_status(
                settings.model_weapon_weights_path,
                detector="weapon",
                promotion_path=settings.model_weapon_promotion_path,
            ),
            "fire_smoke": _model_status(
                settings.model_fire_smoke_weights_path,
                detector="fire_smoke",
                promotion_path=settings.model_fire_smoke_promotion_path,
            ),
            "fire": _model_status(
                settings.model_fire_weights_path
                or settings.model_fire_smoke_weights_path,
                detector="fire" if settings.model_fire_weights_path else None,
                promotion_path=settings.model_fire_promotion_path,
            ),
            "smoke": _model_status(
                settings.model_smoke_weights_path
                or settings.model_fire_smoke_weights_path,
                detector="smoke" if settings.model_smoke_weights_path else None,
                promotion_path=settings.model_smoke_promotion_path,
            ),
        },
        "confidence_thresholds": {
            "person": settings.person_confidence_threshold,
            "weapon": settings.weapon_confidence_threshold,
            "fire": settings.fire_confidence_threshold,
            "smoke": settings.smoke_confidence_threshold,
        },
        "temporal_confirmation": {
            "enabled": settings.temporal_confirmation_enabled,
            "max_gap_seconds": settings.temporal_confirmation_max_gap_seconds,
            "allowed_misses": settings.temporal_confirmation_allowed_misses,
            "required_frames": {
                "weapon": settings.weapon_confirmation_frames,
                "fire": settings.fire_confirmation_frames,
                "smoke": settings.smoke_confirmation_frames,
                "known_person": settings.recognition_confirmation_frames,
            },
            "immediate_confidence": {
                "weapon": settings.weapon_immediate_confidence,
                "fire": settings.fire_immediate_confidence,
                "smoke": settings.smoke_immediate_confidence,
                "known_person": settings.recognition_immediate_confidence,
            },
        },
    }


def _model_status(
    path: str | None,
    *,
    detector: str | None = None,
    promotion_path: str | None = None,
) -> dict[str, object]:
    candidate = Path(path) if path else None
    exists = bool(candidate and candidate.exists())
    size_bytes: int | None = None
    if candidate and candidate.is_file():
        size_bytes = candidate.stat().st_size
    elif candidate and candidate.is_dir():
        size_bytes = sum(item.stat().st_size for item in candidate.rglob("*") if item.is_file())
    payload: dict[str, object] = {
        "path": str(candidate) if candidate else None,
        "configured": candidate is not None,
        "exists": exists,
        "artifact_type": "directory" if candidate and candidate.is_dir() else "file",
        "size_bytes": size_bytes,
    }
    if detector in {"weapon", "fire_smoke", "fire", "smoke"}:
        payload["promotion"] = collect_promotion_status(
            detector=detector,
            model_path=path,
            promotion_path=promotion_path,
            required=settings.environment.strip().lower() == "production",
            min_signatures=settings.model_promotion_min_signatures,
            min_weapon_recall=settings.model_promotion_weapon_min_recall,
        )
    return payload
