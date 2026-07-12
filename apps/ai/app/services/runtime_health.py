import csv
import subprocess
from pathlib import Path

from app.core.config import settings


def collect_runtime_health() -> dict[str, object]:
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
        "models": {
            "person": _model_status(
                settings.model_person_weapon_weights_path or settings.model_weights_path
            ),
            "weapon": _model_status(settings.model_weapon_weights_path),
            "fire_smoke": _model_status(settings.model_fire_smoke_weights_path),
        },
        "confidence_thresholds": {
            "person": settings.person_confidence_threshold,
            "weapon": settings.weapon_confidence_threshold,
            "fire": settings.fire_confidence_threshold,
            "smoke": settings.smoke_confidence_threshold,
        },
    }


def _model_status(path: str | None) -> dict[str, object]:
    candidate = Path(path) if path else None
    exists = bool(candidate and candidate.is_file())
    return {
        "path": str(candidate) if candidate else None,
        "configured": candidate is not None,
        "exists": exists,
        "size_bytes": candidate.stat().st_size if candidate and exists else None,
    }
