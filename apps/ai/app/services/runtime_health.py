import csv
import subprocess

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
    }
