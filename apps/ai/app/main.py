from fastapi import FastAPI

from app.core.config import settings
from app.schemas.inference import (
    InferenceEventDispatchResult,
    InferenceRequest,
    InferenceResult,
)
from app.services.pipeline import InferencePipeline
from app.services.runtime_health import collect_runtime_health

app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description="Independent inference service for weapon, fire, smoke, and person detection.",
)
pipeline = InferencePipeline()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": "aegispro-ai", "status": "ok", "backend": settings.model_backend}


@app.get("/health/runtime")
async def runtime_health() -> dict[str, object]:
    return collect_runtime_health()


@app.post("/v1/inference/run", response_model=InferenceResult)
async def run_inference(payload: InferenceRequest) -> InferenceResult:
    return pipeline.run(payload)


@app.post("/v1/inference/dispatch", response_model=InferenceEventDispatchResult)
async def dispatch_inference(payload: InferenceRequest) -> InferenceEventDispatchResult:
    result = pipeline.run(payload)
    return pipeline.dispatch_events(result)
