import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.core.config import settings
from app.schemas.inference import (
    InferenceBatchRequest,
    InferenceBatchResult,
    InferenceEventDispatchResult,
    InferenceRequest,
    InferenceResult,
)
from app.services.pipeline import InferencePipeline
from app.services.runtime_health import collect_runtime_health


pipeline = InferencePipeline()
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if settings.environment != "test" and settings.model_preload_on_startup:
        await asyncio.to_thread(pipeline.warmup)
    yield


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description="Independent inference service for weapon, fire, smoke, and person detection.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"service": "aegispro-ai", "status": "ok", "backend": settings.model_backend}


@app.get("/health/runtime")
async def runtime_health() -> dict[str, object]:
    return collect_runtime_health(pipeline.primary_backend.snapshot_runtime_state())


@app.post("/v1/inference/run", response_model=InferenceResult)
async def run_inference(payload: InferenceRequest) -> InferenceResult:
    async with inference_lock:
        return await asyncio.to_thread(pipeline.run, payload)


@app.post("/v1/inference/run-batch", response_model=InferenceBatchResult)
async def run_batch_inference(payload: InferenceBatchRequest) -> InferenceBatchResult:
    async with inference_lock:
        results = await asyncio.to_thread(pipeline.run_batch, payload.requests)
    return InferenceBatchResult(results=results)


@app.post("/v1/inference/dispatch", response_model=InferenceEventDispatchResult)
async def dispatch_inference(payload: InferenceRequest) -> InferenceEventDispatchResult:
    async with inference_lock:
        result = await asyncio.to_thread(pipeline.run, payload)
    return await asyncio.to_thread(pipeline.dispatch_events, result)
