from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.db.bootstrap import bootstrap_database
from app.services.continuous_detection import continuous_detection_worker
from app.services.incident_retention import incident_cleanup_worker


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    if settings.auto_create_tables:
        await bootstrap_database()
    if settings.environment != "test":
        incident_cleanup_worker.start()
    if settings.continuous_detection_enabled and settings.environment != "test":
        continuous_detection_worker.start()
    yield
    await incident_cleanup_worker.stop()
    await continuous_detection_worker.stop()


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    description="AegisPro API with auth, camera operations, and AI detection event ingestion.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["system"])
async def root() -> dict[str, str]:
    return {"service": "aegispro-api", "status": "ok"}
