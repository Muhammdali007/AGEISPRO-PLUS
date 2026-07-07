from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from urllib import error, request

from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import engine
from app.schemas.monitoring import AiRuntimeHealth, SystemDependencyStatus, SystemHealthReport


async def read_readiness() -> dict[str, str]:
    database = "ok"
    cache = "ok"
    try:
        async with engine.connect() as connection:
            await connection.execute(text("select 1"))
    except Exception:
        database = "unavailable"

    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        await redis.ping()
    except Exception:
        cache = "unavailable"
    finally:
        await redis.aclose()

    return {
        "status": "ok" if database == "ok" and cache == "ok" else "degraded",
        "database": database,
        "redis": cache,
    }


async def fetch_ai_runtime_health() -> AiRuntimeHealth:
    url = f"{settings.ai_service_url.rstrip('/')}/health/runtime"

    def _load() -> AiRuntimeHealth:
        try:
            with request.urlopen(url, timeout=3) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            return AiRuntimeHealth(
                status="unavailable",
                gpu_available=False,
                telemetry_supported=False,
                detail=str(exc.reason) if getattr(exc, "reason", None) else str(exc),
            )
        except Exception as exc:
            return AiRuntimeHealth(
                status="unavailable",
                gpu_available=False,
                telemetry_supported=False,
                detail=str(exc),
            )
        return AiRuntimeHealth(**payload)

    return await asyncio.to_thread(_load)


async def collect_system_health(_: AsyncSession | None = None) -> SystemHealthReport:
    readiness = await read_readiness()
    ai = await fetch_ai_runtime_health()
    return SystemHealthReport(
        generated_at=datetime.now(UTC),
        api=SystemDependencyStatus(status="ok"),
        database=SystemDependencyStatus(status=readiness["database"]),
        redis=SystemDependencyStatus(status=readiness["redis"]),
        ai=ai,
    )
