from __future__ import annotations

from datetime import UTC, datetime, timedelta
from time import perf_counter

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.alert import AlertStatus
from app.repositories.alerts import AlertRepository
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.incidents import IncidentRepository
from app.schemas.monitoring import (
    DatabaseOptimizationReport,
    OptimizationRecommendation,
    OptimizationReport,
    OptimizationResourceSnapshot,
    RedisOptimizationReport,
    RuntimeOptimizationReport,
)
from app.services.system_health import fetch_ai_runtime_health


class OptimizationService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.incidents = IncidentRepository(session)
        self.alerts = AlertRepository(session)
        self.audit_logs = AuditLogRepository(session)

    async def report(self) -> OptimizationReport:
        now = datetime.now(UTC)
        last_day = now - timedelta(hours=24)

        incidents_total = await self.incidents.count_all()
        incidents_last_24h, _ = await self.incidents.summary_since(last_day)
        active_alerts_total = await self.alerts.count_since(
            started_at=datetime(1970, 1, 1, tzinfo=UTC), status=AlertStatus.active
        )
        alerts_last_24h = await self.alerts.count_since(started_at=last_day)
        audit_logs_total = await self.audit_logs.count_all()
        audit_logs_last_24h = await self.audit_logs.count_since(last_day)

        database = DatabaseOptimizationReport(
            status="ok",
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_recycle_seconds=settings.database_pool_recycle_seconds,
            indexed_paths=[
                "incidents(occurred_at, detection_type)",
                "incidents(status, occurred_at)",
                "alerts(status, created_at)",
                "audit_logs(action, created_at)",
                "audit_logs(resource_type, created_at)",
                "cameras(status, group)",
            ],
            resources=OptimizationResourceSnapshot(
                incidents_total=incidents_total,
                incidents_last_24h=incidents_last_24h,
                active_alerts_total=active_alerts_total,
                alerts_last_24h=alerts_last_24h,
                audit_logs_total=audit_logs_total,
                audit_logs_last_24h=audit_logs_last_24h,
            ),
            detail="Monitoring aggregates are computed through filtered SQL queries to reduce Python-side memory pressure.",
        )

        redis = await self._redis_report()
        runtime = await self._runtime_report()

        return OptimizationReport(
            generated_at=now,
            database=database,
            redis=redis,
            runtime=runtime,
            recommendations=self._recommendations(database, redis, runtime),
        )

    async def _redis_report(self) -> RedisOptimizationReport:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        started = perf_counter()
        try:
            await redis.ping()
            ping_ms = round((perf_counter() - started) * 1000, 2)
            memory_info = await redis.info(section="memory")
            client_info = await redis.info(section="clients")
            stats_info = await redis.info(section="stats")
            return RedisOptimizationReport(
                status="ok",
                ping_ms=ping_ms,
                used_memory_human=memory_info.get("used_memory_human"),
                connected_clients=int(client_info.get("connected_clients", 0)),
                pubsub_channels=int(stats_info.get("pubsub_channels", 0)),
                detail="Redis health verified with a live ping and INFO sampling.",
            )
        except Exception as exc:
            return RedisOptimizationReport(status="unavailable", detail=str(exc))
        finally:
            await redis.aclose()

    async def _runtime_report(self) -> RuntimeOptimizationReport:
        runtime = await fetch_ai_runtime_health()
        return RuntimeOptimizationReport(
            status=runtime.status,
            inference_backend=runtime.inference_backend,
            recognition_backend=runtime.recognition_backend,
            gpu_available=runtime.gpu_available,
            gpu_utilization_percent=runtime.gpu_utilization_percent,
            gpu_memory_used_mb=runtime.gpu_memory_used_mb,
            gpu_memory_total_mb=runtime.gpu_memory_total_mb,
            detail=runtime.detail,
            capacity=runtime.capacity,
            validation_gates=runtime.validation_gates,
        )

    @staticmethod
    def _recommendations(
        database: DatabaseOptimizationReport,
        redis: RedisOptimizationReport,
        runtime: RuntimeOptimizationReport,
    ) -> list[OptimizationRecommendation]:
        recommendations: list[OptimizationRecommendation] = [
            OptimizationRecommendation(
                title="Database-side aggregation",
                detail="Phase 9 monitoring now aggregates incidents and alerts in SQL before shaping dashboard responses.",
                severity="info",
            )
        ]

        if redis.status != "ok":
            recommendations.append(
                OptimizationRecommendation(
                    title="Restore Redis telemetry",
                    detail="Redis is unavailable, which weakens pub/sub fan-out and queue-backed alert workflows.",
                    severity="critical",
                )
            )
        elif redis.ping_ms is not None and redis.ping_ms > 25:
            recommendations.append(
                OptimizationRecommendation(
                    title="Reduce Redis latency",
                    detail=f"Observed Redis ping is {redis.ping_ms} ms. Review host locality and connection reuse before heavier alert fan-out.",
                    severity="warning",
                )
            )

        if runtime.status != "ok":
            recommendations.append(
                OptimizationRecommendation(
                    title="Recover AI runtime telemetry",
                    detail="Optimization planning is limited while the AI runtime health endpoint is unavailable.",
                    severity="critical",
                )
            )
        elif runtime.gpu_available and runtime.gpu_utilization_percent is not None and runtime.gpu_utilization_percent > 90:
            recommendations.append(
                OptimizationRecommendation(
                    title="Watch GPU saturation",
                    detail=f"GPU utilization is {runtime.gpu_utilization_percent}%. Lower inference FPS or scale camera workers before production rollout.",
                    severity="warning",
                )
            )

        gate_status = str(runtime.validation_gates.get("status") or "").lower()
        if gate_status == "missing":
            recommendations.append(
                OptimizationRecommendation(
                    title="Record runtime validation gates",
                    detail="The 8h, 24h, 72h soak or load gate report is missing. Capture the run before promoting this detector stack.",
                    severity="warning",
                )
            )
        elif gate_status == "fail":
            recommendations.append(
                OptimizationRecommendation(
                    title="Do not promote failing runtime gates",
                    detail="At least one soak or load validation gate is failing. Hold rollout until the report is green.",
                    severity="critical",
                )
            )

        if database.resources.audit_logs_total > 10000:
            recommendations.append(
                OptimizationRecommendation(
                    title="Plan audit log retention",
                    detail="Audit volume is growing. Consider archival or partitioning before long-lived production retention windows.",
                    severity="warning",
                )
            )

        return recommendations
