from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.camera import Camera
from app.repositories.alerts import AlertRepository
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.schemas.monitoring import (
    AuditLogPage,
    CameraHealthEntry,
    CameraHealthReport,
    CameraHealthSummary,
    DetectionMixPoint,
    MonitoringKpis,
    MonitoringOverview,
    MonitoringSeriesPoint,
    MonitoringWindow,
)
from app.services.system_health import collect_system_health

STALE_THRESHOLD_MINUTES = 5


class MonitoringService:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self.cameras = CameraRepository(session)
        self.incidents = IncidentRepository(session)
        self.alerts = AlertRepository(session)
        self.audit_logs = AuditLogRepository(session)

    async def overview(self, window: MonitoringWindow) -> MonitoringOverview:
        now = datetime.now(UTC)
        started_at = now - self._window_delta(window)
        cameras = await self.cameras.list()
        incidents = [
            incident
            for incident in await self.incidents.list()
            if self._ensure_utc(incident.occurred_at) >= started_at
        ]
        alerts = [
            alert
            for alert in await self.alerts.list()
            if self._ensure_utc(alert.created_at) >= started_at
        ]

        online = sum(camera.status.value == "online" for camera in cameras)
        average_confidence = (
            round(sum(incident.confidence for incident in incidents) / len(incidents), 4)
            if incidents
            else 0.0
        )

        return MonitoringOverview(
            window=window,
            generated_at=now,
            kpis=MonitoringKpis(
                incident_volume=len(incidents),
                active_alerts=sum(alert.status.value == "active" for alert in alerts),
                online_camera_ratio=round((online / len(cameras)) if cameras else 0.0, 4),
                average_confidence=average_confidence,
            ),
            incidents_over_time=self._bucket_incidents(incidents, started_at, now, window),
            detection_mix=[
                DetectionMixPoint(detection_type=key, count=value)
                for key, value in sorted(
                    Counter(incident.detection_type.value for incident in incidents).items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ],
            camera_health=self._camera_health_summary(cameras, now),
            system_health=await collect_system_health(self.session),
        )

    async def camera_health(self) -> CameraHealthReport:
        now = datetime.now(UTC)
        cameras = await self.cameras.list()
        return CameraHealthReport(
            stale_threshold_minutes=STALE_THRESHOLD_MINUTES,
            generated_at=now,
            summary=self._camera_health_summary(cameras, now),
            entries=[
                CameraHealthEntry(
                    camera_id=camera.id,
                    name=camera.name,
                    status=camera.status,
                    group=camera.group,
                    last_seen_at=camera.last_seen_at,
                    health_checked_at=camera.health_checked_at,
                    stale=self._is_stale(camera, now),
                    detection_enabled=camera.detection_enabled,
                )
                for camera in cameras
            ],
        )

    async def audit_log_page(
        self,
        *,
        action: str | None,
        actor_email: str | None,
        resource_type: str | None,
        date_from: datetime | None,
        date_to: datetime | None,
        limit: int,
        offset: int,
    ) -> AuditLogPage:
        items, total = await self.audit_logs.list(
            action=action,
            actor_email=actor_email,
            resource_type=resource_type,
            date_from=date_from,
            date_to=date_to,
            limit=limit,
            offset=offset,
        )
        return AuditLogPage(items=items, total=total, limit=limit, offset=offset)

    @staticmethod
    def _window_delta(window: MonitoringWindow) -> timedelta:
        return {
            "24h": timedelta(hours=24),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }[window]

    @staticmethod
    def _is_stale(camera: Camera, now: datetime) -> bool:
        if camera.last_seen_at is None:
            return camera.status.value not in {"disabled", "unknown"}
        return now - MonitoringService._ensure_utc(camera.last_seen_at) > timedelta(
            minutes=STALE_THRESHOLD_MINUTES
        )

    def _camera_health_summary(self, cameras: list[Camera], now: datetime) -> CameraHealthSummary:
        groups = Counter(camera.group or "ungrouped" for camera in cameras)
        stale = sum(self._is_stale(camera, now) for camera in cameras)
        return CameraHealthSummary(
            total=len(cameras),
            online=sum(camera.status.value == "online" for camera in cameras),
            offline=sum(camera.status.value == "offline" for camera in cameras),
            degraded=sum(camera.status.value == "degraded" for camera in cameras),
            disabled=sum(camera.status.value == "disabled" for camera in cameras),
            unknown=sum(camera.status.value == "unknown" for camera in cameras),
            stale=stale,
            detection_enabled=sum(camera.detection_enabled for camera in cameras),
            groups=dict(sorted(groups.items())),
        )

    @staticmethod
    def _bucket_incidents(incidents: list, started_at: datetime, now: datetime, window: MonitoringWindow) -> list[MonitoringSeriesPoint]:
        if window == "24h":
            bucket_count = 24
            step = timedelta(hours=1)
            label_format = "%H:%M"
        elif window == "7d":
            bucket_count = 7
            step = timedelta(days=1)
            label_format = "%b %d"
        else:
            bucket_count = 30
            step = timedelta(days=1)
            label_format = "%b %d"

        buckets: list[tuple[datetime, datetime]] = []
        cursor = started_at
        for _ in range(bucket_count):
            next_cursor = cursor + step
            buckets.append((cursor, next_cursor))
            cursor = next_cursor

        points: list[MonitoringSeriesPoint] = []
        for bucket_start, bucket_end in buckets:
            count = sum(
                bucket_start <= MonitoringService._ensure_utc(incident.occurred_at) < bucket_end
                for incident in incidents
            )
            points.append(
                MonitoringSeriesPoint(
                    bucket=bucket_start.isoformat(),
                    label=bucket_start.strftime(label_format),
                    value=count,
                )
            )
        if points and points[-1].bucket > now.isoformat():
            points[-1].label = now.strftime(label_format)
        return points

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
