from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.audit import redact_audit_metadata
from app.models.camera import CameraStatus


MonitoringWindow = Literal["24h", "7d", "30d"]


class MonitoringKpis(BaseModel):
    incident_volume: int
    active_alerts: int
    online_camera_ratio: float
    average_confidence: float


class MonitoringSeriesPoint(BaseModel):
    bucket: str
    label: str
    value: int


class DetectionMixPoint(BaseModel):
    detection_type: str
    count: int


class CameraHealthSummary(BaseModel):
    total: int
    online: int
    offline: int
    degraded: int
    disabled: int
    unknown: int
    stale: int
    detection_enabled: int
    groups: dict[str, int] = Field(default_factory=dict)


class CameraHealthEntry(BaseModel):
    camera_id: UUID
    name: str
    status: CameraStatus
    group: str | None
    last_seen_at: datetime | None
    health_checked_at: datetime | None
    stale: bool
    detection_enabled: bool


class CameraHealthReport(BaseModel):
    stale_threshold_minutes: int
    generated_at: datetime
    summary: CameraHealthSummary
    entries: list[CameraHealthEntry]


class AiRuntimeHealth(BaseModel):
    status: str
    inference_backend: str | None = None
    fallback_backend: str | None = None
    recognition_backend: str | None = None
    recognition_providers: list[str] = Field(default_factory=list)
    model_device: str | None = None
    gpu_available: bool
    gpu_name: str | None = None
    gpu_memory_total_mb: int | None = None
    gpu_memory_used_mb: int | None = None
    gpu_utilization_percent: float | None = None
    telemetry_supported: bool
    detail: str | None = None
    runtime: dict[str, Any] = Field(default_factory=dict)
    capacity: dict[str, Any] = Field(default_factory=dict)
    validation_gates: dict[str, Any] = Field(default_factory=dict)


class SystemDependencyStatus(BaseModel):
    status: str
    detail: str | None = None


class SystemHealthReport(BaseModel):
    generated_at: datetime
    api: SystemDependencyStatus
    database: SystemDependencyStatus
    redis: SystemDependencyStatus
    ai: AiRuntimeHealth


class OptimizationResourceSnapshot(BaseModel):
    incidents_total: int
    incidents_last_24h: int
    active_alerts_total: int
    alerts_last_24h: int
    audit_logs_total: int
    audit_logs_last_24h: int


class DatabaseOptimizationReport(BaseModel):
    status: str
    pool_size: int
    max_overflow: int
    pool_recycle_seconds: int
    indexed_paths: list[str] = Field(default_factory=list)
    resources: OptimizationResourceSnapshot
    detail: str | None = None


class RedisOptimizationReport(BaseModel):
    status: str
    ping_ms: float | None = None
    used_memory_human: str | None = None
    connected_clients: int | None = None
    pubsub_channels: int | None = None
    detail: str | None = None


class RuntimeOptimizationReport(BaseModel):
    status: str
    inference_backend: str | None = None
    recognition_backend: str | None = None
    gpu_available: bool
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: int | None = None
    gpu_memory_total_mb: int | None = None
    detail: str | None = None
    capacity: dict[str, Any] = Field(default_factory=dict)
    validation_gates: dict[str, Any] = Field(default_factory=dict)


class OptimizationRecommendation(BaseModel):
    title: str
    detail: str
    severity: Literal["info", "warning", "critical"]


class OptimizationReport(BaseModel):
    generated_at: datetime
    database: DatabaseOptimizationReport
    redis: RedisOptimizationReport
    runtime: RuntimeOptimizationReport
    recommendations: list[OptimizationRecommendation]


class MonitoringOverview(BaseModel):
    window: MonitoringWindow
    generated_at: datetime
    kpis: MonitoringKpis
    incidents_over_time: list[MonitoringSeriesPoint]
    detection_mix: list[DetectionMixPoint]
    camera_health: CameraHealthSummary
    system_health: SystemHealthReport


class AuditLogEntry(BaseModel):
    id: UUID
    actor_user_id: UUID | None
    actor_email: str | None
    actor_role: str | None
    action: str
    resource_type: str
    resource_id: str | None
    metadata: dict[str, Any] = Field(alias="metadata_")
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True, serialize_by_alias=False)

    @field_validator("metadata", mode="before")
    @classmethod
    def redact_sensitive_metadata(cls, value: Any) -> dict[str, Any]:
        redacted = redact_audit_metadata(value or {})
        return redacted if isinstance(redacted, dict) else {}


class AuditLogPage(BaseModel):
    items: list[AuditLogEntry]
    total: int
    limit: int
    offset: int
