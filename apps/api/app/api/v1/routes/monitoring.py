from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_roles
from app.db.session import get_db
from app.models.user import User, UserRole
from app.schemas.monitoring import (
    AuditLogPage,
    CameraHealthReport,
    MonitoringOverview,
    MonitoringWindow,
    OptimizationReport,
    SystemHealthReport,
)
from app.services.monitoring import MonitoringService
from app.services.optimization import OptimizationService
from app.services.system_health import collect_system_health

router = APIRouter()


def get_monitoring_service(session: AsyncSession = Depends(get_db)) -> MonitoringService:
    return MonitoringService(session)


def get_optimization_service(session: AsyncSession = Depends(get_db)) -> OptimizationService:
    return OptimizationService(session)


@router.get("/overview", response_model=MonitoringOverview)
async def get_monitoring_overview(
    window: MonitoringWindow = Query(default="24h"),
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    monitoring: MonitoringService = Depends(get_monitoring_service),
) -> MonitoringOverview:
    return await monitoring.overview(window)


@router.get("/camera-health", response_model=CameraHealthReport)
async def get_camera_health(
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    monitoring: MonitoringService = Depends(get_monitoring_service),
) -> CameraHealthReport:
    return await monitoring.camera_health()


@router.get("/system-health", response_model=SystemHealthReport)
async def get_system_health(
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    session: AsyncSession = Depends(get_db),
) -> SystemHealthReport:
    return await collect_system_health(session)


@router.get("/optimization", response_model=OptimizationReport)
async def get_optimization_report(
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    optimization: OptimizationService = Depends(get_optimization_service),
) -> OptimizationReport:
    return await optimization.report()


@router.get("/audit-logs", response_model=AuditLogPage)
async def get_audit_logs(
    action: str | None = None,
    actor_email: str | None = None,
    resource_type: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    monitoring: MonitoringService = Depends(get_monitoring_service),
) -> AuditLogPage:
    return await monitoring.audit_log_page(
        action=action,
        actor_email=actor_email,
        resource_type=resource_type,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        offset=offset,
    )
