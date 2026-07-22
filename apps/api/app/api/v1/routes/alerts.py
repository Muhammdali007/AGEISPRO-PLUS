from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, require_roles
from app.db.transactions import transaction_scope
from app.db.session import get_db
from app.models.alert import Alert, AlertStatus
from app.models.incident import IncidentPriority
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.alerts import AlertRepository
from app.repositories.incidents import IncidentRepository
from app.schemas.alerts import AlertCreate, AlertRead
from app.services.audit_logs import AuditLogService
from app.services.transactional_outbox import TransactionalOutboxService

router = APIRouter()


def get_alert_repository(session: AsyncSession = Depends(get_db)) -> AlertRepository:
    return AlertRepository(session)


@router.get("", response_model=list[AlertRead])
async def list_alerts(
    status_filter: AlertStatus | None = None,
    priority: IncidentPriority | None = None,
    incident_id: UUID | None = None,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    alerts: AlertRepository = Depends(get_alert_repository),
) -> list[Alert]:
    return await alerts.list(status=status_filter, priority=priority, incident_id=incident_id)


@router.post("", response_model=AlertRead, status_code=status.HTTP_201_CREATED)
async def create_alert(
    payload: AlertCreate,
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    session: AsyncSession = Depends(get_db),
) -> Alert:
    if not await IncidentRepository(session).get(payload.incident_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    async with transaction_scope(session):
        alert = await AlertRepository(session).create(payload)
    return alert


@router.post("/{alert_id}/acknowledge", response_model=AlertRead)
async def acknowledge_alert(
    alert_id: UUID,
    current_user: User = Depends(get_current_user),
    alerts: AlertRepository = Depends(get_alert_repository),
) -> Alert:
    if current_user.role not in {
        UserRole.administrator,
        UserRole.supervisor,
        UserRole.operator,
    }:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    alert = await alerts.get(alert_id)
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    outbox = TransactionalOutboxService(alerts.session)
    async with transaction_scope(alerts.session) as scope:
        updated_alert = await alerts.acknowledge(alert, current_user.id)
        await outbox.enqueue(
            {
                "type": "alert.acknowledged",
                "alert_id": str(updated_alert.id),
                "incident_id": str(updated_alert.incident_id),
            }
        )
        await AuditLogService(AuditLogRepository(alerts.session)).record(
            actor=current_user,
            action="alerts.acknowledge",
            resource_type="alert",
            resource_id=str(updated_alert.id),
            metadata={"incident_id": str(updated_alert.incident_id)},
        )
    if scope.owns_transaction:
        await outbox.publish_pending()
    return updated_alert


@router.post("/{alert_id}/clear", response_model=AlertRead)
async def clear_alert(
    alert_id: UUID,
    current_user: User = Depends(get_current_user),
    alerts: AlertRepository = Depends(get_alert_repository),
) -> Alert:
    if current_user.role not in {
        UserRole.administrator,
        UserRole.supervisor,
        UserRole.operator,
    }:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
    alert = await alerts.get(alert_id)
    if not alert:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    outbox = TransactionalOutboxService(alerts.session)
    async with transaction_scope(alerts.session) as scope:
        updated_alert = await alerts.clear(alert)
        await outbox.enqueue(
            {
                "type": "alert.cleared",
                "alert_id": str(updated_alert.id),
                "incident_id": str(updated_alert.incident_id),
            }
        )
        await AuditLogService(AuditLogRepository(alerts.session)).record(
            actor=current_user,
            action="alerts.clear",
            resource_type="alert",
            resource_id=str(updated_alert.id),
            metadata={"incident_id": str(updated_alert.incident_id)},
        )
    if scope.owns_transaction:
        await outbox.publish_pending()
    return updated_alert
