from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_roles
from app.db.transactions import transaction_scope
from app.db.session import get_db
from app.models.incident import DetectionType, Incident, IncidentPriority, IncidentStatus
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.alerts import AlertRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.persons import PersonRepository
from app.repositories.users import UserRepository
from app.schemas.alerts import AlertRead
from app.schemas.incidents import (
    IncidentCreate,
    IncidentRead,
    IncidentRetentionPolicyRead,
    IncidentSavePersonRequest,
    IncidentUpdate,
)
from app.schemas.persons import PersonCreate, PersonRead
from app.services.audit_logs import AuditLogService
from app.services.evidence_storage import EvidenceStorageService
from app.services.incident_retention_policy import get_documented_incident_retention_policies
from app.services.persons import PersonService
from app.services.transactional_outbox import TransactionalOutboxService

router = APIRouter()


def get_incident_repository(session: AsyncSession = Depends(get_db)) -> IncidentRepository:
    return IncidentRepository(session)


@router.get("", response_model=list[IncidentRead], response_model_by_alias=False)
async def list_incidents(
    camera_id: UUID | None = None,
    status_filter: IncidentStatus | None = None,
    detection_type: DetectionType | None = None,
    priority: IncidentPriority | None = None,
    assigned_user_id: UUID | None = None,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    incidents: IncidentRepository = Depends(get_incident_repository),
) -> list[Incident]:
    return await incidents.list(
        camera_id=camera_id,
        status=status_filter,
        detection_type=detection_type,
        priority=priority,
        assigned_user_id=assigned_user_id,
    )


@router.post("", response_model=IncidentRead, response_model_by_alias=False, status_code=status.HTTP_201_CREATED)
async def create_incident(
    payload: IncidentCreate,
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    session: AsyncSession = Depends(get_db),
) -> Incident:
    camera = await CameraRepository(session).get(payload.camera_id)
    if not camera:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Camera not found")
    if payload.assigned_user_id and not await UserRepository(session).get_by_id(payload.assigned_user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assigned user not found")
    async with transaction_scope(session):
        incident = await IncidentRepository(session).create(payload)
    return incident


@router.get(
    "/retention-policies",
    response_model=list[IncidentRetentionPolicyRead],
    response_model_by_alias=False,
)
async def list_incident_retention_policies(
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
) -> list[IncidentRetentionPolicyRead]:
    policies = get_documented_incident_retention_policies()
    return [
        IncidentRetentionPolicyRead(
            retention_class=retention_class,
            retention_hours=policy.retention_hours,
            description=policy.description,
        )
        for retention_class, policy in policies.items()
    ]


@router.get("/{incident_id}", response_model=IncidentRead, response_model_by_alias=False)
async def get_incident(
    incident_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    incidents: IncidentRepository = Depends(get_incident_repository),
) -> Incident:
    incident = await incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    return incident


@router.patch("/{incident_id}", response_model=IncidentRead, response_model_by_alias=False)
async def update_incident(
    incident_id: UUID,
    payload: IncidentUpdate,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    session: AsyncSession = Depends(get_db),
) -> Incident:
    incident_repository = IncidentRepository(session)
    incident = await incident_repository.get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    if payload.assigned_user_id and not await UserRepository(session).get_by_id(payload.assigned_user_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Assigned user not found")
    outbox = TransactionalOutboxService(session)
    async with transaction_scope(session) as scope:
        updated_incident = await incident_repository.update(incident, payload)
        await outbox.enqueue(
            {
                "type": "incident.updated",
                "incident_id": str(updated_incident.id),
                "camera_id": str(updated_incident.camera_id),
                "status": updated_incident.status.value,
            }
        )
        await AuditLogService(AuditLogRepository(session)).record(
            actor=current_user,
            action="incidents.update",
            resource_type="incident",
            resource_id=str(updated_incident.id),
            metadata=payload.model_dump(exclude_unset=True),
        )
    if scope.owns_transaction:
        await outbox.publish_pending()
    return updated_incident


@router.delete("/{incident_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_incident(
    incident_id: UUID,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor)),
    session: AsyncSession = Depends(get_db),
) -> None:
    incident_repository = IncidentRepository(session)
    incident = await incident_repository.get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    if incident.legal_hold:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Incident is on legal hold and cannot be archived for deletion",
        )

    outbox = TransactionalOutboxService(session)
    async with transaction_scope(session) as scope:
        await AuditLogService(AuditLogRepository(session)).record(
            actor=current_user,
            action="incidents.delete",
            resource_type="incident",
            resource_id=str(incident.id),
            metadata={
                "camera_id": str(incident.camera_id),
                "detection_type": incident.detection_type.value,
                "priority": incident.priority.value,
                "status": incident.status.value,
                "occurred_at": incident.occurred_at.isoformat(),
                "retention_class": incident.retention_class.value,
                "legal_hold": incident.legal_hold,
                "deletion_mode": "archive_then_async_delete",
            },
        )
        await outbox.enqueue(
            {
                "type": "incident.archived",
                "incident_id": str(incident.id),
                "camera_id": str(incident.camera_id),
                "status": incident.status.value,
            }
        )
        await incident_repository.archive(incident)
    if scope.owns_transaction:
        await outbox.publish_pending()


@router.post("/{incident_id}/save-person", response_model=PersonRead, response_model_by_alias=False)
async def save_incident_person(
    incident_id: UUID,
    payload: IncidentSavePersonRequest,
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    session: AsyncSession = Depends(get_db),
) -> PersonRead:
    incident_repository = IncidentRepository(session)
    incident = await incident_repository.get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")

    image_path = None
    if incident.recognized_identity:
        image_path = incident.recognized_identity.get("face_image_path")
    image_path = image_path or incident.snapshot_path
    if not image_path:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This incident does not include a captured face image yet",
        )

    persons = PersonService(PersonRepository(session))
    try:
        async with transaction_scope(session):
            person = await persons.create(
                PersonCreate(
                    full_name=payload.full_name,
                    person_type=payload.person_type,
                    department=payload.department,
                    reference_id=payload.reference_id,
                    title=payload.title,
                    is_active=payload.is_active,
                    metadata={
                        **payload.metadata,
                        "source_incident_id": str(incident.id),
                        "source_camera_id": str(incident.camera_id),
                        "source_detection_type": incident.detection_type.value,
                    },
                )
            )

            try:
                person = await persons.enroll_face_image_from_storage(
                    person.id,
                    image_path,
                    label=payload.full_name,
                    is_primary=payload.is_primary,
                    metadata={
                        "source_incident_id": str(incident.id),
                        "source_camera_id": str(incident.camera_id),
                        "source_face_image_path": image_path,
                        "source_snapshot_path": incident.snapshot_path,
                    },
                )
            except Exception:
                await persons.delete(person.id)
                raise

            recognized_identity = dict(incident.recognized_identity or {})
            recognized_identity.update(
                {
                    "status": "known",
                    "identity_id": str(person.id),
                    "identity_label": person.full_name,
                    "face_image_path": person.face_profiles[-1]["image_path"] if person.face_profiles else image_path,
                }
            )
            incident.detection_type = DetectionType.known_person
            incident.recognized_identity = recognized_identity
            incident.metadata_ = {
                **incident.metadata_,
                "saved_person_id": str(person.id),
                "saved_from_incident": True,
            }
            await session.flush()
            await session.refresh(incident)
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Reference ID already exists")
    return person


@router.get("/{incident_id}/alerts", response_model=list[AlertRead])
async def list_incident_alerts(
    incident_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    session: AsyncSession = Depends(get_db),
) -> list[AlertRead]:
    incident = await IncidentRepository(session).get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    return await AlertRepository(session).list(incident_id=incident_id)


@router.get("/{incident_id}/snapshot")
async def get_incident_snapshot(
    incident_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    incidents: IncidentRepository = Depends(get_incident_repository),
) -> FileResponse:
    incident = await incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    if not incident.snapshot_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident snapshot not found")
    return FileResponse(EvidenceStorageService().ensure_exists(incident.snapshot_path))


@router.get("/{incident_id}/clip")
async def get_incident_clip(
    incident_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    incidents: IncidentRepository = Depends(get_incident_repository),
) -> FileResponse:
    incident = await incidents.get(incident_id)
    if not incident:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident not found")
    if not incident.clip_path:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Incident clip not found")
    return FileResponse(EvidenceStorageService().ensure_exists(incident.clip_path))
