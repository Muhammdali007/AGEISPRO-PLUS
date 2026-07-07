from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import require_roles
from app.db.session import get_db
from app.models.person import Person
from app.models.user import User, UserRole
from app.repositories.audit_logs import AuditLogRepository
from app.repositories.persons import PersonRepository
from app.schemas.persons import (
    PersonCreate,
    PersonEmbeddingMatchRequest,
    PersonEmbeddingMatchResponse,
    PersonFaceEnrollment,
    PersonRead,
    PersonUpdate,
)
from app.services.persons import PersonService
from app.services.audit_logs import AuditLogService

router = APIRouter()


def get_person_service(session: AsyncSession = Depends(get_db)) -> PersonService:
    return PersonService(PersonRepository(session))


@router.get("", response_model=list[PersonRead], response_model_by_alias=False)
async def list_persons(
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    persons: PersonService = Depends(get_person_service),
) -> list[Person]:
    return await persons.list()


@router.post("/match", response_model=PersonEmbeddingMatchResponse, response_model_by_alias=False)
async def match_person_embeddings(
    payload: PersonEmbeddingMatchRequest,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    persons: PersonService = Depends(get_person_service),
) -> PersonEmbeddingMatchResponse:
    return await persons.match_embeddings(payload)


@router.post("", response_model=PersonRead, response_model_by_alias=False, status_code=status.HTTP_201_CREATED)
async def create_person(
    payload: PersonCreate,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    persons: PersonService = Depends(get_person_service),
) -> Person:
    try:
        person = await persons.create(payload)
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Reference ID already exists")
    await AuditLogService(AuditLogRepository(persons.repository.session)).record(
        actor=current_user,
        action="persons.create",
        resource_type="person",
        resource_id=str(person.id),
        metadata={"reference_id": person.reference_id, "person_type": person.person_type},
    )
    return person


@router.get("/{person_id}", response_model=PersonRead, response_model_by_alias=False)
async def get_person(
    person_id: UUID,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    persons: PersonService = Depends(get_person_service),
) -> Person:
    return await persons.get_or_404(person_id)


@router.patch("/{person_id}", response_model=PersonRead, response_model_by_alias=False)
async def update_person(
    person_id: UUID,
    payload: PersonUpdate,
    _: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    persons: PersonService = Depends(get_person_service),
) -> Person:
    try:
        return await persons.update(person_id, payload)
    except IntegrityError:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Reference ID already exists")


@router.post("/{person_id}/faces", response_model=PersonRead, response_model_by_alias=False)
async def enroll_person_face(
    person_id: UUID,
    payload: PersonFaceEnrollment,
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    persons: PersonService = Depends(get_person_service),
) -> Person:
    person = await persons.enroll_face(person_id, payload)
    await AuditLogService(AuditLogRepository(persons.repository.session)).record(
        actor=current_user,
        action="persons.face_enroll",
        resource_type="person",
        resource_id=str(person.id),
        metadata={"face_count": person.face_image_count},
    )
    return person


@router.post("/{person_id}/faces/upload", response_model=PersonRead, response_model_by_alias=False)
async def upload_person_faces(
    person_id: UUID,
    files: list[UploadFile] = File(...),
    is_primary: bool = Form(False),
    current_user: User = Depends(require_roles(UserRole.administrator, UserRole.supervisor, UserRole.operator)),
    persons: PersonService = Depends(get_person_service),
) -> Person:
    person = await persons.enroll_face_images(person_id, files, is_primary=is_primary)
    await AuditLogService(AuditLogRepository(persons.repository.session)).record(
        actor=current_user,
        action="persons.face_enroll",
        resource_type="person",
        resource_id=str(person.id),
        metadata={"face_count": person.face_image_count, "uploaded_files": len(files)},
    )
    return person


@router.get("/{person_id}/faces/{face_id}/image")
async def get_person_face_image(
    person_id: UUID,
    face_id: str,
    _: User = Depends(
        require_roles(
            UserRole.administrator,
            UserRole.supervisor,
            UserRole.operator,
            UserRole.viewer,
        )
    ),
    persons: PersonService = Depends(get_person_service),
) -> FileResponse:
    face_image = await persons.get_face_image(person_id, face_id)
    if not face_image.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Face image not found")
    return FileResponse(face_image)
