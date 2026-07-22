from collections.abc import AsyncIterator
from io import BytesIO
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException, UploadFile
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_current_user
from app.core.config import settings
from app.db.metadata import Base
from app.db.session import get_db
from app.main import app
from app.models.camera import CameraSourceType
from app.models.incident import DetectionType
from app.models.person import Person
from app.models.person_face_embedding import PersonFaceEmbedding
from app.models.user import User, UserRole
from app.repositories.alerts import AlertRepository
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.repositories.persons import PersonRepository
from app.repositories.users import UserRepository
from app.schemas.cameras import CameraCreate
from app.schemas.detections import DetectionEventIngest, DetectionEventIngestItem, RecognitionStatus
from app.schemas.incidents import IncidentCreate
from app.schemas.persons import PersonCreate, PersonFaceEnrollment
from app.schemas.users import UserCreate
from app.services.detection_events import DetectionEventService
from app.services.camera_detection import CameraDetectionService
from app.services.face_embeddings import FaceEmbeddingError, FaceEmbeddingResult
from app.services.face_enrollment import FaceEnrollmentService
from app.services.persons import PersonService


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as test_session:
        yield test_session

    await engine.dispose()


@pytest_asyncio.fixture
async def client(session: AsyncSession) -> AsyncIterator[AsyncClient]:
    async def override_get_db() -> AsyncIterator[AsyncSession]:
        yield session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client
    app.dependency_overrides.clear()


async def _set_current_user(user: User) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            {"backend": "insightface", "face_count": 2, "det_score": 0.99, "bbox": [0, 0, 100, 100]},
            "exactly one visible face",
        ),
        (
            {"backend": "insightface", "face_count": 1, "det_score": 0.40, "bbox": [0, 0, 100, 100]},
            "confidence is too low",
        ),
        (
            {"backend": "insightface", "face_count": 1, "det_score": 0.99, "bbox": [0, 0, 20, 20]},
            "too small",
        ),
    ],
)
def test_face_enrollment_rejects_low_quality_identity_templates(
    metadata: dict[str, object], message: str
) -> None:
    with pytest.raises(HTTPException, match=message):
        FaceEnrollmentService._validate_enrollment_quality(metadata)


def test_face_enrollment_accepts_score_above_runtime_template_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "recognition_enrollment_min_det_score", 0.60)
    metadata: dict[str, object] = {
        "backend": "insightface",
        "face_count": 1,
        "det_score": 0.69,
        "bbox": [0, 0, 100, 100],
    }

    FaceEnrollmentService._validate_enrollment_quality(metadata)

    assert metadata["enrollment_quality_checked"] is True


@pytest.mark.asyncio
async def test_face_upload_rejects_too_many_images() -> None:
    service = object.__new__(FaceEnrollmentService)
    uploads = [
        UploadFile(BytesIO(b"image-contents"), filename=f"face-{index}.jpg")
        for index in range(6)
    ]
    person = Person(full_name="Test Person", reference_id="TEST-001")

    with pytest.raises(HTTPException, match="Select up to 5 face images"):
        await service.build_enrollments_from_uploads(person, uploads)


@pytest.mark.asyncio
async def test_face_upload_rejects_oversized_image() -> None:
    service = object.__new__(FaceEnrollmentService)
    upload = UploadFile(BytesIO(b"x" * ((10 * 1024 * 1024) + 1)), filename="large.jpg")
    person = Person(full_name="Test Person", reference_id="TEST-001")

    with pytest.raises(HTTPException, match="large.jpg is too large"):
        await service.build_enrollments_from_uploads(person, [upload])


@pytest.mark.asyncio
async def test_face_upload_validation_error_names_the_rejected_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = object.__new__(FaceEnrollmentService)

    def reject_image(_contents: bytes):
        raise HTTPException(
            status_code=422,
            detail="Face detection confidence is too low for reliable enrollment",
        )

    monkeypatch.setattr(service, "_extract_embedding", reject_image)
    upload = UploadFile(BytesIO(b"image-contents"), filename="blurry.jpg")
    person = Person(full_name="Test Person", reference_id="TEST-001")

    with pytest.raises(HTTPException, match="blurry.jpg: Face detection confidence"):
        await service.build_enrollments_from_uploads(person, [upload])


def test_runtime_identity_payload_filters_duplicate_and_low_quality_templates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "recognition_runtime_max_templates_per_person", 2)
    model_name = CameraDetectionService._runtime_embedding_model_name()
    profiles = [
        {
            "id": "best",
            "embedding_vector": [1.0] * 16,
            "embedding_model": model_name,
            "metadata": {"det_score": 0.99},
        },
        {
            "id": "duplicate",
            "embedding_vector": [1.0] * 16,
            "embedding_model": model_name,
            "metadata": {"det_score": 0.98},
        },
        {
            "id": "side",
            "embedding_vector": [1.0, -1.0] * 8,
            "embedding_model": model_name,
            "metadata": {"det_score": 0.90},
        },
        {
            "id": "blurred",
            "embedding_vector": [0.5] * 16,
            "embedding_model": model_name,
            "metadata": {"det_score": 0.30},
        },
    ]

    selected = CameraDetectionService._curate_face_profiles(profiles)

    assert [profile["id"] for profile in selected] == ["best", "side"]


@pytest.mark.asyncio
async def test_runtime_identity_payload_refreshes_stale_embedding_model(
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)
    monkeypatch.setattr(settings, "recognition_backend", "insightface")
    monkeypatch.setattr(settings, "recognition_insightface_model", "buffalo_m")

    person = await PersonRepository(session).create(
        PersonCreate(full_name="Dana Holt", person_type="employee", reference_id="EMP-REFRESH")
    )
    face_path = tmp_path / "faces" / "persons" / "emp-refresh" / "front.jpg"
    face_path.parent.mkdir(parents=True, exist_ok=True)
    face_path.write_bytes(b"stale-face-image")
    await PersonRepository(session).add_face_profile(
        person,
        PersonFaceEnrollment(
            image_path="faces/persons/emp-refresh/front.jpg",
            label="Front profile",
            embedding_vector=[0.1] * 512,
            embedding_model="insightface-buffalo_l",
            is_primary=True,
            metadata={"backend": "insightface", "det_score": 0.99},
        ),
    )

    class StubBackend:
        def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
            assert image_bytes == b"stale-face-image"
            return FaceEmbeddingResult(
                vector=[0.2] * 512,
                model_name="insightface-buffalo_m",
                backend_name="insightface",
                metadata={"backend": "insightface", "det_score": 0.98},
            )

    service = CameraDetectionService(session)
    monkeypatch.setattr(service, "_runtime_face_embedding_backend", lambda: StubBackend())

    prepared = await service._prepare_known_persons_for_recognition([person])
    payload = service._serialize_known_person(prepared[0])

    assert payload["face_profiles"][0]["embedding_model"] == "insightface-buffalo_m"
    assert payload["face_profiles"][0]["embedding_vector"] == [0.2] * 512
    assert payload["face_profiles"][0]["metadata"]["runtime_embedding_refreshed"] is True

    embedding_row = await session.scalar(select(PersonFaceEmbedding))
    assert embedding_row is not None
    assert embedding_row.embedding_model == "insightface-buffalo_m"


@pytest.mark.asyncio
async def test_person_routes_support_crud_and_face_enrollment(
    client: AsyncClient, session: AsyncSession
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="admin@aegispro.local",
            full_name="Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    create_response = await client.post(
        "/api/v1/persons",
        json={
            "full_name": "Dana Holt",
            "person_type": "employee",
            "department": "Operations",
            "reference_id": "EMP-1001",
            "title": "Shift Lead",
            "is_active": True,
        },
    )
    assert create_response.status_code == 201
    created = create_response.json()
    person_id = created["id"]

    list_response = await client.get("/api/v1/persons")
    assert list_response.status_code == 200
    assert list_response.json()[0]["reference_id"] == "EMP-1001"

    face_response = await client.post(
        f"/api/v1/persons/{person_id}/faces",
        json={
            "image_path": "storage/faces/emp-1001/front.jpg",
            "label": "Front profile",
            "embedding_vector": [0.1, 0.2, 0.3, 0.4],
            "embedding_model": "sim-face-v1",
            "is_primary": True,
        },
    )
    assert face_response.status_code == 200
    assert face_response.json()["face_image_count"] == 1
    assert face_response.json()["embedding_count"] == 1

    patch_response = await client.patch(
        f"/api/v1/persons/{person_id}",
        json={"department": "Security", "is_active": False},
    )
    assert patch_response.status_code == 200
    assert patch_response.json()["department"] == "Security"
    assert patch_response.json()["is_active"] is False

    delete_response = await client.delete(f"/api/v1/persons/{person_id}")
    assert delete_response.status_code == 204

    get_response = await client.get(f"/api/v1/persons/{person_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_person_routes_support_multi_image_upload_enrollment(
    client: AsyncClient,
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    admin = await UserRepository(session).create(
        UserCreate(
            email="upload-admin@aegispro.local",
            full_name="Upload Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    create_response = await client.post(
        "/api/v1/persons",
        json={
            "full_name": "Jamie Cross",
            "person_type": "employee",
            "department": "Security",
            "reference_id": "EMP-6006",
            "title": "Analyst",
            "is_active": True,
        },
    )
    assert create_response.status_code == 201
    person_id = create_response.json()["id"]

    upload_response = await client.post(
        f"/api/v1/persons/{person_id}/faces/upload",
        data={"is_primary": "true"},
        files=[
            ("files", ("front.jpg", b"front-face-image", "image/jpeg")),
            ("files", ("side.png", b"side-face-image", "image/png")),
        ],
    )

    assert upload_response.status_code == 200
    payload = upload_response.json()
    assert payload["face_image_count"] == 2
    assert payload["embedding_count"] == 2
    assert all(profile["embedding_model"] == "image-hash-v1" for profile in payload["face_profiles"])
    assert all(profile["embedding_dimensions"] == 16 for profile in payload["face_profiles"])
    assert all(profile["metadata"]["backend"] == "hash" for profile in payload["face_profiles"])

    first_profile = payload["face_profiles"][0]
    stored_path = tmp_path / Path(first_profile["image_path"])
    assert stored_path.is_file()
    assert first_profile["metadata"]["uploaded"] is True

    embedding_rows = list(await session.scalars(select(PersonFaceEmbedding)))
    assert len(embedding_rows) == 2
    assert embedding_rows[0].embedding_dimensions == 16

    image_response = await client.get(
        f"/api/v1/persons/{person_id}/faces/{first_profile['id']}/image"
    )
    assert image_response.status_code == 200
    assert image_response.content == b"front-face-image"


@pytest.mark.asyncio
async def test_person_routes_enforce_write_rbac(client: AsyncClient, session: AsyncSession) -> None:
    operator = await UserRepository(session).create(
        UserCreate(
            email="operator@aegispro.local",
            full_name="Operator",
            role=UserRole.operator,
            password="ChangeMe123!",
        )
    )
    person = await PersonRepository(session).create(
        PersonCreate(full_name="Blocked User", person_type="employee", reference_id="EMP-2002")
    )
    viewer = await UserRepository(session).create(
        UserCreate(
            email="viewer@aegispro.local",
            full_name="Viewer",
            role=UserRole.viewer,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(operator)

    operator_list_response = await client.get("/api/v1/persons")
    assert operator_list_response.status_code == 200

    operator_create_response = await client.post(
        "/api/v1/persons",
        json={"full_name": "Blocked Operator", "person_type": "employee", "reference_id": "EMP-2003"},
    )
    assert operator_create_response.status_code == 403

    operator_patch_response = await client.patch(
        f"/api/v1/persons/{person.id}",
        json={"department": "Security"},
    )
    assert operator_patch_response.status_code == 403

    operator_delete_response = await client.delete(f"/api/v1/persons/{person.id}")
    assert operator_delete_response.status_code == 403

    await _set_current_user(viewer)

    list_response = await client.get("/api/v1/persons")
    assert list_response.status_code == 200

    create_response = await client.post(
        "/api/v1/persons",
        json={"full_name": "Blocked Viewer", "person_type": "employee", "reference_id": "EMP-2004"},
    )
    assert create_response.status_code == 403


@pytest.mark.asyncio
async def test_detection_ingest_creates_known_person_incident_without_alert(
    session: AsyncSession,
) -> None:
    users = UserRepository(session)
    cameras = CameraRepository(session)
    persons_response = await users.create(
        UserCreate(
            email="supervisor@aegispro.local",
            full_name="Supervisor",
            role=UserRole.supervisor,
            password="ChangeMe123!",
        )
    )
    assert persons_response.role is UserRole.supervisor

    person = await PersonRepository(session).create(
        PersonCreate(full_name="Dana Holt", person_type="employee", reference_id="EMP-3003", department="Ops")
    )
    camera = await cameras.create(
        CameraCreate(name="Gate", source_type=CameraSourceType.http, source="http://camera/gate")
    )

    response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="yolo11-face",
            detections=[
                DetectionEventIngestItem(
                    detection_type=DetectionType.person,
                    confidence=0.91,
                    track_id="track-1",
                    identity_id=person.id,
                    identity_label=person.full_name,
                    match_confidence=0.93,
                    recognition_status=RecognitionStatus.known,
                )
            ],
        )
    )

    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    alerts = await AlertRepository(session).list()
    refreshed_person = await PersonRepository(session).get(person.id)

    assert response.incident_count == 1
    assert response.alert_count == 0
    assert incidents[0].detection_type is DetectionType.known_person
    assert incidents[0].recognized_identity is not None
    assert incidents[0].recognized_identity["identity_label"] == "Dana Holt"
    assert len(alerts) == 0
    assert refreshed_person is not None
    assert refreshed_person.recognition_count == 1


@pytest.mark.asyncio
async def test_detection_ingest_creates_unknown_person_incident(session: AsyncSession) -> None:
    cameras = CameraRepository(session)
    from app.services.detection_events import DetectionEventService

    camera = await cameras.create(
        CameraCreate(name="Dock", source_type=CameraSourceType.usb, source="0")
    )

    response = await DetectionEventService(session).ingest(
        DetectionEventIngest(
            camera_id=camera.id,
            model_name="yolo11-face",
            detections=[
                DetectionEventIngestItem(
                    detection_type=DetectionType.person,
                    confidence=0.84,
                    track_id="track-2",
                    recognition_status=RecognitionStatus.unknown,
                    identity_label="Unidentified person",
                )
            ],
        )
    )

    incidents = await IncidentRepository(session).list(camera_id=camera.id)
    assert response.results[0].detection_type is DetectionType.unknown_person
    assert incidents[0].recognized_identity is not None
    assert incidents[0].recognized_identity["status"] == "unknown"


@pytest.mark.asyncio
async def test_unknown_person_incident_can_be_saved_as_known_person(
    client: AsyncClient,
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    admin = await UserRepository(session).create(
        UserCreate(
            email="incident-admin@aegispro.local",
            full_name="Incident Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Lobby", source_type=CameraSourceType.http, source="http://camera/lobby")
    )

    captured_face = tmp_path / "faces" / "detections" / "unknown-visitor.jpg"
    captured_face.parent.mkdir(parents=True, exist_ok=True)
    captured_face.write_bytes(b"detected-face-image")

    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.unknown_person,
            confidence=0.87,
            recognized_identity={
                "status": "unknown",
                "identity_label": "Unidentified person",
                "face_image_path": "faces/detections/unknown-visitor.jpg",
            },
            snapshot_path="faces/detections/unknown-visitor.jpg",
        )
    )

    response = await client.post(
        f"/api/v1/incidents/{incident.id}/save-person",
        json={
            "full_name": "Dana Holt",
            "person_type": "employee",
            "reference_id": "EMP-7007",
            "department": "Security",
            "title": "Guard",
            "is_active": True,
            "is_primary": True,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["full_name"] == "Dana Holt"
    assert payload["reference_id"] == "EMP-7007"
    assert payload["face_image_count"] == 1
    assert payload["embedding_count"] == 1
    assert payload["face_profiles"][0]["metadata"]["source_incident_id"] == str(incident.id)

    refreshed_incident = await IncidentRepository(session).get(incident.id)
    assert refreshed_incident is not None
    assert refreshed_incident.detection_type is DetectionType.known_person
    assert refreshed_incident.recognized_identity is not None
    assert refreshed_incident.recognized_identity["status"] == "known"
    assert refreshed_incident.recognized_identity["identity_label"] == "Dana Holt"


@pytest.mark.asyncio
async def test_save_person_rolls_back_created_person_when_face_enrollment_fails(
    client: AsyncClient,
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)

    admin = await UserRepository(session).create(
        UserCreate(
            email="rollback-admin@aegispro.local",
            full_name="Rollback Admin",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    camera = await CameraRepository(session).create(
        CameraCreate(name="Back Gate", source_type=CameraSourceType.http, source="http://camera/back-gate")
    )

    captured_face = tmp_path / "faces" / "detections" / "broken-visitor.jpg"
    captured_face.parent.mkdir(parents=True, exist_ok=True)
    captured_face.write_bytes(b"detected-face-image")

    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.unknown_person,
            confidence=0.81,
            recognized_identity={
                "status": "unknown",
                "identity_label": "Unidentified person",
                "face_image_path": "faces/detections/broken-visitor.jpg",
            },
            snapshot_path="faces/detections/broken-visitor.jpg",
        )
    )

    async def fail_enrollment(self, person_id, image_path, **kwargs):
        raise HTTPException(status_code=422, detail="No detectable face was found in the provided image.")

    monkeypatch.setattr(PersonService, "enroll_face_image_from_storage", fail_enrollment)

    response = await client.post(
        f"/api/v1/incidents/{incident.id}/save-person",
        json={
            "full_name": "Broken Enrollment",
            "person_type": "employee",
            "reference_id": "EMP-ROLLBACK",
            "department": "Security",
            "title": "Guard",
            "is_active": True,
            "is_primary": True,
        },
    )

    assert response.status_code == 422
    assert await PersonRepository(session).get_by_reference_id("EMP-ROLLBACK") is None


@pytest.mark.asyncio
async def test_face_enrollment_does_not_persist_file_when_embedding_fails_without_fallback(
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)
    monkeypatch.setattr(settings, "recognition_allow_fallback", False)

    person = await PersonRepository(session).create(
        PersonCreate(full_name="Dana Holt", person_type="employee", reference_id="EMP-NOFILE", department="Ops")
    )

    class FailingBackend:
        def extract_embedding(self, image_bytes: bytes):
            raise FaceEmbeddingError("No detectable face was found in the provided image.")

    monkeypatch.setattr(FaceEnrollmentService, "_build_backend", staticmethod(lambda: FailingBackend()))

    service = FaceEnrollmentService()
    source = tmp_path / "faces" / "detections" / "source.jpg"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-face-image")

    with pytest.raises(HTTPException) as exc_info:
        service.build_enrollment_from_stored_image(
            person,
            "faces/detections/source.jpg",
        )

    assert exc_info.value.status_code == 422
    copied_dir = tmp_path / "faces" / "persons"
    assert not copied_dir.exists()


@pytest.mark.asyncio
async def test_person_embedding_match_returns_known_person_match(
    client: AsyncClient,
    session: AsyncSession,
) -> None:
    admin = await UserRepository(session).create(
        UserCreate(
            email="matcher@aegispro.local",
            full_name="Matcher",
            role=UserRole.administrator,
            password="ChangeMe123!",
        )
    )
    await _set_current_user(admin)

    person = await PersonRepository(session).create(
        PersonCreate(
            full_name="Dana Holt",
            person_type="employee",
            reference_id="EMP-8080",
            department="Security",
        )
    )
    await PersonRepository(session).add_face_profile(
        person,
        PersonFaceEnrollment(
            image_path="faces/persons/emp-8080/front.jpg",
            label="Front profile",
            embedding_vector=[0.1, 0.2, 0.3, 0.4],
            embedding_model="sim-face-v1",
            is_primary=True,
        ),
    )

    response = await client.post(
        "/api/v1/persons/match",
        json={
            "embedding_vector": [0.1, 0.2, 0.3, 0.4],
            "embedding_model": "sim-face-v1",
            "top_k": 3,
            "min_similarity": 0.8,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["matched_count"] == 1
    assert payload["results"][0]["full_name"] == "Dana Holt"
    assert payload["results"][0]["reference_id"] == "EMP-8080"
    assert payload["results"][0]["similarity"] >= 0.99
