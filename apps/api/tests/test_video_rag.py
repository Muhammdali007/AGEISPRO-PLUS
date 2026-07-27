from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings
from app.db.metadata import Base
from app.models.camera import CameraSourceType
from app.models.incident import DetectionType, IncidentPriority
from app.models.video_rag import VideoRagChunk, VideoRagIndex, VideoRagIndexStatus
from app.repositories.cameras import CameraRepository
from app.repositories.incidents import IncidentRepository
from app.schemas.cameras import CameraCreate
from app.schemas.incidents import IncidentCreate
from app.schemas.video_rag import VideoRagQueryRequest
from app.services.video_rag_indexing import SampledEvidence, VideoRagIndexer
from app.services.video_rag_query import VideoRagQueryService
from app.services.ollama import OllamaClient, OllamaUnavailableError


@pytest_asyncio.fixture
async def session() -> AsyncIterator[AsyncSession]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with factory() as test_session:
        yield test_session
    await engine.dispose()


class FakeOllama:
    async def embed(self, texts):
        count = len(texts) if isinstance(texts, list) else 1
        return [[1.0] + [0.0] * 767 for _ in range(count)]

    async def extract_filters(self, question, cameras, now_iso):
        return {}

    async def answer(self, question, contexts):
        return f"A weapon detector incident was found [incident:{contexts[0]['incident_id']}]."


class BusyOllama(FakeOllama):
    async def embed(self, texts):
        raise OllamaUnavailableError("busy")

    async def answer(self, question, contexts):
        raise OllamaUnavailableError("busy")


class VisionBusyOllama(FakeOllama):
    async def describe_frames(self, images_base64, offsets):
        raise OllamaUnavailableError("vision timeout")


class AbstainingOllama(FakeOllama):
    async def answer(self, question, contexts):
        return "NOT_FOUND"


@pytest.mark.asyncio
async def test_answer_prompt_requires_formal_concise_responses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = OllamaClient(base_url="http://ollama.test", timeout=1)
    captured: dict[str, object] = {}

    async def fake_chat(prompt, **kwargs):
        captured["prompt"] = prompt
        captured.update(kwargs)
        return {
            "message": {
                "content": (
                    "A weapon was detected at Gate 1 "
                    "[incident:00000000-0000-0000-0000-000000000001]."
                )
            }
        }

    monkeypatch.setattr(client, "_chat", fake_chat)
    await client.answer(
        "Was a weapon detected?",
        [{"incident_id": "00000000-0000-0000-0000-000000000001"}],
    )

    system_prompt = str(captured["system_prompt"])
    assert "concise, formal incident response" in system_prompt
    assert "no more than three short sentences" in system_prompt
    assert "Do not use greetings" in system_prompt
    assert captured["max_tokens"] == 160


async def _incident(session: AsyncSession):
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="Gate 1",
            source_type=CameraSourceType.http,
            source="http://camera/gate-1",
            location="North entrance",
        )
    )
    incident = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.94,
            occurred_at=datetime.now(UTC),
            snapshot_path="incidents/gate/snapshot.jpg",
            bounding_boxes=[{"label": "knife", "x1": 1, "y1": 1, "x2": 2, "y2": 2}],
        )
    )
    return camera, incident


async def _recognized_incident(
    session: AsyncSession,
    *,
    camera_id,
    identity_label: str,
    occurred_at: datetime,
):
    return await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera_id,
            detection_type=DetectionType.known_person,
            priority=IncidentPriority.low,
            confidence=0.91,
            occurred_at=occurred_at,
            snapshot_path=f"incidents/{identity_label.casefold()}/{uuid4()}.jpg",
            recognized_identity={"status": "known", "identity_label": identity_label},
        )
    )


def test_video_rag_models_are_registered() -> None:
    assert {"video_rag_indexes", "video_rag_chunks"} <= set(Base.metadata.tables)


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("List only 2 latest records", 2),
        ("Show the latest three incidents", 3),
        ("Give me top 4", 4),
        ("Show incidents from the last 2 hours", 5),
    ],
)
def test_requested_result_limit_is_extracted_without_confusing_time_counts(
    question: str, expected: int
) -> None:
    assert VideoRagQueryService._requested_result_limit(question, 5) == expected


def test_relative_time_range_is_extracted_deterministically() -> None:
    now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)

    start_at, end_at = VideoRagQueryService._relative_time_range(
        "Show weapon incidents from the last two hours", now=now
    )

    assert start_at == now - timedelta(hours=2)
    assert end_at == now


def test_yesterday_range_uses_pakistan_calendar_day() -> None:
    now = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)

    start_at, end_at = VideoRagQueryService._relative_time_range(
        "What happened yesterday?", now=now
    )

    assert start_at == datetime(2026, 7, 23, 19, 0, tzinfo=UTC)
    assert end_at == datetime(2026, 7, 24, 18, 59, 59, 999999, tzinfo=UTC)


@pytest.mark.asyncio
async def test_indexing_falls_back_to_authoritative_metadata_when_vision_is_busy() -> None:
    indexer = VideoRagIndexer(ollama=VisionBusyOllama())
    description = await indexer._describe_evidence(
        SampledEvidence(images_base64=["image"], offsets=[0.0]),
        uuid4(),
    )

    assert description["visual_available"] is False
    assert description["observations"] == []


@pytest.mark.asyncio
async def test_query_returns_grounded_ranked_incident(session: AsyncSession) -> None:
    camera, incident = await _incident(session)
    session.add(
        VideoRagIndex(
            incident_id=incident.id,
            status=VideoRagIndexStatus.ready,
            indexed_at=datetime.now(UTC),
        )
    )
    session.add(
        VideoRagChunk(
            incident_id=incident.id,
            kind="summary",
            content="Detector event weapon, subtype knife, at Gate 1.",
            metadata_={"authoritative": True},
            embedding=[1.0] + [0.0] * 767,
        )
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(question="Was there a weapon at Gate 1?")
    )

    assert response.answer.startswith("Yes, a weapon (knife) was detected in Gate 1 on ")
    assert response.answer.endswith("with 94% confidence.")
    assert response.evidence[0].incident_id == incident.id
    assert response.evidence[0].camera_name == camera.name
    assert response.evidence[0].snapshot_url.endswith(f"/{incident.id}/snapshot")


@pytest.mark.asyncio
async def test_query_uses_lexical_metadata_when_ollama_is_busy(session: AsyncSession) -> None:
    _, incident = await _incident(session)
    session.add(
        VideoRagIndex(
            incident_id=incident.id,
            status=VideoRagIndexStatus.ready,
            indexed_at=datetime.now(UTC),
        )
    )
    session.add(
        VideoRagChunk(
            incident_id=incident.id,
            kind="summary",
            content="Detector metadata records recognized identity Ali at Gate 1.",
            metadata_={"authoritative": True},
            embedding=[1.0] + [0.0] * 767,
        )
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=BusyOllama()).query(
        VideoRagQueryRequest(question="When was Ali spotted?")
    )

    assert response.evidence[0].incident_id == incident.id
    assert "[incident:" not in response.answer
    assert any("indexed text metadata" in warning for warning in response.warnings)


@pytest.mark.asyncio
async def test_detector_incidents_are_searchable_before_visual_indexing(
    session: AsyncSession,
) -> None:
    _, incident = await _incident(session)
    session.add(VideoRagIndex(incident_id=incident.id, status=VideoRagIndexStatus.queued))
    await session.commit()

    response = await VideoRagQueryService(session, ollama=BusyOllama()).query(
        VideoRagQueryRequest(question="Were there any weapon incidents?")
    )

    assert response.evidence[0].incident_id == incident.id
    assert "weapon" in response.answer
    assert "with 94% confidence." in response.answer
    assert any("detector metadata" in warning for warning in response.warnings)


@pytest.mark.asyncio
async def test_generic_incident_summary_works_without_rag_chunks(session: AsyncSession) -> None:
    _, incident = await _incident(session)

    response = await VideoRagQueryService(session, ollama=BusyOllama()).query(
        VideoRagQueryRequest(question="What incidents happened?")
    )

    assert response.evidence[0].incident_id == incident.id
    assert response.answer.startswith(
        "These are a few incidents recorded: A weapon (knife) was detected in Gate 1 on "
    )
    assert response.answer.endswith("with 94% confidence.")
    assert "status open" not in response.answer


@pytest.mark.asyncio
async def test_query_applies_requested_count_and_relative_time_before_retrieval(
    session: AsyncSession,
) -> None:
    camera, newest = await _incident(session)
    now = datetime.now(UTC)
    recent_second = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.90,
            occurred_at=now - timedelta(minutes=10),
            snapshot_path="incidents/recent-second/snapshot.jpg",
        )
    )
    await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.89,
            occurred_at=now - timedelta(minutes=20),
            snapshot_path="incidents/recent-third/snapshot.jpg",
        )
    )
    await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.88,
            occurred_at=now - timedelta(hours=3),
            snapshot_path="incidents/old/snapshot.jpg",
        )
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(
            question=(
                "When were weapons detected in the last 2 hours? "
                "List only 2 latest records."
            )
        )
    )

    assert len(response.evidence) == 2
    assert [item.incident_id for item in response.evidence] == [newest.id, recent_second.id]
    assert response.answer.count("Yes, a weapon") == 2
    assert "with 94% confidence." in response.answer
    assert "with 90% confidence." in response.answer


@pytest.mark.asyncio
async def test_known_person_question_uses_concise_local_answer(session: AsyncSession) -> None:
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="web",
            source_type=CameraSourceType.http,
            source="http://camera/web",
        )
    )
    incident = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Ali",
        occurred_at=datetime(2026, 7, 24, 11, 53, tzinfo=UTC),
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(question="Was Ali spotted?")
    )

    assert response.evidence[0].incident_id == incident.id
    assert response.answer == (
        "Yes, Ali was spotted in web on 7/24/2026 at 4:53 PM with 91% confidence."
    )


@pytest.mark.asyncio
async def test_yesterday_summary_uses_broad_incident_wording(session: AsyncSession) -> None:
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="web",
            source_type=CameraSourceType.http,
            source="http://camera/web",
        )
    )
    incident = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Ali",
        occurred_at=datetime(2026, 7, 24, 11, 53, tzinfo=UTC),
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(
            question="What happened yesterday?",
            start_at=datetime(2026, 7, 23, 19, 0, tzinfo=UTC),
            end_at=datetime(2026, 7, 24, 18, 59, 59, 999999, tzinfo=UTC),
        )
    )

    assert response.evidence[0].incident_id == incident.id
    assert response.answer == (
        "These are a few incidents recorded yesterday: "
        "Ali was spotted in web on 7/24/2026 at 4:53 PM with 91% confidence."
    )
    assert "[incident:" not in response.answer


@pytest.mark.asyncio
async def test_identity_with_weapon_requires_same_camera_and_time(
    session: AsyncSession,
) -> None:
    camera = await CameraRepository(session).create(
        CameraCreate(
            name="web",
            source_type=CameraSourceType.http,
            source="http://camera/web",
        )
    )
    observed_at = datetime(2026, 7, 24, 11, 53, tzinfo=UTC)
    ali = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Ali",
        occurred_at=observed_at,
    )
    weapon = await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.87,
            occurred_at=observed_at + timedelta(seconds=4),
            snapshot_path="incidents/weapon/snapshot.jpg",
        )
    )
    await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Ali",
        occurred_at=observed_at + timedelta(hours=1),
    )
    misleading_camera = await CameraRepository(session).create(
        CameraCreate(
            name="weapon",
            source_type=CameraSourceType.http,
            source="http://camera/misleading-weapon-name",
        )
    )
    await IncidentRepository(session).create(
        IncidentCreate(
            camera_id=misleading_camera.id,
            detection_type=DetectionType.weapon,
            priority=IncidentPriority.critical,
            confidence=0.99,
            occurred_at=observed_at + timedelta(hours=2),
            snapshot_path="incidents/unrelated-weapon/snapshot.jpg",
        )
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(question="Give me the incident of Ali with a weapon")
    )

    assert {item.incident_id for item in response.evidence} == {ali.id, weapon.id}
    assert response.answer == (
        "Yes, an incident involving Ali and a weapon was recorded in web on "
        "7/24/2026 at 4:53 PM with 91% identity confidence and 87% weapon confidence."
    )
    assert "[incident:" not in response.answer


@pytest.mark.asyncio
async def test_identity_together_query_requires_all_people_in_same_camera_observation(
    session: AsyncSession,
) -> None:
    camera, _ = await _incident(session)
    together_at = datetime(2026, 7, 24, 11, 2, 36, tzinfo=UTC)
    ali = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Ali",
        occurred_at=together_at,
    )
    zain = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Zain",
        occurred_at=together_at + timedelta(seconds=7),
    )
    nearby_zain = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Zain",
        occurred_at=together_at - timedelta(seconds=7),
    )
    newer_zain = await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Zain",
        occurred_at=together_at + timedelta(minutes=1),
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(
            question="When were Ali and Zain seen together? Tell me the time and camera."
        )
    )

    assert "Ali and Zain were recorded on the same camera" in response.answer
    assert "within 15 seconds" in response.answer
    assert camera.name in response.answer
    assert together_at.replace(tzinfo=None).isoformat() in response.answer
    assert {item.incident_id for item in response.evidence} == {ali.id, zain.id}
    assert str(newer_zain.id) not in response.answer
    assert str(nearby_zain.id) not in response.answer


@pytest.mark.asyncio
async def test_identity_together_query_does_not_merge_different_observations(
    session: AsyncSession,
) -> None:
    camera, _ = await _incident(session)
    observed_at = datetime(2026, 7, 24, 11, 2, 36, tzinfo=UTC)
    await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Ali",
        occurred_at=observed_at,
    )
    await _recognized_incident(
        session,
        camera_id=camera.id,
        identity_label="Zain",
        occurred_at=observed_at + timedelta(seconds=16),
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=FakeOllama()).query(
        VideoRagQueryRequest(question="Were Ali and Zain seen together?")
    )

    assert response.answer == (
        "No authoritative recognition records place Ali and Zain on the same "
        "camera within 15 seconds."
    )
    assert response.evidence == []


@pytest.mark.asyncio
async def test_archiving_incident_removes_rag_records(session: AsyncSession) -> None:
    _, incident = await _incident(session)
    session.add(VideoRagIndex(incident_id=incident.id, status=VideoRagIndexStatus.ready))
    session.add(
        VideoRagChunk(
            incident_id=incident.id,
            kind="summary",
            content="Indexed evidence",
            metadata_={},
            embedding=[1.0] + [0.0] * 767,
        )
    )
    await session.commit()

    await IncidentRepository(session).archive(incident)
    await session.commit()

    assert await session.get(VideoRagIndex, incident.id) is None
    count = await session.scalar(
        select(func.count())
        .select_from(VideoRagChunk)
        .where(VideoRagChunk.incident_id == incident.id)
    )
    assert count == 0


@pytest.mark.asyncio
async def test_reconcile_backfills_retained_evidence(
    session: AsyncSession, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "storage_root", tmp_path)
    camera, incident = await _incident(session)
    snapshot = tmp_path / str(incident.snapshot_path)
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"image")

    changed = await VideoRagIndexer(ollama=FakeOllama()).reconcile(session)
    state = await session.get(VideoRagIndex, incident.id)

    assert changed == 1
    assert state is not None
    assert state.status is VideoRagIndexStatus.queued
    assert state.evidence_fingerprint


def test_ungrounded_answer_is_replaced() -> None:
    incident_id = "00000000-0000-0000-0000-000000000001"
    generated = "Something happened [incident:00000000-0000-0000-0000-000000000099]."
    item = type("Item", (), {"incident": type("Incident", (), {"id": incident_id})()})()

    answer, warning = VideoRagQueryService._ground_answer(generated, [item])

    assert "[incident:" not in answer
    assert warning is not None


def test_explicit_not_found_answer_does_not_require_a_citation() -> None:
    incident_id = "00000000-0000-0000-0000-000000000001"
    item = type("Item", (), {"incident": type("Incident", (), {"id": incident_id})()})()

    answer, warning = VideoRagQueryService._ground_answer(
        "NOT_FOUND",
        [item],
        question="What color was the person's backpack?",
    )

    assert answer == (
        "The available incident records do not contain sufficient evidence to answer this question."
    )
    assert incident_id not in answer
    assert warning is None


@pytest.mark.asyncio
async def test_semantically_related_evidence_can_be_rejected_as_insufficient(
    session: AsyncSession,
) -> None:
    _, incident = await _incident(session)
    session.add(
        VideoRagIndex(
            incident_id=incident.id,
            status=VideoRagIndexStatus.ready,
            indexed_at=datetime.now(UTC),
        )
    )
    session.add(
        VideoRagChunk(
            incident_id=incident.id,
            kind="summary",
            content="A person is visible near Gate 1; no backpack color is described.",
            metadata_={"authoritative": False},
            embedding=[1.0] + [0.0] * 767,
        )
    )
    await session.commit()

    response = await VideoRagQueryService(session, ollama=AbstainingOllama()).query(
        VideoRagQueryRequest(question="What color was the person's backpack?")
    )

    assert response.answer == (
        "The available incident records do not contain sufficient evidence to answer this question."
    )
    assert str(incident.id) not in response.answer
