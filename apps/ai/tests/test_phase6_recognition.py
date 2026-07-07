import base64
from io import BytesIO
from uuid import uuid4

from PIL import Image

from app.schemas.inference import InferenceRequest, KnownPersonFaceProfile, KnownPersonProfile
from app.services.pipeline import InferencePipeline
from app.services.recognition import FaceRecognitionService
from app.services.face_embeddings import FaceEmbeddingResult, HashFaceEmbeddingBackend


def build_request(track_hint: str = "known") -> InferenceRequest:
    return InferenceRequest(
        camera_id=uuid4(),
        frame_reference=f"frame-{track_hint}",
        source_type="http",
        requested_detectors=["person"],
        recognition_enabled=True,
        occurrence_hint=track_hint,
        known_persons=[
            KnownPersonProfile(
                person_id=uuid4(),
                full_name="Dana Holt",
                person_type="employee",
                reference_id="EMP-1001",
                face_profiles=[
                    KnownPersonFaceProfile(
                        face_id="front",
                        image_path="storage/faces/dana-front.jpg",
                        embedding_vector=[0.2] * 16,
                        embedding_model="sim-face-v1",
                    )
                ],
            )
        ],
    )


def build_frame_base64() -> str:
    image = Image.new("RGB", (1280, 720), color=(240, 240, 240))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_recognition_schema_and_pipeline_return_face_data() -> None:
    pipeline = InferencePipeline()
    result = pipeline.run(build_request())

    assert result.detections
    detection = result.detections[0]
    assert detection.face_region is not None
    assert detection.recognition is not None
    assert detection.label in {"known_person", "unknown_person"}
    assert detection.recognition.status in {"known", "unknown"}


def test_recognition_deduplicates_track_matches() -> None:
    pipeline = InferencePipeline()
    request = build_request("repeat")

    first = pipeline.run(request)
    second = pipeline.run(request)

    assert first.detections[0].recognition is not None
    assert second.detections[0].recognition is not None
    assert first.detections[0].track_id == second.detections[0].track_id
    assert first.detections[0].recognition.deduplicated is False
    assert second.detections[0].recognition.deduplicated is True


def test_recognition_recomputes_when_known_person_candidates_change() -> None:
    pipeline = InferencePipeline()
    request = build_request("refresh-cache")

    first = pipeline.run(request.model_copy(update={"known_persons": []}))
    first_detection = first.detections[0]
    assert first_detection.recognition is not None
    assert first_detection.recognition.status == "unknown"

    matching_embedding = HashFaceEmbeddingBackend().extract_embedding(
        f"{request.camera_id}:{request.frame_reference}:{first_detection.track_id}".encode("utf-8")
    ).vector
    refreshed_request = request.model_copy(
        update={
            "known_persons": [
                KnownPersonProfile(
                    person_id=request.known_persons[0].person_id,
                    full_name=request.known_persons[0].full_name,
                    person_type=request.known_persons[0].person_type,
                    reference_id=request.known_persons[0].reference_id,
                    face_profiles=[
                        KnownPersonFaceProfile(
                            face_id="front",
                            image_path="storage/faces/dana-front.jpg",
                            embedding_vector=matching_embedding,
                            embedding_model="sim-face-v1",
                        )
                    ],
                )
            ]
        }
    )

    second = pipeline.run(refreshed_request)
    second_detection = second.detections[0]

    assert second_detection.recognition is not None
    assert second_detection.recognition.status == "known"
    assert second_detection.recognition.deduplicated is False


def test_recognition_uses_image_embedding_backend_when_frame_is_available(monkeypatch) -> None:
    class StubBackend:
        def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
            return FaceEmbeddingResult(
                vector=[0.5] * 16,
                model_name="insightface-buffalo_l",
                backend_name="insightface",
                metadata={"backend": "insightface", "deterministic": False},
            )

    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: StubBackend()))
    pipeline = InferencePipeline()
    request = build_request("frame-embedding").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Dana Holt",
                    person_type="employee",
                    reference_id="EMP-1001",
                    face_profiles=[
                        KnownPersonFaceProfile(
                            face_id="front",
                            image_path="storage/faces/dana-front.jpg",
                            embedding_vector=[0.5] * 16,
                            embedding_model="insightface-buffalo_l",
                        )
                    ],
                )
            ],
        }
    )

    result = pipeline.run(request)

    assert result.detections[0].recognition is not None
    assert result.detections[0].recognition.status == "known"
    assert result.detections[0].recognition.embedding_model == "insightface-buffalo_l"
    assert result.detections[0].recognition.metadata["backend"] == "insightface"


def test_dispatch_payload_includes_recognition_metadata(monkeypatch) -> None:
    captured: dict[str, object] = {}
    pipeline = InferencePipeline()

    class DummyResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        captured["body"] = req.data.decode("utf-8")
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("app.services.pipeline.settings.enable_event_callback", True)
    monkeypatch.setattr(
        "app.services.pipeline.settings.api_event_callback_url",
        "http://127.0.0.1:8000/api/v1/detections/ingest",
    )
    monkeypatch.setattr("app.services.pipeline.request.urlopen", fake_urlopen)

    dispatch = pipeline.dispatch_events(pipeline.run(build_request("dispatch")))

    assert dispatch.delivered is True
    assert '"recognition_status"' in captured["body"]
    assert '"face_bounding_box"' in captured["body"]


def test_dispatch_payload_includes_inline_evidence(monkeypatch) -> None:
    captured: dict[str, object] = {}
    pipeline = InferencePipeline()

    class DummyResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        captured["body"] = req.data.decode("utf-8")
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr("app.services.pipeline.settings.enable_event_callback", True)
    monkeypatch.setattr(
        "app.services.pipeline.settings.api_event_callback_url",
        "http://127.0.0.1:8000/api/v1/detections/ingest",
    )
    monkeypatch.setattr("app.services.pipeline.request.urlopen", fake_urlopen)

    request = build_request("dispatch-evidence").model_copy(
        update={
            "include_evidence": True,
            "frame_content_base64": "c25hcHNob3Q=",
            "frame_content_type": "image/jpeg",
        }
    )
    dispatch = pipeline.dispatch_events(pipeline.run(request))

    assert dispatch.delivered is True
    assert '"snapshot_evidence"' in captured["body"]
    assert '"face_image_evidence"' in captured["body"]


def test_dispatch_payload_includes_service_token_header(monkeypatch) -> None:
    captured: dict[str, object] = {}
    pipeline = InferencePipeline()

    class DummyResponse:
        status = 202

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req, timeout=0):
        captured["service_token"] = dict(req.header_items()).get("X-service-token")
        return DummyResponse()

    monkeypatch.setattr("app.services.pipeline.settings.enable_event_callback", True)
    monkeypatch.setattr(
        "app.services.pipeline.settings.api_event_callback_url",
        "http://127.0.0.1:8000/api/v1/detections/ingest",
    )
    monkeypatch.setattr("app.services.pipeline.settings.api_event_callback_token", "phase5-service-token")
    monkeypatch.setattr("app.services.pipeline.request.urlopen", fake_urlopen)

    dispatch = pipeline.dispatch_events(pipeline.run(build_request("dispatch-token")))

    assert dispatch.delivered is True
    assert captured["service_token"] == "phase5-service-token"
