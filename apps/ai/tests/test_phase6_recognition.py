import base64
from io import BytesIO
from uuid import uuid4

from PIL import Image

from app.schemas.inference import (
    FaceRegion,
    InferenceBox,
    InferenceRecognition,
    InferenceRequest,
    KnownPersonFaceProfile,
    KnownPersonProfile,
)
from app.services.pipeline import InferencePipeline
from app.services.recognition import FaceRecognitionService
from app.services.face_embeddings import (
    FaceEmbeddingError,
    FaceEmbeddingResult,
    HashFaceEmbeddingBackend,
)


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


def test_face_evidence_crops_validated_face_region() -> None:
    image = Image.new("RGB", (120, 80), color=(240, 240, 240))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    request = build_request("crop").model_copy(
        update={
            "include_evidence": True,
            "frame_content_base64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
            "frame_content_type": "image/jpeg",
        }
    )

    evidence = InferencePipeline._face_evidence(
        request,
        FaceRegion(x1=10, y1=12, x2=50, y2=42, confidence=0.95),
    )

    assert evidence is not None
    cropped = Image.open(BytesIO(base64.b64decode(evidence.content_base64)))
    assert cropped.size == (40, 30)


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


def test_recognition_reports_incompatible_known_person_template_dimensions(monkeypatch) -> None:
    class StubBackend:
        def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
            return FaceEmbeddingResult(
                vector=[0.5] * 512,
                model_name="insightface-buffalo_m",
                backend_name="insightface",
                metadata={"backend": "insightface", "dimensions": 512},
            )

    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: StubBackend()))
    pipeline = InferencePipeline()
    request = build_request("dimension-mismatch").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Dana Holt",
                    person_type="employee",
                    face_profiles=[
                        KnownPersonFaceProfile(
                            face_id="old-hash",
                            image_path="storage/faces/dana-front.jpg",
                            embedding_vector=[0.5] * 16,
                            embedding_model="image-hash-v1",
                        )
                    ],
                )
            ],
        }
    )

    result = pipeline.run(request)
    recognition = result.detections[0].recognition

    assert recognition is not None
    assert recognition.status == "unknown"
    assert recognition.metadata["known_person_count"] == 1
    assert recognition.metadata["source_template_count"] == 1
    assert recognition.metadata["eligible_template_count"] == 1
    assert recognition.metadata["compatible_template_count"] == 0
    assert recognition.metadata["query_embedding_dimensions"] == 512
    assert recognition.metadata["skipped_template_dimension_count"] == 1
    assert recognition.metadata["skipped_template_dimensions"] == [16]


def test_recognition_recomputes_real_frames_when_tracker_id_is_reused(monkeypatch) -> None:
    vectors = iter(([0.5] * 16, [-0.5] * 16))

    class ChangingBackend:
        def extract_embedding(self, image_bytes: bytes) -> FaceEmbeddingResult:
            return FaceEmbeddingResult(
                vector=next(vectors),
                model_name="insightface-buffalo_l",
                backend_name="insightface",
                metadata={"backend": "insightface"},
            )

    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: ChangingBackend()))
    request = build_request("reused-track").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Muhammad Ali",
                    person_type="student",
                    face_profiles=[
                        KnownPersonFaceProfile(
                            face_id="front",
                            embedding_vector=[0.5] * 16,
                            embedding_model="insightface-buffalo_l",
                        )
                    ],
                )
            ],
        }
    )
    pipeline = InferencePipeline()

    first = pipeline.run(request).detections[0].recognition
    second = pipeline.run(request).detections[0].recognition

    assert first is not None and first.status == "known"
    assert second is not None and second.status == "unknown"
    assert second.deduplicated is False


def test_recognition_analyzes_full_frame_once_and_reuses_short_lived_cache(monkeypatch) -> None:
    class BatchBackend:
        calls = 0

        def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
            self.calls += 1
            return [
                FaceEmbeddingResult(
                    vector=[1.0, 0.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [30.0, 20.0, 70.0, 65.0],
                        "det_score": 0.98,
                    },
                ),
                FaceEmbeddingResult(
                    vector=[0.0, 1.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [230.0, 25.0, 270.0, 70.0],
                        "det_score": 0.97,
                    },
                ),
            ]

    backend = BatchBackend()
    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: backend))
    monkeypatch.setattr("app.services.recognition.settings.recognition_refresh_seconds", 2.0)
    request = build_request("batch-frame").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="First Person",
                    person_type="employee",
                    face_profiles=[KnownPersonFaceProfile(face_id="first", embedding_vector=[1.0, 0.0])],
                ),
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Second Person",
                    person_type="employee",
                    face_profiles=[KnownPersonFaceProfile(face_id="second", embedding_vector=[0.0, 1.0])],
                ),
            ],
        }
    )
    detections = [
        InferenceBox(x1=0, y1=0, x2=120, y2=300, confidence=0.9, label="person"),
        InferenceBox(x1=180, y1=0, x2=320, y2=300, confidence=0.9, label="person"),
    ]
    service = FaceRecognitionService()

    first = service.enrich_detections(request, detections)
    second = service.enrich_detections(request, detections)

    assert backend.calls == 1
    assert [item.recognition.identity_label for item in first] == ["First Person", "Second Person"]
    assert all(item.recognition and item.recognition.status == "known" for item in first)
    assert all(item.recognition and item.recognition.deduplicated for item in second)
    assert all(
        item.recognition and item.recognition.metadata["frame_recognition_cache_hit"]
        for item in second
    )


def test_recognition_creates_known_person_from_unboxed_face(monkeypatch) -> None:
    class FaceOnlyBackend:
        def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
            return [
                FaceEmbeddingResult(
                    vector=[1.0, 0.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [310.0, 80.0, 390.0, 170.0],
                        "det_score": 0.99,
                    },
                )
            ]

    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: FaceOnlyBackend()))
    request = build_request("face-only").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Registered Face",
                    person_type="employee",
                    face_profiles=[KnownPersonFaceProfile(face_id="front", embedding_vector=[1.0, 0.0])],
                )
            ],
        }
    )

    result = FaceRecognitionService().enrich_detections(request, [])

    assert len(result) == 1
    assert result[0].label == "known_person"
    assert result[0].object_label == "face"
    assert result[0].face_region is not None
    assert result[0].recognition is not None
    assert result[0].recognition.identity_label == "Registered Face"
    assert result[0].recognition.metadata["face_only_detection"] is True


def test_recognition_adds_unassigned_face_when_person_box_used_another_face(monkeypatch) -> None:
    class MixedFaceBackend:
        def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
            return [
                FaceEmbeddingResult(
                    vector=[1.0, 0.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [30.0, 20.0, 70.0, 65.0],
                        "det_score": 0.98,
                    },
                ),
                FaceEmbeddingResult(
                    vector=[0.0, 1.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [330.0, 80.0, 390.0, 155.0],
                        "det_score": 0.97,
                    },
                ),
            ]

    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: MixedFaceBackend()))
    request = build_request("mixed-face").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="First Face",
                    face_profiles=[KnownPersonFaceProfile(face_id="first", embedding_vector=[1.0, 0.0])],
                ),
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Second Face",
                    face_profiles=[KnownPersonFaceProfile(face_id="second", embedding_vector=[0.0, 1.0])],
                ),
            ],
        }
    )
    person = InferenceBox(x1=0, y1=0, x2=120, y2=300, confidence=0.9, label="person")

    result = FaceRecognitionService().enrich_detections(request, [person])

    assert len(result) == 2
    assert [item.recognition.identity_label for item in result] == ["First Face", "Second Face"]
    assert result[1].recognition.metadata["face_only_detection"] is True


def test_recognition_cache_follows_stable_track_during_fast_motion(monkeypatch) -> None:
    class BatchBackend:
        calls = 0

        def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
            self.calls += 1
            return [
                FaceEmbeddingResult(
                    vector=[1.0, 0.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [30.0, 20.0, 70.0, 65.0],
                        "det_score": 0.98,
                    },
                )
            ]

    backend = BatchBackend()
    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: backend))
    monkeypatch.setattr("app.services.recognition.settings.recognition_refresh_seconds", 2.0)
    request = build_request("moving-frame").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [
                KnownPersonProfile(
                    person_id=uuid4(),
                    full_name="Moving Person",
                    face_profiles=[
                        KnownPersonFaceProfile(face_id="front", embedding_vector=[1.0, 0.0])
                    ],
                )
            ],
        }
    )
    service = FaceRecognitionService()
    first_box = InferenceBox(
        x1=0,
        y1=0,
        x2=120,
        y2=300,
        confidence=0.9,
        label="person",
        track_id="pe-t1",
    )
    moved_box = first_box.model_copy(
        update={"x1": 300.0, "x2": 420.0}
    )

    first = service.enrich_detections(request, [first_box])[0]
    second = service.enrich_detections(request, [moved_box])[0]

    assert backend.calls == 1
    assert first.recognition and first.recognition.status == "known"
    assert second.recognition and second.recognition.deduplicated is True
    assert second.face_region and second.face_region.x1 >= moved_box.x1


def test_recognition_cache_age_starts_after_face_analysis(monkeypatch) -> None:
    clock = [0.0]

    class SlowBatchBackend:
        calls = 0

        def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
            self.calls += 1
            clock[0] = 1.0
            return [
                FaceEmbeddingResult(
                    vector=[1.0, 0.0],
                    model_name="insightface-buffalo_m",
                    backend_name="insightface",
                    metadata={
                        "backend": "insightface",
                        "bbox": [30.0, 20.0, 70.0, 65.0],
                        "det_score": 0.98,
                    },
                )
            ]

    backend = SlowBatchBackend()
    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: backend))
    monkeypatch.setattr("app.services.recognition.monotonic", lambda: clock[0])
    monkeypatch.setattr("app.services.recognition.settings.recognition_refresh_seconds", 0.5)
    request = build_request("slow-analysis").model_copy(
        update={
            "frame_content_base64": build_frame_base64(),
            "frame_content_type": "image/jpeg",
            "known_persons": [],
        }
    )
    detection = InferenceBox(
        x1=0,
        y1=0,
        x2=120,
        y2=300,
        confidence=0.9,
        label="person",
        track_id="pe-t1",
    )
    service = FaceRecognitionService()

    service.enrich_detections(request, [detection])
    clock[0] = 1.4
    second = service.enrich_detections(request, [detection])[0]

    assert backend.calls == 1
    assert second.recognition and second.recognition.deduplicated is True


def test_recognition_warmup_initializes_batch_backend(monkeypatch) -> None:
    class WarmupBackend:
        calls = 0

        def extract_embeddings(self, image_bytes: bytes) -> list[FaceEmbeddingResult]:
            self.calls += 1
            assert image_bytes.startswith(b"\xff\xd8")
            raise FaceEmbeddingError("No detectable face was found in the provided image.")

    backend = WarmupBackend()
    monkeypatch.setattr(FaceRecognitionService, "_build_backend", staticmethod(lambda: backend))

    FaceRecognitionService().warmup()

    assert backend.calls == 1


def test_recognition_aggregates_multiple_known_person_templates(monkeypatch) -> None:
    monkeypatch.setattr("app.services.recognition.settings.recognition_match_threshold", 0.70)
    service = FaceRecognitionService()
    person = KnownPersonProfile(
        person_id=uuid4(),
        full_name="Multi Angle Person",
        person_type="employee",
        face_profiles=[
            KnownPersonFaceProfile(face_id="front", embedding_vector=[1.0, 0.0]),
            KnownPersonFaceProfile(face_id="left", embedding_vector=[0.95, 0.05]),
            KnownPersonFaceProfile(face_id="right", embedding_vector=[0.8, 0.2]),
        ],
    )

    result = service._match_known_person(
        FaceRegion(x1=0, y1=0, x2=40, y2=40, confidence=0.95),
        [1.0, 0.0],
        [person],
    )

    assert result.status == "known"
    assert result.identity_label == "Multi Angle Person"
    assert result.metadata["template_count"] == 3
    assert result.metadata["max_template_score"] >= result.match_confidence


def test_recognition_ignores_duplicate_and_low_quality_templates(monkeypatch) -> None:
    monkeypatch.setattr("app.services.recognition.settings.recognition_template_min_det_score", 0.60)
    service = FaceRecognitionService()
    person = KnownPersonProfile(
        person_id=uuid4(),
        full_name="Curated Person",
        face_profiles=[
            KnownPersonFaceProfile(
                face_id="front",
                embedding_vector=[1.0, 0.0],
                metadata={"det_score": 0.99},
            ),
            KnownPersonFaceProfile(
                face_id="duplicate",
                embedding_vector=[1.0, 0.0],
                metadata={"det_score": 0.98},
            ),
            KnownPersonFaceProfile(
                face_id="blurred",
                embedding_vector=[0.0, 1.0],
                metadata={"det_score": 0.40},
            ),
        ],
    )

    result = service._match_known_person(
        FaceRegion(x1=0, y1=0, x2=40, y2=40, confidence=0.95),
        [1.0, 0.0],
        [person],
    )

    assert result.status == "known"
    assert result.metadata["source_template_count"] == 3
    assert result.metadata["template_count"] == 1


def test_recognition_rejects_low_quality_or_tiny_runtime_faces(monkeypatch) -> None:
    monkeypatch.setattr("app.services.recognition.settings.recognition_min_face_detection_score", 0.50)
    monkeypatch.setattr("app.services.recognition.settings.recognition_min_face_size", 32)
    detection = InferenceBox(x1=0, y1=0, x2=300, y2=400, confidence=0.95, label="person")
    faces = [
        FaceEmbeddingResult(
            vector=[1.0, 0.0],
            model_name="insightface-buffalo_m",
            backend_name="insightface",
            metadata={"bbox": [20.0, 20.0, 100.0, 100.0], "det_score": 0.30},
        ),
        FaceEmbeddingResult(
            vector=[1.0, 0.0],
            model_name="insightface-buffalo_m",
            backend_name="insightface",
            metadata={"bbox": [120.0, 20.0, 140.0, 42.0], "det_score": 0.99},
        ),
    ]

    assert FaceRecognitionService._best_face_index(detection, faces, set()) is None


def test_live_known_identity_is_only_held_for_same_near_threshold_candidate(monkeypatch) -> None:
    monkeypatch.setattr("app.services.recognition.settings.recognition_match_threshold", 0.60)
    monkeypatch.setattr("app.services.recognition.settings.recognition_min_margin", 0.10)
    identity_id = uuid4()
    detection = InferenceBox(x1=0, y1=0, x2=100, y2=200, confidence=0.95, label="person")
    payload = build_request("dashboard_live_scan").model_copy(
        update={"occurrence_hint": "dashboard_live_scan"}
    )
    known = InferenceRecognition(
        status="known",
        identity_id=identity_id,
        identity_label="Dana Holt",
        match_confidence=0.75,
    )
    near_match = InferenceRecognition(
        status="unknown",
        identity_label="Unknown visitor",
        match_confidence=0.58,
        metadata={"candidate_identity_id": str(identity_id), "match_margin": 0.20},
    )
    different_person = InferenceRecognition(
        status="unknown",
        identity_label="Unknown visitor",
        match_confidence=0.58,
        metadata={"candidate_identity_id": str(uuid4()), "match_margin": 0.20},
    )

    holding_service = FaceRecognitionService()
    holding_service._stabilize_live_recognition(payload, detection, known)
    held = holding_service._stabilize_live_recognition(payload, detection, near_match)

    switching_service = FaceRecognitionService()
    switching_service._stabilize_live_recognition(payload, detection, known)
    switched = switching_service._stabilize_live_recognition(payload, detection, different_person)

    assert held.status == "known"
    assert held.deduplicated is True
    assert switched.status == "unknown"


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
