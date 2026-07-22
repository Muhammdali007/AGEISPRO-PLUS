import sys
from types import SimpleNamespace
from uuid import uuid4

from app.schemas.inference import InferenceBox, InferenceRequest
from app.services.backends import InferenceBackendUnavailableError, UltralyticsInferenceBackend
from app.services.pipeline import InferencePipeline

TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8"
    "z8BQDwAFgwJ/l8QlxQAAAABJRU5ErkJggg=="
)


def build_request() -> InferenceRequest:
    return InferenceRequest(
        camera_id=uuid4(),
        frame_reference="frame-real-backend",
        source_type="http",
        frame_content_base64=TINY_PNG_BASE64,
        frame_content_type="image/png",
        requested_detectors=["person", "fire"],
    )


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeBoxes:
    xyxy = FakeTensor([[10, 20, 110, 220], [140, 40, 260, 190]])
    conf = FakeTensor([0.94, 0.81])
    cls = FakeTensor([0, 1])
    id = FakeTensor([7, 11])


class FakeResult:
    names = {0: "person", 1: "fire"}
    boxes = FakeBoxes()


class FakeWeaponBoxes:
    xyxy = FakeTensor([[12, 30, 100, 120]])
    conf = FakeTensor([0.91])
    cls = FakeTensor([0])
    id = FakeTensor([5])


class FakeWeaponResult:
    names = {0: "knife"}
    boxes = FakeWeaponBoxes()


class FakeModel:
    names = {0: "person", 1: "fire", 2: "smoke"}

    def track(self, **kwargs):
        return [FakeResult()]

    def predict(self, **kwargs):
        return [FakeResult()]


class FakeBatchedModel(FakeModel):
    def predict(self, **kwargs):
        source = kwargs.get("source")
        if isinstance(source, list):
            return [FakeResult() for _ in source]
        return [FakeResult()]


class FakeWeaponOnlyModel:
    names = {0: "knife", 1: "scissors", 2: "gun"}

    def track(self, **kwargs):
        return [FakeWeaponResult()]

    def predict(self, **kwargs):
        return [FakeWeaponResult()]


class FakePersonOnlyModel:
    names = {0: "person"}

    def track(self, **kwargs):
        return []

    def predict(self, **kwargs):
        return []


def test_ultralytics_backend_parses_tracked_detections(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.pipeline.settings.allow_backend_fallback", False)
    monkeypatch.setattr("app.services.backends.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.backends.settings.allow_backend_fallback", False)
    monkeypatch.setattr(UltralyticsInferenceBackend, "_load_model", lambda self, weights_path: FakeModel())

    pipeline = InferencePipeline()
    result = pipeline.run(build_request())

    assert len(result.detections) == 2
    assert result.metadata["backend"] == "ultralytics"
    assert result.detections[0].label == "person"
    assert result.detections[0].track_id == "pe-7"
    assert result.detections[1].label == "fire"
    assert result.detections[1].track_id == "fi-11"
    assert result.inference_fps > 0


def test_pipeline_falls_back_to_simulated_backend(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.pipeline.settings.model_fallback_backend", "simulated")
    monkeypatch.setattr("app.services.pipeline.settings.allow_backend_fallback", True)
    monkeypatch.setattr("app.services.backends.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.backends.settings.model_fallback_backend", "simulated")
    monkeypatch.setattr("app.services.backends.settings.allow_backend_fallback", True)

    def fail_load_model(self, weights_path):
        raise InferenceBackendUnavailableError("ultralytics unavailable in test")

    monkeypatch.setattr(UltralyticsInferenceBackend, "_load_model", fail_load_model)

    pipeline = InferencePipeline()
    result = pipeline.run(build_request())

    assert result.metadata["backend"] == "simulated"
    assert result.metadata["backend_fallback"] is True
    assert result.metadata["fallback_backend"] == "simulated"
    assert result.detections


def test_ultralytics_backend_routes_detectors_to_dedicated_models(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.pipeline.settings.allow_backend_fallback", False)
    monkeypatch.setattr("app.services.backends.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.backends.settings.allow_backend_fallback", False)
    monkeypatch.setattr(
        "app.services.backends.settings.model_person_weapon_weights_path",
        "storage/models/yolo11n.pt",
    )
    monkeypatch.setattr(
        "app.services.backends.settings.model_fire_smoke_weights_path",
        "storage/models/fire-smoke.pt",
    )
    monkeypatch.setattr("app.services.backends.settings.model_fire_weights_path", None)
    monkeypatch.setattr("app.services.backends.settings.model_smoke_weights_path", None)
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", None)

    model_by_path = {
        "storage/models/yolo11n.pt": FakeWeaponOnlyModel(),
        "storage/models/fire-smoke.pt": FakeModel(),
    }

    monkeypatch.setattr(
        UltralyticsInferenceBackend,
        "_load_model",
        lambda self, weights_path: model_by_path[weights_path],
    )

    pipeline = InferencePipeline()
    result = pipeline.run(
        build_request().model_copy(update={"requested_detectors": ["weapon", "fire", "smoke"]})
    )

    active_models = result.metadata["active_models"]
    assert len(active_models) == 2
    assert {item["weights_path"] for item in active_models} == {
        "storage/models/yolo11n.pt",
        "storage/models/fire-smoke.pt",
    }
    assert {detection.label for detection in result.detections} == {"weapon", "fire"}


def test_ultralytics_backend_prefers_dedicated_fire_and_smoke_models(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_fire_weights_path", "fire.pt")
    monkeypatch.setattr("app.services.backends.settings.model_smoke_weights_path", "smoke.pt")
    monkeypatch.setattr(
        "app.services.backends.settings.model_fire_smoke_weights_path",
        "fire-smoke.pt",
    )

    assignments = UltralyticsInferenceBackend()._assign_detectors_to_models(["fire", "smoke"])

    assert assignments == {"fire.pt": ["fire"], "smoke.pt": ["smoke"]}


def test_ultralytics_backend_preloads_only_effective_detector_models(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weights_path", "general.pt")
    monkeypatch.setattr(
        "app.services.backends.settings.model_person_weapon_weights_path",
        "general.pt",
    )
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr(
        "app.services.backends.settings.model_fire_smoke_weights_path",
        "fire-smoke.pt",
    )
    monkeypatch.setattr("app.services.backends.settings.model_fire_weights_path", "fire.pt")
    monkeypatch.setattr("app.services.backends.settings.model_smoke_weights_path", "smoke.pt")

    assert UltralyticsInferenceBackend()._configured_model_paths() == [
        "general.pt",
        "weapon.pt",
        "fire.pt",
        "smoke.pt",
    ]


def test_ultralytics_backend_uses_fixed_openvino_image_size(tmp_path) -> None:
    model_path = tmp_path / "hazard_openvino_model"
    model_path.mkdir()
    (model_path / "metadata.yaml").write_text("imgsz:\n- 320\n- 320\n", encoding="utf-8")
    calls: list[dict] = []

    class RecordingModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            return []

    backend = UltralyticsInferenceBackend()
    backend._run_model_invocation(
        model=RecordingModel(),
        model_path=str(model_path),
        images=[object()],
        requested_classes=[0, 1],
        confidence=0.1,
        use_tracking=False,
    )

    assert calls[0]["imgsz"] == 320


def test_ultralytics_backend_normalizes_weapon_detector_aliases(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "storage/models/weapon.pt")
    monkeypatch.setattr(
        "app.services.backends.settings.model_person_weapon_weights_path",
        "storage/models/weapon.pt",
    )

    backend = UltralyticsInferenceBackend()
    model = FakeWeaponOnlyModel()

    assignments = backend._assign_detectors_to_models(["knife", "scissor", "gun"])
    requested_classes = backend._resolve_requested_classes(model, assignments["storage/models/weapon.pt"])
    detections = backend._parse_results([FakeWeaponResult()], ["knife", "scissor", "gun"])

    assert assignments == {"storage/models/weapon.pt": ["weapon"]}
    assert requested_classes == [0, 1, 2]
    assert detections[0].label == "weapon"


def test_ultralytics_backend_excludes_ambiguous_weapon_classes(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_excluded_labels", ["scissors"])
    backend = UltralyticsInferenceBackend()
    result = type(
        "ScissorsResult",
        (),
        {
            "names": {0: "scissors"},
            "boxes": type(
                "ScissorsBoxes",
                (),
                {
                    "xyxy": FakeTensor([[10, 10, 100, 100]]),
                    "conf": FakeTensor([0.99]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    assert backend._parse_results([result], ["weapon"]) == []


def test_weapon_exclusion_matches_canonical_singular_and_plural_labels(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_excluded_labels", ["scissor"])
    backend = UltralyticsInferenceBackend()
    result = type(
        "ScissorsResult",
        (),
        {
            "names": {0: "scissors"},
            "boxes": type(
                "ScissorsBoxes",
                (),
                {
                    "xyxy": FakeTensor([[10, 10, 100, 100]]),
                    "conf": FakeTensor([0.99]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    assert backend._parse_results([result], ["weapon"], model_path=None) == []


def test_ambiguous_general_exclusion_does_not_hide_specialist_class(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_excluded_labels", ["scissors"])
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")
    backend = UltralyticsInferenceBackend()
    result = type(
        "SpecialistScissorsResult",
        (),
        {
            "names": {0: "scissors"},
            "orig_shape": (480, 640),
            "boxes": type(
                "SpecialistScissorsBoxes",
                (),
                {
                    "xyxy": FakeTensor([[10, 10, 100, 100]]),
                    "conf": FakeTensor([0.99]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    detections = backend._parse_results([result], ["weapon"], model_path="weapon.pt")

    assert len(detections) == 1
    assert detections[0].object_label == "scissors"


def test_ultralytics_backend_keeps_only_strongest_box_per_threat() -> None:
    detections = [
        InferenceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.91, label="smoke"),
        InferenceBox(x1=20, y1=20, x2=30, y2=30, confidence=0.70, label="smoke"),
        InferenceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.88, label="weapon"),
        InferenceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.95, label="person"),
    ]

    limited = UltralyticsInferenceBackend._limit_frame_detections(detections)

    assert [detection.label for detection in limited] == ["smoke", "weapon", "person"]
    assert next(detection for detection in limited if detection.label == "smoke").confidence == 0.91


def test_ultralytics_backend_rejects_smoke_contained_inside_person(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.smoke_max_person_coverage", 0.70)
    smoke = InferenceBox(x1=20, y1=20, x2=80, y2=80, confidence=0.20, label="smoke")
    person = InferenceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.90, label="person")

    filtered = UltralyticsInferenceBackend._reject_cross_class_conflicts([person, smoke])

    assert filtered == [person]


def test_ultralytics_backend_keeps_smoke_outside_person(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.smoke_max_person_coverage", 0.70)
    smoke = InferenceBox(x1=70, y1=20, x2=130, y2=80, confidence=0.20, label="smoke")
    person = InferenceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.90, label="person")

    filtered = UltralyticsInferenceBackend._reject_cross_class_conflicts([person, smoke])

    assert filtered == [person, smoke]


def test_ultralytics_backend_ensembles_specialist_and_general_weapon_models(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")
    monkeypatch.setattr("app.services.backends.settings.model_weapon_ensemble_general", True)

    assignments = UltralyticsInferenceBackend()._assign_detectors_to_models(["person", "weapon"])

    assert assignments == {
        "general.pt": ["person", "weapon"],
        "weapon.pt": ["weapon"],
    }


def test_ultralytics_backend_prefers_specialist_weapon_model_by_default(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")
    monkeypatch.setattr("app.services.backends.settings.model_weapon_ensemble_general", False)

    assignments = UltralyticsInferenceBackend()._assign_detectors_to_models(["person", "weapon"])

    assert assignments == {
        "general.pt": ["person"],
        "weapon.pt": ["weapon"],
    }


def test_ultralytics_backend_reports_detectors_missing_from_model(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.pipeline.settings.allow_backend_fallback", False)
    monkeypatch.setattr("app.services.backends.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.backends.settings.allow_backend_fallback", False)
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "generic.pt")
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", None)
    monkeypatch.setattr("app.services.backends.settings.model_fire_smoke_weights_path", None)
    monkeypatch.setattr("app.services.backends.settings.model_fire_weights_path", None)
    monkeypatch.setattr("app.services.backends.settings.model_smoke_weights_path", None)
    monkeypatch.setattr("app.services.backends.settings.model_weights_path", "generic.pt")
    monkeypatch.setattr(
        UltralyticsInferenceBackend,
        "_load_model",
        lambda self, weights_path: FakePersonOnlyModel(),
    )

    result = InferencePipeline().run(
        build_request().model_copy(
            update={"requested_detectors": ["weapon", "fire", "smoke", "person"]}
        )
    )

    assert result.metadata["unsupported_requested_detectors"] == ["weapon", "fire", "smoke"]


def test_ultralytics_backend_batches_multi_camera_requests(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.pipeline.settings.allow_backend_fallback", False)
    monkeypatch.setattr("app.services.backends.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.backends.settings.allow_backend_fallback", False)
    monkeypatch.setattr(
        UltralyticsInferenceBackend,
        "_load_model",
        lambda self, weights_path: FakeBatchedModel(),
    )

    pipeline = InferencePipeline()
    results = pipeline.run_batch(
        [
            build_request().model_copy(update={"frame_reference": "batch-1"}),
            build_request().model_copy(update={"frame_reference": "batch-2"}),
        ]
    )

    assert len(results) == 2
    assert all(result.metadata["batch_size"] == 2 for result in results)
    assert all(result.metadata["batched"] is True for result in results)
    assert all(result.metadata["active_models"][0]["mode"] == "predict" for result in results)
    assert all(len(result.detections) == 2 for result in results)


def test_weapon_training_labels_are_normalized(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    backend = UltralyticsInferenceBackend()
    model = type(
        "TrainedWeaponModel",
        (),
        {"names": {0: "kitchen_knife", 1: "shotgun", 2: "handgun", 3: "other_weapon"}},
    )()

    assert backend._resolve_requested_classes(model, ["weapon"]) == [0, 1, 2, 3]


def test_weapon_labels_are_canonicalized_for_display(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_excluded_labels", [])
    backend = UltralyticsInferenceBackend()
    result = type(
        "HandgunResult",
        (),
        {
            "names": {0: "handgun"},
            "orig_shape": (480, 640),
            "boxes": type(
                "HandgunBoxes",
                (),
                {
                    "xyxy": FakeTensor([[20, 30, 120, 130]]),
                    "conf": FakeTensor([0.87]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    detections = backend._parse_results([result], ["weapon"], model_path="multiclass.pt")

    assert detections[0].label == "weapon"
    assert detections[0].object_label == "pistol"
    assert detections[0].source_model_path == "multiclass.pt"
    assert "source_model_path" not in detections[0].model_dump()


def test_general_weapon_subtype_requires_and_enriches_specialist_detection(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")
    monkeypatch.setattr("app.services.backends.settings.model_weapon_ensemble_general", True)
    backend = UltralyticsInferenceBackend()
    specialist = InferenceBox(
        x1=10,
        y1=10,
        x2=150,
        y2=180,
        confidence=0.92,
        label="weapon",
        object_label="other_weapon",
        source_model_path="weapon.pt",
    )
    confirmed_knife = InferenceBox(
        x1=25,
        y1=30,
        x2=130,
        y2=160,
        confidence=0.98,
        label="weapon",
        object_label="knife",
        source_model_path="general.pt",
    )
    standalone_scissors = InferenceBox(
        x1=300,
        y1=30,
        x2=380,
        y2=120,
        confidence=0.95,
        label="weapon",
        object_label="scissors",
        source_model_path="general.pt",
    )

    confirmed = backend._confirm_specialist_weapon_detections(
        [specialist, confirmed_knife, standalone_scissors]
    )
    merged = backend._deduplicate_detections(confirmed)

    assert len(merged) == 1
    assert merged[0].confidence == specialist.confidence
    assert merged[0].source_model_path == "weapon.pt"
    assert merged[0].object_label == "knife"


def test_numeric_cuda_device_falls_back_to_cpu_when_cuda_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_device", "0")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )

    assert UltralyticsInferenceBackend()._resolve_device() == "cpu"


def test_nested_same_label_detections_are_deduplicated() -> None:
    detections = [
        InferenceBox(x1=10, y1=10, x2=200, y2=300, confidence=0.95, label="person"),
        InferenceBox(x1=30, y1=40, x2=180, y2=280, confidence=0.90, label="person"),
    ]

    result = UltralyticsInferenceBackend._deduplicate_detections(detections)

    assert result == [detections[0]]
