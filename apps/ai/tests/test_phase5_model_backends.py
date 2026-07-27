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


def test_ultralytics_backend_splits_batches_for_static_openvino_model(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "person_openvino_model"
    model_path.mkdir()
    (model_path / "metadata.yaml").write_text(
        "batch: 1\nimgsz:\n- 640\n- 640\n",
        encoding="utf-8",
    )
    calls: list[dict] = []

    class StaticBatchModel:
        def predict(self, **kwargs):
            calls.append(kwargs)
            assert not isinstance(kwargs["source"], list)
            assert "batch" not in kwargs
            return [FakeResult()]

    monkeypatch.setattr("app.services.backends.settings.model_batch_size", 8)
    backend = UltralyticsInferenceBackend()
    results = backend._run_model_invocation(
        model=StaticBatchModel(),
        model_path=str(model_path),
        images=[object(), object(), object()],
        requested_classes=[0],
        confidence=0.1,
        use_tracking=False,
    )

    assert len(calls) == 3
    assert len(results) == 3


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


def test_ultralytics_backend_rejects_tall_edge_strip_threat_false_positive(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.threat_edge_strip_margin_ratio", 0.04)
    monkeypatch.setattr("app.services.backends.settings.threat_edge_strip_max_width_ratio", 0.18)
    monkeypatch.setattr("app.services.backends.settings.threat_edge_strip_min_height_ratio", 0.45)
    monkeypatch.setattr("app.services.backends.settings.threat_edge_strip_min_aspect_ratio", 4.0)
    backend = UltralyticsInferenceBackend()
    result = type(
        "EdgeTreeResult",
        (),
        {
            "names": {0: "smoke"},
            "orig_shape": (540, 960),
            "boxes": type(
                "EdgeTreeBoxes",
                (),
                {
                    "xyxy": FakeTensor([[860, 20, 940, 520]]),
                    "conf": FakeTensor([0.88]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    assert backend._parse_results([result], ["smoke"]) == []


def test_ultralytics_backend_rejects_weak_generic_model_weapon_false_positive(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_generic_weapon_min_confidence", 0.65)
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "specialist.pt")
    backend = UltralyticsInferenceBackend()
    result = type(
        "FireSceneWeaponResult",
        (),
        {
            "names": {0: "weapon"},
            "orig_shape": (960, 540),
            "boxes": type(
                "FireSceneWeaponBoxes",
                (),
                {
                    "xyxy": FakeTensor([[80, 120, 460, 210]]),
                    "conf": FakeTensor([0.58]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    assert backend._parse_results([result], ["weapon"], model_path="general.pt") == []


def test_ultralytics_backend_keeps_specialist_weapon_below_generic_floor(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_generic_weapon_min_confidence", 0.65)
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "specialist.pt")
    backend = UltralyticsInferenceBackend()
    result = type(
        "SpecialistWeaponResult",
        (),
        {
            "names": {0: "weapon"},
            "orig_shape": (720, 1280),
            "boxes": type(
                "SpecialistWeaponBoxes",
                (),
                {
                    "xyxy": FakeTensor([[373, 310, 843, 533]]),
                    "conf": FakeTensor([0.58]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    detections = backend._parse_results(
        [result],
        ["weapon"],
        model_path="specialist.pt",
    )

    assert len(detections) == 1
    assert detections[0].label == "weapon"
    assert detections[0].confidence == 0.58


def test_ultralytics_backend_rejects_twenty_percent_weapon_false_positive(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.weapon_confidence_threshold", 0.25)
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "specialist.pt")
    backend = UltralyticsInferenceBackend()
    result = type(
        "WeakSpecialistWeaponResult",
        (),
        {
            "names": {0: "weapon"},
            "orig_shape": (960, 540),
            "boxes": type(
                "WeakSpecialistWeaponBoxes",
                (),
                {
                    "xyxy": FakeTensor([[65, 510, 505, 600]]),
                    "conf": FakeTensor([0.20]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    assert backend._parse_results(
        [result],
        ["weapon"],
        model_path="specialist.pt",
    ) == []


def test_ultralytics_backend_keeps_specific_weapon_below_generic_floor(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_generic_weapon_min_confidence", 0.65)
    backend = UltralyticsInferenceBackend()
    result = type(
        "SpecificWeaponResult",
        (),
        {
            "names": {0: "pistol"},
            "orig_shape": (480, 640),
            "boxes": type(
                "SpecificWeaponBoxes",
                (),
                {
                    "xyxy": FakeTensor([[100, 100, 180, 180]]),
                    "conf": FakeTensor([0.42]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    detections = backend._parse_results([result], ["weapon"], model_path="general.pt")

    assert len(detections) == 1
    assert detections[0].object_label == "pistol"


def test_ultralytics_backend_keeps_centered_tall_smoke_plume(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.threat_edge_strip_margin_ratio", 0.04)
    backend = UltralyticsInferenceBackend()
    result = type(
        "SmokePlumeResult",
        (),
        {
            "names": {0: "smoke"},
            "orig_shape": (540, 960),
            "boxes": type(
                "SmokePlumeBoxes",
                (),
                {
                    "xyxy": FakeTensor([[430, 20, 510, 520]]),
                    "conf": FakeTensor([0.88]),
                    "cls": FakeTensor([0]),
                    "id": None,
                },
            )(),
        },
    )()

    detections = backend._parse_results([result], ["smoke"])

    assert len(detections) == 1
    assert detections[0].label == "smoke"


def test_ultralytics_backend_keeps_only_strongest_box_per_threat() -> None:
    detections = [
        InferenceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.91, label="smoke"),
        InferenceBox(x1=20, y1=20, x2=30, y2=30, confidence=0.70, label="smoke"),
        InferenceBox(x1=40, y1=40, x2=50, y2=50, confidence=0.65, label="smoke"),
        InferenceBox(x1=60, y1=60, x2=70, y2=70, confidence=0.60, label="smoke"),
        InferenceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.88, label="weapon"),
        InferenceBox(x1=0, y1=0, x2=10, y2=10, confidence=0.95, label="person"),
    ]

    limited = UltralyticsInferenceBackend._limit_frame_detections(detections)

    assert [detection.label for detection in limited] == ["smoke", "smoke", "smoke", "weapon", "person"]
    assert [detection.confidence for detection in limited if detection.label == "smoke"] == [0.91, 0.70, 0.65]


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


def test_ultralytics_backend_keeps_confident_smoke_inside_person(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.smoke_max_person_coverage", 0.70)
    monkeypatch.setattr("app.services.backends.settings.smoke_person_conflict_min_confidence", 0.35)
    smoke = InferenceBox(x1=20, y1=20, x2=80, y2=80, confidence=0.48, label="smoke")
    person = InferenceBox(x1=0, y1=0, x2=100, y2=100, confidence=0.90, label="person")

    filtered = UltralyticsInferenceBackend._reject_cross_class_conflicts([person, smoke])

    assert filtered == [person, smoke]


def test_general_weapon_fallback_keeps_high_confidence_when_specialist_misses(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")
    monkeypatch.setattr("app.services.backends.settings.model_weapon_ensemble_general", True)
    monkeypatch.setattr("app.services.backends.settings.model_weapon_general_fallback_confidence", 0.72)
    general_weapon = InferenceBox(
        x1=10,
        y1=20,
        x2=100,
        y2=120,
        confidence=0.78,
        label="weapon",
        object_label="knife",
        source_model_path="general.pt",
    )

    filtered = UltralyticsInferenceBackend._confirm_specialist_weapon_detections([general_weapon])

    assert filtered == [general_weapon]


def test_general_weapon_fallback_rejects_weak_general_hit_when_specialist_misses(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")
    monkeypatch.setattr("app.services.backends.settings.model_weapon_ensemble_general", True)
    monkeypatch.setattr("app.services.backends.settings.model_weapon_general_fallback_confidence", 0.72)
    general_weapon = InferenceBox(
        x1=10,
        y1=20,
        x2=100,
        y2=120,
        confidence=0.55,
        label="weapon",
        object_label="knife",
        source_model_path="general.pt",
    )

    filtered = UltralyticsInferenceBackend._confirm_specialist_weapon_detections([general_weapon])

    assert filtered == []


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
