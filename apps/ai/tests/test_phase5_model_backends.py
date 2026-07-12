from uuid import uuid4

from app.schemas.inference import InferenceRequest
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


def test_ultralytics_backend_ensembles_specialist_and_general_weapon_models(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    monkeypatch.setattr("app.services.backends.settings.model_person_weapon_weights_path", "general.pt")

    assignments = UltralyticsInferenceBackend()._assign_detectors_to_models(["person", "weapon"])

    assert assignments == {
        "general.pt": ["person", "weapon"],
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


def test_weapon_training_labels_are_normalized(monkeypatch) -> None:
    monkeypatch.setattr("app.services.backends.settings.model_weapon_weights_path", "weapon.pt")
    backend = UltralyticsInferenceBackend()
    model = type(
        "TrainedWeaponModel",
        (),
        {"names": {0: "kitchen_knife", 1: "shotgun", 2: "handgun"}},
    )()

    assert backend._resolve_requested_classes(model, ["weapon"]) == [0, 1, 2]
