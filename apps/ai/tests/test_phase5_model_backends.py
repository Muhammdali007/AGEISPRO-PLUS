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


class FakeModel:
    names = {0: "person", 1: "fire", 2: "smoke"}

    def track(self, **kwargs):
        return [FakeResult()]


def test_ultralytics_backend_parses_tracked_detections(monkeypatch) -> None:
    monkeypatch.setattr("app.services.pipeline.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.pipeline.settings.allow_backend_fallback", False)
    monkeypatch.setattr("app.services.backends.settings.model_backend", "ultralytics")
    monkeypatch.setattr("app.services.backends.settings.allow_backend_fallback", False)
    monkeypatch.setattr(UltralyticsInferenceBackend, "_load_model", lambda self: FakeModel())

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

    def fail_load_model(self):
        raise InferenceBackendUnavailableError("ultralytics unavailable in test")

    monkeypatch.setattr(UltralyticsInferenceBackend, "_load_model", fail_load_model)

    pipeline = InferencePipeline()
    result = pipeline.run(build_request())

    assert result.metadata["backend"] == "simulated"
    assert result.metadata["backend_fallback"] is True
    assert result.metadata["fallback_backend"] == "simulated"
    assert result.detections
