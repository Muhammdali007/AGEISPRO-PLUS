import base64
import hashlib
from dataclasses import dataclass, field
from io import BytesIO
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.inference import InferenceBox, InferenceRequest


class InferenceBackendError(RuntimeError):
    pass


class InferenceBackendUnavailableError(InferenceBackendError):
    pass


class InferenceBackendInputError(InferenceBackendError):
    pass


@dataclass
class BackendInferenceOutput:
    detections: list[InferenceBox]
    backend_name: str
    model_name: str
    model_version: str
    inference_fps: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend:
    backend_name = "unknown"

    def infer(self, payload: InferenceRequest) -> BackendInferenceOutput:
        raise NotImplementedError


class SimulatedInferenceBackend(InferenceBackend):
    backend_name = "simulated"

    def infer(self, payload: InferenceRequest) -> BackendInferenceOutput:
        supported = [label for label in payload.requested_detectors if label in {"weapon", "fire", "smoke", "person"}]
        if not supported:
            return BackendInferenceOutput(
                detections=[],
                backend_name=self.backend_name,
                model_name=settings.model_name,
                model_version=settings.model_version,
                inference_fps=settings.inference_fps,
            )

        seed = hashlib.sha256(
            f"{payload.camera_id}:{payload.frame_reference}:{payload.occurrence_hint}".encode("utf-8")
        ).hexdigest()
        detections: list[InferenceBox] = []

        for index, label in enumerate(supported):
            value = int(seed[index * 8 : index * 8 + 8], 16)
            confidence = 0.55 + ((value % 40) / 100)
            if confidence < settings.confidence_threshold:
                continue
            left = float(10 + (value % 120))
            top = float(20 + ((value // 7) % 90))
            width = float(120 + ((value // 11) % 80))
            height = float(140 + ((value // 13) % 70))
            detections.append(
                InferenceBox(
                    x1=left,
                    y1=top,
                    x2=left + width,
                    y2=top + height,
                    confidence=round(min(confidence, 0.99), 2),
                    label=label,
                    track_id=f"{label[:2]}-{value % 9999}",
                )
            )

        return BackendInferenceOutput(
            detections=detections,
            backend_name=self.backend_name,
            model_name=settings.model_name,
            model_version=settings.model_version,
            inference_fps=settings.inference_fps,
        )


class UltralyticsInferenceBackend(InferenceBackend):
    backend_name = "ultralytics"

    def __init__(self) -> None:
        self._model = None

    def infer(self, payload: InferenceRequest) -> BackendInferenceOutput:
        if not payload.frame_content_base64:
            raise InferenceBackendInputError(
                "The ultralytics backend requires frame_content_base64 for model-backed inference."
            )

        image = self._decode_image(payload.frame_content_base64)
        model = self._load_model()
        requested_classes = self._resolve_requested_classes(model, payload.requested_detectors)

        started_at = perf_counter()
        results = model.track(
            source=image,
            conf=settings.confidence_threshold,
            classes=requested_classes or None,
            device=settings.model_device,
            imgsz=settings.model_image_size,
            half=settings.model_half_precision,
            persist=settings.model_track_persist,
            tracker=settings.model_tracker_config,
            verbose=False,
        )
        elapsed = max(perf_counter() - started_at, 1e-6)
        detections = self._parse_results(results, payload.requested_detectors)

        return BackendInferenceOutput(
            detections=detections,
            backend_name=self.backend_name,
            model_name=settings.model_name,
            model_version=settings.model_version,
            inference_fps=round(1 / elapsed, 2),
            metadata={
                "weights_path": settings.model_weights_path,
                "tracker": settings.model_tracker_config,
                "requested_classes": requested_classes,
            },
        )

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceBackendUnavailableError(
                "Ultralytics is not installed. Install `aegispro-ai[model]` to enable YOLO11/ByteTrack inference."
            ) from exc

        self._model = YOLO(settings.model_weights_path)
        return self._model

    @staticmethod
    def _decode_image(frame_content_base64: str) -> Image.Image:
        try:
            raw = base64.b64decode(frame_content_base64)
        except ValueError as exc:
            raise InferenceBackendInputError("frame_content_base64 is not valid base64 image data.") from exc

        try:
            return Image.open(BytesIO(raw)).convert("RGB")
        except UnidentifiedImageError as exc:
            raise InferenceBackendInputError("frame_content_base64 did not contain a readable image.") from exc

    def _resolve_requested_classes(self, model: Any, requested_detectors: list[str]) -> list[int]:
        names = getattr(model, "names", {}) or {}
        requested = {label.strip().lower() for label in requested_detectors}
        class_ids: list[int] = []

        for class_id, class_name in names.items():
            normalized = self._normalize_model_label(str(class_name))
            if normalized in requested:
                class_ids.append(int(class_id))

        return class_ids

    def _parse_results(self, results: Any, requested_detectors: list[str]) -> list[InferenceBox]:
        requested = {label.strip().lower() for label in requested_detectors}
        detections: list[InferenceBox] = []

        for result in results:
            names = getattr(result, "names", {}) or {}
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            xyxy_rows = self._to_rows(getattr(boxes, "xyxy", []))
            confidences = self._to_flat_list(getattr(boxes, "conf", []))
            class_ids = self._to_flat_list(getattr(boxes, "cls", []))
            track_ids = self._to_flat_list(getattr(boxes, "id", [])) if getattr(boxes, "id", None) is not None else []

            for index, coords in enumerate(xyxy_rows):
                class_id = int(class_ids[index]) if index < len(class_ids) else -1
                raw_label = str(names.get(class_id, class_id))
                label = self._normalize_model_label(raw_label)
                if label not in requested:
                    continue

                confidence = float(confidences[index]) if index < len(confidences) else settings.confidence_threshold
                if confidence < settings.confidence_threshold:
                    continue

                track_id = None
                if index < len(track_ids):
                    track_id = f"{label[:2]}-{int(float(track_ids[index]))}"
                else:
                    track_id = f"{label[:2]}-{index}"

                detections.append(
                    InferenceBox(
                        x1=float(coords[0]),
                        y1=float(coords[1]),
                        x2=float(coords[2]),
                        y2=float(coords[3]),
                        confidence=round(min(confidence, 0.99), 2),
                        label=label,
                        track_id=track_id,
                    )
                )

        return detections

    @staticmethod
    def _to_flat_list(value: Any) -> list[float]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            raw = value.tolist()
            if isinstance(raw, list):
                return [float(item) for item in raw]
            return [float(raw)]
        if isinstance(value, list):
            return [float(item) for item in value]
        return [float(value)]

    @staticmethod
    def _to_rows(value: Any) -> list[list[float]]:
        if value is None:
            return []
        if hasattr(value, "tolist"):
            raw = value.tolist()
            if isinstance(raw, list):
                return [[float(cell) for cell in row] for row in raw]
        return [[float(cell) for cell in row] for row in value]

    @staticmethod
    def _normalize_model_label(label: str) -> str:
        normalized = label.strip().lower().replace("-", "_").replace(" ", "_")
        for detector, aliases in settings.model_label_aliases.items():
            if normalized == detector or normalized in aliases:
                return detector
        return normalized


def build_inference_backend(backend_name: str) -> InferenceBackend:
    normalized = backend_name.strip().lower().replace("-", "_")
    if normalized == "simulated":
        return SimulatedInferenceBackend()
    if normalized in {"ultralytics", "yolo11"}:
        return UltralyticsInferenceBackend()
    raise InferenceBackendUnavailableError(f"Unsupported inference backend: {backend_name}")
