import base64
import hashlib
import os
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.inference import InferenceBox, InferenceRequest

PROJECT_ROOT = Path(__file__).resolve().parents[4]


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
        supported = [
            label
            for label in normalize_requested_detectors(payload.requested_detectors)
            if label in {"weapon", "fire", "smoke", "person"}
        ]
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
        self._models: dict[str, Any] = {}

    def infer(self, payload: InferenceRequest) -> BackendInferenceOutput:
        if not payload.frame_content_base64:
            raise InferenceBackendInputError(
                "The ultralytics backend requires frame_content_base64 for model-backed inference."
            )

        image = self._decode_image(payload.frame_content_base64)
        assignments = self._assign_detectors_to_models(payload.requested_detectors)
        if not assignments:
            return BackendInferenceOutput(
                detections=[],
                backend_name=self.backend_name,
                model_name=settings.model_name,
                model_version=settings.model_version,
                inference_fps=settings.inference_fps,
                metadata={
                    "active_models": [],
                    "unsupported_requested_detectors": payload.requested_detectors,
                },
            )

        started_at = perf_counter()
        detections: list[InferenceBox] = []
        active_models: list[dict[str, Any]] = []
        unsupported = normalize_requested_detectors(payload.requested_detectors)
        for model_path, detectors in assignments.items():
            model = self._load_model(model_path)
            requested_classes = self._resolve_requested_classes(model, detectors)
            supported_detectors = self._resolve_supported_detectors(model, detectors)
            if not requested_classes:
                continue

            unsupported = [detector for detector in unsupported if detector not in supported_detectors]

            inference_options = {
                "source": image,
                "conf": min(
                    self._confidence_threshold(detector) for detector in supported_detectors
                ),
                "classes": requested_classes,
                "device": settings.model_device,
                "imgsz": settings.model_image_size,
                "verbose": False,
            }
            if settings.model_half_precision:
                inference_options["half"] = True
            if "person" in supported_detectors:
                results = model.track(
                    **inference_options,
                    persist=settings.model_track_persist,
                    tracker=settings.model_tracker_config,
                )
            else:
                # Hazard confidence thresholds are intentionally lower than the
                # tracker admission threshold, so direct prediction must retain them.
                results = model.predict(**inference_options)
            detections.extend(self._parse_results(results, detectors))
            active_models.append(
                {
                    "weights_path": model_path,
                    "requested_detectors": detectors,
                    "requested_classes": requested_classes,
                }
            )

        elapsed = max(perf_counter() - started_at, 1e-6)
        return BackendInferenceOutput(
            detections=self._limit_frame_detections(
                self._deduplicate_detections(detections)
            ),
            backend_name=self.backend_name,
            model_name=settings.model_name,
            model_version=settings.model_version,
            inference_fps=round(1 / elapsed, 2),
            metadata={
                "weights_path": settings.model_weights_path,
                "active_models": active_models,
                "tracker": settings.model_tracker_config,
                "unsupported_requested_detectors": unsupported,
            },
        )

    def _load_model(self, weights_path: str):
        if weights_path in self._models:
            return self._models[weights_path]

        if "YOLO_CONFIG_DIR" not in os.environ:
            yolo_config_dir = PROJECT_ROOT / "storage" / "ultralytics"
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
        if "MPLCONFIGDIR" not in os.environ:
            matplotlib_config_dir = PROJECT_ROOT / "storage" / "matplotlib"
            matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceBackendUnavailableError(
                "Ultralytics is not installed. Install `aegispro-ai[model]` to enable YOLO11/ByteTrack inference."
            ) from exc

        self._models[weights_path] = YOLO(weights_path)
        return self._models[weights_path]

    def _assign_detectors_to_models(self, requested_detectors: list[str]) -> dict[str, list[str]]:
        assignments: dict[str, list[str]] = {}
        requested = normalize_requested_detectors(requested_detectors)
        for detector in requested:
            for model_path in self._model_paths_for_detector(detector):
                detectors = assignments.setdefault(model_path, [])
                if detector not in detectors:
                    detectors.append(detector)
        return assignments

    def _model_paths_for_detector(self, detector: str) -> list[str]:
        if detector == "person":
            return [settings.model_person_weapon_weights_path or settings.model_weights_path]
        if detector == "weapon":
            specialist = settings.model_weapon_weights_path
            general = settings.model_person_weapon_weights_path or settings.model_weights_path
            return list(dict.fromkeys(path for path in (specialist, general) if path))
        if detector in {"fire", "smoke"}:
            return [settings.model_fire_smoke_weights_path or settings.model_weights_path]
        return [settings.model_weights_path]

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
        requested = set(normalize_requested_detectors(requested_detectors))
        class_ids: list[int] = []

        for class_id, class_name in names.items():
            normalized = self._normalize_model_label(str(class_name))
            if normalized in requested:
                class_ids.append(int(class_id))

        return class_ids

    def _resolve_supported_detectors(self, model: Any, requested_detectors: list[str]) -> set[str]:
        names = getattr(model, "names", {}) or {}
        requested = set(normalize_requested_detectors(requested_detectors))
        return {
            normalized
            for class_name in names.values()
            if (normalized := self._normalize_model_label(str(class_name))) in requested
        }

    def _parse_results(self, results: Any, requested_detectors: list[str]) -> list[InferenceBox]:
        requested = set(normalize_requested_detectors(requested_detectors))
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

                object_label = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
                threshold = self._confidence_threshold(label)
                confidence = float(confidences[index]) if index < len(confidences) else threshold
                if confidence < threshold:
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
                        object_label=object_label,
                        track_id=track_id,
                    )
                )

        return detections

    @staticmethod
    def _confidence_threshold(label: str) -> float:
        return {
            "person": settings.person_confidence_threshold,
            "weapon": settings.weapon_confidence_threshold,
            "fire": settings.fire_confidence_threshold,
            "smoke": settings.smoke_confidence_threshold,
        }.get(label, settings.confidence_threshold)

    @classmethod
    def _deduplicate_detections(cls, detections: list[InferenceBox]) -> list[InferenceBox]:
        kept: list[InferenceBox] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                existing.label == candidate.label and cls._intersection_over_union(existing, candidate) >= 0.6
                for existing in kept
            ):
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _limit_frame_detections(detections: list[InferenceBox]) -> list[InferenceBox]:
        strongest_weapon = next(
            (detection for detection in detections if detection.label == "weapon"),
            None,
        )
        non_weapons = [detection for detection in detections if detection.label != "weapon"][:10]
        return ([strongest_weapon] if strongest_weapon else []) + non_weapons

    @staticmethod
    def _intersection_over_union(first: InferenceBox, second: InferenceBox) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
        second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
        union = first_area + second_area - intersection
        return intersection / union if union > 0 else 0.0

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


def normalize_requested_detectors(requested_detectors: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw_label in requested_detectors:
        label = UltralyticsInferenceBackend._normalize_model_label(raw_label)
        if label not in normalized:
            normalized.append(label)
    return normalized


def build_inference_backend(backend_name: str) -> InferenceBackend:
    normalized = backend_name.strip().lower().replace("-", "_")
    if normalized == "simulated":
        return SimulatedInferenceBackend()
    if normalized in {"ultralytics", "yolo11"}:
        return UltralyticsInferenceBackend()
    raise InferenceBackendUnavailableError(f"Unsupported inference backend: {backend_name}")
