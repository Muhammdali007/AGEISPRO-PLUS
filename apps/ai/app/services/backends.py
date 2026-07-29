import base64
import hashlib
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter
from typing import Any

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.inference import InferenceBox, InferenceRequest

_SERVICE_FILE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (
        parent
        for parent in _SERVICE_FILE.parents
        if (parent / "docker-compose.yml").exists() or (parent / "storage").exists()
    ),
    _SERVICE_FILE.parents[min(2, len(_SERVICE_FILE.parents) - 1)],
)


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


@dataclass
class PreparedInferencePayload:
    payload: InferenceRequest
    image: Image.Image
    assignments: dict[str, list[str]]
    unsupported: list[str]
    detections: list[InferenceBox] = field(default_factory=list)
    active_models: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BatchInvocationKey:
    model_path: str
    detectors: tuple[str, ...]
    requested_classes: tuple[int, ...]
    confidence: float
    use_tracking: bool


class InferenceBackend:
    backend_name = "unknown"

    def infer(self, payload: InferenceRequest) -> BackendInferenceOutput:
        raise NotImplementedError

    def infer_batch(self, payloads: list[InferenceRequest]) -> list[BackendInferenceOutput]:
        return [self.infer(payload) for payload in payloads]

    def warmup(self) -> None:
        return None

    def snapshot_runtime_state(self) -> dict[str, Any]:
        return {}


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
        self._model_image_sizes: dict[str, int | list[int]] = {}
        self._model_batch_sizes: dict[str, int] = {}
        self._resolved_device = self._resolve_device()
        self._runtime_stats: dict[str, Any] = {
            "single_requests_total": 0,
            "batch_requests_total": 0,
            "images_processed_total": 0,
            "last_batch_size": 0,
            "max_observed_batch_size": 0,
            "warmup_completed_at": None,
            "warmup_model_count": 0,
            "warmup_inference_seconds": None,
        }

    def infer(self, payload: InferenceRequest) -> BackendInferenceOutput:
        return self._infer_payloads([payload], allow_tracking=settings.model_enable_tracking)[0]

    def infer_batch(self, payloads: list[InferenceRequest]) -> list[BackendInferenceOutput]:
        return self._infer_payloads(payloads, allow_tracking=False)

    def warmup(self) -> None:
        self._prepare_runtime_environment()
        if not settings.model_preload_on_startup:
            return
        started_at = perf_counter()
        warmed_models = 0
        for model_path in self._configured_model_paths():
            model = self._load_model(model_path)
            image_size = self._image_size_for_model(model_path)
            warmup_width, warmup_height = (
                (image_size, image_size)
                if isinstance(image_size, int)
                else (image_size[1], image_size[0])
            )
            warmup_image = Image.new(
                "RGB",
                (warmup_width, warmup_height),
                color=(0, 0, 0),
            )
            options: dict[str, Any] = {
                "source": warmup_image,
                "conf": 0.99,
                "device": self._resolved_device,
                "imgsz": image_size,
                "verbose": False,
            }
            if settings.model_half_precision:
                options["half"] = True
            self._to_result_list(model.predict(**options))
            warmed_models += 1
        self._runtime_stats["warmup_model_count"] = warmed_models
        self._runtime_stats["warmup_inference_seconds"] = round(
            perf_counter() - started_at,
            3,
        )
        self._runtime_stats["warmup_completed_at"] = datetime.now(UTC).isoformat()

    def snapshot_runtime_state(self) -> dict[str, Any]:
        return {
            "runtime": {
                "requested_device": settings.model_device,
                "resolved_device": str(self._resolved_device) if self._resolved_device is not None else None,
                "tracking_enabled": settings.model_enable_tracking,
                "runtime_autoinstall_enabled": settings.model_runtime_autoinstall,
                "preload_on_startup": settings.model_preload_on_startup,
                "loaded_model_paths": sorted(self._models),
                "loaded_model_count": len(self._models),
                **self._runtime_stats,
            },
            "capacity": {
                "batching_enabled": True,
                "max_batch_size": settings.model_batch_size,
            },
        }

    def _infer_payloads(
        self,
        payloads: list[InferenceRequest],
        *,
        allow_tracking: bool,
    ) -> list[BackendInferenceOutput]:
        prepared_payloads = [self._prepare_payload(payload) for payload in payloads]
        if not prepared_payloads:
            return []

        started_at = perf_counter()
        grouped_invocations: dict[BatchInvocationKey, dict[str, Any]] = {}

        for payload_index, prepared in enumerate(prepared_payloads):
            for model_path, detectors in prepared.assignments.items():
                model = self._load_model(model_path)
                requested_classes = self._resolve_requested_classes(model, detectors)
                supported_detectors = self._resolve_supported_detectors(model, detectors)
                if not requested_classes:
                    continue

                prepared.unsupported = [
                    detector for detector in prepared.unsupported if detector not in supported_detectors
                ]
                use_tracking = (
                    allow_tracking
                    and len(prepared_payloads) == 1
                    # USB camera frames arrive as independent browser snapshots.
                    # Persistent tracking would reuse motion state from an old
                    # frame after network/inference delays.
                    and prepared.payload.source_type != "usb"
                    and set(detectors) == {"person"}
                    and supported_detectors == {"person"}
                )
                confidence = min(
                    self._confidence_threshold(detector) for detector in supported_detectors
                )
                key = BatchInvocationKey(
                    model_path=model_path,
                    detectors=tuple(detectors),
                    requested_classes=tuple(requested_classes),
                    confidence=confidence,
                    use_tracking=use_tracking,
                )
                invocation = grouped_invocations.setdefault(
                    key,
                    {
                        "model": model,
                        "indices": [],
                        "images": [],
                        "detectors": detectors,
                        "requested_classes": requested_classes,
                        "use_tracking": use_tracking,
                    },
                )
                invocation["indices"].append(payload_index)
                invocation["images"].append(prepared.image)
                prepared.active_models.append(
                    {
                        "weights_path": model_path,
                        "requested_detectors": detectors,
                        "requested_classes": requested_classes,
                        "mode": "track" if use_tracking else "predict",
                        "image_size": self._image_size_for_model(model_path),
                    }
                )

        for key, invocation in grouped_invocations.items():
            results = self._run_model_invocation(
                model=invocation["model"],
                model_path=key.model_path,
                images=invocation["images"],
                requested_classes=invocation["requested_classes"],
                confidence=key.confidence,
                use_tracking=invocation["use_tracking"],
            )
            results_list = self._to_result_list(results)
            for result_index, payload_index in enumerate(invocation["indices"]):
                if result_index >= len(results_list):
                    continue
                prepared_payloads[payload_index].detections.extend(
                    self._parse_results(
                        [results_list[result_index]],
                        invocation["detectors"],
                        model_path=key.model_path,
                    )
                )

        elapsed = max(perf_counter() - started_at, 1e-6)
        batch_size = len(prepared_payloads)
        self._record_runtime_stats(batch_size)
        effective_fps = round(batch_size / elapsed, 2)

        return [
            BackendInferenceOutput(
                detections=self._limit_frame_detections(
                    self._reject_cross_class_conflicts(
                        self._deduplicate_detections(
                            self._confirm_specialist_weapon_detections(prepared.detections)
                        )
                    )
                ),
                backend_name=self.backend_name,
                model_name=settings.model_name,
                model_version=settings.model_version,
                inference_fps=effective_fps,
                metadata={
                    "weights_path": settings.model_weights_path,
                    "active_models": prepared.active_models,
                    "detector_ms": round(elapsed * 1000, 2),
                    "tracker": settings.model_tracker_config if settings.model_enable_tracking else None,
                    "unsupported_requested_detectors": prepared.unsupported,
                    "batched": batch_size > 1,
                    "batch_size": batch_size,
                },
            )
            for prepared in prepared_payloads
        ]

    def _prepare_payload(self, payload: InferenceRequest) -> PreparedInferencePayload:
        if not payload.frame_content_base64:
            raise InferenceBackendInputError(
                "The ultralytics backend requires frame_content_base64 for model-backed inference."
            )

        assignments = self._assign_detectors_to_models(payload.requested_detectors)
        return PreparedInferencePayload(
            payload=payload,
            image=self._decode_image(payload.frame_content_base64),
            assignments=assignments,
            unsupported=normalize_requested_detectors(payload.requested_detectors),
        )

    def _record_runtime_stats(self, batch_size: int) -> None:
        if batch_size > 1:
            self._runtime_stats["batch_requests_total"] += 1
        else:
            self._runtime_stats["single_requests_total"] += 1
        self._runtime_stats["images_processed_total"] += batch_size
        self._runtime_stats["last_batch_size"] = batch_size
        self._runtime_stats["max_observed_batch_size"] = max(
            self._runtime_stats["max_observed_batch_size"],
            batch_size,
        )

    def _configured_model_paths(self) -> list[str]:
        fire_path = settings.model_fire_weights_path or settings.model_fire_smoke_weights_path
        smoke_path = settings.model_smoke_weights_path or settings.model_fire_smoke_weights_path
        return list(
            dict.fromkeys(
                path
                for path in (
                    settings.model_weights_path,
                    settings.model_person_weapon_weights_path,
                    settings.model_weapon_weights_path,
                    fire_path,
                    smoke_path,
                )
                if path
            )
        )

    def _prepare_runtime_environment(self) -> None:
        if "YOLO_CONFIG_DIR" not in os.environ:
            yolo_config_dir = PROJECT_ROOT / "storage" / "ultralytics"
            yolo_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
        if "MPLCONFIGDIR" not in os.environ:
            matplotlib_config_dir = PROJECT_ROOT / "storage" / "matplotlib"
            matplotlib_config_dir.mkdir(parents=True, exist_ok=True)
            os.environ["MPLCONFIGDIR"] = str(matplotlib_config_dir)
        os.environ["YOLO_AUTOINSTALL"] = "true" if settings.model_runtime_autoinstall else "false"

    def _resolve_device(self) -> str | None:
        if settings.model_device is not None:
            requested = settings.model_device.strip().lower()
            if requested.startswith("intel:"):
                requested_openvino_device = requested.split(":", 1)[1].upper()
                try:
                    from openvino import Core

                    available = {device.split(".", 1)[0].upper() for device in Core().available_devices}
                    if requested_openvino_device in available:
                        return f"intel:{requested_openvino_device.lower()}"
                    if "CPU" in available:
                        return "intel:cpu"
                except ImportError:
                    return "cpu"
            if requested == "cpu":
                return "cpu"
            if requested.isdigit():
                try:
                    import torch

                    return requested if torch.cuda.is_available() else "cpu"
                except ImportError:
                    return "cpu"
            return settings.model_device
        try:
            import torch
        except ImportError:
            return None
        return "0" if torch.cuda.is_available() else "cpu"

    def _load_model(self, weights_path: str):
        if weights_path in self._models:
            return self._models[weights_path]

        self._prepare_runtime_environment()
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise InferenceBackendUnavailableError(
                "Ultralytics is not installed. Install `aegispro-ai[model]` to enable YOLO11/ByteTrack inference."
            ) from exc

        self._models[weights_path] = YOLO(weights_path)
        return self._models[weights_path]

    def _run_model_invocation(
        self,
        *,
        model: Any,
        model_path: str,
        images: list[Image.Image],
        requested_classes: list[int],
        confidence: float,
        use_tracking: bool,
    ) -> list[Any]:
        inference_options: dict[str, Any] = {
            "conf": confidence,
            "classes": requested_classes,
            "device": self._resolved_device,
            "imgsz": self._image_size_for_model(model_path),
            "iou": settings.model_iou_threshold,
            "max_det": settings.model_max_detections,
            "verbose": False,
        }
        if settings.model_half_precision:
            inference_options["half"] = True
        if use_tracking:
            results = model.track(
                **inference_options,
                source=images[0],
                persist=settings.model_track_persist,
                tracker=settings.model_tracker_config,
            )
            return self._to_result_list(results)

        # OpenVINO exports are commonly static (for example, batch: 1).
        # Passing several images to such a model makes Ultralytics build an
        # incompatible NCHW tensor and the entire continuous scan fails.
        model_batch_size = self._batch_size_for_model(model_path)
        results: list[Any] = []
        for offset in range(0, len(images), model_batch_size):
            image_batch = images[offset : offset + model_batch_size]
            batch_options = {
                **inference_options,
                "source": image_batch[0] if len(image_batch) == 1 else image_batch,
            }
            if len(image_batch) > 1:
                batch_options["batch"] = len(image_batch)
            results.extend(self._to_result_list(model.predict(**batch_options)))
        return results

    def _batch_size_for_model(self, weights_path: str) -> int:
        cached = self._model_batch_sizes.get(weights_path)
        if cached is not None:
            return cached

        batch_size = settings.model_batch_size
        metadata_path = Path(weights_path) / "metadata.yaml"
        if metadata_path.is_file():
            try:
                import yaml

                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                configured_batch = metadata.get("batch")
                if isinstance(configured_batch, int) and configured_batch > 0:
                    batch_size = min(batch_size, configured_batch)
            except (ImportError, OSError, TypeError, ValueError):
                batch_size = settings.model_batch_size

        self._model_batch_sizes[weights_path] = max(1, batch_size)
        return self._model_batch_sizes[weights_path]

    def _image_size_for_model(self, weights_path: str) -> int | list[int]:
        cached = self._model_image_sizes.get(weights_path)
        if cached is not None:
            return cached

        image_size: int | list[int] = settings.model_image_size
        metadata_path = Path(weights_path) / "metadata.yaml"
        if metadata_path.is_file():
            try:
                import yaml

                metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
                configured_size = metadata.get("imgsz")
                if isinstance(configured_size, int) and configured_size > 0:
                    image_size = configured_size
                elif isinstance(configured_size, (list, tuple)) and len(configured_size) >= 2:
                    height, width = int(configured_size[0]), int(configured_size[1])
                    if height > 0 and width > 0:
                        image_size = height if height == width else [height, width]
            except (ImportError, OSError, TypeError, ValueError):
                image_size = settings.model_image_size

        self._model_image_sizes[weights_path] = image_size
        return image_size

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
            if specialist and not settings.model_weapon_ensemble_general:
                return [specialist]
            return list(dict.fromkeys(path for path in (specialist, general) if path))
        if detector == "fire":
            return [
                settings.model_fire_weights_path
                or settings.model_fire_smoke_weights_path
                or settings.model_weights_path
            ]
        if detector == "smoke":
            return [
                settings.model_smoke_weights_path
                or settings.model_fire_smoke_weights_path
                or settings.model_weights_path
            ]
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

    def _parse_results(
        self,
        results: Any,
        requested_detectors: list[str],
        *,
        model_path: str | None = None,
    ) -> list[InferenceBox]:
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
            track_ids = (
                self._to_flat_list(getattr(boxes, "id", []))
                if getattr(boxes, "id", None) is not None
                else []
            )

            for index, coords in enumerate(xyxy_rows):
                class_id = int(class_ids[index]) if index < len(class_ids) else -1
                raw_label = str(names.get(class_id, class_id))
                label = self._normalize_model_label(raw_label)
                if label not in requested:
                    continue

                raw_object_label = raw_label.strip().lower().replace("-", "_").replace(" ", "_")
                general_weapon_path = (
                    settings.model_person_weapon_weights_path or settings.model_weights_path
                )
                object_label = (
                    self._canonical_weapon_type(raw_object_label)
                    if label == "weapon"
                    else raw_object_label
                )
                excluded_weapon_labels = {
                    self._canonical_weapon_type(excluded_label)
                    for excluded_label in settings.model_weapon_excluded_labels
                }
                if (
                    label == "weapon"
                    and object_label in excluded_weapon_labels
                    and (model_path is None or model_path == general_weapon_path)
                ):
                    # Exclusions protect subtype enrichment from ambiguous classes in
                    # a general model (for example, COCO scissors). They must not
                    # suppress the same class from a purpose-trained weapon model.
                    continue
                if (
                    label == "weapon"
                    and self._frame_coverage(coords, getattr(result, "orig_shape", None))
                    > settings.model_weapon_max_frame_coverage
                ):
                    # Camera hard negatives showed the specialist occasionally
                    # classifying nearly the entire frame as one weapon. A real
                    # localized weapon box cannot plausibly occupy this area.
                    continue
                if label in {"weapon", "fire", "smoke"} and self._looks_like_edge_strip_false_positive(
                    coords,
                    getattr(result, "orig_shape", None),
                ):
                    continue
                threshold = self._confidence_threshold(label)
                confidence = float(confidences[index]) if index < len(confidences) else threshold
                if confidence < threshold:
                    continue
                if (
                    label == "weapon"
                    and object_label in {"weapon", "other_weapon"}
                    and confidence < settings.model_generic_weapon_min_confidence
                ):
                    # "Weapon" and "other_weapon" are ambiguous catch-all
                    # classes. Apply the stricter floor even to the specialist
                    # model: weak scene textures such as flames, roof lines,
                    # benches, and railings otherwise become operator-visible
                    # weapon boxes before temporal confirmation settles.
                    # Specific classes such as pistol or knife continue to use
                    # the calibrated weapon threshold.
                    continue

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
                        source_model_path=model_path,
                        track_id=track_id,
                    )
                )

        return detections

    @classmethod
    def _confirm_specialist_weapon_detections(
        cls,
        detections: list[InferenceBox],
    ) -> list[InferenceBox]:
        """Use the general model for names, while the weapon model remains the alert gate."""
        specialist_path = settings.model_weapon_weights_path
        general_path = settings.model_person_weapon_weights_path or settings.model_weights_path
        if (
            not specialist_path
            or not settings.model_weapon_ensemble_general
            or specialist_path == general_path
        ):
            return detections

        specialist_weapons = [
            detection
            for detection in detections
            if detection.label == "weapon" and detection.source_model_path == specialist_path
        ]
        if not specialist_weapons:
            return [
                detection
                for detection in detections
                if detection.label != "weapon"
                or detection.confidence >= settings.model_weapon_general_fallback_confidence
            ]

        return [
            detection
            for detection in detections
            if detection.label != "weapon"
            or detection.source_model_path == specialist_path
            or any(
                cls._intersection_over_union(detection, specialist) >= 0.10
                or cls._smaller_box_coverage(detection, specialist) >= 0.50
                for specialist in specialist_weapons
            )
        ]

    @staticmethod
    def _canonical_weapon_type(raw_label: str) -> str:
        return settings.model_weapon_type_aliases.get(raw_label, raw_label or "other_weapon")

    @staticmethod
    def _frame_coverage(coords: list[float], original_shape: Any) -> float:
        if not original_shape or len(original_shape) < 2:
            return 0.0
        try:
            frame_height = float(original_shape[0])
            frame_width = float(original_shape[1])
            box_width = max(0.0, float(coords[2]) - float(coords[0]))
            box_height = max(0.0, float(coords[3]) - float(coords[1]))
        except (TypeError, ValueError, IndexError):
            return 0.0
        frame_area = frame_width * frame_height
        return (box_width * box_height) / frame_area if frame_area > 0 else 0.0

    @staticmethod
    def _looks_like_edge_strip_false_positive(coords: list[float], original_shape: Any) -> bool:
        if not original_shape or len(original_shape) < 2:
            return False
        try:
            frame_height = float(original_shape[0])
            frame_width = float(original_shape[1])
            left = float(coords[0])
            top = float(coords[1])
            right = float(coords[2])
            bottom = float(coords[3])
        except (TypeError, ValueError, IndexError):
            return False

        box_width = max(0.0, right - left)
        box_height = max(0.0, bottom - top)
        if frame_width <= 0 or frame_height <= 0 or box_width <= 0:
            return False

        aspect_ratio = box_height / box_width
        touches_vertical_edge = (
            left <= frame_width * settings.threat_edge_strip_margin_ratio
            or right >= frame_width * (1 - settings.threat_edge_strip_margin_ratio)
        )
        return (
            touches_vertical_edge
            and aspect_ratio >= settings.threat_edge_strip_min_aspect_ratio
            and box_width / frame_width <= settings.threat_edge_strip_max_width_ratio
            and box_height / frame_height >= settings.threat_edge_strip_min_height_ratio
        )

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
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(kept)
                    if existing.label == candidate.label
                    and (
                        cls._intersection_over_union(existing, candidate) >= 0.6
                        or cls._smaller_box_coverage(existing, candidate) >= 0.80
                    )
                ),
                None,
            )
            if duplicate_index is not None:
                existing = kept[duplicate_index]
                if existing.label == "weapon":
                    specialist_path = settings.model_weapon_weights_path
                    existing_is_specialist = bool(
                        specialist_path and existing.source_model_path == specialist_path
                    )
                    candidate_is_specialist = bool(
                        specialist_path and candidate.source_model_path == specialist_path
                    )
                    existing_specific = existing.object_label not in {None, "weapon", "other_weapon"}
                    candidate_specific = candidate.object_label not in {None, "weapon", "other_weapon"}
                    if existing_is_specialist != candidate_is_specialist:
                        specialist = existing if existing_is_specialist else candidate
                        enrichment = candidate if existing_is_specialist else existing
                        enrichment_specific = enrichment.object_label not in {
                            None,
                            "weapon",
                            "other_weapon",
                        }
                        kept[duplicate_index] = specialist.model_copy(
                            update={
                                "object_label": (
                                    enrichment.object_label
                                    if enrichment_specific
                                    else specialist.object_label
                                )
                            }
                        )
                    elif candidate_specific and not existing_specific:
                        kept[duplicate_index] = existing.model_copy(
                            update={"object_label": candidate.object_label}
                        )
                continue
            kept.append(candidate)
        return kept

    @staticmethod
    def _smaller_box_coverage(first: InferenceBox, second: InferenceBox) -> float:
        left = max(first.x1, second.x1)
        top = max(first.y1, second.y1)
        right = min(first.x2, second.x2)
        bottom = min(first.y2, second.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first.x2 - first.x1) * max(0.0, first.y2 - first.y1)
        second_area = max(0.0, second.x2 - second.x1) * max(0.0, second.y2 - second.y1)
        smaller_area = min(first_area, second_area)
        return intersection / smaller_area if smaller_area > 0 else 0.0

    @staticmethod
    def _limit_frame_detections(detections: list[InferenceBox]) -> list[InferenceBox]:
        kept: list[InferenceBox] = []
        threat_counts: dict[str, int] = {}
        other_count = 0
        for detection in detections:
            if detection.label in {"weapon", "fire", "smoke"}:
                count = threat_counts.get(detection.label, 0)
                if count >= settings.model_max_threat_detections_per_type:
                    continue
                threat_counts[detection.label] = count + 1
            else:
                if other_count >= 10:
                    continue
                other_count += 1
            kept.append(detection)
        return kept

    @classmethod
    def _reject_cross_class_conflicts(cls, detections: list[InferenceBox]) -> list[InferenceBox]:
        persons = [detection for detection in detections if detection.label == "person"]
        return [
            detection
            for detection in detections
            if not (
                detection.label == "smoke"
                and detection.confidence < settings.smoke_person_conflict_min_confidence
                and any(
                    cls._intersection_coverage(detection, person)
                    >= settings.smoke_max_person_coverage
                    for person in persons
                )
            )
        ]

    @staticmethod
    def _intersection_coverage(candidate: InferenceBox, other: InferenceBox) -> float:
        left = max(candidate.x1, other.x1)
        top = max(candidate.y1, other.y1)
        right = min(candidate.x2, other.x2)
        bottom = min(candidate.y2, other.y2)
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        candidate_area = max(0.0, candidate.x2 - candidate.x1) * max(
            0.0, candidate.y2 - candidate.y1
        )
        return intersection / candidate_area if candidate_area else 0.0

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
    def _to_result_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return list(value)

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
