from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_GATES: dict[str, tuple[str, ...]] = {
    "weapon": ("weapon_recall_at_selected_threshold",),
    "fire_smoke": (
        "fire_recall_at_selected_threshold",
        "smoke_recall_at_selected_threshold",
    ),
    "fire": ("fire_recall_at_selected_threshold",),
    "smoke": ("smoke_recall_at_selected_threshold",),
}


def resolve_promotion_path(model_path: str | None, promotion_path: str | None = None) -> Path | None:
    if promotion_path:
        return Path(promotion_path)
    if not model_path:
        return None
    model_file = Path(model_path)
    suffix = f"{model_file.suffix}.promotion.json" if model_file.suffix else ".promotion.json"
    return model_file.with_suffix(suffix)


def validate_promotion_manifest(
    *,
    detector: str,
    model_path: str,
    promotion_path: str | None = None,
    min_signatures: int = 1,
    min_weapon_recall: float = 0.90,
) -> dict[str, Any]:
    model_file = Path(model_path)
    manifest_file = resolve_promotion_path(model_path, promotion_path)
    if manifest_file is None:
        raise ValueError(f"{detector} promotion manifest path is not configured.")
    if not manifest_file.is_file():
        raise ValueError(f"{detector} promotion manifest is missing: {manifest_file}")

    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{detector} promotion manifest is not valid JSON: {manifest_file}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"{detector} promotion manifest must be a JSON object: {manifest_file}")

    model_id = _require_non_empty_string(payload, "model_id", detector)
    checkpoint_sha256 = _require_non_empty_string(payload, "checkpoint_sha256", detector)
    actual_sha256 = _sha256_model(model_file)
    if checkpoint_sha256.lower() != actual_sha256:
        raise ValueError(
            f"{detector} promotion manifest hash does not match {model_file.name}: "
            f"expected {actual_sha256}, found {checkpoint_sha256}"
        )

    selected_threshold = _require_probability(payload, "selected_operating_threshold", detector)
    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{detector} promotion manifest must list the licensed datasets used.")
    for index, dataset in enumerate(datasets):
        if not isinstance(dataset, dict):
            raise ValueError(f"{detector} dataset entry #{index + 1} must be an object.")
        _require_non_empty_string(dataset, "name", detector)
        _require_non_empty_string(dataset, "license", detector)

    holdout = payload.get("independent_holdout")
    if not isinstance(holdout, dict):
        raise ValueError(f"{detector} promotion manifest must define an independent_holdout object.")
    _require_non_empty_string(holdout, "name", detector)
    _require_non_empty_string(holdout, "license", detector)
    if holdout.get("independent_from_training") is not True:
        raise ValueError(
            f"{detector} promotion manifest must state independent_holdout.independent_from_training=true."
        )

    per_class_metrics = payload.get("per_class_metrics")
    if not isinstance(per_class_metrics, dict) or not per_class_metrics:
        raise ValueError(f"{detector} promotion manifest must include per_class_metrics.")
    for class_name, metrics in per_class_metrics.items():
        if not isinstance(metrics, dict):
            raise ValueError(f"{detector} per_class_metrics.{class_name} must be an object.")
        _require_probability(metrics, "precision", detector, context=f"per_class_metrics.{class_name}")
        _require_probability(metrics, "recall", detector, context=f"per_class_metrics.{class_name}")

    gates = payload.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError(f"{detector} promotion manifest must include non-empty gates.")

    gates_by_name: dict[str, dict[str, Any]] = {}
    for index, gate in enumerate(gates):
        if not isinstance(gate, dict):
            raise ValueError(f"{detector} gate entry #{index + 1} must be an object.")
        gate_name = _require_non_empty_string(gate, "name", detector)
        actual = _require_probability(gate, "actual", detector, context=f"gates.{gate_name}")
        minimum = _require_probability(gate, "minimum", detector, context=f"gates.{gate_name}")
        if gate.get("passed") is not True:
            raise ValueError(f"{detector} gate {gate_name} is not approved for promotion.")
        if actual < minimum:
            raise ValueError(
                f"{detector} gate {gate_name} reports actual {actual:.3f} below minimum {minimum:.3f}."
            )
        gates_by_name[gate_name] = gate

    for gate_name in REQUIRED_GATES.get(detector, ()):
        if gate_name not in gates_by_name:
            raise ValueError(f"{detector} promotion manifest is missing the {gate_name} gate.")

    if detector == "weapon":
        weapon_gate = gates_by_name["weapon_recall_at_selected_threshold"]
        weapon_minimum = float(weapon_gate["minimum"])
        if weapon_minimum < min_weapon_recall:
            raise ValueError(
                "weapon promotion manifest must require weapon_recall_at_selected_threshold "
                f">= {min_weapon_recall:.2f}; found {weapon_minimum:.2f}."
            )

    signatures = payload.get("signatures")
    if not isinstance(signatures, list) or len(signatures) < min_signatures:
        raise ValueError(
            f"{detector} promotion manifest requires at least {min_signatures} signature(s)."
        )
    for index, signature in enumerate(signatures):
        if not isinstance(signature, dict):
            raise ValueError(f"{detector} signature entry #{index + 1} must be an object.")
        _require_non_empty_string(signature, "name", detector)
        _require_non_empty_string(signature, "role", detector)
        _require_non_empty_string(signature, "signed_at", detector)

    return {
        "valid": True,
        "model_id": model_id,
        "promotion_path": str(manifest_file),
        "selected_operating_threshold": selected_threshold,
        "datasets": [
            {"name": dataset["name"], "license": dataset["license"]}
            for dataset in datasets
        ],
        "independent_holdout": {
            "name": holdout["name"],
            "license": holdout["license"],
            "independent_from_training": True,
        },
        "gates": [
            {
                "name": gate["name"],
                "minimum": float(gate["minimum"]),
                "actual": float(gate["actual"]),
                "passed": True,
            }
            for gate in gates
        ],
        "signatures": [
            {
                "name": signature["name"],
                "role": signature["role"],
                "signed_at": signature["signed_at"],
            }
            for signature in signatures
        ],
        "checkpoint_sha256": actual_sha256,
    }


def collect_promotion_status(
    *,
    detector: str,
    model_path: str | None,
    promotion_path: str | None = None,
    required: bool = False,
    min_signatures: int = 1,
    min_weapon_recall: float = 0.90,
) -> dict[str, Any]:
    manifest_file = resolve_promotion_path(model_path, promotion_path)
    status: dict[str, Any] = {
        "required": required,
        "configured": manifest_file is not None,
        "path": str(manifest_file) if manifest_file else None,
        "exists": bool(manifest_file and manifest_file.is_file()),
        "valid": False,
    }
    if not model_path:
        status["detail"] = "No model checkpoint configured."
        return status
    if manifest_file is None:
        status["detail"] = "No promotion manifest path configured."
        return status
    if not manifest_file.is_file():
        status["detail"] = "Promotion manifest missing."
        return status

    try:
        manifest = validate_promotion_manifest(
            detector=detector,
            model_path=model_path,
            promotion_path=str(manifest_file),
            min_signatures=min_signatures,
            min_weapon_recall=min_weapon_recall,
        )
    except ValueError as exc:
        status["detail"] = str(exc)
        return status

    status.update(manifest)
    return status


def _require_non_empty_string(payload: dict[str, Any], field_name: str, detector: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{detector} promotion manifest requires a non-empty {field_name} field.")
    return value.strip()


def _require_probability(
    payload: dict[str, Any],
    field_name: str,
    detector: str,
    *,
    context: str | None = None,
) -> float:
    value = payload.get(field_name)
    scope = f"{context}.{field_name}" if context else field_name
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{detector} promotion manifest requires numeric {scope}.") from exc
    if parsed < 0 or parsed > 1:
        raise ValueError(f"{detector} promotion manifest requires {scope} between 0 and 1.")
    return parsed


def _sha256_model(path: Path) -> str:
    """Hash a checkpoint file or an exported-model directory deterministically."""
    if path.is_dir():
        digest = hashlib.sha256()
        files = sorted(item for item in path.rglob("*") if item.is_file())
        if not files:
            raise ValueError(f"Model directory is empty: {path}")
        for item in files:
            relative = item.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(item.stat().st_size.to_bytes(8, "big"))
            _update_digest(digest, item)
        return digest.hexdigest()
    if not path.is_file():
        raise ValueError(f"Model checkpoint is missing: {path}")
    digest = hashlib.sha256()
    _update_digest(digest, path)
    return digest.hexdigest()


def _update_digest(digest: Any, path: Path) -> None:
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
