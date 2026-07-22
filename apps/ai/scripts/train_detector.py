from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.training.threshold_calibration import metrics_at_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, and conditionally promote an AegisPro YOLO detector."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument(
        "--holdout-data",
        type=Path,
        help="Optional independent dataset YAML used for the promotion comparison.",
    )
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--freeze-layers", type=int, default=0)
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="disk")
    parser.add_argument("--single-class", action="store_true")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--save-period", type=int, default=5)
    parser.add_argument(
        "--defer-intermediate-checkpoints",
        action="store_true",
        help="Write the final best/last checkpoint only; useful on slow CPU or synced filesystems.",
    )
    parser.add_argument("--project", type=Path, default=Path("storage/training-runs"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-weights", required=True, type=Path)
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        help="Checkpoint to beat. Defaults to output-weights when it already exists.",
    )
    parser.add_argument("--minimum-map50", type=float, default=0.0)
    parser.add_argument("--minimum-precision", type=float, default=0.0)
    parser.add_argument("--minimum-recall", type=float, default=0.0)
    parser.add_argument(
        "--operating-threshold",
        type=float,
        help="Runtime confidence threshold to include in the promotion gate.",
    )
    parser.add_argument(
        "--operating-class",
        help="Class name to gate when the checkpoint contains multiple classes.",
    )
    parser.add_argument("--minimum-operating-precision", type=float, default=0.0)
    parser.add_argument("--minimum-operating-recall", type=float, default=0.0)
    parser.add_argument(
        "--minimum-operating-precision-improvement",
        type=float,
        default=0.0,
        help="Required precision gain over the baseline at the same runtime threshold.",
    )
    parser.add_argument(
        "--minimum-map50-improvement",
        type=float,
        default=0.0,
        help="Required candidate mAP50 gain over the baseline on the same evaluation set.",
    )
    parser.add_argument(
        "--force-promote",
        action="store_true",
        help="Copy the candidate despite failed gates; intended only for isolated experiments.",
    )
    parser.add_argument(
        "--export-openvino",
        action="store_true",
        help="Export a promoted checkpoint to an FP16 OpenVINO directory.",
    )
    parser.add_argument(
        "--export-image-size",
        type=int,
        default=640,
        help="Fixed OpenVINO input size; keep this aligned with AI_MODEL_IMAGE_SIZE.",
    )
    return parser


def _metrics_payload(metrics: Any) -> dict[str, float]:
    return {
        "map50": round(float(metrics.box.map50), 6),
        "map50_95": round(float(metrics.box.map), 6),
        "precision": round(float(metrics.box.mp), 6),
        "recall": round(float(metrics.box.mr), 6),
        "fitness": round(float(metrics.fitness), 6),
    }


def _cache_value(value: str) -> bool | str:
    return False if value == "false" else value


def _confidence_curve(metrics: Any, label: str) -> tuple[list[float], list[list[float]]]:
    for x_values, y_values, x_label, y_label in metrics.curves_results:
        if x_label == "Confidence" and y_label == label:
            return x_values.tolist(), y_values.tolist()
    raise RuntimeError(f"Ultralytics did not return a {label}-confidence curve.")


def _operating_metrics(metrics: Any, class_name: str | None, threshold: float) -> dict[str, Any]:
    confidence, precision_by_class = _confidence_curve(metrics, "Precision")
    recall_confidence, recall_by_class = _confidence_curve(metrics, "Recall")
    f1_confidence, f1_by_class = _confidence_curve(metrics, "F1")
    if confidence != recall_confidence or confidence != f1_confidence:
        raise RuntimeError("Ultralytics returned confidence curves on different grids.")

    class_indexes = list(getattr(metrics.box, "ap_class_index", []))
    if len(class_indexes) != len(precision_by_class):
        class_indexes = list(range(len(precision_by_class)))
    names = getattr(metrics, "names", {})
    available = [str(names.get(int(index), index)) for index in class_indexes]
    if class_name is None:
        if len(available) != 1:
            raise ValueError(
                "--operating-class is required for a multi-class checkpoint; "
                f"available classes: {', '.join(available)}"
            )
        selected_index = 0
        class_name = available[0]
    else:
        try:
            selected_index = available.index(class_name)
        except ValueError as exc:
            raise ValueError(
                f"Operating class {class_name!r} is absent; available classes: {', '.join(available)}"
            ) from exc

    selected = metrics_at_threshold(
        confidence,
        precision_by_class[selected_index],
        recall_by_class[selected_index],
        f1_by_class[selected_index],
        threshold,
    )
    return {"class": class_name, **asdict(selected)}


def main() -> int:
    args = build_parser().parse_args()
    if args.operating_threshold is not None and not 0 <= args.operating_threshold <= 1:
        raise ValueError("--operating-threshold must be between 0 and 1.")
    from ultralytics import YOLO

    data_path = args.data.resolve()
    evaluation_path = (args.holdout_data or args.data).resolve()
    output_path = args.output_weights.resolve()
    baseline_path = (
        args.baseline_weights.resolve()
        if args.baseline_weights
        else (output_path if output_path.is_file() else None)
    )

    model = YOLO(args.weights)
    train_result = model.train(
        data=str(data_path),
        imgsz=args.image_size,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        lr0=args.learning_rate,
        warmup_epochs=args.warmup_epochs,
        freeze=args.freeze_layers or None,
        project=str(args.project.resolve()),
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        cache=_cache_value(args.cache),
        cos_lr=True,
        optimizer="AdamW",
        close_mosaic=min(10, max(1, args.epochs // 3)),
        seed=args.seed,
        deterministic=True,
        single_cls=args.single_class,
        save=not args.defer_intermediate_checkpoints,
        save_period=-1 if args.defer_intermediate_checkpoints else args.save_period,
        plots=False,
    )
    run_dir = Path(train_result.save_dir)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        best_weights = run_dir / "weights" / "last.pt"
    candidate_validation = YOLO(str(best_weights)).val(
        data=str(evaluation_path),
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        single_cls=args.single_class,
    )
    candidate_metrics = _metrics_payload(candidate_validation)
    candidate_operating_metrics = (
        _operating_metrics(
            candidate_validation,
            args.operating_class,
            args.operating_threshold,
        )
        if args.operating_threshold is not None
        else None
    )
    baseline_metrics: dict[str, float] | None = None
    baseline_operating_metrics: dict[str, Any] | None = None
    if baseline_path and baseline_path.is_file() and baseline_path != best_weights.resolve():
        baseline_validation = YOLO(str(baseline_path)).val(
            data=str(evaluation_path),
            imgsz=args.image_size,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            plots=False,
            verbose=False,
            single_cls=args.single_class,
        )
        baseline_metrics = _metrics_payload(baseline_validation)
        baseline_operating_metrics = (
            _operating_metrics(
                baseline_validation,
                args.operating_class,
                args.operating_threshold,
            )
            if args.operating_threshold is not None
            else None
        )

    gates = {
        "minimum_map50": candidate_metrics["map50"] >= args.minimum_map50,
        "minimum_precision": candidate_metrics["precision"] >= args.minimum_precision,
        "minimum_recall": candidate_metrics["recall"] >= args.minimum_recall,
        "baseline_map50_improvement": (
            baseline_metrics is None
            or candidate_metrics["map50"]
            >= baseline_metrics["map50"] + args.minimum_map50_improvement
        ),
        "minimum_operating_precision": (
            candidate_operating_metrics is None
            or candidate_operating_metrics["precision"] >= args.minimum_operating_precision
        ),
        "minimum_operating_recall": (
            candidate_operating_metrics is None
            or candidate_operating_metrics["recall"] >= args.minimum_operating_recall
        ),
        "baseline_operating_precision_improvement": (
            candidate_operating_metrics is None
            or baseline_operating_metrics is None
            or candidate_operating_metrics["precision"]
            >= baseline_operating_metrics["precision"]
            + args.minimum_operating_precision_improvement
        ),
    }
    promoted = all(gates.values()) or args.force_promote
    candidate_output = output_path.with_name(f"{output_path.stem}.candidate{output_path.suffix}")
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, candidate_output)

    exported_path: str | None = None
    if promoted:
        shutil.copy2(best_weights, output_path)
        if args.export_openvino:
            exported_path = str(
                YOLO(str(output_path)).export(
                    format="openvino",
                    imgsz=args.export_image_size,
                    half=True,
                    dynamic=False,
                )
            )

    summary = {
        "completed_at": datetime.now(UTC).isoformat(),
        "run_name": args.run_name,
        "training_data": str(data_path),
        "evaluation_data": str(evaluation_path),
        "candidate_weights": str(candidate_output),
        "baseline_weights": str(baseline_path) if baseline_path else None,
        "candidate_metrics": candidate_metrics,
        "candidate_operating_metrics": candidate_operating_metrics,
        "baseline_metrics": baseline_metrics,
        "baseline_operating_metrics": baseline_operating_metrics,
        "gates": gates,
        "forced": args.force_promote,
        "promoted": promoted,
        "output_weights": str(output_path) if promoted else None,
        "openvino_export": exported_path,
    }
    summary_path = run_dir / "aegispro-evaluation.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if promoted:
        print(f"Promoted checkpoint to {output_path}")
        return 0
    print(f"Candidate retained at {candidate_output}; production checkpoint was not replaced.")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
