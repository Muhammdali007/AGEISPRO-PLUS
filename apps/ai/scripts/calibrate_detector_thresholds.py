from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.training.threshold_calibration import metrics_at_threshold, select_threshold


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure class-specific precision/recall at runtime confidence thresholds "
            "and recommend thresholds that satisfy an alert-quality gate."
        )
    )
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--thresholds",
        default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60,0.70",
        help="Comma-separated deployed thresholds to report.",
    )
    parser.add_argument("--minimum-precision", type=float, default=0.70)
    parser.add_argument("--minimum-recall", type=float, default=0.10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--project", type=Path, default=Path("storage/evaluation-runs"))
    parser.add_argument("--run-name", default="threshold-calibration")
    return parser


def _curve(metrics: Any, label: str) -> tuple[list[float], list[list[float]]]:
    for x_values, y_values, x_label, y_label in metrics.curves_results:
        if x_label == "Confidence" and y_label == label:
            return x_values.tolist(), y_values.tolist()
    raise RuntimeError(f"Ultralytics did not return a {label}-confidence curve.")


def _class_names(metrics: Any, curve_count: int) -> list[str]:
    class_indexes = list(getattr(metrics.box, "ap_class_index", []))
    if len(class_indexes) != curve_count:
        class_indexes = list(range(curve_count))
    names = getattr(metrics, "names", {})
    return [str(names.get(int(index), index)) for index in class_indexes]


def main() -> int:
    args = build_parser().parse_args()
    if not (0 <= args.minimum_precision <= 1 and 0 <= args.minimum_recall <= 1):
        raise ValueError("Minimum precision and recall must be between 0 and 1.")
    requested_thresholds = [float(value.strip()) for value in args.thresholds.split(",") if value.strip()]
    if not requested_thresholds or any(value < 0 or value > 1 for value in requested_thresholds):
        raise ValueError("Thresholds must contain values between 0 and 1.")

    from ultralytics import YOLO

    metrics = YOLO(args.weights).val(
        data=str(args.data.resolve()),
        split=args.split,
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        plots=False,
        verbose=False,
        project=str(args.project.resolve()),
        name=args.run_name,
        exist_ok=True,
    )
    confidence, precision_by_class = _curve(metrics, "Precision")
    recall_confidence, recall_by_class = _curve(metrics, "Recall")
    f1_confidence, f1_by_class = _curve(metrics, "F1")
    if confidence != recall_confidence or confidence != f1_confidence:
        raise RuntimeError("Ultralytics returned confidence curves on different grids.")

    class_names = _class_names(metrics, len(precision_by_class))
    classes: dict[str, Any] = {}
    all_constraints_met = True
    for index, class_name in enumerate(class_names):
        precision = precision_by_class[index]
        recall = recall_by_class[index]
        f1 = f1_by_class[index]
        selected, constraints_met = select_threshold(
            confidence,
            precision,
            recall,
            f1,
            minimum_precision=args.minimum_precision,
            minimum_recall=args.minimum_recall,
        )
        all_constraints_met = all_constraints_met and constraints_met
        classes[class_name] = {
            "constraints_met": constraints_met,
            "selected": asdict(selected),
            "requested_thresholds": [
                asdict(metrics_at_threshold(confidence, precision, recall, f1, threshold))
                for threshold in requested_thresholds
            ],
        }

    payload = {
        "completed_at": datetime.now(UTC).isoformat(),
        "weights": str(Path(args.weights).resolve()),
        "data": str(args.data.resolve()),
        "split": args.split,
        "image_size": args.image_size,
        "quality_gate": {
            "minimum_precision": args.minimum_precision,
            "minimum_recall": args.minimum_recall,
            "passed": all_constraints_met,
        },
        "validation_summary": {
            "map50": round(float(metrics.box.map50), 6),
            "map50_95": round(float(metrics.box.map), 6),
            "precision_at_best_f1": round(float(metrics.box.mp), 6),
            "recall_at_best_f1": round(float(metrics.box.mr), 6),
        },
        "classes": classes,
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if all_constraints_met else 3


if __name__ == "__main__":
    raise SystemExit(main())
