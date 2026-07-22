from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_PRESETS = {
    "s": {"weights": "yolo11s.pt", "imgsz": 960, "batch": 8},
    "m": {"weights": "yolo11m.pt", "imgsz": 960, "batch": 8},
    "l": {"weights": "yolo11l.pt", "imgsz": 1280, "batch": 4},
    "x": {"weights": "yolo11x.pt", "imgsz": 1280, "batch": 2},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a staged weapon detector for AegisPro with Ultralytics YOLO."
    )
    parser.add_argument("--data", required=True, type=Path, help="Path to dataset.yaml.")
    parser.add_argument(
        "--model-size",
        choices=sorted(MODEL_PRESETS),
        default="m",
        help="Model size preset. 'm' is the recommended default for staged CCTV weapon evaluation.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Optional explicit base checkpoint. Overrides the preset weights name.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=Path("storage/training-runs"))
    parser.add_argument("--run-name", type=str, default="weapon-detector")
    parser.add_argument("--output-weights", type=Path, default=Path("storage/models/weapon.pt"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ultralytics import YOLO

    preset = MODEL_PRESETS[args.model_size]
    weights = args.weights or preset["weights"]
    model = YOLO(weights)
    result = model.train(
        data=str(args.data.resolve()),
        imgsz=int(preset["imgsz"]),
        batch=int(preset["batch"]),
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        cache=True,
        cos_lr=True,
        optimizer="AdamW",
        close_mosaic=10,
    )

    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.output_weights)
    print(f"Saved checkpoint to {args.output_weights.resolve()}")


if __name__ == "__main__":
    main()
