from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and export an AegisPro YOLO detector.")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--project", type=Path, default=Path("storage/training-runs"))
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-weights", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model.train(
        data=str(args.data.resolve()),
        imgsz=args.image_size,
        batch=args.batch,
        epochs=args.epochs,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(args.project.resolve()),
        name=args.run_name,
        exist_ok=True,
        pretrained=True,
        cache=False,
        cos_lr=True,
        optimizer="AdamW",
        close_mosaic=min(10, max(1, args.epochs // 3)),
        seed=20260711,
        deterministic=True,
    )
    best_weights = Path(result.save_dir) / "weights" / "best.pt"
    args.output_weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, args.output_weights)
    print(f"Saved checkpoint to {args.output_weights.resolve()}")


if __name__ == "__main__":
    main()
