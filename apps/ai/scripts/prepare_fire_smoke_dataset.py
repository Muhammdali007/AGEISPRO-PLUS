from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create deterministic, leakage-free YOLO train/validation/test splits."
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260711)
    return parser


def prepare_dataset(
    source_root: Path,
    output_root: Path,
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Path:
    if validation_fraction <= 0 or test_fraction <= 0:
        raise ValueError("Validation and test fractions must be greater than zero.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave room for training data.")

    source_images = source_root / "images"
    source_labels = source_root / "labels"
    images = sorted(
        path for path in source_images.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No images found in {source_images}")

    random.Random(seed).shuffle(images)
    test_count = max(1, round(len(images) * test_fraction))
    validation_count = max(1, round(len(images) * validation_fraction))
    split_images = {
        "test": images[:test_count],
        "val": images[test_count : test_count + validation_count],
        "train": images[test_count + validation_count :],
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for split, items in split_images.items():
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image in items:
            shutil.copy2(image, image_dir / image.name)
            source_label = source_labels / f"{image.stem}.txt"
            destination_label = label_dir / f"{image.stem}.txt"
            if source_label.exists():
                shutil.copy2(source_label, destination_label)
            else:
                destination_label.write_text("", encoding="utf-8")

    dataset_yaml = output_root / "data.yaml"
    yaml_path = output_root.resolve().as_posix()
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {yaml_path}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: fire",
                "  1: smoke",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml


def main() -> None:
    args = build_parser().parse_args()
    dataset_yaml = prepare_dataset(
        args.source_root.resolve(),
        args.output_root.resolve(),
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    print(f"Prepared dataset: {dataset_yaml}")


if __name__ == "__main__":
    main()
