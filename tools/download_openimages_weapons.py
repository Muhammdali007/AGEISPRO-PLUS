#!/usr/bin/env python3
"""Build a compact YOLO weapon-detection dataset from Open Images boxes."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


CLASS_MIDS = {
    "/m/0gxl3": "handgun",
    "/m/058qzx": "kitchen_knife",
    "/m/04ctx": "knife",
    "/m/06c54": "rifle",
    "/m/01lsmm": "scissors",
    "/m/06nrc": "shotgun",
    "/m/083kb": "weapon",
}

CLASS_NAMES = list(dict.fromkeys(CLASS_MIDS.values()))
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default="storage")
    parser.add_argument("--out-dir", default="storage/datasets/weapons_openimages_yolo")
    parser.add_argument("--max-train-per-class", type=int, default=60)
    parser.add_argument("--max-val-per-class", type=int, default=25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def read_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    by_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["LabelName"] in CLASS_MIDS:
                by_image[row["ImageID"]].append(row)
    return by_image


def choose_images(
    by_image: dict[str, list[dict[str, str]]], limit_per_class: int, seed: int
) -> set[str]:
    rng = random.Random(seed)
    class_to_images: dict[str, list[str]] = defaultdict(list)
    for image_id, rows in by_image.items():
        seen = {CLASS_MIDS[row["LabelName"]] for row in rows}
        for class_name in seen:
            class_to_images[class_name].append(image_id)

    chosen: set[str] = set()
    for class_name in CLASS_NAMES:
        image_ids = class_to_images[class_name]
        rng.shuffle(image_ids)
        chosen.update(image_ids[:limit_per_class])
    return chosen


def yolo_lines(rows: list[dict[str, str]]) -> list[str]:
    lines = []
    for row in rows:
        class_name = CLASS_MIDS[row["LabelName"]]
        xmin = float(row["XMin"])
        xmax = float(row["XMax"])
        ymin = float(row["YMin"])
        ymax = float(row["YMax"])
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0 or height <= 0:
            continue
        lines.append(
            f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} "
            f"{width:.6f} {height:.6f}"
        )
    return lines


def download_image(split: str, image_id: str, dest: Path, retries: int) -> bool:
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    for attempt in range(1, retries + 1):
        try:
            urlretrieve(url, dest)
            return True
        except (HTTPError, URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"warning: failed {split}/{image_id}: {exc}", file=sys.stderr)
                return False
            time.sleep(1.5 * attempt)
    return False


def prepare_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_split(
    source_split: str,
    yolo_split: str,
    by_image: dict[str, list[dict[str, str]]],
    selected: set[str],
    out_dir: Path,
    retries: int,
) -> tuple[int, int]:
    image_count = 0
    box_count = 0
    for image_id in sorted(selected):
        image_dest = out_dir / "images" / yolo_split / f"{image_id}.jpg"
        label_dest = out_dir / "labels" / yolo_split / f"{image_id}.txt"
        lines = yolo_lines(by_image[image_id])
        if not lines:
            continue
        if not image_dest.exists() and not download_image(source_split, image_id, image_dest, retries):
            continue
        label_dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        image_count += 1
        box_count += len(lines)
    return image_count, box_count


def main() -> int:
    args = parse_args()
    metadata_dir = Path(args.metadata_dir)
    out_dir = Path(args.out_dir)
    validation_csv = metadata_dir / "validation-annotations-bbox.csv"
    test_csv = metadata_dir / "test-annotations-bbox.csv"
    missing = [p for p in (validation_csv, test_csv) if not p.exists()]
    if missing:
        print("missing metadata CSVs: " + ", ".join(str(p) for p in missing), file=sys.stderr)
        return 2

    prepare_dirs(out_dir)
    train_annotations = read_annotations(test_csv)
    val_annotations = read_annotations(validation_csv)
    train_ids = choose_images(train_annotations, args.max_train_per_class, args.seed)
    val_ids = choose_images(val_annotations, args.max_val_per_class, args.seed + 1)

    train_images, train_boxes = write_dataset_split(
        "test", "train", train_annotations, train_ids, out_dir, args.retries
    )
    val_images, val_boxes = write_dataset_split(
        "validation", "val", val_annotations, val_ids, out_dir, args.retries
    )

    yaml = [
        f"path: {out_dir.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml.extend(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (out_dir / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")

    shutil.copy2(validation_csv, out_dir / "source-validation-annotations-bbox.csv")
    shutil.copy2(test_csv, out_dir / "source-test-annotations-bbox.csv")
    print(f"train: {train_images} images, {train_boxes} boxes")
    print(f"val: {val_images} images, {val_boxes} boxes")
    print(f"dataset: {out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
