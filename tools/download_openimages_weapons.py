#!/usr/bin/env python3
"""Build a reproducible YOLO weapon dataset from Open Images bounding boxes.

The default output is a single ``weapon`` class because that is the class used by
the AegisPro runtime checkpoint.  Visually similar tools and handheld objects are
added as empty-label hard negatives to reduce false alarms.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


CLASS_MIDS = {
    "/m/0gxl3": "pistol",
    "/m/058qzx": "knife",
    "/m/04ctx": "knife",
    "/m/06c54": "rifle",
    "/m/01lsmm": "scissors",
    "/m/06nrc": "shotgun",
    "/m/083kb": "other_weapon",
}

# These are frequent CCTV false-positive shapes for small weapon detectors.
HARD_NEGATIVE_MIDS = {
    "/m/0dv5r": "camera",
    "/m/01d380": "drill",
    "/m/01kb5b": "flashlight",
    "/m/03wvsk": "hair_dryer",
    "/m/050k8": "mobile_phone",
    "/m/0k1tl": "pen",
    "/m/0qjjc": "remote_control",
    "/m/01bms0": "screwdriver",
    "/m/07k1x": "tool",
    "/m/012xff": "toothbrush",
}

CLASS_NAMES = list(dict.fromkeys(CLASS_MIDS.values()))
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("storage"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("storage/datasets/weapons_openimages_yolo")
    )
    parser.add_argument(
        "--max-train-per-class",
        type=int,
        default=0,
        help="Maximum positive images per source class; 0 keeps every available image.",
    )
    parser.add_argument(
        "--max-val-per-class",
        type=int,
        default=0,
        help="Maximum validation images per source class; 0 keeps every available image.",
    )
    parser.add_argument("--train-hard-negatives", type=int, default=500)
    parser.add_argument("--val-hard-negatives", type=int, default=150)
    parser.add_argument(
        "--preserve-classes",
        action="store_true",
        help=(
            "Keep canonical weapon subtypes (knife, scissors, pistol, rifle, shotgun, "
            "and other_weapon) instead of collapsing every box to weapon."
        ),
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--retries", type=int, default=4)
    return parser.parse_args()


def read_annotations(
    path: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, set[str]]]:
    positives: dict[str, list[dict[str, str]]] = defaultdict(list)
    negative_categories: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label_mid = row["LabelName"]
            image_id = row["ImageID"]
            if label_mid in CLASS_MIDS:
                positives[image_id].append(row)
            elif label_mid in HARD_NEGATIVE_MIDS:
                negative_categories[image_id].add(HARD_NEGATIVE_MIDS[label_mid])

    # An image is never a hard negative if it has any weapon annotation.
    for image_id in positives:
        negative_categories.pop(image_id, None)
    return positives, negative_categories


def choose_positive_images(
    by_image: dict[str, list[dict[str, str]]], limit_per_class: int, seed: int
) -> set[str]:
    rng = random.Random(seed)
    class_to_images: dict[str, list[str]] = defaultdict(list)
    for image_id, rows in by_image.items():
        for class_name in {CLASS_MIDS[row["LabelName"]] for row in rows}:
            class_to_images[class_name].append(image_id)

    chosen: set[str] = set()
    for class_name in CLASS_NAMES:
        image_ids = sorted(class_to_images[class_name])
        rng.shuffle(image_ids)
        limit = limit_per_class if limit_per_class > 0 else len(image_ids)
        chosen.update(image_ids[:limit])
    return chosen


def choose_hard_negatives(
    candidates: dict[str, set[str]], limit: int, seed: int
) -> list[str]:
    if limit <= 0:
        return []
    rng = random.Random(seed)
    # Greedily balance categories rather than letting common phones dominate.
    category_to_images: dict[str, list[str]] = defaultdict(list)
    for image_id, categories in candidates.items():
        for category in categories:
            category_to_images[category].append(image_id)
    for image_ids in category_to_images.values():
        rng.shuffle(image_ids)

    chosen: list[str] = []
    selected: set[str] = set()
    categories = sorted(category_to_images)
    while len(chosen) < limit:
        made_progress = False
        rng.shuffle(categories)
        for category in categories:
            while category_to_images[category]:
                image_id = category_to_images[category].pop()
                if image_id in selected:
                    continue
                selected.add(image_id)
                chosen.append(image_id)
                made_progress = True
                break
            if len(chosen) >= limit:
                break
        if not made_progress:
            break
    return chosen


def _box_iou(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def yolo_lines(rows: list[dict[str, str]], preserve_classes: bool) -> list[str]:
    lines: list[str] = []
    kept: list[tuple[int, tuple[float, float, float, float]]] = []
    for row in rows:
        xmin = max(0.0, min(1.0, float(row["XMin"])))
        xmax = max(0.0, min(1.0, float(row["XMax"])))
        ymin = max(0.0, min(1.0, float(row["YMin"])))
        ymax = max(0.0, min(1.0, float(row["YMax"])))
        if xmax <= xmin or ymax <= ymin:
            continue
        class_id = CLASS_TO_ID[CLASS_MIDS[row["LabelName"]]] if preserve_classes else 0
        box = (xmin, ymin, xmax, ymax)
        if any(class_id == kept_id and _box_iou(box, kept_box) >= 0.85 for kept_id, kept_box in kept):
            continue
        kept.append((class_id, box))
        x_center = (xmin + xmax) / 2
        y_center = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        )
    return lines


def download_image(split: str, image_id: str, dest: Path, retries: int) -> bool:
    url = f"https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
    partial = dest.with_suffix(f"{dest.suffix}.part")
    for attempt in range(1, retries + 1):
        try:
            urlretrieve(url, partial)
            partial.replace(dest)
            return True
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            partial.unlink(missing_ok=True)
            if attempt == retries:
                print(f"warning: failed {split}/{image_id}: {exc}", file=sys.stderr)
                return False
            time.sleep(min(8.0, 1.5 * attempt))
    return False


def prepare_dirs(out_dir: Path) -> None:
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_dataset_split(
    *,
    source_split: str,
    yolo_split: str,
    positives: dict[str, list[dict[str, str]]],
    positive_ids: set[str],
    negative_ids: list[str],
    out_dir: Path,
    retries: int,
    workers: int,
    preserve_classes: bool,
) -> dict[str, int]:
    selected = sorted(positive_ids | set(negative_ids))

    def ensure_image(image_id: str) -> tuple[str, bool]:
        destination = out_dir / "images" / yolo_split / f"{image_id}.jpg"
        return image_id, destination.is_file() or download_image(
            source_split, image_id, destination, retries
        )

    successful: set[str] = set()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(ensure_image, image_id) for image_id in selected]
        for index, future in enumerate(as_completed(futures), start=1):
            image_id, ok = future.result()
            if ok:
                successful.add(image_id)
            if index % 100 == 0 or index == len(futures):
                print(f"{yolo_split}: downloaded/verified {index}/{len(futures)}")

    box_count = 0
    positive_count = 0
    negative_count = 0
    for image_id in sorted(successful):
        lines = yolo_lines(positives.get(image_id, []), preserve_classes)
        label_dest = out_dir / "labels" / yolo_split / f"{image_id}.txt"
        label_dest.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        if lines:
            positive_count += 1
            box_count += len(lines)
        else:
            negative_count += 1
    return {
        "images": len(successful),
        "positive_images": positive_count,
        "hard_negative_images": negative_count,
        "boxes": box_count,
        "failed_images": len(selected) - len(successful),
    }


def main() -> int:
    args = parse_args()
    metadata_dir = args.metadata_dir.resolve()
    out_dir = args.out_dir.resolve()
    validation_csv = metadata_dir / "validation-annotations-bbox.csv"
    test_csv = metadata_dir / "test-annotations-bbox.csv"
    missing = [path for path in (validation_csv, test_csv) if not path.is_file()]
    if missing:
        print("missing metadata CSVs: " + ", ".join(str(path) for path in missing), file=sys.stderr)
        return 2

    prepare_dirs(out_dir)
    train_annotations, train_negative_candidates = read_annotations(test_csv)
    val_annotations, val_negative_candidates = read_annotations(validation_csv)
    train_ids = choose_positive_images(
        train_annotations, args.max_train_per_class, args.seed
    )
    val_ids = choose_positive_images(val_annotations, args.max_val_per_class, args.seed + 1)
    train_negatives = choose_hard_negatives(
        train_negative_candidates, args.train_hard_negatives, args.seed + 2
    )
    val_negatives = choose_hard_negatives(
        val_negative_candidates, args.val_hard_negatives, args.seed + 3
    )

    train_stats = write_dataset_split(
        source_split="test",
        yolo_split="train",
        positives=train_annotations,
        positive_ids=train_ids,
        negative_ids=train_negatives,
        out_dir=out_dir,
        retries=args.retries,
        workers=args.workers,
        preserve_classes=args.preserve_classes,
    )
    val_stats = write_dataset_split(
        source_split="validation",
        yolo_split="val",
        positives=val_annotations,
        positive_ids=val_ids,
        negative_ids=val_negatives,
        out_dir=out_dir,
        retries=args.retries,
        workers=args.workers,
        preserve_classes=args.preserve_classes,
    )

    names = CLASS_NAMES if args.preserve_classes else ["weapon"]
    yaml_lines = [
        f"path: {out_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    stats = {
        "source": "Open Images V6 bounding boxes",
        "license": "CC BY 4.0 for annotations; verify individual image licenses before deployment",
        "seed": args.seed,
        "single_class": not args.preserve_classes,
        "negative_categories": sorted(HARD_NEGATIVE_MIDS.values()),
        "train": train_stats,
        "val": val_stats,
    }
    (out_dir / "dataset-stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "SOURCE.md").write_text(
        "# Dataset source\n\n"
        "Images and bounding-box annotations are selected from Open Images V6. "
        "Annotations are distributed under CC BY 4.0; image-level license metadata "
        "must be retained and reviewed for the intended deployment. The split and "
        "selection seed are recorded in `dataset-stats.json`.\n",
        encoding="utf-8",
    )
    shutil.copy2(validation_csv, out_dir / "source-validation-annotations-bbox.csv")
    shutil.copy2(test_csv, out_dir / "source-test-annotations-bbox.csv")
    print(json.dumps(stats, indent=2))
    print(f"dataset: {out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
