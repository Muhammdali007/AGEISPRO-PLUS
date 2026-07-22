from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a YOLO dataset before starting an expensive training run."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--verify-images",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--check-duplicates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reject duplicate image content and train/validation/test leakage.",
    )
    return parser


def _resolve_root(data_path: Path, payload: dict[str, Any]) -> Path:
    configured = Path(str(payload.get("path", ".")))
    if configured.is_absolute() and configured.exists():
        return configured.resolve()
    return (data_path.parent / configured).resolve()


def _split_paths(root: Path, raw: Any) -> list[Path]:
    values = raw if isinstance(raw, list) else [raw]
    resolved: list[Path] = []
    for value in values:
        if not value:
            continue
        candidate = Path(str(value))
        resolved.append(candidate if candidate.is_absolute() else root / candidate)
    return resolved


def _images_in(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".txt":
        return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not path.is_dir():
        return []
    return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = build_parser().parse_args()
    data_path = args.data.resolve()
    payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Dataset YAML must contain an object.")
    root = _resolve_root(data_path, payload)
    raw_names = payload.get("names", {})
    class_count = len(raw_names) if isinstance(raw_names, (dict, list)) else int(payload.get("nc", 0))

    errors: list[str] = []
    split_stats: dict[str, Any] = {}
    content_locations: dict[str, list[dict[str, str]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        if split not in payload:
            continue
        image_paths: list[Path] = []
        for split_path in _split_paths(root, payload[split]):
            image_paths.extend(_images_in(split_path))
        boxes_by_class: Counter[int] = Counter()
        empty_labels = 0
        missing_labels = 0
        invalid_labels = 0
        corrupt_images = 0
        duplicate_images = 0
        for image_path in image_paths:
            if args.verify_images:
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError):
                    corrupt_images += 1
                    errors.append(f"corrupt image: {image_path}")
            label_path = _label_path(image_path)
            if not label_path.is_file():
                missing_labels += 1
                errors.append(f"missing label: {label_path}")
                continue
            if args.check_duplicates:
                content_hash = _sha256(image_path)
                if content_locations[content_hash]:
                    duplicate_images += 1
                content_locations[content_hash].append(
                    {
                        "split": split,
                        "image": str(image_path),
                        "label_sha256": _sha256(label_path),
                    }
                )
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not lines:
                empty_labels += 1
            for line_number, line in enumerate(lines, start=1):
                parts = line.split()
                try:
                    class_id = int(parts[0])
                    coordinates = [float(value) for value in parts[1:5]]
                    valid = (
                        len(parts) == 5
                        and 0 <= class_id < class_count
                        and all(0.0 <= value <= 1.0 for value in coordinates)
                        and coordinates[2] > 0
                        and coordinates[3] > 0
                    )
                except (IndexError, TypeError, ValueError):
                    valid = False
                    class_id = -1
                if not valid:
                    invalid_labels += 1
                    errors.append(f"invalid label: {label_path}:{line_number}")
                    continue
                boxes_by_class[class_id] += 1
        split_stats[split] = {
            "images": len(image_paths),
            "boxes": sum(boxes_by_class.values()),
            "boxes_by_class": dict(sorted(boxes_by_class.items())),
            "empty_labels": empty_labels,
            "missing_labels": missing_labels,
            "invalid_labels": invalid_labels,
            "corrupt_images": corrupt_images,
            "duplicate_images": duplicate_images,
        }

    duplicate_groups = [locations for locations in content_locations.values() if len(locations) > 1]
    cross_split_leaks = [
        locations
        for locations in duplicate_groups
        if len({location["split"] for location in locations}) > 1
    ]
    label_conflicts = [
        locations
        for locations in duplicate_groups
        if len({location["label_sha256"] for location in locations}) > 1
    ]
    for locations in duplicate_groups:
        errors.append(
            "duplicate image content: "
            + ", ".join(f"{item['split']}={item['image']}" for item in locations)
        )
    for locations in label_conflicts:
        errors.append(
            "conflicting labels for duplicate image: "
            + ", ".join(item["image"] for item in locations)
        )

    report = {
        "data": str(data_path),
        "root": str(root),
        "class_count": class_count,
        "splits": split_stats,
        "duplicate_content_groups": len(duplicate_groups),
        "cross_split_leaks": len(cross_split_leaks),
        "label_conflicts": len(label_conflicts),
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors[:100],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
