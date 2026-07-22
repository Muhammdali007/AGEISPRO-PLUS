from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")
TARGET_CLASSES = {"fire": 0, "smoke": 1}


def prepare_dataset(
    source: Path,
    output: Path,
    *,
    validation_fraction: float = 0.20,
    test_fraction: float = 0.10,
    seed: int = 20260718,
) -> Path:
    """Prepare a flat ``images``/``labels`` source with deterministic disjoint splits.

    This compatibility entry point is useful for small local corpora. The CLI
    below handles already-split, multi-source datasets and class remapping.
    """
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions cannot be negative.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must leave a training split.")

    source = source.resolve()
    output = output.resolve()
    images = sorted(
        item
        for item in (source / "images").iterdir()
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )
    random.Random(seed).shuffle(images)
    validation_count = round(len(images) * validation_fraction)
    test_count = round(len(images) * test_fraction)
    split_images = {
        "test": images[:test_count],
        "val": images[test_count : test_count + validation_count],
        "train": images[test_count + validation_count :],
    }

    for split, paths in split_images.items():
        for image_path in paths:
            image_destination = output / "images" / split / image_path.name
            _link_or_copy(image_path, image_destination, copy_files=True)
            label_source = source / "labels" / f"{image_path.stem}.txt"
            label_destination = output / "labels" / split / f"{image_path.stem}.txt"
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            if label_source.is_file():
                shutil.copy2(label_source, label_destination)
            else:
                label_destination.write_text("", encoding="utf-8")

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: fire",
                "  1: smoke",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return yaml_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize and stratify YOLO fire/smoke datasets for AegisPro training."
    )
    parser.add_argument("--source-data", required=True, type=Path, action="append")
    parser.add_argument(
        "--source-license",
        action="append",
        default=[],
        help="License/SPDX note aligned with each --source-data entry.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-train", type=int, default=4000)
    parser.add_argument("--max-val", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy instead of hard-linking source images on the same volume.",
    )
    return parser


def _names_by_id(payload: dict[str, Any]) -> dict[int, str]:
    raw = payload.get("names", {})
    if isinstance(raw, list):
        return {index: str(name).strip().lower() for index, name in enumerate(raw)}
    if isinstance(raw, dict):
        return {int(index): str(name).strip().lower() for index, name in raw.items()}
    raise ValueError("Dataset YAML must define names as a list or mapping.")


def _dataset_root(data_path: Path, payload: dict[str, Any]) -> Path:
    configured = Path(str(payload.get("path", ".")))
    candidate = configured if configured.is_absolute() else data_path.parent / configured
    if candidate.exists():
        return candidate.resolve()
    # Old copied datasets often contain a now-invalid absolute path. Their YAML
    # parent is the portable and unambiguous fallback.
    return data_path.parent.resolve()


def _image_paths(root: Path, value: Any) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for raw in values:
        candidate = Path(str(raw))
        candidate = candidate if candidate.is_absolute() else root / candidate
        if candidate.is_dir():
            images.extend(
                item for item in candidate.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS
            )
    return sorted(set(images))


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def _normalized_lines(label_path: Path, names: dict[int, str]) -> tuple[list[str], str]:
    normalized: list[str] = []
    present: set[str] = set()
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                source_id = int(parts[0])
            except ValueError:
                continue
            source_name = names.get(source_id, "")
            if source_name not in TARGET_CLASSES:
                continue
            try:
                center_x, center_y, width, height = [float(value) for value in parts[1:5]]
            except ValueError:
                continue
            left = max(0.0, center_x - width / 2)
            top = max(0.0, center_y - height / 2)
            right = min(1.0, center_x + width / 2)
            bottom = min(1.0, center_y + height / 2)
            if right <= left or bottom <= top:
                continue
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            width = right - left
            height = bottom - top
            normalized.append(
                f"{TARGET_CLASSES[source_name]} {center_x:.6f} {center_y:.6f} "
                f"{width:.6f} {height:.6f}"
            )
            present.add(source_name)
    category = "both" if len(present) == 2 else (next(iter(present)) if present else "none")
    return normalized, category


def _select_stratified(
    records: list[tuple[Path, list[str], str, str]], limit: int, seed: int
) -> list[tuple[Path, list[str], str, str]]:
    if limit <= 0 or len(records) <= limit:
        return records
    rng = random.Random(seed)
    groups: dict[str, list[tuple[Path, list[str], str, str]]] = defaultdict(list)
    for record in records:
        groups[record[2]].append(record)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[tuple[Path, list[str], str, str]] = []
    categories = [category for category in ("fire", "smoke", "both", "none") if groups[category]]
    # Round-robin gives rare hazard combinations a fair share while retaining
    # hard-negative examples from the source dataset.
    while len(selected) < limit and categories:
        for category in list(categories):
            if not groups[category]:
                categories.remove(category)
                continue
            selected.append(groups[category].pop())
            if len(selected) >= limit:
                break
    return selected


def _link_or_copy(source: Path, destination: Path, copy_files: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if not copy_files:
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def main() -> int:
    args = build_parser().parse_args()
    out_dir = args.out_dir.resolve()
    split_records: dict[str, list[tuple[Path, list[str], str, str]]] = defaultdict(list)
    sources: list[dict[str, Any]] = []

    for source_index, data_path_raw in enumerate(args.source_data, start=1):
        data_path = data_path_raw.resolve()
        payload = yaml.safe_load(data_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Dataset YAML must contain an object: {data_path}")
        root = _dataset_root(data_path, payload)
        names = _names_by_id(payload)
        source_prefix = f"source{source_index}"
        license_note = (
            args.source_license[source_index - 1]
            if source_index - 1 < len(args.source_license)
            else "unspecified - review required before promotion"
        )
        sources.append(
            {"yaml": str(data_path), "root": str(root), "names": names, "license": license_note}
        )
        for split in ("train", "val", "test"):
            if split not in payload:
                continue
            for image_path in _image_paths(root, payload[split]):
                lines, category = _normalized_lines(_label_path(image_path), names)
                split_records[split].append((image_path, lines, category, source_prefix))

    limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    stats: dict[str, Any] = {}
    for split, records in split_records.items():
        selected = _select_stratified(records, limits[split], args.seed + len(stats))
        categories: Counter[str] = Counter()
        boxes: Counter[str] = Counter()
        used_names: Counter[str] = Counter()
        for image_path, lines, category, source_prefix in selected:
            base_name = f"{source_prefix}_{image_path.stem}"
            suffix_index = used_names[base_name]
            used_names[base_name] += 1
            stem = base_name if suffix_index == 0 else f"{base_name}_{suffix_index}"
            destination = out_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
            _link_or_copy(image_path, destination, args.copy_files)
            label_destination = out_dir / "labels" / split / f"{stem}.txt"
            label_destination.parent.mkdir(parents=True, exist_ok=True)
            label_destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            categories[category] += 1
            for line in lines:
                boxes["fire" if line.startswith("0 ") else "smoke"] += 1
        stats[split] = {
            "images": len(selected),
            "categories": dict(sorted(categories.items())),
            "boxes": dict(sorted(boxes.items())),
        }

    yaml_lines = [f"path: {out_dir.as_posix()}", "train: images/train", "val: images/val"]
    if "test" in stats:
        yaml_lines.append("test: images/test")
    yaml_lines.extend(["names:", "  0: fire", "  1: smoke"])
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    manifest = {
        "seed": args.seed,
        "target_classes": TARGET_CLASSES,
        "sources": sources,
        "splits": stats,
    }
    (out_dir / "dataset-stats.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    print(f"dataset: {out_dir / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
