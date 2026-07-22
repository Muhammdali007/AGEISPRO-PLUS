from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build feedback-enriched weapon and fire/smoke training datasets."
    )
    parser.add_argument("--weapon-base", required=True, type=Path)
    parser.add_argument("--fire-smoke-base", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--knife-positive",
        action="append",
        default=[],
        metavar="IMAGE,X1,Y1,X2,Y2",
        help="Camera knife image and pixel bounding box; may be provided more than once.",
    )
    parser.add_argument(
        "--fire-positive",
        action="append",
        default=[],
        metavar="IMAGE,X1,Y1,X2,Y2",
        help="Reviewed camera fire image and pixel bounding box; may be provided more than once.",
    )
    parser.add_argument(
        "--smoke-positive",
        action="append",
        default=[],
        metavar="IMAGE,X1,Y1,X2,Y2",
        help="Reviewed camera smoke image and pixel bounding box; may be provided more than once.",
    )
    parser.add_argument(
        "--hard-negative",
        action="append",
        default=[],
        type=Path,
        help="Camera frame containing no fire, smoke, or weapon.",
    )
    parser.add_argument(
        "--weapon-hard-negative",
        action="append",
        default=[],
        type=Path,
        help="Reviewed frame containing no weapon; it may contain unrelated detector classes.",
    )
    parser.add_argument(
        "--fire-smoke-hard-negative",
        action="append",
        default=[],
        type=Path,
        help="Reviewed frame containing neither fire nor smoke.",
    )
    parser.add_argument(
        "--smoke-hard-negative",
        action="append",
        default=[],
        type=Path,
        help="Reviewed frame containing no smoke; it may contain fire or weapons.",
    )
    return parser


def _copy_dataset(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _collapse_weapon_classes(dataset: Path) -> None:
    for label_path in (dataset / "labels").rglob("*.txt"):
        rewritten: list[str] = []
        kept_boxes: list[tuple[float, float, float, float]] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) >= 5:
                box = tuple(map(float, parts[1:5]))
                if any(_box_iou(box, kept) >= 0.80 for kept in kept_boxes):
                    continue
                kept_boxes.append(box)
                rewritten.append(" ".join(["0", *parts[1:]]))
        label_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")


def _keep_smoke_only(dataset: Path) -> None:
    for label_path in (dataset / "labels").rglob("*.txt"):
        rewritten: list[str] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) >= 5 and parts[0] == "1":
                rewritten.append(" ".join(["0", *parts[1:]]))
        label_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")


def _sanitize_yolo_labels(dataset: Path) -> None:
    """Clamp inherited boxes to the image plane and discard zero-area labels."""
    for label_path in (dataset / "labels").rglob("*.txt"):
        rewritten: list[str] = []
        for raw_line in label_path.read_text(encoding="utf-8").splitlines():
            parts = raw_line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                center_x, center_y, width, height = map(float, parts[1:5])
            except (TypeError, ValueError):
                continue
            if class_id < 0 or not all(
                math.isfinite(value) for value in (center_x, center_y, width, height)
            ):
                continue
            left = max(0.0, center_x - width / 2)
            top = max(0.0, center_y - height / 2)
            right = min(1.0, center_x + width / 2)
            bottom = min(1.0, center_y + height / 2)
            if right <= left or bottom <= top:
                continue
            rewritten.append(
                f"{class_id} {(left + right) / 2:.6f} {(top + bottom) / 2:.6f} "
                f"{right - left:.6f} {bottom - top:.6f}"
            )
        label_path.write_text("\n".join(rewritten) + ("\n" if rewritten else ""), encoding="utf-8")


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    def corners(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        center_x, center_y, width, height = box
        return (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        )

    first_box, second_box = corners(first), corners(second)
    left = max(first_box[0], second_box[0])
    top = max(first_box[1], second_box[1])
    right = min(first_box[2], second_box[2])
    bottom = min(first_box[3], second_box[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = first[2] * first[3]
    second_area = second[2] * second[3]
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _copy_feedback_image(image_path: Path, dataset: Path, stem: str, label: str) -> None:
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    suffix = image_path.suffix.lower() or ".jpg"
    shutil.copy2(image_path, image_dir / f"{stem}{suffix}")
    (label_dir / f"{stem}.txt").write_text(label, encoding="utf-8")


def _positive_label(
    image_path: Path,
    coords: tuple[float, float, float, float],
    *,
    class_id: int = 0,
) -> str:
    with Image.open(image_path) as image:
        width, height = image.size
    x1, y1, x2, y2 = coords
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid bounding box {coords} for {image_path} ({width}x{height}).")
    center_x = ((x1 + x2) / 2) / width
    center_y = ((y1 + y2) / 2) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}\n"


def _parse_positive(raw: str) -> tuple[Path, tuple[float, float, float, float]]:
    image_raw, x1, y1, x2, y2 = raw.rsplit(",", 4)
    return Path(image_raw).resolve(), tuple(map(float, (x1, y1, x2, y2)))


def _write_yaml(dataset: Path, names: list[str]) -> None:
    lines = [
        f"path: {dataset.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
    ]
    if (dataset / "images" / "test").is_dir():
        lines.append("test: images/test")
    lines.append("names:")
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (dataset / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reviewed_paths(paths: list[Path]) -> list[Path]:
    unique: dict[Path, None] = {}
    for raw_path in paths:
        path = raw_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Reviewed feedback image does not exist: {path}")
        try:
            with Image.open(path) as image:
                image.verify()
        except OSError as exc:
            raise ValueError(f"Reviewed feedback image is unreadable: {path}") from exc
        unique[path] = None
    return list(unique)


def main() -> None:
    args = build_parser().parse_args()
    weapon_dataset = args.output_root / "weapon-single-class"
    fire_smoke_dataset = args.output_root / "fire-smoke-hard-negatives"
    smoke_dataset = args.output_root / "smoke-single-class"
    _copy_dataset(args.weapon_base.resolve(), weapon_dataset)
    _copy_dataset(args.fire_smoke_base.resolve(), fire_smoke_dataset)
    _copy_dataset(args.fire_smoke_base.resolve(), smoke_dataset)
    _collapse_weapon_classes(weapon_dataset)
    _keep_smoke_only(smoke_dataset)
    _sanitize_yolo_labels(weapon_dataset)
    _sanitize_yolo_labels(fire_smoke_dataset)
    _sanitize_yolo_labels(smoke_dataset)

    positive_paths: list[Path] = []
    for index, raw in enumerate(args.knife_positive, start=1):
        image_path, coords = _parse_positive(raw)
        positive_paths.append(image_path)
        label = _positive_label(image_path, coords)
        _copy_feedback_image(image_path, weapon_dataset, f"camera_knife_positive_{index}", label)

    fire_positive_paths: list[Path] = []
    for index, raw in enumerate(args.fire_positive, start=1):
        image_path, coords = _parse_positive(raw)
        fire_positive_paths.append(image_path)
        label = _positive_label(image_path, coords, class_id=0)
        _copy_feedback_image(image_path, fire_smoke_dataset, f"camera_fire_positive_{index}", label)

    smoke_positive_paths: list[Path] = []
    for index, raw in enumerate(args.smoke_positive, start=1):
        image_path, coords = _parse_positive(raw)
        smoke_positive_paths.append(image_path)
        _copy_feedback_image(
            image_path,
            fire_smoke_dataset,
            f"camera_smoke_positive_{index}",
            _positive_label(image_path, coords, class_id=1),
        )
        _copy_feedback_image(
            image_path,
            smoke_dataset,
            f"camera_smoke_positive_{index}",
            _positive_label(image_path, coords),
        )

    global_negative_paths = _reviewed_paths(args.hard_negative)
    weapon_negative_paths = _reviewed_paths([*global_negative_paths, *args.weapon_hard_negative])
    fire_smoke_negative_paths = _reviewed_paths(
        [*global_negative_paths, *args.fire_smoke_hard_negative]
    )
    smoke_negative_paths = _reviewed_paths(
        [*fire_smoke_negative_paths, *args.smoke_hard_negative]
    )
    contradictory_paths = set(positive_paths) & set(weapon_negative_paths)
    if contradictory_paths:
        raise ValueError(
            "The same image cannot be a weapon positive and weapon hard negative: "
            + ", ".join(map(str, sorted(contradictory_paths)))
        )
    hazard_positive_paths = set(fire_positive_paths) | set(smoke_positive_paths)
    contradictory_hazard_paths = hazard_positive_paths & set(fire_smoke_negative_paths)
    if contradictory_hazard_paths:
        raise ValueError(
            "The same image cannot be a fire/smoke positive and fire/smoke hard negative: "
            + ", ".join(map(str, sorted(contradictory_hazard_paths)))
        )
    contradictory_smoke_paths = set(smoke_positive_paths) & set(smoke_negative_paths)
    if contradictory_smoke_paths:
        raise ValueError(
            "The same image cannot be a smoke positive and smoke hard negative: "
            + ", ".join(map(str, sorted(contradictory_smoke_paths)))
        )

    for index, image_path in enumerate(weapon_negative_paths, start=1):
        _copy_feedback_image(image_path, weapon_dataset, f"camera_weapon_negative_{index}", "")

    for index, image_path in enumerate(fire_smoke_negative_paths, start=1):
        _copy_feedback_image(image_path, fire_smoke_dataset, f"camera_fire_smoke_negative_{index}", "")
    for index, image_path in enumerate(smoke_negative_paths, start=1):
        _copy_feedback_image(image_path, smoke_dataset, f"camera_smoke_negative_{index}", "")

    _write_yaml(weapon_dataset, ["weapon"])
    _write_yaml(fire_smoke_dataset, ["fire", "smoke"])
    _write_yaml(smoke_dataset, ["smoke"])
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "policy": (
            "Only human-reviewed detector-specific negatives were added. Weapon positives are not "
            "implicitly treated as fire/smoke negatives."
        ),
        "weapon_positives": [str(path) for path in positive_paths],
        "fire_positives": [str(path) for path in fire_positive_paths],
        "smoke_positives": [str(path) for path in smoke_positive_paths],
        "global_hard_negatives": [str(path) for path in global_negative_paths],
        "weapon_hard_negatives": [str(path) for path in weapon_negative_paths],
        "fire_smoke_hard_negatives": [str(path) for path in fire_smoke_negative_paths],
        "smoke_hard_negatives": [str(path) for path in smoke_negative_paths],
    }
    (args.output_root / "feedback-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Weapon dataset: {weapon_dataset / 'data.yaml'}")
    print(f"Fire/smoke dataset: {fire_smoke_dataset / 'data.yaml'}")
    print(f"Smoke dataset: {smoke_dataset / 'data.yaml'}")


if __name__ == "__main__":
    main()
