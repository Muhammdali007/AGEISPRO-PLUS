from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLIT_ALIASES = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}


@dataclass(frozen=True)
class YoloObbAnnotation:
    class_id: int
    points: tuple[float, float, float, float, float, float, float, float]


@dataclass(frozen=True)
class YoloBox:
    class_name: str
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo_row(self, class_id: int) -> str:
        return (
            f"{class_id} "
            f"{self.x_center:.6f} {self.y_center:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


def load_class_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Class map must be a JSON object.")
    return {str(key).strip().lower(): str(value).strip().lower() for key, value in payload.items()}


def parse_source_names(raw_names: str | None) -> list[str]:
    if not raw_names:
        raise ValueError("Source class names are required when the source dataset YAML is unavailable.")
    names = [name.strip().lower() for name in raw_names.split(",") if name.strip()]
    if not names:
        raise ValueError("Source class names cannot be empty.")
    return names


def parse_obb_annotation(line: str) -> YoloObbAnnotation:
    parts = line.strip().split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 values in OBB row, received {len(parts)}: {line!r}")
    class_id = int(parts[0])
    points = tuple(float(value) for value in parts[1:])
    return YoloObbAnnotation(class_id=class_id, points=points)  # type: ignore[arg-type]


def obb_to_yolo_box(annotation: YoloObbAnnotation, class_name: str) -> YoloBox:
    xs = annotation.points[0::2]
    ys = annotation.points[1::2]
    min_x = max(min(xs), 0.0)
    min_y = max(min(ys), 0.0)
    max_x = min(max(xs), 1.0)
    max_y = min(max(ys), 1.0)
    width = max(max_x - min_x, 1e-6)
    height = max(max_y - min_y, 1e-6)
    x_center = min_x + (width / 2)
    y_center = min_y + (height / 2)
    return YoloBox(
        class_name=class_name,
        x_center=min(max(x_center, 0.0), 1.0),
        y_center=min(max(y_center, 0.0), 1.0),
        width=min(max(width, 1e-6), 1.0),
        height=min(max(height, 1e-6), 1.0),
    )


def find_split_directory(dataset_root: Path, split_name: str) -> Path | None:
    for alias in DEFAULT_SPLIT_ALIASES[split_name]:
        direct = dataset_root / alias
        if direct.is_dir():
            return direct

        for images_name in ("images", "imgs"):
            nested = dataset_root / images_name / alias
            if nested.is_dir():
                return nested.parent.parent / alias if nested.parent.parent.name != dataset_root.name else nested

        for candidate in dataset_root.rglob(alias):
            if candidate.is_dir():
                return candidate
    return None


def resolve_image_and_label_roots(split_dir: Path) -> tuple[Path, Path]:
    if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
        return split_dir / "images", split_dir / "labels"

    if split_dir.name in {"train", "training", "val", "valid", "validation", "test", "testing"}:
        parent = split_dir.parent
        if parent.name == "images" and (parent.parent / "labels" / split_dir.name).is_dir():
            return split_dir, parent.parent / "labels" / split_dir.name
        if parent.name == "labels" and (parent.parent / "images" / split_dir.name).is_dir():
            return parent.parent / "images" / split_dir.name, split_dir
        if (parent / "images" / split_dir.name).is_dir() and (parent / "labels" / split_dir.name).is_dir():
            return parent / "images" / split_dir.name, parent / "labels" / split_dir.name

    image_dir = next((path for path in split_dir.rglob("*") if path.is_dir() and path.name == "images"), None)
    label_dir = next((path for path in split_dir.rglob("*") if path.is_dir() and path.name == "labels"), None)
    if image_dir and label_dir:
        return image_dir, label_dir

    if split_dir.name == "images":
        label_dir = split_dir.parent / "labels"
        if label_dir.is_dir():
            return split_dir, label_dir

    raise ValueError(f"Unable to find image/label folders under {split_dir}")


def discover_dataset_layout(dataset_root: Path) -> dict[str, tuple[Path, Path]]:
    layout: dict[str, tuple[Path, Path]] = {}
    for split_name in ("train", "val", "test"):
        split_dir = find_split_directory(dataset_root, split_name)
        if split_dir is None:
            continue
        layout[split_name] = resolve_image_and_label_roots(split_dir)
    if "train" not in layout or "val" not in layout:
        raise ValueError(
            "Dataset layout must expose at least train and val splits with images and labels."
        )
    return layout


def collect_target_names(source_names: list[str], class_map: dict[str, str]) -> list[str]:
    mapped_names = [class_map.get(name, name) for name in source_names]
    target_names: list[str] = []
    for name in mapped_names:
        if name not in target_names:
            target_names.append(name)
    return target_names


def convert_dataset(
    *,
    dataset_root: Path,
    output_root: Path,
    source_names: list[str],
    class_map: dict[str, str],
    copy_images: bool = False,
) -> Path:
    layout = discover_dataset_layout(dataset_root)
    target_names = collect_target_names(source_names, class_map)
    target_index = {name: index for index, name in enumerate(target_names)}

    for split_name, (image_root, label_root) in layout.items():
        output_image_root = output_root / "images" / split_name
        output_label_root = output_root / "labels" / split_name
        output_image_root.mkdir(parents=True, exist_ok=True)
        output_label_root.mkdir(parents=True, exist_ok=True)

        image_paths = [
            path for path in image_root.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        for image_path in image_paths:
            relative = image_path.relative_to(image_root)
            target_image_path = output_image_root / relative
            target_image_path.parent.mkdir(parents=True, exist_ok=True)
            if copy_images:
                shutil.copy2(image_path, target_image_path)
            else:
                if target_image_path.exists():
                    target_image_path.unlink()
                target_image_path.symlink_to(image_path.resolve())

            label_path = label_root / relative.with_suffix(".txt")
            target_label_path = output_label_root / relative.with_suffix(".txt")
            target_label_path.parent.mkdir(parents=True, exist_ok=True)
            rows: list[str] = []
            if label_path.is_file():
                for line in label_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    annotation = parse_obb_annotation(line)
                    source_name = source_names[annotation.class_id]
                    target_name = class_map.get(source_name, source_name)
                    converted = obb_to_yolo_box(annotation, target_name)
                    rows.append(converted.to_yolo_row(target_index[target_name]))
            target_label_path.write_text("\n".join(rows), encoding="utf-8")

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test" if "test" in layout else "",
                f"names: {target_names}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return dataset_yaml
