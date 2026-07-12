from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.training.weapon_dataset import convert_dataset, load_class_map, parse_source_names  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a YOLO OBB weapon dataset into standard YOLO detect labels for AegisPro."
    )
    parser.add_argument(
        "--source-root",
        required=True,
        type=Path,
        help="Root directory of the source OBB dataset.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Directory where the converted detect dataset will be written.",
    )
    parser.add_argument(
        "--source-names",
        required=True,
        help="Comma-separated source class names, in dataset class-id order.",
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=None,
        help="Optional JSON file mapping source class names to target class names.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into the output dataset instead of creating symlinks.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset_yaml = convert_dataset(
        dataset_root=args.source_root.resolve(),
        output_root=args.output_root.resolve(),
        source_names=parse_source_names(args.source_names),
        class_map=load_class_map(args.class_map.resolve()) if args.class_map else {},
        copy_images=args.copy_images,
    )
    print(f"Prepared dataset: {dataset_yaml}")


if __name__ == "__main__":
    main()
