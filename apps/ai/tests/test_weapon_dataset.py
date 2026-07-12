from pathlib import Path

from app.training.weapon_dataset import convert_dataset, obb_to_yolo_box, parse_obb_annotation


def test_parse_obb_annotation_and_convert_to_detect_box() -> None:
    annotation = parse_obb_annotation("2 0.10 0.20 0.30 0.20 0.30 0.50 0.10 0.50")
    converted = obb_to_yolo_box(annotation, "pistol")

    assert converted.class_name == "pistol"
    assert round(converted.x_center, 2) == 0.20
    assert round(converted.y_center, 2) == 0.35
    assert round(converted.width, 2) == 0.20
    assert round(converted.height, 2) == 0.30


def test_convert_dataset_writes_detect_labels(tmp_path: Path) -> None:
    dataset_root = tmp_path / "source"
    train_images = dataset_root / "images" / "train"
    train_labels = dataset_root / "labels" / "train"
    val_images = dataset_root / "images" / "val"
    val_labels = dataset_root / "labels" / "val"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    val_images.mkdir(parents=True)
    val_labels.mkdir(parents=True)

    (train_images / "frame1.jpg").write_bytes(b"img")
    (val_images / "frame2.jpg").write_bytes(b"img")
    (train_labels / "frame1.txt").write_text(
        "0 0.10 0.20 0.30 0.20 0.30 0.50 0.10 0.50",
        encoding="utf-8",
    )
    (val_labels / "frame2.txt").write_text(
        "1 0.40 0.10 0.70 0.10 0.70 0.40 0.40 0.40",
        encoding="utf-8",
    )

    output_root = tmp_path / "prepared"
    dataset_yaml = convert_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        source_names=["knife", "handgun"],
        class_map={"handgun": "pistol"},
        copy_images=True,
    )

    assert dataset_yaml.is_file()
    assert (output_root / "images" / "train" / "frame1.jpg").is_file()
    assert (output_root / "labels" / "train" / "frame1.txt").read_text(encoding="utf-8").startswith("0 ")
    assert (output_root / "labels" / "val" / "frame2.txt").read_text(encoding="utf-8").startswith("1 ")
    assert "names: ['knife', 'pistol']" in dataset_yaml.read_text(encoding="utf-8")
