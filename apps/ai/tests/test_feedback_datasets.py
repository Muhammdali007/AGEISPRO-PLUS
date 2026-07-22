import json
import sys
from pathlib import Path

from PIL import Image

from scripts.prepare_feedback_datasets import main


def _base_dataset(root: Path, *, classes: int) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 32), color=(32, 32, 32)).save(
            root / "images" / split / f"base-{split}.jpg"
        )
        (root / "labels" / split / f"base-{split}.txt").write_text(
            f"{min(classes - 1, 0)} 0.5 0.5 0.2 0.2\n",
            encoding="utf-8",
        )


def test_detector_specific_negative_is_not_copied_to_other_detectors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    weapon_base = tmp_path / "weapon-base"
    fire_base = tmp_path / "fire-base"
    _base_dataset(weapon_base, classes=1)
    _base_dataset(fire_base, classes=2)
    (fire_base / "labels" / "train" / "base-train.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n1 0.5 0.5 0 0\n",
        encoding="utf-8",
    )
    knife = tmp_path / "knife.jpg"
    smoke = tmp_path / "smoke.jpg"
    weapon_negative = tmp_path / "phone.jpg"
    Image.new("RGB", (100, 100), color=(120, 120, 120)).save(knife)
    Image.new("RGB", (100, 100), color=(180, 180, 180)).save(smoke)
    Image.new("RGB", (100, 100), color=(20, 40, 80)).save(weapon_negative)
    output = tmp_path / "feedback"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_feedback_datasets.py",
            "--weapon-base",
            str(weapon_base),
            "--fire-smoke-base",
            str(fire_base),
            "--output-root",
            str(output),
            "--knife-positive",
            f"{knife},10,10,80,80",
            "--smoke-positive",
            f"{smoke},20,20,90,90",
            "--weapon-hard-negative",
            str(weapon_negative),
        ],
    )

    main()

    assert (output / "weapon-single-class" / "images" / "train" / "camera_knife_positive_1.jpg").is_file()
    assert (output / "weapon-single-class" / "images" / "train" / "camera_weapon_negative_1.jpg").is_file()
    assert (
        output / "fire-smoke-hard-negatives" / "images" / "train" / "camera_smoke_positive_1.jpg"
    ).is_file()
    assert (
        output / "smoke-single-class" / "images" / "train" / "camera_smoke_positive_1.jpg"
    ).is_file()
    assert (
        output / "fire-smoke-hard-negatives" / "labels" / "train" / "camera_smoke_positive_1.txt"
    ).read_text(encoding="utf-8").startswith("1 ")
    assert (
        output / "smoke-single-class" / "labels" / "train" / "camera_smoke_positive_1.txt"
    ).read_text(encoding="utf-8").startswith("0 ")
    assert (
        output / "smoke-single-class" / "labels" / "train" / "base-train.txt"
    ).read_text(encoding="utf-8") == ""
    manifest = json.loads((output / "feedback-manifest.json").read_text(encoding="utf-8"))
    assert manifest["weapon_hard_negatives"] == [str(weapon_negative.resolve())]
    assert manifest["smoke_positives"] == [str(smoke.resolve())]
    assert manifest["fire_smoke_hard_negatives"] == []
