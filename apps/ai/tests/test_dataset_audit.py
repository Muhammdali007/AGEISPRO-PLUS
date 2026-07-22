import json
import sys
from pathlib import Path

from PIL import Image

from scripts.audit_yolo_dataset import main


def test_audit_rejects_duplicate_content_across_splits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "labels" / split).mkdir(parents=True)

    image_path = dataset / "images" / "train" / "frame.jpg"
    Image.new("RGB", (16, 16), color=(255, 64, 0)).save(image_path)
    (dataset / "images" / "val" / "copy.jpg").write_bytes(image_path.read_bytes())
    (dataset / "labels" / "train" / "frame.txt").write_text("", encoding="utf-8")
    (dataset / "labels" / "val" / "copy.txt").write_text("", encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: fire\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_yolo_dataset.py", "--data", str(data_yaml), "--report", str(report_path)],
    )

    assert main() == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is False
    assert report["duplicate_content_groups"] == 1
    assert report["cross_split_leaks"] == 1
