from pathlib import Path

from scripts.prepare_fire_smoke_dataset import prepare_dataset


def test_prepare_dataset_creates_disjoint_splits_and_empty_negative_labels(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    for index in range(20):
        (source / "images" / f"frame-{index}.jpg").write_bytes(b"image")
        if index != 19:
            (source / "labels" / f"frame-{index}.txt").write_text(
                f"{index % 2} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
            )

    output = tmp_path / "prepared"
    yaml_path = prepare_dataset(
        source,
        output,
        validation_fraction=0.2,
        test_fraction=0.1,
        seed=42,
    )

    names_by_split = {
        split: {path.name for path in (output / "images" / split).iterdir()}
        for split in ("train", "val", "test")
    }
    assert len(names_by_split["train"]) == 14
    assert len(names_by_split["val"]) == 4
    assert len(names_by_split["test"]) == 2
    assert names_by_split["train"].isdisjoint(names_by_split["val"])
    assert names_by_split["train"].isdisjoint(names_by_split["test"])
    assert names_by_split["val"].isdisjoint(names_by_split["test"])
    assert yaml_path.exists()
    assert len(list((output / "labels").rglob("*.txt"))) == 20
