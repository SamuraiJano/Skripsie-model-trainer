from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from PIL import Image

from naartjie_cv.datasets import (
    dfine_loader_smoke,
    discover_training_export,
    ultralytics_loader_smoke,
    validate_training_export,
)
from naartjie_cv.train import TrainingOptions, synthetic_asset_summary, train_dfine_export


def make_export(tmp_path: Path) -> Path:
    root = tmp_path / "fixture-training-export"
    detection = root / "yolo26_detection"
    segmentation = root / "yolo26_segmentation"
    dfine = root / "dfine_detection"
    for task_root in (detection, segmentation):
        for split in ("train", "val"):
            (task_root / "images" / split).mkdir(parents=True)
            (task_root / "labels" / split).mkdir(parents=True)
    for split in ("train", "val"):
        (dfine / "images" / split).mkdir(parents=True)
    (dfine / "annotations").mkdir()
    rows = []
    for number, split in enumerate(("train", "val")):
        name = f"image-{number}.png"
        for target in (detection, segmentation, dfine):
            Image.new("RGB", (32, 24), (20 + number, 80, 120)).save(target / "images" / split / name)
        (detection / "labels" / split / f"image-{number}.txt").write_text("0 0.500000 0.500000 0.500000 0.500000\n", encoding="utf-8")
        (segmentation / "labels" / split / f"image-{number}.txt").write_text("0 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000 0.250000 0.750000\n", encoding="utf-8")
        rows.append({"image_uuid": f"image-{number}", "split": split})
        coco = {
            "images": [{"id": 1, "file_name": name, "width": 32, "height": 24}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "bbox": [8.0, 6.0, 16.0, 12.0], "area": 192.0, "iscrowd": 0}],
            "categories": [{"id": 0, "name": "fruit"}],
        }
        (dfine / "annotations" / f"instances_{split}.json").write_text(json.dumps(coco), encoding="utf-8")
    data = {"path": ".", "train": "images/train", "val": "images/val", "names": {0: "fruit"}}
    (detection / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    data["names"] = {0: "defect"}
    (segmentation / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (dfine / "dataset_info.yaml").write_text(yaml.safe_dump({
        "annotations": {"train": "annotations/instances_train.json", "val": "annotations/instances_val.json"},
        "remap_mscoco_category": False,
        "category_id_policy": "zero_based_contiguous",
    }, sort_keys=False), encoding="utf-8")
    (root / "synthetic_assets").mkdir()
    (root / "synthetic_assets" / "assets_manifest.json").write_text(json.dumps({"counts": {"foregrounds": 0, "backgrounds": 0}, "foregrounds": [], "backgrounds": []}), encoding="utf-8")
    class_item = lambda name: {"uuid": f"00000000-0000-0000-0000-00000000000{'1' if name == 'fruit' else '2'}", "id": 0, "export_index": 0, "name": name, "display_name": name.title()}
    manifest = {
        "split": {"enabled": ["train", "val"], "counts": {"train": 1, "val": 1}, "seed": 42},
        "class_mappings": {"detection": [class_item("fruit")], "segmentation": [class_item("defect")]},
        "images": rows,
        "export_hash": "fixture",
    }
    (root / "export_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_discovery_validation_and_synthetic_boundary(tmp_path: Path) -> None:
    root = make_export(tmp_path)
    assert discover_training_export(root).root == root.resolve()
    report = validate_training_export(root)
    assert report["ok"], report["errors"]
    assert synthetic_asset_summary(root)["foregrounds"] == 0


def test_rejects_malformed_detection_labels(tmp_path: Path) -> None:
    root = make_export(tmp_path)
    (root / "yolo26_detection" / "labels" / "train" / "image-0.txt").write_text("0 0.5 0.5 -0.1 0.2\n", encoding="utf-8")
    report = validate_training_export(root)
    assert not report["ok"]
    assert any("out-of-range" in error or "non-positive" in error for error in report["errors"])


def test_installed_ultralytics_loaders_accept_export(tmp_path: Path) -> None:
    root = make_export(tmp_path)
    assert ultralytics_loader_smoke(root, "detect")["images"] == 1
    assert ultralytics_loader_smoke(root, "segment")["images"] == 1


def test_dfine_dry_run_uses_official_nano_config(tmp_path: Path) -> None:
    configured = os.getenv("DFINE_ROOT")
    if not configured:
        pytest.skip("Set DFINE_ROOT to run the official D-FINE integration test")
    root = make_export(tmp_path)
    result = train_dfine_export(root, configured, TrainingOptions(device="cpu", output_dir=str(tmp_path / "runs")), execute=False)
    config = Path(result["config"]).read_text(encoding="utf-8")
    assert "dfine_hgnetv2_n_custom.yml" in config
    assert "remap_mscoco_category: false" in config
    smoke = dfine_loader_smoke(root, configured)
    assert smoke["images"] == 1


def test_trainer_has_no_inference_frontend_or_camera_backend() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "foreground" / "package.json").exists()
    assert not (repository / "background" / "src" / "CameraManager.cpp").exists()
