from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

from naartjie_cv.datasets import (
    dfine_loader_smoke,
    discover_training_export,
    require_valid_yolo_training_dataset,
    ultralytics_loader_smoke,
    validate_training_export,
    validate_yolo_training_dataset,
)
from naartjie_cv.notebook_ui import TRAINING_TASKS, _project_root, create_training_dashboard
import naartjie_cv.train as train_module
from naartjie_cv.train import TrainingOptions, synthetic_asset_summary, train_dfine_export, train_yolo_export


def make_yolo_dataset(tmp_path: Path, task: str, *, include_val: bool = True) -> Path:
    root = tmp_path / f"standalone-{task}"
    splits = ("train", "val") if include_val else ("train",)
    for split in splits:
        (root / "images" / split).mkdir(parents=True)
        (root / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (32, 24), (40, 80, 120)).save(root / "images" / split / f"{split}.png")
        label = (
            "0 0.500000 0.500000 0.500000 0.500000\n"
            if task == "detect"
            else "0 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000\n"
        )
        (root / "labels" / split / f"{split}.txt").write_text(label, encoding="utf-8")
    data = {"train": "images/train", "val": "images/val", "names": {0: "Naartjie"}}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


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


def test_valid_standalone_detection_dataset(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    report = validate_yolo_training_dataset(root, "detect")
    assert report["ok"], report["errors"]
    assert report["discovery"] == {
        "root": str(root.resolve()),
        "data_yaml": str((root / "data.yaml").resolve()),
        "task": "detect",
        "names": {0: "Naartjie"},
    }
    assert require_valid_yolo_training_dataset(root, "detect") == (root / "data.yaml").resolve()


def test_valid_standalone_segmentation_dataset(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "segment")
    report = validate_yolo_training_dataset(root, "segment")
    assert report["ok"], report["errors"]
    assert report["discovery"]["task"] == "segment"


def test_rejects_detection_labels_without_five_columns(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    (root / "labels" / "train" / "train.txt").write_text("0 0.5 0.5 0.5\n", encoding="utf-8")
    report = validate_yolo_training_dataset(root, "detect")
    assert not report["ok"]
    assert any("malformed detect label" in error for error in report["errors"])


def test_rejects_invalid_detection_coordinates(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    (root / "labels" / "train" / "train.txt").write_text("0 1.1 0.5 0.5 0.5\n", encoding="utf-8")
    report = validate_yolo_training_dataset(root, "detect")
    assert not report["ok"]
    assert any("out-of-range coordinate" in error for error in report["errors"])


def test_rejects_missing_validation_folder(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect", include_val=False)
    report = validate_yolo_training_dataset(root, "detect")
    assert not report["ok"]
    assert any("val image directory is missing" in error for error in report["errors"])


def test_complete_bundle_remains_compatible_with_task_validation(tmp_path: Path) -> None:
    root = make_export(tmp_path)
    detect = validate_yolo_training_dataset(root, "detect")
    segment = validate_yolo_training_dataset(root, "segment")
    assert detect["ok"], detect["errors"]
    assert segment["ok"], segment["errors"]
    assert Path(detect["discovery"]["data_yaml"]) == root / "yolo26_detection" / "data.yaml"
    assert Path(segment["discovery"]["data_yaml"]) == root / "yolo26_segmentation" / "data.yaml"
    assert validate_training_export(root)["ok"]


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

    standalone = make_yolo_dataset(tmp_path, "detect")
    assert ultralytics_loader_smoke(standalone, "detect")["images"] == 1


def test_yolo_detect_model_and_metadata_without_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    selected_models: list[str] = []

    class Scalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def item(self) -> float:
            return self.value

    class Results:
        results_dict = {
            "metrics/precision(B)": Scalar(0.91),
            "metrics/recall(B)": Scalar(0.82),
            "metrics/mAP50(B)": Scalar(0.73),
            "metrics/mAP50-95(B)": Scalar(0.64),
        }

    class FakeYOLO:
        def __init__(self, model_name: str) -> None:
            selected_models.append(model_name)
            self.trainer = None

        def train(self, **arguments: object) -> Results:
            run_dir = Path(str(arguments["project"])) / str(arguments["name"])
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights" / "best.pt").touch()
            (run_dir / "weights" / "last.pt").touch()
            (run_dir / "results.png").touch()
            (run_dir / "results.csv").write_text(
                "epoch,train/box_loss,val/box_loss,metrics/precision(B)\n0,0.31,0.42,0.91\n",
                encoding="utf-8",
            )
            self.trainer = SimpleNamespace(save_dir=run_dir, metrics=Results())
            return Results()

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setattr(train_module, "_runtime", lambda: {"python": "test", "cuda_available": False})

    result = train_yolo_export(
        root,
        "detect",
        TrainingOptions(device="cpu", output_dir=str(tmp_path / "runs"), run_name="standalone-run"),
    )

    assert selected_models == ["yolo26n.pt"]
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert metadata["dataset"]["manifest"] is None
    assert metadata["dataset"]["export_hash"] is None
    assert metadata["dataset"]["class_mapping"] == {"0": "Naartjie"}
    assert metadata["artifacts"]["final_metrics"] == result["final_metrics"]
    metrics = json.loads(Path(result["final_metrics"]).read_text(encoding="utf-8"))
    assert metrics["metrics/precision(B)"] == pytest.approx(0.91)
    assert metrics["metrics/recall(B)"] == pytest.approx(0.82)
    assert metrics["metrics/mAP50(B)"] == pytest.approx(0.73)
    assert metrics["metrics/mAP50-95(B)"] == pytest.approx(0.64)
    assert metrics["train/box_loss"] == pytest.approx(0.31)
    assert metrics["val/box_loss"] == pytest.approx(0.42)


def test_model_trainer_notebook_covers_detection_and_segmentation() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "model_trainer_ui.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert _project_root(project_root) == project_root.resolve()
    assert [(label, task, model) for label, task, model in TRAINING_TASKS] == [
        ("Object detection", "detect", "yolo26n.pt"),
        ("Instance segmentation", "segment", "yolo26n-seg.pt"),
    ]
    assert notebook["nbformat"] == 4
    assert "create_training_dashboard" in source
    assert "object detection" in source.lower()
    assert "instance segmentation" in source.lower()
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_notebook_dashboard_constructs_when_optional_dependency_is_installed() -> None:
    widgets = pytest.importorskip("ipywidgets")
    project_root = Path(__file__).resolve().parents[1]
    dashboard = create_training_dashboard(project_root)
    assert isinstance(dashboard, widgets.VBox)
    task_controls = [child for child in dashboard.children if isinstance(child, widgets.ToggleButtons)]
    assert len(task_controls) == 1
    assert tuple(task_controls[0].options) == (
        ("Object detection", "detect"),
        ("Instance segmentation", "segment"),
    )


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
