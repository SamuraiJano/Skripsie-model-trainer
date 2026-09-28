from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from PIL import Image

from naartjie_cv.datasets import (
    discover_training_export,
    prepare_dfine_seg_dataset,
    require_valid_yolo_training_dataset,
    ultralytics_loader_smoke,
    validate_training_export,
    validate_yolo_training_dataset,
)
import naartjie_cv.dfine_notebook_ui as dfine_ui
from naartjie_cv.dfine_notebook_ui import DFINE_TASKS, create_dfine_training_dashboard
from naartjie_cv.notebook_ui import (
    TRAINING_TASKS,
    YOLO_PARAMETER_GROUPS,
    _project_root,
    create_training_dashboard,
)
import naartjie_cv.train as train_module
from naartjie_cv.train import (
    DFineSegOptions,
    TrainingOptions,
    YoloTrainingOptions,
    synthetic_asset_summary,
    train_dfine_export,
    train_dfine_seg_export,
    train_yolo_export,
)
from naartjie_cv.yolo_export import (
    create_yolo_export_dashboard,
    discover_yolo_checkpoints,
    export_yolo_model,
    resolve_yolo_checkpoint,
    supported_export_arguments,
)


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
    data = {"path": ".", "train": "images/train", "val": "images/val", "names": {0: "Naartjie"}}
    (root / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return root


def add_test_split(root: Path, task: str) -> None:
    (root / "images" / "test").mkdir(parents=True)
    (root / "labels" / "test").mkdir(parents=True)
    Image.new("RGB", (32, 24), (40, 80, 120)).save(root / "images" / "test" / "test.png")
    label = (
        "0 0.500000 0.500000 0.500000 0.500000\n"
        if task == "detect"
        else "0 0.250000 0.250000 0.750000 0.250000 0.750000 0.750000\n"
    )
    (root / "labels" / "test" / "test.txt").write_text(label, encoding="utf-8")
    data_yaml = root / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    data["test"] = "images/test"
    data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


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


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_prepare_dfine_seg_dataset_from_yolo_splits(tmp_path: Path, task: str) -> None:
    root = make_yolo_dataset(tmp_path, task)
    prepared = prepare_dfine_seg_dataset(root, task, tmp_path / f"prepared-{task}")

    assert prepared["task"] == task
    assert prepared["classes"] == {0: "Naartjie"}
    assert prepared["splits"]["train"]["images"] == 1
    assert prepared["splits"]["val"]["images"] == 1
    assert Path(prepared["root"], "images", "train", "train.png").is_file()

    train_json = json.loads(Path(prepared["splits"]["train"]["json"]).read_text(encoding="utf-8"))
    assert train_json["images"][0]["file_name"] == "train/train.png"
    annotation = train_json["annotations"][0]
    assert annotation["bbox"][2:] == pytest.approx([16.0, 12.0])
    if task == "segment":
        assert len(annotation["segmentation"][0]) == 6
        assert annotation["area"] > 0
    else:
        assert annotation["segmentation"] == []


def test_prepare_dfine_seg_clips_rounded_edge_box_to_image(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    # Six-decimal YOLO values can put the calculated top edge just below zero.
    (root / "labels" / "train" / "train.txt").write_text(
        "0 0.592581 0.015457 0.079258 0.030915\n",
        encoding="utf-8",
    )

    prepared = prepare_dfine_seg_dataset(root, "detect", tmp_path / "prepared-edge")
    payload = json.loads(Path(prepared["splits"]["train"]["json"]).read_text(encoding="utf-8"))
    x, y, width, height = payload["annotations"][0]["bbox"]

    assert x >= 0
    assert y == 0
    assert width > 0
    assert height > 0


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
    submitted_arguments: list[dict[str, object]] = []

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
            submitted_arguments.append(arguments)
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
        TrainingOptions(
            device="cpu",
            output_dir=str(tmp_path / "runs"),
            run_name="standalone-run",
            yolo=YoloTrainingOptions(
                patience=25,
                cache="disk",
                save_period=5,
                mosaic=0.25,
                extra_args={"end2end": True},
            ),
        ),
    )

    assert selected_models == ["yolo26n.pt"]
    submitted_data_yaml = Path(str(submitted_arguments[0]["data"]))
    submitted_data = yaml.safe_load(submitted_data_yaml.read_text(encoding="utf-8"))
    assert submitted_data_yaml != root / "data.yaml"
    assert Path(submitted_data["path"]) == root.resolve()
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert metadata["dataset"]["manifest"] is None
    assert metadata["dataset"]["export_hash"] is None
    assert metadata["dataset"]["class_mapping"] == {"0": "Naartjie"}
    assert metadata["artifacts"]["final_metrics"] == result["final_metrics"]
    assert metadata["artifacts"]["training_data_yaml"] == str(submitted_data_yaml)
    assert metadata["training_arguments"]["patience"] == 25
    assert metadata["training_arguments"]["cache"] == "disk"
    assert metadata["training_arguments"]["save_period"] == 5
    assert metadata["training_arguments"]["mosaic"] == pytest.approx(0.25)
    assert metadata["training_arguments"]["end2end"] is True
    assert metadata["training_arguments"]["show_labels"] is False
    assert metadata["training_arguments"]["line_width"] == 2
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
    assert "create_yolo_export_dashboard" in source
    assert "yolo26n-detect-20260822-002413" not in source
    assert "object detection" in source.lower()
    assert "instance segmentation" in source.lower()
    assert all(
        output.get("output_type") != "error"
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        for output in cell.get("outputs", [])
    )


def _make_checkpoint_project(tmp_path: Path) -> tuple[Path, Path, Path]:
    project_root = tmp_path / "model_training"
    (project_root / "src" / "naartjie_cv").mkdir(parents=True)
    older = project_root / "runs" / "yolo26" / "older" / "weights" / "best.pt"
    newer = project_root / "runs" / "yolo26" / "newer" / "weights" / "best.pt"
    for checkpoint in (older, newer):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"checkpoint")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    return project_root, older.resolve(), newer.resolve()


def test_yolo_export_discovers_and_selects_newest_best_checkpoint(tmp_path: Path) -> None:
    project_root, older, newer = _make_checkpoint_project(tmp_path)

    assert discover_yolo_checkpoints(project_root)[:2] == (newer, older)
    assert resolve_yolo_checkpoint(project_root=project_root) == newer
    assert resolve_yolo_checkpoint(newer.relative_to(project_root), project_root) == newer


def test_yolo_export_validates_format_specific_arguments() -> None:
    engine_arguments = supported_export_arguments("tensorrt")
    onnx_arguments = supported_export_arguments("onnx")

    assert {"imgsz", "device", "workspace", "quantize"} <= engine_arguments
    assert {"dynamic", "opset", "simplify"} <= onnx_arguments
    assert "workspace" not in onnx_arguments


def test_yolo_export_runs_with_verified_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project_root, _, checkpoint = _make_checkpoint_project(tmp_path)
    calls: dict[str, Any] = {}

    class FakeYOLO:
        def __init__(self, model: str) -> None:
            calls["model"] = model

        def export(self, **kwargs: Any) -> str:
            calls["arguments"] = kwargs
            output = checkpoint.with_suffix(".onnx")
            output.write_bytes(b"onnx")
            return str(output)

    monkeypatch.setattr("ultralytics.YOLO", FakeYOLO)
    output = export_yolo_model(
        checkpoint,
        "onnx",
        project_root=project_root,
        imgsz=320,
        dynamic=True,
        simplify=True,
    )

    assert output == checkpoint.with_suffix(".onnx")
    assert calls == {
        "model": str(checkpoint),
        "arguments": {"format": "onnx", "imgsz": 320, "dynamic": True, "simplify": True},
    }


def test_yolo_export_dashboard_constructs_with_current_formats(tmp_path: Path) -> None:
    widgets = pytest.importorskip("ipywidgets")
    project_root, _, _ = _make_checkpoint_project(tmp_path)
    dashboard = create_yolo_export_dashboard(project_root)

    assert isinstance(dashboard, widgets.VBox)
    format_controls = [
        control
        for row in dashboard.children
        if isinstance(row, widgets.HBox)
        for control in row.children
        if isinstance(control, widgets.Dropdown) and control.description == "Format"
    ]
    assert len(format_controls) == 1
    assert {value for _, value in format_controls[0].options} >= {"onnx", "engine", "openvino"}


def test_yolo_resume_applies_safe_worker_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    run_dir = tmp_path / "runs" / "yolo26" / "interrupted-run"
    checkpoint = run_dir / "weights" / "last.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    (run_dir / "results.csv").write_text("epoch,time\n44,11830\n", encoding="utf-8")
    selected_models: list[str] = []
    submitted_arguments: list[dict[str, object]] = []

    class FakeYOLO:
        def __init__(self, model_name: str) -> None:
            selected_models.append(model_name)
            self.trainer = None

        def train(self, **arguments: object) -> None:
            submitted_arguments.append(arguments)
            self.trainer = SimpleNamespace(save_dir=run_dir, metrics=None)

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)
    monkeypatch.setattr(train_module, "_runtime", lambda: {"python": "test", "cuda_available": False})

    result = train_yolo_export(
        root,
        "detect",
        TrainingOptions(resume=str(checkpoint), workers=0, output_dir=str(tmp_path / "unused")),
    )

    assert selected_models == [str(checkpoint.resolve())]
    assert submitted_arguments == [{"resume": True, "workers": 0}]
    assert Path(result["run_dir"]) == run_dir.resolve()
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert metadata["training_arguments"] == {"resume": str(checkpoint.resolve()), "workers": 0}


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

    def descendants(widget: object) -> list[object]:
        result = [widget]
        for child in getattr(widget, "children", ()):
            result.extend(descendants(child))
        return result

    controls = descendants(dashboard)
    descriptions = {getattr(control, "description", "") for control in controls}
    assert "Early-stop patience" in descriptions
    assert "Dataset cache" in descriptions
    assert "Checkpoint period" in descriptions
    assert "Copy-paste" in descriptions
    assert "I understand that training can run for a long time" in descriptions
    assert task_controls[0].style.button_width == "220px"
    assert all(
        getattr(control.layout, "width", None) == "100%"
        for control in controls
        if getattr(control, "description", "") in {"Early-stop patience", "Dataset cache", "Checkpoint period"}
    )


def test_dashboard_groups_cover_every_yolo_training_option() -> None:
    from dataclasses import fields

    grouped = {name for _, _, names in YOLO_PARAMETER_GROUPS for name in names}
    configurable = {item.name for item in fields(YoloTrainingOptions)} - {"extra_args"}
    assert grouped == configurable


def test_cli_accepts_advanced_yolo_training_parameters() -> None:
    from scripts.model_trainer import parser

    args = parser().parse_args(
        [
            "yolo-detect",
            "dataset",
            "--batch",
            "0.75",
            "--patience",
            "20",
            "--cache",
            "disk",
            "--no-plots",
            "--classes",
            "0,2",
            "--yolo-args-json",
            '{"end2end":true}',
        ]
    )
    assert args.batch == pytest.approx(0.75)
    assert args.patience == 20
    assert args.cache == "disk"
    assert args.plots is False
    assert args.classes == [0, 2]
    assert args.yolo_args_json == {"end2end": True}


def test_cli_accepts_official_dfine_training_parameters() -> None:
    from scripts.model_trainer import parser

    args = parser().parse_args(
        [
            "dfine-detect",
            "dataset",
            "--dfine-root",
            "upstream",
            "--imgsz",
            "768",
            "--dfine-base-lr",
            "0.001",
            "--dfine-backbone-lr",
            "0.0001",
            "--no-dfine-ema",
            "--dfine-config-json",
            '{"print_freq":25}',
        ]
    )
    assert args.imgsz == 768
    assert args.dfine_base_lr == pytest.approx(0.001)
    assert args.dfine_backbone_lr == pytest.approx(0.0001)
    assert args.dfine_ema is False
    assert args.dfine_config_json == {"print_freq": 25}


def test_dfine_seg_dry_run_generates_task_specific_config(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "segment")
    upstream = tmp_path / "D-FINE-seg"
    (upstream / "src" / "dl").mkdir(parents=True)
    (upstream / "src" / "dl" / "train.py").write_text("# fixture\n", encoding="utf-8")
    (upstream / "src" / "dl" / "dataset.py").write_text("# fixture\n", encoding="utf-8")
    (upstream / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "train": {
                    "ddp": {"enabled": False, "n_gpus": 1},
                    "mosaic_augs": {"mosaic_prob": None, "no_mosaic_epochs": 5},
                    "augs": {},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = train_dfine_seg_export(
        root,
        "segment",
        upstream,
        DFineSegOptions(
            model_size="s",
            epochs=12,
            batch=2,
            early_stopping=4,
            output_dir=str(tmp_path / "runs"),
            run_name="segment-fixture",
            extra_config={
                "train": {
                    "keep_ratio": True,
                    "freeze_except_mask": True,
                    "mask_batch_size": 48,
                    "augs": {"left_right_flip": 0.5, "rotation_p": 0.2},
                }
            },
        ),
        python_executable=sys.executable,
        execute=False,
    )

    config = yaml.safe_load(Path(result["config"]).read_text(encoding="utf-8"))
    assert config["task"] == "segment"
    assert config["model_name"] == "s"
    assert config["train"]["coco_dataset"] is True
    assert config["train"]["epochs"] == 12
    assert config["train"]["early_stopping"] == 4
    assert config["train"]["keep_ratio"] is True
    assert config["train"]["freeze_except_mask"] is True
    assert config["train"]["mask_batch_size"] == 48
    assert config["train"]["augs"]["left_right_flip"] == pytest.approx(0.5)
    assert config["train"]["augs"]["rotation_p"] == pytest.approx(0.2)
    assert config["train"]["label_to_name"] == {0: "Naartjie"}
    assert Path(result["prepared_dataset"], "train.json").is_file()
    assert result["executed"] is False


def test_dfine_notebook_covers_official_detection_and_dfine_seg() -> None:
    project_root = Path(__file__).resolve().parents[1]
    notebook_path = project_root / "notebooks" / "dfine_model_trainer_ui.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert notebook["nbformat"] == 4
    assert "create_dfine_training_dashboard" in source
    assert "object detection only" in source
    assert "class_id x1 y1 x2 y2" in source
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")


def test_dfine_dashboard_constructs_with_all_task_choices() -> None:
    widgets = pytest.importorskip("ipywidgets")
    project_root = Path(__file__).resolve().parents[1]
    dashboard = create_dfine_training_dashboard(project_root)

    assert isinstance(dashboard, widgets.VBox)
    task_controls = [child for child in dashboard.children if isinstance(child, widgets.RadioButtons)]
    assert len(task_controls) == 1
    assert tuple(task_controls[0].options) == DFINE_TASKS
    assert task_controls[0].layout.width == "100%"
    assert task_controls[0].value == "dfine_seg_segment"

    paths = dashboard.children[5]
    official_settings = dashboard.children[7]
    assert len({id(control.layout) for control in paths.children}) == len(paths.children)
    assert paths.children[0].layout.display != "none"  # dataset root
    assert Path(paths.children[0].value) == project_root / "dataset_seg_rgb"
    assert paths.children[1].layout.display == "none"  # inactive official checkout
    assert paths.children[3].layout.display != "none"  # D-FINE-seg checkout
    assert official_settings.layout.display == "none"
    seg_fields = {control.description: control for control in dashboard.children[8].children[0].children}
    assert seg_fields["Model size"].value == "s"
    assert seg_fields["Epochs"].value == 120
    assert seg_fields["Image size"].value == 640
    assert seg_fields["Physical batch size"].value == 2
    assert seg_fields["Early-stop patience"].value == 20
    advanced_fields = {
        control.description: control
        for control in dashboard.children[8].children[2].children[0].children[1].children
    }
    assert advanced_fields["Gradient accumulation"].value == 4
    assert advanced_fields["Mosaic probability"].value == "0"
    assert seg_fields["AMP dtype"].value == dfine_ui._preferred_amp_dtype()
    mask_fields = {
        control.description: control
        for control in dashboard.children[8].children[1].children[0].children[1].children
    }
    assert mask_fields["Mask evaluation batch"].value == 16
    assert mask_fields["Preserve image aspect ratio"].value is True
    task_controls[0].value = "official_detect"
    assert Path(paths.children[0].value) == project_root / "dataset_detect_rgb"
    assert paths.children[1].layout.display != "none"  # official checkout
    assert paths.children[3].layout.display == "none"  # inactive D-FINE-seg checkout
    assert paths.children[5].layout.display != "none"  # output directory
    assert all(control.layout.display != "none" for control in official_settings.children)
    essentials, advanced = official_settings.children
    batch = next(control for control in essentials.children if control.description == "Total batch size")
    assert isinstance(batch, widgets.BoundedIntText)
    assert batch.min == 1
    assert any(control.description == "Image size" for control in essentials.children)
    assert advanced.get_title(0) == "Official D-FINE advanced settings"
    assert advanced.selected_index == 0
    paths.children[0].value = str(project_root / "my_custom_dataset")
    task_controls[0].value = "dfine_seg_segment"
    assert Path(paths.children[0].value) == project_root / "my_custom_dataset"
    progress_panel = dashboard.children[14]
    assert isinstance(progress_panel, widgets.VBox)
    assert progress_panel.layout.display == "none"
    assert isinstance(progress_panel.children[0], widgets.IntProgress)

    def descendants(control: Any) -> list[Any]:
        return [control, *(child for item in getattr(control, "children", ()) for child in descendants(item))]

    setting_types = (
        widgets.Text, widgets.Textarea, widgets.IntText, widgets.FloatText,
        widgets.Dropdown, widgets.RadioButtons, widgets.Checkbox,
    )
    assert all(
        control.tooltip
        for control in descendants(dashboard)
        if isinstance(control, setting_types) and control.description
    )


def test_dfine_dashboard_segmentation_controls_reach_training_config(monkeypatch: Any) -> None:
    pytest.importorskip("ipywidgets")
    dashboard = create_dfine_training_dashboard()
    task = dashboard.children[2]
    segment_settings = dashboard.children[8].children[1]
    assert segment_settings.layout.display != "none"
    assert segment_settings.selected_index == 0
    controls = {
        control.description: control
        for control in segment_settings.children[0].children[1].children
    }
    controls["Preserve image aspect ratio"].value = True
    controls["Train mask head only"].value = True
    controls["Mask evaluation batch"].value = 48
    controls["Horizontal flip chance"].value = 0.6
    controls["Rotation chance"].value = 0.2
    controls["Max rotation degrees"].value = 15
    controls["Blur chance"].value = 0.1
    assert "--extra-config-json" in dashboard.children[9].value
    assert "mask_batch_size" in dashboard.children[9].value

    captured: dict[str, Any] = {}

    def fake_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["task"] = args[1]
        captured["options"] = args[3]
        captured["execute"] = kwargs["execute"]
        return {"config": "fixture.yaml"}

    monkeypatch.setattr(dfine_ui, "train_dfine_seg_export", fake_train)
    dashboard.children[10].children[2].click()  # Prepare dry run

    assert captured["task"] == "segment"
    assert captured["execute"] is False
    train_config = captured["options"].extra_config["train"]
    assert train_config["keep_ratio"] is True
    assert train_config["freeze_except_mask"] is True
    assert train_config["mask_batch_size"] == 48
    assert train_config["augs"]["left_right_flip"] == pytest.approx(0.6)
    assert train_config["augs"]["rotation_p"] == pytest.approx(0.2)
    assert train_config["augs"]["rotation_degree"] == pytest.approx(15)
    assert train_config["augs"]["blur"] == pytest.approx(0.1)

    task.value = "dfine_seg_detect"
    assert segment_settings.layout.display == "none"
    assert "mask_batch_size" not in dashboard.children[9].value


def test_dfine_dashboard_official_controls_reach_training_options(monkeypatch: Any) -> None:
    pytest.importorskip("ipywidgets")
    dashboard = create_dfine_training_dashboard()
    dashboard.children[2].value = "official_detect"
    essentials, advanced = dashboard.children[7].children
    fields = {control.description: control for control in essentials.children}
    fields["Image size"].value = 768
    advanced_fields = {
        control.description: control for control in advanced.children[0].children[1].children
    }
    advanced_fields["Base learning rate"].value = "0.001"
    advanced_fields["Backbone learning rate"].value = "0.0001"
    advanced_fields["Warmup steps"].value = 200
    advanced_fields["Exponential moving average"].value = False
    advanced.children[0].children[2].value = '{"print_freq":25}'

    captured: dict[str, Any] = {}

    def fake_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured["options"] = args[2]
        captured["execute"] = kwargs["execute"]
        return {"config": "fixture.yml"}

    monkeypatch.setattr(dfine_ui, "train_dfine_export", fake_train)
    dashboard.children[10].children[2].click()  # Prepare dry run

    options = captured["options"]
    assert captured["execute"] is False
    assert options.imgsz == 768
    assert options.dfine_base_lr == pytest.approx(0.001)
    assert options.dfine_backbone_lr == pytest.approx(0.0001)
    assert options.dfine_warmup_steps == 200
    assert options.dfine_use_ema is False
    assert options.dfine_extra_config == {"print_freq": 25}
    preview = dashboard.children[9].value
    assert "--imgsz" in preview
    assert "--dfine-base-lr" in preview
    assert "--no-dfine-ema" in preview


def test_dfine_dashboard_summarizes_negative_images(monkeypatch: Any) -> None:
    pytest.importorskip("ipywidgets")
    dashboard = create_dfine_training_dashboard()
    dashboard.children[2].value = "dfine_seg_segment"
    warnings = [f"segment negative image: train/image-{i}.png" for i in range(43)]
    monkeypatch.setattr(
        dfine_ui,
        "validate_yolo_training_dataset",
        lambda *_: {"ok": True, "errors": [], "warnings": warnings, "discovery": {}},
    )

    dashboard.children[10].children[0].click()  # Validate dataset

    assert "43 images have empty or missing labels" in dashboard.children[13].value


def test_official_dfine_rejects_automatic_batch_value() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        train_dfine_export(
            "unused-dataset",
            "unused-upstream",
            TrainingOptions(batch=-1),
            execute=False,
        )


def test_training_process_forwards_live_stdout_and_stderr(tmp_path: Path) -> None:
    streamed: list[str] = []
    console_log = tmp_path / "trainer_console.log"
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"

    train_module._run_training_process(
        [
            sys.executable,
            "-u",
            "-c",
            (
                "import sys; "
                "print('Epoch: [0/2]  [ 0/10] eta: 0:00:10'); "
                "sys.stderr.write('Epoch 1/2: 50%\\r'); sys.stderr.flush()"
            ),
        ],
        cwd=tmp_path,
        environment=environment,
        output_callback=streamed.append,
        output_log=console_log,
    )

    combined = "".join(streamed)
    assert "Epoch: [0/2]" in combined
    assert "Epoch 1/2: 50%" in combined
    assert "Epoch: [0/2]" in console_log.read_text(encoding="utf-8")


def test_dfine_dashboard_runs_training_without_blocking_widget_loop(monkeypatch: Any) -> None:
    pytest.importorskip("ipywidgets")
    dashboard = create_dfine_training_dashboard()
    confirmation = dashboard.children[11]
    train_button = dashboard.children[12]
    status = dashboard.children[13]
    progress_panel = dashboard.children[14]

    def fake_train(*args: Any, **kwargs: Any) -> dict[str, Any]:
        callback = kwargs["output_callback"]
        callback("Trainer console log: fixture.log\n")
        time.sleep(0.3)
        callback("Epoch 1/2: 50%\r")
        return {"test_metrics": None, "console_log": "fixture.log"}

    monkeypatch.setattr(dfine_ui, "train_dfine_seg_export", fake_train)

    async def exercise_dashboard() -> None:
        confirmation.value = True
        started = time.monotonic()
        train_button.click()
        assert time.monotonic() - started < 0.15
        assert train_button.disabled is True
        await asyncio.sleep(0.7)
        assert "completed successfully" in status.value
        assert progress_panel.children[0].value == 100
        assert train_button.disabled is False

    asyncio.run(exercise_dashboard())


def test_dfine_dry_run_uses_official_nano_config(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    upstream = tmp_path / "D-FINE"
    (upstream / "src" / "data" / "dataset").mkdir(parents=True)
    (upstream / "configs" / "dfine" / "custom").mkdir(parents=True)
    (upstream / "train.py").write_text("# fixture\n", encoding="utf-8")
    (upstream / "src" / "data" / "dataset" / "coco_dataset.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    (upstream / "configs" / "dfine" / "include").mkdir(parents=True)
    (upstream / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_custom.yml").write_text(
        yaml.safe_dump(
            {
                "task": "detection",
                "optimizer": {
                    "params": [
                        {"params": "backbone", "lr": 0.0004},
                        {"params": "encoder", "weight_decay": 0},
                    ]
                },
                "train_dataloader": {
                    "dataset": {"transforms": {"ops": [{"type": "Resize", "size": [640, 640]}]}}
                },
            }
        ),
        encoding="utf-8",
    )
    (upstream / "configs" / "dfine" / "include" / "dataloader.yml").write_text(
        yaml.safe_dump(
            {"val_dataloader": {"dataset": {"transforms": {"ops": [{"type": "Resize", "size": [640, 640]}]}}}}
        ),
        encoding="utf-8",
    )
    result = train_dfine_export(
        root,
        upstream,
        TrainingOptions(
            device="cpu",
            imgsz=768,
            workers=0,
            output_dir=str(tmp_path / "runs"),
            run_name="official-fixture",
            dfine_base_lr=0.001,
            dfine_backbone_lr=0.0001,
            dfine_weight_decay=0.0002,
            dfine_clip_max_norm=0.2,
            dfine_warmup_steps=200,
            dfine_use_ema=False,
            dfine_extra_config={"print_freq": 25},
        ),
        execute=False,
    )
    config = yaml.safe_load(Path(result["config"]).read_text(encoding="utf-8"))
    assert "dfine_hgnetv2_n_custom.yml" in config["__include__"][0]
    assert config["remap_mscoco_category"] is False
    assert config["print_freq"] == 25
    assert config["eval_spatial_size"] == [768, 768]
    assert config["train_dataloader"]["dataset"]["transforms"]["ops"][0]["size"] == [768, 768]
    assert config["val_dataloader"]["dataset"]["transforms"]["ops"][0]["size"] == [768, 768]
    assert config["train_dataloader"]["collate_fn"]["base_size"] == 768
    assert config["optimizer"]["lr"] == pytest.approx(0.001)
    assert config["optimizer"]["params"][0]["lr"] == pytest.approx(0.0001)
    assert config["optimizer"]["weight_decay"] == pytest.approx(0.0002)
    assert config["clip_max_norm"] == pytest.approx(0.2)
    assert config["lr_warmup_scheduler"]["warmup_duration"] == 200
    assert config["use_ema"] is False
    assert Path(result["prepared_dataset"], "train.json").is_file()
    assert "prepared_dataset" in config["train_dataloader"]["dataset"]["img_folder"]


def test_dfine_resume_stages_stage1_checkpoint_in_new_run(tmp_path: Path) -> None:
    root = make_yolo_dataset(tmp_path, "detect")
    upstream = tmp_path / "D-FINE"
    (upstream / "src" / "data" / "dataset").mkdir(parents=True)
    (upstream / "configs" / "dfine" / "custom").mkdir(parents=True)
    (upstream / "train.py").write_text("# fixture\n", encoding="utf-8")
    (upstream / "src" / "data" / "dataset" / "coco_dataset.py").write_text(
        "# fixture\n", encoding="utf-8"
    )
    (upstream / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_custom.yml").write_text(
        "task: detection\n", encoding="utf-8"
    )
    source_run = tmp_path / "previous-run"
    source_run.mkdir()
    checkpoint = source_run / "last.pth"
    checkpoint.write_bytes(b"last stage-one state")
    (source_run / "best_stg1.pth").write_bytes(b"best stage-one state")

    result = train_dfine_export(
        root,
        upstream,
        TrainingOptions(
            device="cpu",
            workers=0,
            output_dir=str(tmp_path / "runs"),
            run_name="resumed-run",
            resume=str(checkpoint),
        ),
        execute=False,
    )

    staged = Path(result["resume_stage1"])
    assert staged == Path(result["run_dir"]) / "best_stg1.pth"
    assert staged.read_bytes() == b"best stage-one state"
    assert not staged.samefile(source_run / "best_stg1.pth")
    metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
    assert metadata["artifacts"]["resume_stage1"] == str(staged)


def test_trainer_has_no_inference_frontend_or_camera_backend() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert not (repository / "foreground" / "package.json").exists()
    assert not (repository / "background" / "src" / "CameraManager.cpp").exists()
