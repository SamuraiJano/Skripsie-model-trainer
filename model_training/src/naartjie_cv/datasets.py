from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


YoloTask = Literal["detect", "segment"]
YOLO_IMAGE_SUFFIXES = {
    ".bmp",
    ".dng",
    ".heic",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".pfm",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class TrainingExport:
    root: Path
    manifest: Path
    yolo_detection: Path
    yolo_segmentation: Path
    dfine_detection: Path
    synthetic_assets: Path

    def to_dict(self) -> dict[str, str]:
        return {key: str(value) for key, value in asdict(self).items()}


def discover_training_export(root: str | Path) -> TrainingExport:
    export_root = Path(root).expanduser().resolve()
    if not export_root.is_dir():
        raise FileNotFoundError(f"Training export root does not exist: {export_root}")
    discovered = TrainingExport(
        root=export_root,
        manifest=export_root / "export_manifest.json",
        yolo_detection=export_root / "yolo26_detection" / "data.yaml",
        yolo_segmentation=export_root / "yolo26_segmentation" / "data.yaml",
        dfine_detection=export_root / "dfine_detection" / "dataset_info.yaml",
        synthetic_assets=export_root / "synthetic_assets" / "assets_manifest.json",
    )
    missing = [str(path.relative_to(export_root)) for path in asdict(discovered).values() if isinstance(path, Path) and path != export_root and not path.is_file()]
    if missing:
        raise FileNotFoundError("Export is incomplete; missing: " + ", ".join(missing))
    return discovered


def _names(payload: dict[str, Any]) -> dict[int, str]:
    value = payload.get("names") or {}
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    return dict(sorted((int(index), str(name)) for index, name in value.items()))


def _yolo_task(task: str) -> YoloTask:
    if task not in {"detect", "segment"}:
        raise ValueError(f"YOLO task must be 'detect' or 'segment'; received {task!r}")
    return task


def _display_path(path: Path, validation_root: Path) -> str:
    try:
        return str(path.relative_to(validation_root))
    except ValueError:
        return str(path)


def _yolo_data_yaml_candidates(root: Path, task: YoloTask) -> list[Path]:
    nested = "yolo26_detection" if task == "detect" else "yolo26_segmentation"
    return [root / "data.yaml", root / nested / "data.yaml"]


def _yolo_dataset_root(
    validation_root: Path,
    yaml_path: Path,
    payload: dict[str, Any],
    errors: list[str],
) -> Path | None:
    configured = payload.get("path")
    dataset_root = yaml_path.parent.resolve()
    if configured not in (None, ""):
        configured_path = Path(str(configured)).expanduser()
        dataset_root = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (yaml_path.parent / configured_path).resolve()
        )
    try:
        dataset_root.relative_to(validation_root.resolve())
    except ValueError:
        errors.append(f"{_display_path(yaml_path, validation_root)} path escapes its validation root")
        return None
    return dataset_root


def _matching_label_path(image: Path, dataset_root: Path, split: str) -> Path:
    relative = image.relative_to(dataset_root)
    parts = list(relative.parts)
    if "images" in parts:
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        return (dataset_root.joinpath(*parts)).with_suffix(".txt")
    return dataset_root / "labels" / split / f"{image.stem}.txt"


def _validate_yolo(
    validation_root: Path,
    yaml_path: Path,
    task: str,
    errors: list[str],
    warnings: list[str],
) -> dict[int, str]:
    task = _yolo_task(task)
    try:
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"cannot read {_display_path(yaml_path, validation_root)}: {exc}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{_display_path(yaml_path, validation_root)} must contain a YAML mapping")
        return {}
    try:
        names = _names(payload)
    except (AttributeError, TypeError, ValueError) as exc:
        errors.append(f"{_display_path(yaml_path, validation_root)} has invalid class names: {exc}")
        names = {}
    if not names:
        errors.append(f"{_display_path(yaml_path, validation_root)} has no classes")
    if list(names) != list(range(len(names))):
        errors.append(f"{_display_path(yaml_path, validation_root)} class IDs are not zero-based and contiguous")
    for required_split in ("train", "val"):
        if not payload.get(required_split):
            errors.append(f"{task} {required_split} split is not configured in {_display_path(yaml_path, validation_root)}")

    dataset_root = _yolo_dataset_root(validation_root, yaml_path, payload, errors)
    if dataset_root is None:
        return names
    for split in ("train", "val", "test"):
        configured = payload.get(split)
        if not configured:
            continue
        if not isinstance(configured, (str, Path)):
            errors.append(f"{task} {split} split must configure one image directory")
            continue
        configured_path = Path(configured).expanduser()
        image_dir = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (dataset_root / configured_path).resolve()
        )
        try:
            image_dir.relative_to(dataset_root)
        except ValueError:
            errors.append(f"{_display_path(yaml_path, validation_root)} {split} path escapes its dataset root")
            continue
        if not image_dir.is_dir():
            errors.append(f"{task} {split} image directory is missing: {image_dir}")
            continue
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in YOLO_IMAGE_SUFFIXES
        )
        if split in {"train", "val"} and not images:
            errors.append(f"{task} {split} split has no images")
        for image in images:
            label = _matching_label_path(image, dataset_root, split)
            if not label.exists():
                warnings.append(f"{task} negative image has no label file: {image.name}")
                continue
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                warnings.append(f"{task} negative image: {split}/{image.name}")
            for row_number, row in enumerate(rows, start=1):
                expected = len(row) == 5 if task == "detect" else len(row) >= 7 and len(row[1:]) % 2 == 0
                if not expected:
                    errors.append(f"malformed {task} label {_display_path(label, validation_root)}:{row_number}")
                    continue
                try:
                    class_id = int(row[0])
                    coordinates = [float(value) for value in row[1:]]
                except ValueError:
                    errors.append(f"non-numeric {task} label {_display_path(label, validation_root)}:{row_number}")
                    continue
                if class_id not in names:
                    errors.append(f"unknown class {class_id} in {_display_path(label, validation_root)}:{row_number}")
                if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in coordinates):
                    errors.append(f"out-of-range coordinate in {_display_path(label, validation_root)}:{row_number}")
                if task == "detect" and (coordinates[2] <= 0 or coordinates[3] <= 0):
                    errors.append(f"non-positive box size in {_display_path(label, validation_root)}:{row_number}")
    return names


def validate_yolo_training_dataset(root: str | Path, task: str) -> dict[str, Any]:
    task = _yolo_task(task)
    validation_root = Path(root).expanduser().resolve()
    discovery: dict[str, Any] = {
        "root": str(validation_root),
        "data_yaml": None,
        "task": task,
        "names": {},
    }
    if not validation_root.is_dir():
        return {
            "ok": False,
            "errors": [f"YOLO dataset root does not exist: {validation_root}"],
            "warnings": [],
            "discovery": discovery,
        }

    candidates = _yolo_data_yaml_candidates(validation_root, task)
    yaml_path = next((path for path in candidates if path.is_file()), None)
    if yaml_path is None:
        checked = ", ".join(_display_path(path, validation_root) for path in candidates)
        return {
            "ok": False,
            "errors": [f"YOLO {task} data.yaml is missing; checked: {checked}"],
            "warnings": [],
            "discovery": discovery,
        }

    errors: list[str] = []
    warnings: list[str] = []
    names = _validate_yolo(validation_root, yaml_path, task, errors, warnings)
    discovery.update({"data_yaml": str(yaml_path), "names": names})
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": sorted(set(warnings)),
        "discovery": discovery,
    }


def require_valid_yolo_training_dataset(root: str | Path, task: str) -> Path:
    report = validate_yolo_training_dataset(root, task)
    if not report["ok"]:
        raise ValueError(f"YOLO {task} dataset validation failed:\n- " + "\n- ".join(report["errors"]))
    return Path(report["discovery"]["data_yaml"])


def _validate_dfine(bundle: TrainingExport, expected_names: dict[int, str], errors: list[str], warnings: list[str]) -> None:
    info = yaml.safe_load(bundle.dfine_detection.read_text(encoding="utf-8")) or {}
    if info.get("remap_mscoco_category") is not False:
        errors.append("D-FINE dataset_info.yaml must set remap_mscoco_category: false")
    if str(info.get("category_id_policy")) != "zero_based_contiguous":
        errors.append("D-FINE category policy must be zero_based_contiguous")
    annotations = info.get("annotations") or {}
    for split, relative in annotations.items():
        path = bundle.dfine_detection.parent / str(relative)
        if not path.is_file():
            errors.append(f"D-FINE {split} annotations are missing: {relative}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        if categories != expected_names:
            errors.append(f"D-FINE {split} categories {categories} do not match detection classes {expected_names}")
        image_ids = {int(item["id"]) for item in payload.get("images", [])}
        for image in payload.get("images", []):
            file_path = bundle.dfine_detection.parent / "images" / str(split) / str(image["file_name"])
            if not file_path.is_file():
                errors.append(f"D-FINE image is missing: {file_path.relative_to(bundle.root)}")
        if not payload.get("annotations"):
            warnings.append(f"D-FINE {split} has no positive boxes")
        for annotation in payload.get("annotations", []):
            bbox = annotation.get("bbox") or []
            if int(annotation.get("image_id", -1)) not in image_ids:
                errors.append(f"D-FINE {split} annotation {annotation.get('id')} has an unknown image_id")
            if int(annotation.get("category_id", -1)) not in categories:
                errors.append(f"D-FINE {split} annotation {annotation.get('id')} has an unknown category_id")
            if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                errors.append(f"D-FINE {split} annotation {annotation.get('id')} has an invalid pixel xywh bbox")
            if float(annotation.get("area", -1)) <= 0 or int(annotation.get("iscrowd", -1)) != 0:
                errors.append(f"D-FINE {split} annotation {annotation.get('id')} has invalid area/iscrowd")


def validate_training_export(root: str | Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        bundle = discover_training_export(root)
    except (FileNotFoundError, ValueError) as exc:
        return {"ok": False, "errors": [str(exc)], "warnings": [], "discovery": None}
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    enabled = list(manifest.get("split", {}).get("enabled") or [])
    if "train" not in enabled or "val" not in enabled:
        errors.append("export_manifest.json must contain non-empty train and val splits")
    if "test" not in enabled:
        for directory in bundle.root.glob("*/images/test"):
            errors.append(f"empty/disabled test artifact should not exist: {directory.relative_to(bundle.root)}")
    detect_names = _validate_yolo(bundle.root, bundle.yolo_detection, "detect", errors, warnings)
    segment_names = _validate_yolo(bundle.root, bundle.yolo_segmentation, "segment", errors, warnings)
    manifest_detection = {
        int(item["export_index"]): str(item["name"])
        for item in manifest.get("class_mappings", {}).get("detection", [])
    }
    manifest_segmentation = {
        int(item["export_index"]): str(item["name"])
        for item in manifest.get("class_mappings", {}).get("segmentation", [])
    }
    if detect_names != manifest_detection:
        errors.append("YOLO detection names do not match export_manifest.json")
    if segment_names != manifest_segmentation:
        errors.append("YOLO segmentation names do not match export_manifest.json")
    _validate_dfine(bundle, detect_names, errors, warnings)
    assets = json.loads(bundle.synthetic_assets.read_text(encoding="utf-8"))
    for kind in ("foregrounds", "backgrounds"):
        for item in assets.get(kind, []):
            path = safe_asset = (bundle.synthetic_assets.parent / str(item["path"])).resolve()
            try:
                safe_asset.relative_to(bundle.synthetic_assets.parent.resolve())
            except ValueError:
                errors.append(f"synthetic asset path escapes export: {item['path']}")
                continue
            if not path.is_file():
                errors.append(f"synthetic asset is missing: {item['path']}")
    return {"ok": not errors, "errors": errors, "warnings": sorted(set(warnings)), "discovery": bundle.to_dict(), "manifest": manifest}


def require_valid_training_export(root: str | Path) -> TrainingExport:
    report = validate_training_export(root)
    if not report["ok"]:
        raise ValueError("Training export validation failed:\n- " + "\n- ".join(report["errors"]))
    return discover_training_export(root)


def ultralytics_loader_smoke(root: str | Path, task: str) -> dict[str, Any]:
    """Load one real sample through the installed Ultralytics dataset class."""
    from ultralytics.data.dataset import YOLODataset

    yaml_path = require_valid_yolo_training_dataset(root, task)
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = _names(payload)
    errors: list[str] = []
    dataset_root = _yolo_dataset_root(Path(root).expanduser().resolve(), yaml_path, payload, errors)
    if dataset_root is None:
        raise ValueError("YOLO dataset path resolution failed:\n- " + "\n- ".join(errors))
    train_path = Path(str(payload["train"])).expanduser()
    image_dir = train_path.resolve() if train_path.is_absolute() else (dataset_root / train_path).resolve()
    dataset = YOLODataset(
        img_path=str(image_dir),
        data={"names": names, "nc": len(names)},
        task=task,
        imgsz=64,
        augment=False,
        cache=False,
        batch_size=1,
    )
    if not len(dataset):
        raise ValueError(f"Ultralytics {task} loader found no training images")
    sample = dataset.get_image_and_label(0)
    return {"task": task, "images": len(dataset), "sample_shape": list(sample["img"].shape), "classes": names}


def dfine_loader_smoke(root: str | Path, upstream_dir: str | Path) -> dict[str, Any]:
    """Instantiate the current official D-FINE CocoDetection and load one item."""
    bundle = require_valid_training_export(root)
    upstream = Path(upstream_dir).expanduser().resolve()
    loader_file = upstream / "src" / "data" / "dataset" / "coco_dataset.py"
    if not loader_file.is_file():
        raise FileNotFoundError(f"Official D-FINE loader is missing: {loader_file}")
    dataset = bundle.dfine_detection.parent
    script = (
        "import json; "
        "from src.data.dataset.coco_dataset import CocoDetection; "
        f"d=CocoDetection(r'{dataset / 'images' / 'train'}',r'{dataset / 'annotations' / 'instances_train.json'}',None,False,False); "
        "image,target=d.load_item(0); "
        "print(json.dumps({'images':len(d),'labels':target['labels'].tolist(),'boxes':list(target['boxes'].shape)}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=upstream,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"Official D-FINE loader smoke failed with exit code {completed.returncode}:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    line = next((value for value in reversed(completed.stdout.splitlines()) if value.startswith("{")), "")
    if not line:
        raise RuntimeError(f"D-FINE loader smoke did not return JSON:\n{completed.stdout}\n{completed.stderr}")
    result = json.loads(line)
    result["upstream"] = str(upstream)
    return result
