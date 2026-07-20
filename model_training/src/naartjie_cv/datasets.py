from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


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


def _validate_yolo(bundle: TrainingExport, yaml_path: Path, task: str, errors: list[str], warnings: list[str]) -> dict[int, str]:
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    names = _names(payload)
    if list(names) != list(range(len(names))):
        errors.append(f"{yaml_path.relative_to(bundle.root)} class IDs are not zero-based and contiguous")
    dataset_root = yaml_path.parent
    for split in ("train", "val", "test"):
        configured = payload.get(split)
        if not configured:
            continue
        image_dir = (dataset_root / configured).resolve()
        try:
            image_dir.relative_to(dataset_root.resolve())
        except ValueError:
            errors.append(f"{yaml_path.relative_to(bundle.root)} {split} path escapes its portable dataset root")
            continue
        if not image_dir.is_dir():
            errors.append(f"{task} {split} image directory is missing: {image_dir}")
            continue
        images = sorted(path for path in image_dir.iterdir() if path.is_file())
        if split in {"train", "val"} and not images:
            errors.append(f"{task} {split} split has no images")
        label_dir = dataset_root / "labels" / split
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                warnings.append(f"{task} negative image has no label file: {image.name}")
                continue
            rows = [line.split() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not rows:
                warnings.append(f"{task} negative image: {split}/{image.name}")
            for row_number, row in enumerate(rows, start=1):
                expected = len(row) == 5 if task == "detect" else len(row) >= 7 and len(row[1:]) % 2 == 0
                if not expected:
                    errors.append(f"malformed {task} label {label.relative_to(bundle.root)}:{row_number}")
                    continue
                try:
                    class_id = int(row[0])
                    coordinates = [float(value) for value in row[1:]]
                except ValueError:
                    errors.append(f"non-numeric {task} label {label.relative_to(bundle.root)}:{row_number}")
                    continue
                if class_id not in names:
                    errors.append(f"unknown class {class_id} in {label.relative_to(bundle.root)}:{row_number}")
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    errors.append(f"out-of-range coordinate in {label.relative_to(bundle.root)}:{row_number}")
                if task == "detect" and (coordinates[2] <= 0 or coordinates[3] <= 0):
                    errors.append(f"non-positive box size in {label.relative_to(bundle.root)}:{row_number}")
    return names


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
    detect_names = _validate_yolo(bundle, bundle.yolo_detection, "detect", errors, warnings)
    segment_names = _validate_yolo(bundle, bundle.yolo_segmentation, "segment", errors, warnings)
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

    bundle = require_valid_training_export(root)
    yaml_path = bundle.yolo_detection if task == "detect" else bundle.yolo_segmentation
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = _names(payload)
    dataset = YOLODataset(
        img_path=str(yaml_path.parent / payload["train"]),
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
