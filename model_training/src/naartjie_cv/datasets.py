from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from PIL import Image


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


def _link_or_copy(source: Path, destination: Path) -> str:
    """Create a storage-efficient dataset view, falling back to a normal copy."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            x1 * y2 - x2 * y1
            for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
        )
    ) / 2.0


def prepare_dfine_seg_dataset(
    root: str | Path,
    task: Literal["detect", "segment"],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Stage a validated YOLO dataset as COCO JSON for D-FINE-seg.

    D-FINE-seg can read YOLO text labels directly, but its YOLO loader expects a
    flat images/labels directory plus generated CSV split files. Converting the
    existing data.yaml splits to COCO JSON preserves the source export, supports
    nested train/val directories, and works for both boxes and polygons.
    """
    task = _yolo_task(task)
    yaml_path = require_valid_yolo_training_dataset(root, task)
    validation_root = Path(root).expanduser().resolve()
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    errors: list[str] = []
    dataset_root = _yolo_dataset_root(validation_root, yaml_path, payload, errors)
    if dataset_root is None:
        raise ValueError("YOLO dataset path resolution failed:\n- " + "\n- ".join(errors))

    names = _names(payload)
    prepared_root = Path(output_dir).expanduser().resolve()
    try:
        prepared_root.relative_to(validation_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "D-FINE-seg staging output must be outside the labelled source dataset"
        )
    prepared_root.mkdir(parents=True, exist_ok=True)
    categories = [
        {"id": class_id, "name": name, "supercategory": ""}
        for class_id, name in names.items()
    ]
    split_reports: dict[str, Any] = {}
    link_modes: set[str] = set()

    for split in ("train", "val", "test"):
        configured = payload.get(split)
        if not configured:
            continue
        configured_path = Path(str(configured)).expanduser()
        image_dir = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (dataset_root / configured_path).resolve()
        )
        images = sorted(
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in YOLO_IMAGE_SUFFIXES
        )
        coco_images: list[dict[str, Any]] = []
        coco_annotations: list[dict[str, Any]] = []
        annotation_id = 1

        for image_id, image_path in enumerate(images, start=1):
            relative_image = image_path.relative_to(image_dir)
            staged_relative = Path(split) / relative_image
            staged_image = prepared_root / "images" / staged_relative
            link_modes.add(_link_or_copy(image_path, staged_image))
            with Image.open(image_path) as opened:
                width, height = opened.size
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": staged_relative.as_posix(),
                    "width": width,
                    "height": height,
                }
            )

            label_path = _matching_label_path(image_path, dataset_root, split)
            if not label_path.is_file():
                continue
            for row_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                values = line.split()
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
                if task == "detect":
                    xc, yc, box_width, box_height = coordinates
                    # YOLO labels are commonly rounded to six decimals.  A box that
                    # touches an image edge can therefore become a few millionths of
                    # a pixel negative after xywh -> xyxy conversion.  Albumentations
                    # 2 validates those values strictly, so keep the staged COCO box
                    # inside the actual image bounds before D-FINE-seg loads it.
                    half_width = box_width * width / 2.0
                    half_height = box_height * height / 2.0
                    x1 = min(float(width), max(0.0, xc * width - half_width))
                    y1 = min(float(height), max(0.0, yc * height - half_height))
                    x2 = min(float(width), max(0.0, xc * width + half_width))
                    y2 = min(float(height), max(0.0, yc * height + half_height))
                    pixel_width = x2 - x1
                    pixel_height = y2 - y1
                    if pixel_width <= 0 or pixel_height <= 0:
                        raise ValueError(f"Degenerate detection box in {label_path}:{row_number}")
                    bbox = [x1, y1, pixel_width, pixel_height]
                    area = pixel_width * pixel_height
                    segmentation: list[list[float]] = []
                else:
                    points = [
                        (
                            min(float(width), max(0.0, coordinates[index] * width)),
                            min(float(height), max(0.0, coordinates[index + 1] * height)),
                        )
                        for index in range(0, len(coordinates), 2)
                    ]
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
                    area = _polygon_area(points)
                    if area <= 0 or bbox[2] <= 0 or bbox[3] <= 0:
                        raise ValueError(
                            f"Degenerate segmentation polygon in {label_path}:{row_number}"
                        )
                    segmentation = [[coordinate for point in points for coordinate in point]]
                coco_annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    }
                )
                annotation_id += 1

        annotation_path = prepared_root / f"{split}.json"
        annotation_path.write_text(
            json.dumps(
                {
                    "info": {
                        "description": f"Model Trainer D-FINE-seg {task} staging dataset",
                        "source_data_yaml": str(yaml_path),
                    },
                    "images": coco_images,
                    "annotations": coco_annotations,
                    "categories": categories,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        split_reports[split] = {
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "json": str(annotation_path),
        }

    return {
        "root": str(prepared_root),
        "source_root": str(validation_root),
        "source_data_yaml": str(yaml_path),
        "task": task,
        "format": "COCO JSON with bbox" + (" and polygon segmentation" if task == "segment" else ""),
        "classes": names,
        "splits": split_reports,
        "image_materialization": sorted(link_modes),
    }


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


def dfine_loader_smoke(
    root: str | Path,
    upstream_dir: str | Path,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Instantiate the current official D-FINE CocoDetection and load one item."""
    upstream = Path(upstream_dir).expanduser().resolve()
    loader_file = upstream / "src" / "data" / "dataset" / "coco_dataset.py"
    if not loader_file.is_file():
        raise FileNotFoundError(f"Official D-FINE loader is missing: {loader_file}")
    with tempfile.TemporaryDirectory(prefix="model-trainer-dfine-smoke-") as temporary:
        prepared = prepare_dfine_seg_dataset(root, "detect", Path(temporary) / "prepared")
        dataset = Path(prepared["root"])
        script = (
            "import json; "
            "from src.data.dataset.coco_dataset import CocoDetection; "
            f"d=CocoDetection(r'{dataset / 'images'}',r'{dataset / 'train.json'}',None,False,False); "
            "image,target=d.load_item(0); "
            "print(json.dumps({'images':len(d),'labels':target['labels'].tolist(),'boxes':list(target['boxes'].shape)}))"
        )
        completed = subprocess.run(
            [str(python_executable or sys.executable), "-c", script],
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
