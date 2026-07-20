from __future__ import annotations

import csv
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from .datasets import require_valid_training_export, require_valid_yolo_training_dataset


@dataclass
class TrainingOptions:
    epochs: int = 100
    imgsz: int = 640
    batch: int | float = 16
    workers: int = 8
    seed: int = 42
    device: str = "0"
    amp: bool = True
    output_dir: str = "runs"
    run_name: str | None = None
    resume: str | None = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _json_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (RuntimeError, TypeError, ValueError):
            return None
        if isinstance(scalar, bool):
            return int(scalar)
        if isinstance(scalar, int):
            return scalar
        if isinstance(scalar, float):
            return scalar if math.isfinite(scalar) else None
    return None


def _metric_values(source: Any) -> dict[str, int | float]:
    mapping = source if isinstance(source, Mapping) else getattr(source, "results_dict", None)
    if not isinstance(mapping, Mapping):
        return {}
    metrics: dict[str, int | float] = {}
    for key, value in mapping.items():
        scalar = _json_number(value)
        if scalar is not None:
            metrics[str(key)] = scalar
    return metrics


def _final_csv_values(path: Path) -> dict[str, int | float]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    metrics: dict[str, int | float] = {}
    for key, value in rows[-1].items():
        if key is None or value is None or not value.strip():
            continue
        try:
            scalar = float(value.strip())
        except ValueError:
            continue
        if math.isfinite(scalar):
            metrics[key.strip()] = scalar
    return metrics


def _ultralytics_final_metrics(results: Any, trainer: Any, results_csv: Path) -> dict[str, int | float]:
    metrics: dict[str, int | float] = {}
    for source in (results, getattr(trainer, "metrics", None)):
        metrics.update(_metric_values(source))
    for key, value in _final_csv_values(results_csv).items():
        metrics.setdefault(key, value)
    return metrics


def _yolo_dataset_metadata(export_root: str | Path, data_yaml: Path, task: str) -> dict[str, Any]:
    root = Path(export_root).expanduser().resolve()
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    raw_names = payload.get("names") or {}
    names = (
        {index: str(name) for index, name in enumerate(raw_names)}
        if isinstance(raw_names, list)
        else dict(sorted((int(index), str(name)) for index, name in raw_names.items()))
    )
    manifest_path = root / "export_manifest.json"
    manifest: dict[str, Any] | None = None
    if manifest_path.is_file():
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            candidate = None
        if isinstance(candidate, dict):
            manifest = candidate
    mapping_key = "detection" if task == "detect" else "segmentation"
    manifest_mapping = (
        manifest.get("class_mappings", {}).get(mapping_key)
        if manifest is not None
        else None
    )
    return {
        "root": str(root),
        "data_yaml": str(data_yaml),
        "manifest": str(manifest_path) if manifest is not None else None,
        "export_hash": manifest.get("export_hash") if manifest is not None else None,
        "split": manifest.get("split") if manifest is not None else None,
        "class_mapping": manifest_mapping if manifest_mapping is not None else names,
    }


def _runtime() -> dict[str, Any]:
    import torch
    import ultralytics

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _validate_options(options: TrainingOptions) -> None:
    if options.epochs <= 0 or options.imgsz <= 0 or options.workers < 0:
        raise ValueError("epochs/imgsz must be positive and workers cannot be negative")
    if isinstance(options.batch, (int, float)) and options.batch == 0:
        raise ValueError("batch cannot be zero")
    if options.device not in {"cpu", "mps"}:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {options.device!r} was requested, but PyTorch reports CUDA unavailable")


def _cuda_visible_devices(device: str) -> str:
    """Translate user-facing ``cuda:0`` notation for CUDA_VISIBLE_DEVICES."""
    value = device.strip().lower()
    if value.startswith("cuda:"):
        value = value.removeprefix("cuda:")
    if not value or any(not part.strip().isdigit() for part in value.split(",")):
        raise ValueError(f"D-FINE GPU device must be an index such as '0' or 'cuda:0'; received {device!r}")
    return value


def train_yolo_export(
    export_root: str | Path,
    task: Literal["detect", "segment"],
    options: TrainingOptions | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    options = options or TrainingOptions()
    _validate_options(options)
    data_yaml = require_valid_yolo_training_dataset(export_root, task)
    model_name = "yolo26n.pt" if task == "detect" else "yolo26n-seg.pt"
    run_name = options.run_name or f"yolo26n-{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project_dir = Path(options.output_dir).expanduser().resolve() / "yolo26"
    run_dir = project_dir / run_name

    if options.resume:
        checkpoint = Path(options.resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        model = YOLO(str(checkpoint))
        results = model.train(resume=True)
        arguments: dict[str, Any] = {"resume": str(checkpoint)}
    else:
        arguments = {
            "data": str(data_yaml),
            "task": task,
            "epochs": options.epochs,
            "imgsz": options.imgsz,
            "batch": options.batch,
            "workers": options.workers,
            "seed": options.seed,
            "device": options.device,
            "amp": options.amp,
            "project": str(project_dir),
            "name": run_name,
            "exist_ok": False,
            "deterministic": True,
            "plots": True,
            "save": True,
            "val": True,
        }
        model = YOLO(model_name)
        results = model.train(**arguments)

    # Ultralytics owns the final save directory. This is especially important
    # for resume, which continues the checkpoint's original run rather than the
    # provisional name calculated above.
    trainer_save_dir = getattr(getattr(model, "trainer", None), "save_dir", None)
    if trainer_save_dir:
        run_dir = Path(trainer_save_dir).expanduser().resolve()
    elif options.resume:
        run_dir = checkpoint.parents[1]

    results_csv = run_dir / "results.csv"
    final_metrics_path = run_dir / "final_metrics.json"
    final_metrics = _ultralytics_final_metrics(results, getattr(model, "trainer", None), results_csv)
    _write_json(final_metrics_path, final_metrics)
    metadata = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "trainer": "ultralytics",
        "model": model_name,
        "task": task,
        "options": asdict(options),
        "training_arguments": arguments,
        "dataset": _yolo_dataset_metadata(export_root, data_yaml, task),
        "runtime": _runtime(),
        "artifacts": {
            "best": str(run_dir / "weights" / "best.pt"),
            "last": str(run_dir / "weights" / "last.pt"),
            "metrics_csv": str(results_csv),
            "final_metrics": str(final_metrics_path),
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    return {
        "run_dir": str(run_dir),
        "metadata": str(run_dir / "run_metadata.json"),
        "final_metrics": str(final_metrics_path),
        "result_type": type(results).__name__,
    }


def _dfine_config(bundle_root: Path, upstream: Path, run_dir: Path, options: TrainingOptions) -> Path:
    manifest = json.loads((bundle_root / "export_manifest.json").read_text(encoding="utf-8"))
    classes = manifest["class_mappings"]["detection"]
    dataset = bundle_root / "dfine_detection"
    upstream_config = upstream / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_custom.yml"
    if not upstream_config.is_file():
        raise FileNotFoundError(f"Official D-FINE-N custom config is missing: {upstream_config}")
    payload = {
        "__include__": [str(upstream_config)],
        "output_dir": str(run_dir),
        "num_classes": len(classes),
        "remap_mscoco_category": False,
        "epochs": options.epochs,
        "train_dataloader": {
            "total_batch_size": int(options.batch),
            "num_workers": options.workers,
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(dataset / "images" / "train"),
                "ann_file": str(dataset / "annotations" / "instances_train.json"),
                "return_masks": False,
                "remap_mscoco_category": False,
            },
        },
        "val_dataloader": {
            "total_batch_size": int(options.batch),
            "num_workers": options.workers,
            "dataset": {
                "type": "CocoDetection",
                "img_folder": str(dataset / "images" / "val"),
                "ann_file": str(dataset / "annotations" / "instances_val.json"),
                "return_masks": False,
                "remap_mscoco_category": False,
            },
        },
    }
    destination = run_dir / "dfine_n_training.yml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def train_dfine_export(
    export_root: str | Path,
    upstream_dir: str | Path,
    options: TrainingOptions | None = None,
    *,
    execute: bool = True,
) -> dict[str, Any]:
    options = options or TrainingOptions()
    _validate_options(options)
    bundle = require_valid_training_export(export_root)
    upstream = Path(upstream_dir).expanduser().resolve()
    train_script = upstream / "train.py"
    if not train_script.is_file() or not (upstream / "src" / "data" / "dataset" / "coco_dataset.py").is_file():
        raise FileNotFoundError(f"Not an official Peterande/D-FINE checkout: {upstream}")
    run_name = options.run_name or f"dfine-n-detect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(options.output_dir).expanduser().resolve() / "dfine" / run_name
    config = _dfine_config(bundle.root, upstream, run_dir, options)
    command = [sys.executable, str(train_script), "-c", str(config), "--seed", str(options.seed), "--output-dir", str(run_dir)]
    if options.amp:
        command.append("--use-amp")
    if options.resume:
        checkpoint = Path(options.resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        command.extend(["--resume", str(checkpoint)])
    manifest = json.loads(bundle.manifest.read_text(encoding="utf-8"))
    metadata = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "trainer": "Peterande/D-FINE",
        "model": "D-FINE-N",
        "task": "detect",
        "upstream": str(upstream),
        "command": command,
        "options": asdict(options),
        "dataset": {"root": str(bundle.root), "export_hash": manifest.get("export_hash"), "split": manifest.get("split"), "class_mapping": manifest["class_mappings"]["detection"]},
        "artifacts": {"run_dir": str(run_dir), "config": str(config), "checkpoints": str(run_dir / "*.pth"), "log": str(run_dir / "log.txt")},
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    if execute:
        environment = os.environ.copy()
        if options.device not in {"cpu", "mps"}:
            environment["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices(options.device)
        subprocess.run(command, cwd=upstream, env=environment, check=True)
    return {"run_dir": str(run_dir), "config": str(config), "metadata": str(run_dir / "run_metadata.json"), "command": command, "executed": execute}


def synthetic_asset_summary(export_root: str | Path) -> dict[str, Any]:
    bundle = require_valid_training_export(export_root)
    payload = json.loads(bundle.synthetic_assets.read_text(encoding="utf-8"))
    return {
        "foregrounds": int(payload.get("counts", {}).get("foregrounds", 0)),
        "backgrounds": int(payload.get("counts", {}).get("backgrounds", 0)),
        "manifest": str(bundle.synthetic_assets),
        "extension_boundary": "Generators consume this manifest and write a new dataset version; training never mutates the labelled export.",
    }


# Compatibility wrapper for older imports. It deliberately accepts only a new
# Data Labeller export root and never performs prediction or deployment export.
def train_yolo(export_root: str | Path, variant: str = "segment") -> Any:
    task: Literal["detect", "segment"] = "detect" if variant == "detect" else "segment"
    return train_yolo_export(export_root, task)
