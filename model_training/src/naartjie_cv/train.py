from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

from .datasets import (
    YOLO_IMAGE_SUFFIXES,
    prepare_dfine_seg_dataset,
    require_valid_training_export,
    require_valid_yolo_training_dataset,
)
from .experiment_tracking import dataset_identity, dataset_split_integrity, record_validation, update_metadata
from .colour import colour_metadata, dataset_colour, file_sha256, prepare_colour_dataset
from .runtime import seed_training
from .training_policy import MAX_EPOCHS, MIN_EPOCHS, PATIENCE, WORKERS, policy_metadata

YOLO_FINAL_CHECKPOINTS = {"detect": "yolo26n.pt", "segment": "yolo26n-seg.pt"}
OFFICIAL_DFINE_NANO_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_n_coco.pth"


def official_dfine_checkpoint(upstream: Path) -> str:
    """Prefer the original Nano checkpoint cached inside the upstream checkout."""
    local = upstream / "pretrained" / "dfine_n_coco.pth"
    return str(local.resolve()) if local.is_file() else OFFICIAL_DFINE_NANO_URL


@dataclass
class YoloTrainingOptions:
    """Ultralytics train-mode settings exposed by the CLI and notebook UI."""

    time: float | None = None
    patience: int = PATIENCE
    min_epochs_before_early_stop: int = MIN_EPOCHS
    save: bool = True
    save_period: int = -1
    cache: bool | Literal["ram", "disk"] = False
    pretrained: bool = True
    cls_remap: bool = True
    optimizer: str = "AdamW"
    verbose: bool = True
    deterministic: bool = True
    single_cls: bool = False
    rect: bool = False
    cos_lr: bool = False
    close_mosaic: int = 10
    fraction: float = 1.0
    profile: bool = False
    freeze: int | list[int] | None = None
    multi_scale: float = 0.0
    compile: bool | str = False
    overlap_mask: bool = True
    mask_ratio: int = 4
    dropout: float = 0.0
    val: bool = True
    split: str = "val"
    save_json: bool = False
    conf: float | None = None
    iou: float = 0.7
    max_det: int = 300
    agnostic_nms: bool = False
    plots: bool = True
    classes: list[int] | None = None
    # Preserve the project's current plotting choices while making them editable.
    show_labels: bool = False
    line_width: int = 2
    lr0: float = 0.001
    lrf: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_lr: float = 0.0
    box: float = 7.5
    cls: float = 0.5
    cls_pw: float = 0.0
    dfl: float = 1.5
    rle: float = 1.0
    nbs: int = 16
    distill_model: str | None = None
    dis: float = 6.0
    hsv_h: float = 0.0
    hsv_s: float = 0.0
    hsv_v: float = 0.0
    degrees: float = 10.0
    translate: float = 0.1
    scale: float = 0.25
    shear: float = 0.0
    perspective: float = 0.0
    flipud: float = 0.0
    fliplr: float = 0.5
    bgr: float = 0.0
    mosaic: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0
    copy_paste: float = 0.0
    copy_paste_mode: str = "flip"
    auto_augment: str | None = None
    erasing: float = 0.0
    # Escape hatch for a newer installed Ultralytics version. Managed run/path
    # keys are rejected below so custom values cannot bypass this workflow.
    extra_args: dict[str, Any] = field(default_factory=lambda: {"augmentations": []})


def final_yolo_options(task: str = "detect") -> YoloTrainingOptions:
    """Shared final-experiment defaults for the notebook, CLI and Python API."""
    options = YoloTrainingOptions()
    if task == "detect":
        return replace(options, mosaic=0.5)
    return options


@dataclass
class TrainingOptions:
    epochs: int = MAX_EPOCHS
    imgsz: int = 640
    batch: int | float = 16
    workers: int = WORKERS
    seed: int = 0
    colour_space: Literal["auto", "rgb", "hsv"] = "auto"
    device: str = "0"
    amp: bool = True
    output_dir: str = "runs"
    run_name: str | None = None
    resume: str | None = None
    yolo: YoloTrainingOptions = field(default_factory=YoloTrainingOptions)
    dfine_base_lr: float | None = None
    dfine_backbone_lr: float | None = None
    dfine_weight_decay: float | None = None
    dfine_clip_max_norm: float | None = None
    dfine_warmup_steps: int | None = None
    dfine_use_ema: bool = True
    dfine_extra_config: dict[str, Any] = field(default_factory=dict)


def preferred_dfine_amp_dtype() -> str:
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except (ImportError, RuntimeError):
        pass
    return "float16"


@dataclass
class DFineSegOptions:
    """D-FINE-seg settings shared by detection and instance segmentation."""

    model_size: Literal["n", "s", "m", "l", "x"] = "n"
    epochs: int = MAX_EPOCHS
    imgsz: int = 640
    batch: int = 2
    # Loading in the trainer avoids spawned PyTorch workers on Windows.
    # Explicit worker counts remain available for machines with enough RAM.
    workers: int = 0
    seed: int = 0
    deterministic: bool = True
    colour_space: Literal["auto", "rgb", "hsv"] = "auto"
    device: str = "cuda"
    amp: bool = True
    amp_dtype: Literal["bfloat16", "float16"] = field(default_factory=preferred_dfine_amp_dtype)
    early_stopping: int = PATIENCE
    min_epochs_before_early_stop: int = MIN_EPOCHS
    grad_accumulation: int = 8
    clip_max_norm: float = 0.1
    use_wandb: bool = False
    use_ema: bool = True
    use_scheduler: bool = True
    use_muon: bool = True
    aux_optimizer: Literal["adamw", "adan"] = "adan"
    base_lr: float | None = None
    backbone_lr: float | None = None
    weight_decay: float = 0.000125
    mosaic_probability: float | None = 0.0
    no_mosaic_epochs: int = 10
    conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    pretrained_dataset: Literal["coco", "obj2coco"] = "coco"
    pretrained_model: str | None = None
    visualize_validation: bool = True
    output_dir: str = "runs"
    run_name: str | None = None
    extra_config: dict[str, Any] = field(default_factory=dict)


def final_dfine_seg_options(task: str = "segment") -> DFineSegOptions:
    """Project ArgoHA recipes: Nano detection and Nano segmentation."""
    options = DFineSegOptions()
    if task == "detect":
        return replace(
            options, model_size="n", batch=4, grad_accumulation=4,
            use_muon=False, aux_optimizer="adamw", base_lr=0.0008,
            backbone_lr=0.0004, weight_decay=0.0001, mosaic_probability=0.5,
        )
    return options


def _final_dfine_extra_config(task: str) -> dict[str, Any]:
    return {"train": {
        "decision_metrics": ["mAP_50_95" if task == "detect" else "mAP_50_95_mask"],
        "keep_ratio": True, "freeze_except_mask": False, "mask_batch_size": 16,
        "max_walltime_min": None, "ignore_background_epochs": 0,
        # The project detection scheduler otherwise homogenizes these LRs.
        "preserve_lr_groups": task == "detect",
        "augs": {
            "left_right_flip": 0.5, "up_down_flip": 0.0, "rotate_90": 0.0,
            "rotation_degree": 10.0, "rotation_p": 0.05, "blur": 0.0,
            "to_gray": 0.0, "gamma": 0.0, "brightness": 0.0, "noise": 0.0,
            "coarse_dropout": 0.0, "multiscale_prob": 0.0,
        },
    }}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _resolved_yolo_data_yaml(source: Path, project_dir: Path, task: str) -> Path:
    """Write a stable Ultralytics config whose dataset root is unambiguous.

    Ultralytics resolves an explicit relative ``path`` from the process working
    directory, not from ``data.yaml``. Notebook kernels therefore break valid
    portable configs such as ``path: .``. The workflow validator resolves paths
    relative to the source YAML, so persist that same resolved root in a derived
    config used only for training.
    """
    source = source.expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YOLO data config must contain a mapping: {source}")
    configured_root = payload.get("path")
    if configured_root in (None, ""):
        dataset_root = source.parent
    else:
        configured_path = Path(str(configured_root)).expanduser()
        dataset_root = (
            configured_path.resolve()
            if configured_path.is_absolute()
            else (source.parent / configured_path).resolve()
        )
    resolved_payload = dict(payload)
    resolved_payload["path"] = dataset_root.as_posix()
    fingerprint = hashlib.sha256(
        str(source).encode("utf-8") + b"\0" + source.read_bytes()
    ).hexdigest()[:16]
    destination = project_dir / "_dataset_configs" / f"{task}-{fingerprint}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    contents = yaml.safe_dump(resolved_payload, sort_keys=False, allow_unicode=True)
    if not destination.is_file() or destination.read_text(encoding="utf-8") != contents:
        destination.write_text(contents, encoding="utf-8", newline="\n")
    return destination.resolve()


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


def _yolo_test_image_count(data_yaml: Path) -> int:
    """Count an optional held-out test split using the same root rules as training."""
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    configured = payload.get("test")
    if not configured:
        return 0
    root = Path(str(payload.get("path") or ".")).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    directory = Path(str(configured)).expanduser()
    if not directory.is_absolute():
        directory = root / directory
    return sum(
        path.is_file() and path.suffix.lower() in YOLO_IMAGE_SUFFIXES
        for path in directory.rglob("*")
    )


def _dfine_seg_test_metrics(path: Path) -> dict[str, int | float]:
    """Extract the held-out test row written by D-FINE-seg's final evaluator."""
    if not path.is_file():
        raise FileNotFoundError(f"D-FINE-seg did not write final evaluation metrics: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if (item.get("") or item.get("Unnamed: 0")) == "test"), None)
    if row is None:
        raise RuntimeError(f"D-FINE-seg did not evaluate the configured test split: {path}")
    metrics: dict[str, int | float] = {}
    for key, value in row.items():
        if not key or value in (None, ""):
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            metrics[key] = number
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
        "split_integrity": dataset_split_integrity(data_yaml),
    }


def _runtime() -> dict[str, Any]:
    import torch
    import ultralytics
    from .runtime import library_versions

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "libraries": library_versions(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _validate_options(options: TrainingOptions) -> None:
    if options.epochs <= 0 or options.imgsz <= 0 or options.workers < 0:
        raise ValueError("epochs/imgsz must be positive and workers cannot be negative")
    if isinstance(options.batch, (int, float)) and options.batch == 0:
        raise ValueError("batch cannot be zero")
    yolo = options.yolo
    if yolo.time is not None and yolo.time <= 0:
        raise ValueError("time must be positive when supplied")
    if yolo.patience < 0 or yolo.save_period < -1 or yolo.close_mosaic < 0:
        raise ValueError("patience/close_mosaic cannot be negative and save_period must be at least -1")
    if yolo.min_epochs_before_early_stop < 0:
        raise ValueError("Minimum epochs cannot be negative")
    if not 0 < yolo.fraction <= 1:
        raise ValueError("fraction must be greater than 0 and at most 1")
    unit_interval_settings = (
        "conf",
        "iou",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "translate",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
    )
    for name in unit_interval_settings:
        value = getattr(yolo, name)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    protected = {"data", "task", "model", "mode", "project", "name", "exist_ok", "resume", "val", "split", "save"}
    conflicts = protected.intersection(yolo.extra_args)
    if conflicts:
        raise ValueError(f"extra_args cannot override managed settings: {', '.join(sorted(conflicts))}")
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


def resolved_yolo_arguments(options: TrainingOptions, data_yaml: Path, task: str, project: Path, name: str) -> dict[str, Any]:
    """The exact fresh-run arguments, shared with the final preflight."""
    yolo = asdict(options.yolo)
    yolo.pop("min_epochs_before_early_stop")  # Implemented by our stopper, not an Ultralytics argument.
    extra = yolo.pop("extra_args")
    yolo = {key: value for key, value in yolo.items() if value is not None or key == "auto_augment"}
    yolo.update(extra)
    return {"data": str(data_yaml), "task": task, "epochs": options.epochs, "imgsz": options.imgsz,
            "batch": options.batch, "workers": options.workers, "seed": options.seed, "device": options.device,
            "amp": options.amp, "project": str(project), "name": name, "exist_ok": False, **yolo}


def train_yolo_export(
    export_root: str | Path,
    task: Literal["detect", "segment"],
    options: TrainingOptions | None = None,
) -> dict[str, Any]:
    from ultralytics import YOLO

    options = options or TrainingOptions(yolo=final_yolo_options(task))
    _validate_options(options)
    if not options.yolo.val or options.yolo.split != "val" or not options.yolo.save:
        raise ValueError("YOLO training must validate on val and save a checkpoint; test is reserved for final evaluation")
    original_data_yaml = require_valid_yolo_training_dataset(export_root, task)
    original_root = str(Path(export_root).expanduser().resolve())
    export_root = prepare_colour_dataset(export_root, task, Path(options.output_dir).expanduser().resolve() / "yolo26" / "_colour_datasets", options.colour_space)
    data_yaml = require_valid_yolo_training_dataset(export_root, task)
    seed_training(options.seed, options.yolo.deterministic)
    test_images = _yolo_test_image_count(data_yaml)
    model_name = YOLO_FINAL_CHECKPOINTS[task]
    model_variant = "yolo26n"
    run_name = options.run_name or f"{model_variant}-{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    project_dir = Path(options.output_dir).expanduser().resolve() / "yolo26"
    run_dir = project_dir / run_name
    training_data_yaml: Path | None = None

    def track_initialisation(trainer: Any) -> None:
        from .yolo_training import ValidationEarlyStopping

        trainer.stopper = ValidationEarlyStopping(trainer.args.patience, options.yolo.min_epochs_before_early_stop, trainer.epochs)
        trainer._model_trainer_last_epoch = getattr(trainer, "start_epoch", 0) - 1
        metadata_path = Path(trainer.save_dir) / "run_metadata.json"
        previous = json.loads(metadata_path.read_text(encoding="utf-8")) if options.resume and metadata_path.is_file() else {}
        trainer._model_trainer_optimizer_steps = previous.get("actual_optimizer_steps", 0)
        history = metadata_path.parent / "validation_history.jsonl"
        if options.resume and history.is_file():
            for line in history.read_text(encoding="utf-8").splitlines():
                observation = json.loads(line)
                if observation["epoch"] <= getattr(trainer, "start_epoch", 0):
                    trainer.stopper(observation["epoch"], observation["value"])
        def count_step(*_):
            trainer._model_trainer_optimizer_steps += 1
        trainer.optimizer.register_step_post_hook(count_step)
        actual_args = vars(trainer.args).copy()
        physical = trainer.batch_size
        effective = physical * trainer.accumulate
        metric = "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)"
        from ultralytics.utils import LOGGER

        LOGGER.info(f"Physical batch size: {physical}; accumulation: {trainer.accumulate}; effective batch size: {effective}; AMP: {bool(trainer.amp)}")
        policy = policy_metadata(trainer.epochs, options.yolo.min_epochs_before_early_stop, trainer.args.patience, metric)
        if previous:
            policy.update(best_metric=previous.get("best_metric"), best_epoch=previous.get("best_epoch"))
        batches = len(trainer.train_loader) if hasattr(trainer, "train_loader") else None
        update_metadata(
            Path(trainer.save_dir) / "run_metadata.json",
            trainer="ultralytics", model=model_name, model_variant=model_variant, task=task,
            **policy,
            options=asdict(options), training_arguments=actual_args, runtime=_runtime(),
            dataset={**_yolo_dataset_metadata(original_root, original_data_yaml, task),
                     "prepared_root": str(export_root), **dataset_identity(data_yaml)},
            optimisation={
                "optimizer": type(trainer.optimizer).__name__, "physical_batch_size": physical,
                "effective_batch_size": effective, "gradient_accumulation": trainer.accumulate,
                "amp": bool(trainer.amp), "scheduler": "cosine" if trainer.args.cos_lr else "linear",
                "scheduler_epochs": trainer.epochs,
                "optimizer_steps_per_epoch": batches if trainer.accumulate == 1 else None,
                "total_optimizer_steps": batches * trainer.epochs if batches is not None and trainer.accumulate == 1 else None,
                "resolved_workers": trainer.train_loader.num_workers if hasattr(trainer, "train_loader") else actual_args["workers"],
                "warmup_epochs": trainer.args.warmup_epochs,
                "parameter_groups": [{key: group[key] for key in ("lr", "initial_lr", "weight_decay", "betas", "momentum", "eps") if key in group} for group in trainer.optimizer.param_groups],
            },
            selection=previous.get("selection") or {"split": "val", "metric": metric, "best_epoch": None, "best_validation_metric": None,
                       "best_checkpoint_path": str(Path(trainer.save_dir) / "weights" / "best.pt")},
            experiment={"architecture": "YOLO26", "colour_space": dataset_colour(data_yaml),
                        "model_size": "Nano", "task": task, "input_representation": colour_metadata(data_yaml),
                        "source_dataset": original_root, "dataset_path": str(data_yaml),
                        "pretrained_checkpoint": model_name if not options.resume else options.resume,
                        "pretrained_checkpoint_sha256": file_sha256(Path(model_name)) if Path(model_name).is_file() and not options.resume else None,
                        "image_size": actual_args["imgsz"], "epochs": trainer.epochs,
                        "batch_size": physical, "seed": actual_args["seed"],
                        "deterministic": actual_args["deterministic"], "amp": bool(trainer.amp),
                        "ema": True, "gradient_clip_norm": 10.0,
                        "optimizer_and_augmentations": actual_args},
        )
        LOGGER.info("Experiment configuration saved to " + str(Path(trainer.save_dir) / "run_metadata.json"))

    def track_validation(trainer: Any) -> None:
        # Ultralytics emits this callback again after its final evaluation.
        if trainer.epoch == getattr(trainer, "_model_trainer_last_epoch", -1):
            return
        metric = "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)"
        metrics = _metric_values(trainer.metrics)
        if metric in metrics:
            record_validation(Path(trainer.save_dir) / "run_metadata.json", epoch=trainer.epoch + 1,
                              metric=metric, value=float(metrics[metric]),
                              checkpoint=Path(trainer.save_dir) / "weights" / "best.pt", metrics=metrics, replace_equal=True)
            trainer._model_trainer_last_epoch = trainer.epoch
            update_metadata(Path(trainer.save_dir) / "run_metadata.json",
                            actual_optimizer_steps=trainer._model_trainer_optimizer_steps,
                            early_stopping_state=vars(trainer.stopper.state))
        else:
            raise ValueError(f"Required validation-only selection metric is missing: {metric}")

    def attach_tracking(model: Any) -> None:
        if hasattr(model, "add_callback"):
            model.add_callback("on_pretrain_routine_end", track_initialisation)
            model.add_callback("on_fit_epoch_end", track_validation)
            model.add_callback("on_train_epoch_start", prevent_batch_reduction)

    def prevent_batch_reduction(trainer: Any) -> None:
        # Ultralytics 8.4.95 silently halves batch on first-epoch OOM.
        # Controlled pairs must fail at their configured physical batch.
        trainer._oom_retries = 3

    def execute_training(model: Any, **kwargs: Any) -> Any:
        try:
            result = model.train(**kwargs)
        except BaseException as error:
            trainer = getattr(model, "trainer", None)
            directory = Path(getattr(trainer, "save_dir", run_dir))
            update_metadata(directory / "run_metadata.json",
                            **{key: value for key, value in policy_metadata(options.epochs, options.yolo.min_epochs_before_early_stop,
                                options.yolo.patience, "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)").items()
                               if key not in {"best_metric", "best_epoch", "actual_stop_epoch", "stop_reason"}},
                            actual_stop_epoch=max(0, getattr(trainer, "_model_trainer_last_epoch", -1) + 1),
                            stop_reason="manual_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                            actual_optimizer_steps=getattr(trainer, "_model_trainer_optimizer_steps", 0),
                            training_error=str(error))
            raise
        trainer = getattr(model, "trainer", None)
        if trainer is not None and hasattr(trainer, "_model_trainer_last_epoch"):
            state = getattr(getattr(trainer, "stopper", None), "state", None)
            completed = getattr(trainer, "_model_trainer_last_epoch", -1) + 1
            reason = "max_epochs" if completed >= trainer.epochs else (
                "early_stopping" if state is not None and state.should_stop else "manual_interrupt")
            update_metadata(Path(trainer.save_dir) / "run_metadata.json", actual_stop_epoch=completed,
                            stop_reason=reason, actual_optimizer_steps=getattr(trainer, "_model_trainer_optimizer_steps", 0))
        return result

    if options.resume:
        checkpoint = Path(options.resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        model = YOLO(str(checkpoint))
        attach_tracking(model)
        # Ultralytics restores the original run configuration from last.pt, but
        # explicitly permits a worker-count override. This matters for recovery
        # from Windows/Jupyter data-loader deadlocks: setting workers=0 keeps
        # loading in the trainer process while preserving model/optimizer state.
        if task == "segment":
            from .yolo_training import MaskSelectionTrainer

            results = execute_training(model, trainer=MaskSelectionTrainer, resume=True, workers=options.workers)
        else:
            results = execute_training(model, resume=True, workers=options.workers)
        arguments: dict[str, Any] = {
            "resume": str(checkpoint),
            "workers": options.workers,
        }
    else:
        training_data_yaml = _resolved_yolo_data_yaml(data_yaml, project_dir, task)
        arguments = resolved_yolo_arguments(options, training_data_yaml, task, project_dir, run_name)
        model = YOLO(model_name)
        attach_tracking(model)
        if task == "segment":
            from .yolo_training import MaskSelectionTrainer

            results = execute_training(model, trainer=MaskSelectionTrainer, **arguments)
        else:
            results = execute_training(model, **arguments)

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
    test_metrics_path: Path | None = None
    test_checkpoint: Path | None = None
    if test_images:
        best = run_dir / "weights" / "best.pt"
        test_checkpoint = best
        if not test_checkpoint.is_file():
            raise FileNotFoundError(f"No best validation YOLO checkpoint is available for test evaluation in {run_dir}")
        test_model = YOLO(str(test_checkpoint))
        test_results = test_model.val(
            data=str(training_data_yaml or _resolved_yolo_data_yaml(data_yaml, project_dir, task)),
            split="test",
            imgsz=options.imgsz,
            batch=1,
            workers=options.workers,
            device=options.device,
            project=str(run_dir),
            name="test_eval",
            exist_ok=True,
            plots=options.yolo.plots,
        )
        test_metrics_path = run_dir / "test_metrics.json"
        test_values = _metric_values(test_results)
        if not test_values:
            raise RuntimeError("YOLO test evaluation returned no performance metrics")
        _write_json(
            test_metrics_path,
            {
                "split": "test",
                "images": test_images,
                "checkpoint": str(test_checkpoint),
                "metrics": test_values,
            },
        )
    metadata = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "trainer": "ultralytics",
        "model_family": "YOLO26",
        "model_variant": model_variant,
        "pretrained_checkpoint": model_name if not options.resume else options.resume,
        "model": model_name,
        "task": task,
        "options": asdict(options),
        "training_arguments": arguments,
        "dataset": {**_yolo_dataset_metadata(original_root, original_data_yaml, task),
                    "prepared_root": str(export_root), **dataset_identity(data_yaml)},
        "runtime": _runtime(),
        "artifacts": {
            "training_data_yaml": str(training_data_yaml) if training_data_yaml else None,
            "best": str(run_dir / "weights" / "best.pt"),
            "last": str(run_dir / "weights" / "last.pt"),
            "metrics_csv": str(results_csv),
            "final_metrics": str(final_metrics_path),
            "test_metrics": str(test_metrics_path) if test_metrics_path else None,
            "test_checkpoint": str(test_checkpoint) if test_checkpoint else None,
        },
    }
    # Keep the per-epoch selection and actual optimiser/batch records produced
    # by callbacks, including any arguments restored by an explicit resume.
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["training_arguments"] = recorded.get("training_arguments", arguments)
        metadata["runtime"] = recorded.get("runtime", metadata["runtime"])
    update_metadata(metadata_path, **metadata)
    return {
        "run_dir": str(run_dir),
        "metadata": str(run_dir / "run_metadata.json"),
        "final_metrics": str(final_metrics_path),
        "test_metrics": str(test_metrics_path) if test_metrics_path else None,
        "test_images": test_images,
        "result_type": type(results).__name__,
    }


def _dfine_config(
    upstream: Path,
    run_dir: Path,
    prepared: dict[str, Any],
    options: TrainingOptions,
) -> Path:
    classes = prepared["classes"]
    dataset = Path(prepared["root"])
    upstream_config = upstream / "configs" / "dfine" / "custom" / "dfine_hgnetv2_n_custom.yml"
    if not upstream_config.is_file():
        raise FileNotFoundError(f"Official D-FINE-N custom config is missing: {upstream_config}")
    if not isinstance(options.dfine_extra_config, dict):
        raise ValueError("Official D-FINE advanced config must be a mapping")
    payload: dict[str, Any] = {"__include__": [str(upstream_config)], "print_freq": 10}
    _deep_merge(payload, options.dfine_extra_config)
    payload.update(
        {
            "__include__": [str(upstream_config)],
            "output_dir": str(run_dir),
            "num_classes": len(classes),
            "remap_mscoco_category": False,
            "epochs": options.epochs,
            "use_ema": options.dfine_use_ema,
        }
    )
    for split, annotation in (("train", "train.json"), ("val", "val.json")):
        dataloader = payload.setdefault(f"{split}_dataloader", {})
        if not isinstance(dataloader, dict):
            raise ValueError(f"Official D-FINE {split}_dataloader must be a mapping")
        dataloader.update({"total_batch_size": int(options.batch), "num_workers": options.workers})
        split_dataset = dataloader.setdefault("dataset", {})
        if not isinstance(split_dataset, dict):
            raise ValueError(f"Official D-FINE {split} dataset config must be a mapping")
        split_dataset.update(
            {
                "type": "CocoDetection",
                "img_folder": str(dataset / "images"),
                "ann_file": str(dataset / annotation),
                "return_masks": False,
                "remap_mscoco_category": False,
            }
        )

    optimizer = payload.setdefault("optimizer", {})
    if not isinstance(optimizer, dict):
        raise ValueError("Official D-FINE optimizer config must be a mapping")
    if options.dfine_base_lr is not None:
        optimizer["lr"] = options.dfine_base_lr
    if options.dfine_backbone_lr is not None:
        custom_config = yaml.safe_load(upstream_config.read_text(encoding="utf-8")) or {}
        groups = custom_config.get("optimizer", {}).get("params", [])
        backbone_groups = [
            group for group in groups if isinstance(group, dict) and "backbone" in group.get("params", "") and "lr" in group
        ]
        if not backbone_groups:
            raise ValueError("Official D-FINE custom config has no backbone learning-rate groups")
        for group in backbone_groups:
            group["lr"] = options.dfine_backbone_lr
        optimizer["params"] = groups
    if options.dfine_weight_decay is not None:
        optimizer["weight_decay"] = options.dfine_weight_decay
    if options.dfine_clip_max_norm is not None:
        payload["clip_max_norm"] = options.dfine_clip_max_norm
    if options.dfine_warmup_steps is not None:
        warmup = payload.setdefault("lr_warmup_scheduler", {})
        if not isinstance(warmup, dict):
            raise ValueError("Official D-FINE warmup scheduler config must be a mapping")
        warmup["warmup_duration"] = options.dfine_warmup_steps

    # Always resolve transforms: inherited photometric distort and multiscale
    # collation must be disabled even at the native 640 default.
    custom_config = yaml.safe_load(upstream_config.read_text(encoding="utf-8")) or {}
    dataloader_config = yaml.safe_load(
        (upstream / "configs" / "dfine" / "include" / "dataloader.yml").read_text(encoding="utf-8")
    ) or {}
    for split, source in (("train", custom_config), ("val", dataloader_config)):
        source_ops = source.get(f"{split}_dataloader", {}).get("dataset", {}).get("transforms", {}).get("ops")
        if source_ops is None:
            source_ops = dataloader_config.get(f"{split}_dataloader", {}).get("dataset", {}).get("transforms", {}).get("ops", [])
        override_ops = options.dfine_extra_config.get(f"{split}_dataloader", {}).get("dataset", {}).get("transforms", {}).get("ops")
        ops = [dict(item) for item in (override_ops if override_ops is not None else source_ops)
               if override_ops is not None or item.get("type") != "RandomPhotometricDistort"]
        resize = next((item for item in ops if item.get("type") == "Resize"), None)
        if resize is None:
            raise ValueError(f"Official D-FINE {split} transforms have no Resize operation")
        resize["size"] = [options.imgsz, options.imgsz]
        payload[f"{split}_dataloader"].setdefault("dataset", {}).setdefault("transforms", {})["ops"] = ops
    payload["eval_spatial_size"] = [options.imgsz, options.imgsz]
    payload["train_dataloader"].setdefault("collate_fn", {})["base_size"] = options.imgsz
    payload["train_dataloader"]["collate_fn"].setdefault("base_size_repeat", None)
    payload["seed"] = options.seed
    payload["deterministic"] = options.yolo.deterministic
    payload["colour_space"] = prepared.get("colour_space", "rgb")
    payload["input_representation"] = colour_metadata(Path(prepared["source_data_yaml"]))
    payload["use_amp"] = options.amp
    destination = run_dir / "dfine_n_training.yml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def _stage_dfine_resume_checkpoint(checkpoint: Path, run_dir: Path) -> Path:
    """Seed the stage-one checkpoint expected by D-FINE in a new run directory.

    At the augmentation transition, the official solver reloads
    ``output_dir/best_stg1.pth``. Resumed jobs use a fresh output directory, so
    carry the source run's best stage-one checkpoint forward. A standalone
    checkpoint may not have that companion; in that case it is the best
    available baseline and is staged under the name expected by the solver.
    """
    companion = checkpoint.with_name("best_stg1.pth")
    source = companion if companion.is_file() else checkpoint
    destination = run_dir / "best_stg1.pth"
    if destination.exists():
        return destination
    # Do not hard-link checkpoints: torch.save() may later replace or truncate
    # the destination, and the source run must remain an immutable recovery
    # point.
    shutil.copy2(source, destination)
    return destination


def _dfine_test_config(config: Path, prepared: dict[str, Any], run_dir: Path) -> Path:
    """Evaluate on held-out test images without altering training/validation config."""
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    dataset = payload.setdefault("val_dataloader", {}).setdefault("dataset", {})
    dataset["ann_file"] = str(Path(prepared["root"]) / "test.json")
    dataset["img_folder"] = str(Path(prepared["root"]) / "images")
    payload["output_dir"] = str(run_dir / "test_eval")
    destination = run_dir / "dfine_n_test.yml"
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def train_dfine_export(
    export_root: str | Path,
    upstream_dir: str | Path,
    options: TrainingOptions | None = None,
    *,
    python_executable: str | Path | None = None,
    execute: bool = True,
    output_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    # This official Nano route is separate from the final ArgoHA experiments.
    options = options or TrainingOptions(batch=8)
    _validate_options(options)
    if isinstance(options.batch, bool) or not isinstance(options.batch, int) or options.batch <= 0:
        raise ValueError(
            "Official D-FINE batch size must be a positive integer; automatic batch values "
            "such as -1 are only supported by Ultralytics and D-FINE-seg"
        )
    for name in ("dfine_base_lr", "dfine_backbone_lr"):
        value = getattr(options, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"Official D-FINE {name} must be positive")
    if options.dfine_weight_decay is not None and (
        not math.isfinite(options.dfine_weight_decay) or options.dfine_weight_decay < 0
    ):
        raise ValueError("Official D-FINE weight decay cannot be negative")
    if options.dfine_clip_max_norm is not None and (
        not math.isfinite(options.dfine_clip_max_norm) or options.dfine_clip_max_norm < 0
    ):
        raise ValueError("Official D-FINE gradient clip norm cannot be negative")
    if options.dfine_warmup_steps is not None and options.dfine_warmup_steps < 0:
        raise ValueError("Official D-FINE warmup steps cannot be negative")
    upstream = Path(upstream_dir).expanduser().resolve()
    train_script = upstream / "train.py"
    if not train_script.is_file() or not (upstream / "src" / "data" / "dataset" / "coco_dataset.py").is_file():
        raise FileNotFoundError(f"Not an official Peterande/D-FINE checkout: {upstream}")
    run_name = options.run_name or f"dfine-n-detect-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path(options.output_dir).expanduser().resolve() / "dfine" / run_name
    original_root = str(Path(export_root).expanduser().resolve())
    export_root = prepare_colour_dataset(export_root, "detect", run_dir.parent / "_colour_datasets", options.colour_space)
    # Official D-FINE requires COCO JSON. Stage either a standalone YOLO
    # detection dataset or a complete Data Labeller bundle into a stable COCO
    # view without modifying or duplicating the source images when hard links
    # are available.
    prepared = prepare_dfine_seg_dataset(
        export_root,
        "detect",
        run_dir / "prepared_dataset",
    )
    config = _dfine_config(upstream, run_dir, prepared, options)
    test_images = prepared["splits"].get("test", {}).get("images", 0)
    test_config = _dfine_test_config(config, prepared, run_dir) if test_images else None
    test_metrics = run_dir / "test_metrics.json" if test_images else None
    console_log = run_dir / "trainer_console.log"
    executable = _resolve_python_executable(python_executable)
    runner = Path(__file__).resolve().parents[2] / "scripts" / "dfine_train_runner.py"
    command = [str(executable), str(runner), "--upstream", str(upstream), "-c", str(config), "--seed", str(options.seed), "--output-dir", str(run_dir)]
    if options.amp:
        command.append("--use-amp")
    resume_stage1: Path | None = None
    if options.resume:
        checkpoint = Path(options.resume).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint}")
        resume_stage1 = _stage_dfine_resume_checkpoint(checkpoint, run_dir)
        command.extend(["--resume", str(checkpoint)])
    else:
        command.extend(["--tuning", official_dfine_checkpoint(upstream)])
    metadata = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "trainer": "Peterande/D-FINE",
        "model": "D-FINE-N",
        "task": "detect",
        "upstream": str(upstream),
        "python": str(executable),
        "command": command,
        "options": asdict(options),
        "experiment": {"architecture": "Peterande/D-FINE-N", "colour_space": dataset_colour(Path(prepared["source_data_yaml"])),
                       "model_size": "Nano", "task": "detect", "input_representation": colour_metadata(Path(prepared["source_data_yaml"])),
                       "source_dataset": original_root, "dataset_path": prepared["root"],
                       "pretrained_checkpoint": options.resume or command[-1], "image_size": [options.imgsz] * 2,
                       "epochs": options.epochs, "batch_size": options.batch, "seed": options.seed,
                       "deterministic": options.yolo.deterministic, "amp": options.amp, "ema": options.dfine_use_ema},
        "effective_training_config": yaml.safe_load(config.read_text(encoding="utf-8")),
        "dataset": prepared,
        "artifacts": {
            "run_dir": str(run_dir),
            "prepared_dataset": prepared["root"],
            "config": str(config),
            "checkpoints": str(run_dir / "*.pth"),
            "resume_stage1": str(resume_stage1) if resume_stage1 else None,
            "log": str(run_dir / "log.txt"),
            "console_log": str(console_log),
            "test_config": str(test_config) if test_config else None,
            "test_metrics": str(test_metrics) if test_metrics else None,
        },
    }
    _write_json(run_dir / "run_metadata.json", metadata)
    if execute:
        environment = os.environ.copy()
        if options.device not in {"cpu", "mps"}:
            environment["CUDA_VISIBLE_DEVICES"] = _cuda_visible_devices(options.device)
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONHASHSEED"] = str(options.seed)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if output_callback:
            output_callback(f"Trainer console log: {console_log}\n")
        _run_training_process(
            command,
            cwd=upstream,
            environment=environment,
            output_callback=output_callback,
            output_log=console_log,
        )
        if test_config and test_metrics:
            checkpoint = next(
                (path for path in (run_dir / "best_stg2.pth", run_dir / "best_stg1.pth", run_dir / "last.pth") if path.is_file()),
                None,
            )
            if checkpoint is None:
                raise FileNotFoundError(f"No official D-FINE checkpoint is available for test evaluation in {run_dir}")
            runner = Path(__file__).resolve().parents[2] / "scripts" / "dfine_test_runner.py"
            test_command = [
                str(executable), str(runner), "--upstream", str(upstream), "--config", str(test_config),
                "--checkpoint", str(checkpoint), "--output", str(test_metrics), "--images", str(test_images),
            ]
            if output_callback:
                output_callback("Training complete; evaluating the held-out test split.\n")
            _run_training_process(
                test_command,
                cwd=upstream,
                environment=environment,
                output_callback=output_callback,
                output_log=console_log,
            )
    return {
        "run_dir": str(run_dir),
        "prepared_dataset": prepared["root"],
        "config": str(config),
        "metadata": str(run_dir / "run_metadata.json"),
        "resume_stage1": str(resume_stage1) if resume_stage1 else None,
        "command": command,
        "test_config": str(test_config) if test_config else None,
        "test_metrics": str(test_metrics) if execute and test_metrics else None,
        "test_images": test_images,
        "console_log": str(console_log),
        "executed": execute,
    }


_DFINE_SEG_LEARNING_RATES = {
    "n": {"base_lr": 0.0008, "backbone_lr": 0.0004},
    "s": {"base_lr": 0.00025, "backbone_lr": 0.00006},
    "m": {"base_lr": 0.00015, "backbone_lr": 0.00002},
    "l": {"base_lr": 0.00016, "backbone_lr": 0.00001},
    "x": {"base_lr": 0.0002, "backbone_lr": 0.000002},
}


def _deep_merge(target: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = value
    return target


def _resolve_python_executable(value: str | Path | None) -> Path:
    requested = str(value or sys.executable)
    candidate = Path(requested).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    discovered = shutil.which(requested)
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError(f"Training Python executable does not exist: {requested}")


class TrainingProcessError(subprocess.CalledProcessError):
    """Keep subprocess details while showing the trainer's actual failure."""

    def __init__(self, returncode: int, command: list[str], output: str, log_path: Path | None = None):
        super().__init__(returncode, command, output=output)
        self.log_path = log_path

    def __str__(self) -> str:
        tail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", self.output or "")[-8192:].strip()
        traceback_start = tail.find("Traceback (most recent call last):")
        if traceback_start >= 0:
            tail = tail[traceback_start:]
        else:
            tail = "\n".join(tail.splitlines()[-20:])
        message = f"Training process exited with status {self.returncode}."
        if tail:
            message += "\n" + tail
        ram_failure = re.search(r"\b(?:MemoryError|_ArrayMemoryError)\b", tail, re.IGNORECASE) or any(
            marker in tail.lower() for marker in ("unable to allocate", "paging file is too small", "defaultcpuallocator")
        )
        if ram_failure:
            message += (
                "\nSystem RAM allocation failed. Set Data workers to 0 (CLI: --workers 0) "
                "and close other memory-heavy applications before retrying."
            )
        elif re.search(r"DataLoader worker.*exited unexpectedly", tail, re.IGNORECASE):
            message += (
                "\nAn image-loading worker stopped unexpectedly; the log does not identify why. "
                "Set Data workers to 0 (CLI: --workers 0) before retrying. "
                "This loads images in the trainer process and surfaces image-loading errors directly. "
                "Restart the notebook kernel and rerun the setup/dashboard cells to load updated defaults."
            )
        if self.log_path is not None:
            message += f"\nFull trainer log: {self.log_path}"
        return message


def _run_training_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
    output_callback: Callable[[str], None] | None = None,
    output_log: str | Path | None = None,
) -> None:
    """Run a trainer and optionally mirror its output from a durable log.

    A file-backed stream prevents a notebook or extension-host restart from
    invalidating the trainer's stdout handle. The callback is best-effort: a
    dashboard rendering failure must not terminate a long-running training job.
    """
    if output_log is None and output_callback is None:
        subprocess.run(command, cwd=cwd, env=environment, check=True)
        return

    if output_log is not None:
        log_path = Path(output_log).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        popen_options: dict[str, Any] = {}
        if os.name == "nt":
            popen_options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        else:
            popen_options["start_new_session"] = True

        with log_path.open("ab", buffering=0) as sink:
            start_offset = sink.tell()
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=sink,
                stderr=subprocess.STDOUT,
                **popen_options,
            )
            pending = ""
            callback = output_callback
            if callback is None:
                def console_callback(fragment: str) -> None:
                    sys.stdout.write(fragment)
                    sys.stdout.flush()

                callback = console_callback

            def emit(text: str) -> None:
                nonlocal pending, callback
                if callback is None:
                    return
                for character in text:
                    pending += character
                    if character in "\r\n":
                        try:
                            callback(pending)
                        except Exception:
                            # The log remains authoritative if the notebook UI
                            # disappears or rejects further widget messages.
                            callback = None
                        pending = ""

            try:
                with log_path.open("rb") as source:
                    source.seek(start_offset)
                    while process.poll() is None:
                        chunk = source.read()
                        if chunk:
                            emit(chunk.decode("utf-8", errors="replace"))
                        else:
                            time.sleep(0.1)
                    chunk = source.read()
                    if chunk:
                        emit(chunk.decode("utf-8", errors="replace"))
                if pending and callback is not None:
                    try:
                        callback(pending)
                    except Exception:
                        pass
                return_code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if return_code:
            with log_path.open("rb") as source:
                source.seek(0, os.SEEK_END)
                source.seek(max(start_offset, source.tell() - 16384))
                tail = source.read().decode("utf-8", errors="replace")
            raise TrainingProcessError(return_code, command, tail, log_path)
        return

    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    pending = ""
    tail = ""
    try:
        while True:
            character = process.stdout.read(1)
            if character == "":
                break
            pending += character
            if character in "\r\n":
                tail = (tail + pending)[-16384:]
                output_callback(pending)
                pending = ""
        if pending:
            tail = (tail + pending)[-16384:]
            output_callback(pending)
        return_code = process.wait()
    except Exception:
        process.terminate()
        process.wait()
        raise
    if return_code:
        raise TrainingProcessError(return_code, command, tail)


def _validate_dfine_seg_options(options: DFineSegOptions) -> None:
    if options.model_size not in _DFINE_SEG_LEARNING_RATES:
        raise ValueError("D-FINE-seg model_size must be one of n, s, m, l, or x")
    if options.epochs <= 0 or options.imgsz <= 0:
        raise ValueError("D-FINE-seg epochs and image size must be positive")
    if options.batch == 0 or options.workers < 0 or options.early_stopping < 0:
        raise ValueError("D-FINE-seg batch cannot be zero; workers/patience cannot be negative")
    if options.min_epochs_before_early_stop < 0:
        raise ValueError("Minimum epochs cannot be negative")
    if options.grad_accumulation <= 0 or options.no_mosaic_epochs < 0:
        raise ValueError("Gradient accumulation must be positive and no-mosaic epochs cannot be negative")
    for name in ("conf_threshold", "iou_threshold", "mosaic_probability"):
        value = getattr(options, name)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if options.aux_optimizer not in {"adamw", "adan"}:
        raise ValueError("D-FINE-seg auxiliary optimizer must be adamw or adan")


def _dfine_seg_config(
    upstream: Path,
    run_dir: Path,
    prepared: dict[str, Any],
    task: Literal["detect", "segment"],
    options: DFineSegOptions,
) -> Path:
    base_config = upstream / "config.yaml"
    train_module = upstream / "src" / "dl" / "train.py"
    dataset_module = upstream / "src" / "dl" / "dataset.py"
    if not base_config.is_file() or not train_module.is_file() or not dataset_module.is_file():
        raise FileNotFoundError(f"Not an ArgoHA/D-FINE-seg checkout: {upstream}")
    payload = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"D-FINE-seg config must contain a YAML mapping: {base_config}")
    if not isinstance(options.extra_config, dict):
        raise ValueError("D-FINE-seg extra_config must be a mapping")
    _deep_merge(payload, _final_dfine_extra_config(task))
    _deep_merge(payload, options.extra_config)

    model_dir = run_dir / "model"
    prepared_root = Path(prepared["root"])
    learning_rates = _DFINE_SEG_LEARNING_RATES[options.model_size]
    if options.pretrained_model:
        configured_pretrained = Path(options.pretrained_model).expanduser()
        pretrained = (
            configured_pretrained.resolve()
            if configured_pretrained.is_absolute()
            else (upstream / configured_pretrained).resolve()
        )
    else:
        prefix = "dfine_seg_" if task == "segment" else "dfine_"
        pretrained = (
            upstream
            / "pretrained"
            / f"{prefix}{options.model_size}_{options.pretrained_dataset}.pt"
        ).resolve()

    payload.update(
        {
            "project_name": "model-trainer",
            "exp_name": run_dir.name,
            "exp": run_dir.name,
            "model_name": options.model_size,
            "task": task,
            "now_dir": datetime.now().strftime("%Y-%m-%d"),
        }
    )
    train = payload.setdefault("train", {})
    decision_metrics = train.setdefault("decision_metrics", ["mAP_50_95" if task == "detect" else "mAP_50_95_mask"])
    supported_metrics = {
        "f1", "iou", "precision", "recall", "mAP_50", "mAP_50_95",
        "mAP_50_mask", "mAP_50_95_mask",
    }
    if (
        not isinstance(decision_metrics, list)
        or not decision_metrics
        or any(not isinstance(metric, str) or metric not in supported_metrics for metric in decision_metrics)
        or (task == "detect" and any(metric.endswith("_mask") for metric in decision_metrics))
    ):
        raise ValueError("D-FINE-seg train.decision_metrics must be a non-empty list of supported task metrics")
    train.update(
        {
            "root": str(run_dir.parent),
            "pretrained_dataset": options.pretrained_dataset,
            "pretrained_model_path": str(pretrained),
            "coco_dataset": True,
            "data_path": str(prepared_root),
            "path_to_test_data": str(prepared_root / "images" / "test"),
            "path_to_save": str(model_dir),
            "debug_img_path": str(run_dir / "debug_images"),
            "eval_preds_path": str(run_dir / "eval_preds"),
            "bench_img_path": str(run_dir / "bench_images"),
            "infer_path": str(run_dir / "inference"),
            "device": options.device,
            "label_to_name": prepared["classes"],
            "use_one_class": False,
            "img_size": [options.imgsz, options.imgsz],
            "in_channels": 3,
            "to_visualize_eval": options.visualize_validation,
            "debug_img_processing": False,
            "amp_enabled": options.amp,
            "amp_dtype": options.amp_dtype,
            "clip_max_norm": options.clip_max_norm,
            "batch_size": options.batch,
            "b_accum_steps": options.grad_accumulation,
            "epochs": options.epochs,
            "early_stopping": options.early_stopping,
            "min_epochs_before_early_stop": options.min_epochs_before_early_stop,
            "num_workers": options.workers,
            "conf_thresh": options.conf_threshold,
            "iou_thresh": options.iou_threshold,
            "use_wandb": options.use_wandb,
            "use_ema": options.use_ema,
            "use_scheduler": options.use_scheduler,
            "use_muon": options.use_muon,
            "aux_optimizer": options.aux_optimizer,
            "base_lr": (
                options.base_lr if options.base_lr is not None else learning_rates["base_lr"]
            ),
            "backbone_lr": (
                options.backbone_lr
                if options.backbone_lr is not None
                else learning_rates["backbone_lr"]
            ),
            "weight_decay": options.weight_decay,
            "seed": options.seed,
            "deterministic": options.deterministic,
            "cudnn_fixed": options.deterministic,
            "colour_space": prepared.get("colour_space", "rgb"),
            "input_representation": {key: prepared[key] for key in (
                "colour_space", "hsv_encoding", "hsv_storage_order", "tensor_channel_order", "tensor_scaling",
                "hue_range", "saturation_range", "value_range") if key in prepared},
        }
    )
    train.setdefault("ddp", {}).update({"enabled": False, "n_gpus": 1})
    train.setdefault("mosaic_augs", {}).update(
        {
            "mosaic_prob": options.mosaic_probability,
            "no_mosaic_epochs": options.no_mosaic_epochs,
        }
    )

    destination = run_dir / "dfine_seg_training.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return destination


def _ensure_dfine_seg_pretrained(config: Path, *, automatic: bool) -> None:
    """Use Windows' trusted TLS stack when Hugging Face is unavailable to Python SSL."""
    payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    pretrained = Path(str(payload.get("train", {}).get("pretrained_model_path", "")))
    if pretrained.is_file():
        return
    if not automatic:
        raise FileNotFoundError(f"D-FINE-seg pretrained checkpoint does not exist: {pretrained}")
    if os.name != "nt":
        # Upstream D-FINE-seg downloads the checkpoint with huggingface_hub.
        return
    curl = shutil.which("curl.exe")
    if not curl:
        # Keep the upstream downloader as the fallback on non-standard Windows installs.
        return

    pretrained.parent.mkdir(parents=True, exist_ok=True)
    partial = pretrained.with_suffix(pretrained.suffix + ".part")
    if partial.exists():
        partial.unlink()
    url = (
        "https://huggingface.co/ArgoSA/D-FINE-seg/resolve/main/"
        + pretrained.name
    )
    try:
        subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--retry-delay",
                "2",
                "--output",
                str(partial),
                url,
            ],
            check=True,
        )
        if not partial.is_file() or partial.stat().st_size < 1_000_000:
            size = partial.stat().st_size if partial.is_file() else 0
            raise RuntimeError(
                f"Downloaded D-FINE-seg checkpoint is unexpectedly small ({size} bytes): {url}"
            )
        partial.replace(pretrained)
    except Exception:
        if partial.exists():
            partial.unlink()
        raise


def train_dfine_seg_export(
    export_root: str | Path,
    task: Literal["detect", "segment"],
    upstream_dir: str | Path,
    options: DFineSegOptions | None = None,
    *,
    python_executable: str | Path | None = None,
    execute: bool = True,
    output_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Prepare and train ArgoHA/D-FINE-seg without mutating the labelled export."""
    options = options or final_dfine_seg_options(task)
    _validate_dfine_seg_options(options)
    if task == "segment" and options.pretrained_dataset != "coco" and not options.pretrained_model:
        raise ValueError(
            "D-FINE-seg publishes automatic instance-segmentation checkpoints for COCO only; "
            "select COCO or provide an explicit pretrained model"
        )
    upstream = Path(upstream_dir).expanduser().resolve()
    executable = _resolve_python_executable(python_executable)
    run_name = options.run_name or (
        f"dfine-seg-{options.model_size}-{task}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    run_dir = Path(options.output_dir).expanduser().resolve() / "dfine-seg" / run_name
    original_root = str(Path(export_root).expanduser().resolve())
    export_root = prepare_colour_dataset(export_root, task, run_dir.parent / "_colour_datasets", options.colour_space)
    prepared = prepare_dfine_seg_dataset(export_root, task, run_dir / "prepared_dataset")
    test_images = prepared["splits"].get("test", {}).get("images", 0)
    test_metrics_path = run_dir / "test_metrics.json" if test_images else None
    console_log = run_dir / "trainer_console.log"
    config = _dfine_seg_config(upstream, run_dir, prepared, task, options)
    runner = Path(__file__).resolve().parents[2] / "scripts" / "dfine_seg_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Model Trainer D-FINE-seg runner is missing: {runner}")
    command = [
        str(executable),
        str(runner),
        "--upstream",
        str(upstream),
        "--config",
        str(config),
        "--action",
        "train",
    ]
    metadata = {
        "schema_version": "1.0",
        "created_at": _utc_now(),
        "trainer": "ArgoHA/D-FINE-seg",
        **policy_metadata(options.epochs, options.min_epochs_before_early_stop, options.early_stopping,
                          "mAP_50_95" if task == "detect" else "mAP_50_95_mask"),
        "model_family": "D-FINE" if task == "detect" else "D-FINE-seg",
        "model_variant": options.model_size.upper(),
        "model": f"D-FINE-seg-{options.model_size.upper()}",
        "task": task,
        "upstream": str(upstream),
        "python": str(executable),
        "command": command,
        "options": asdict(options),
        "experiment": {"architecture": "ArgoHA/D-FINE-seg", "colour_space": dataset_colour(Path(prepared["source_data_yaml"])),
                       "recipe": "Project D-FINE detection recipe" if task == "detect" else "Project D-FINE segmentation recipe",
                       "model_size": options.model_size.upper(), "task": task,
                       "input_representation": colour_metadata(Path(prepared["source_data_yaml"])),
                       "source_dataset": original_root, "dataset_path": prepared["root"]},
        "effective_training_config": yaml.safe_load(config.read_text(encoding="utf-8")),
        "augmentation": {"mixup": 0.0, "mixup_supported": False},
        "dataset": {**prepared, **dataset_identity(Path(prepared["source_data_yaml"]))},
        "selection": {"split": "val", "metric": yaml.safe_load(config.read_text(encoding="utf-8"))["train"]["decision_metrics"],
                      "best_epoch": None, "best_validation_metric": None,
                      "best_checkpoint_path": str(run_dir / "model" / "model.pt")},
        "artifacts": {
            "run_dir": str(run_dir),
            "model_dir": str(run_dir / "model"),
            "config": str(config),
            "best": str(run_dir / "model" / "model.pt"),
            "last": str(run_dir / "model" / "last.pt"),
            "log": str(run_dir / "model" / "train_log.txt"),
            "console_log": str(console_log),
            "test_metrics": str(test_metrics_path) if test_metrics_path else None,
        },
    }
    metadata_path = run_dir / "run_metadata.json"
    _write_json(metadata_path, metadata)
    if execute:
        _ensure_dfine_seg_pretrained(config, automatic=not bool(options.pretrained_model))
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["PYTHONHASHSEED"] = str(options.seed)
        environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if output_callback:
            output_callback(f"Trainer console log: {console_log}\n")
        try:
            _run_training_process(
                command,
                cwd=upstream,
                environment=environment,
                output_callback=output_callback,
                output_log=console_log,
            )
        except BaseException as error:
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
            update_metadata(metadata_path, actual_stop_epoch=recorded.get("actual_stop_epoch") or recorded.get("completed_epochs", 0),
                            stop_reason="manual_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                            training_error=recorded.get("training_error") or str(error),
                            training_process_error=str(error))
            raise
        if test_metrics_path:
            _write_json(
                test_metrics_path,
                {
                    "split": "test",
                    "images": test_images,
                    "checkpoint": str(run_dir / "model" / "model.pt"),
                    "metrics": _dfine_seg_test_metrics(run_dir / "model" / "metrics.csv"),
                },
            )
    return {
        "run_dir": str(run_dir),
        "model_dir": str(run_dir / "model"),
        "prepared_dataset": prepared["root"],
        "config": str(config),
        "metadata": str(metadata_path),
        "command": command,
        "test_metrics": str(test_metrics_path) if execute and test_metrics_path else None,
        "test_images": test_images,
        "console_log": str(console_log),
        "executed": execute,
    }


def dfine_seg_loader_smoke(
    export_root: str | Path,
    task: Literal["detect", "segment"],
    upstream_dir: str | Path,
    options: DFineSegOptions | None = None,
    *,
    python_executable: str | Path | None = None,
) -> dict[str, Any]:
    """Load one prepared batch through the selected D-FINE-seg checkout."""
    options = options or final_dfine_seg_options(task)
    with tempfile.TemporaryDirectory(prefix="model-trainer-dfine-seg-smoke-") as temporary:
        smoke_options = replace(
            options,
            batch=1,
            workers=0,
            output_dir=temporary,
            run_name="loader-smoke",
            visualize_validation=False,
        )
        dry_run = train_dfine_seg_export(
            export_root,
            task,
            upstream_dir,
            smoke_options,
            python_executable=python_executable,
            execute=False,
        )
        command = [
            *dry_run["command"][:-1],
            "smoke",
        ]
        completed = subprocess.run(
            command,
            cwd=Path(upstream_dir).expanduser().resolve(),
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            raise RuntimeError(
                f"D-FINE-seg loader smoke failed with exit code {completed.returncode}:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        prefix = "MODEL_TRAINER_RESULT="
        line = next(
            (value.removeprefix(prefix) for value in reversed(completed.stdout.splitlines()) if value.startswith(prefix)),
            "",
        )
        if not line:
            raise RuntimeError(
                f"D-FINE-seg loader smoke did not return a result:\n"
                f"{completed.stdout}\n{completed.stderr}"
            )
        result = json.loads(line)
        result["upstream"] = str(Path(upstream_dir).expanduser().resolve())
        return result


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
