"""Final controlled-experiment checks; no model update or training epoch."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import math
from types import SimpleNamespace

import torch
import yaml

from .experiment_tracking import dataset_identity
from .runtime import seed_training
from .train import TrainingOptions, YOLO_FINAL_CHECKPOINTS, final_yolo_options, resolved_yolo_arguments
from .training_policy import policy_metadata, DIAGNOSTIC_EPOCHS, optimizer_steps_per_epoch


ALLOWED_CONFIG_DIFFERENCES = {
    "yolo": ("data", "name", "project", "save_dir"),
    "project_dfine": ("exp", "exp_name", "train.root", "train.data_path", "train.path_to_test_data",
                      "train.path_to_save", "train.debug_img_path", "train.eval_preds_path", "train.bench_img_path",
                      "train.infer_path", "train.colour_space", "train.input_representation"),
    "official_dfine": ("output_dir", "colour_space", "input_representation",
                       "train_dataloader.dataset.img_folder", "train_dataloader.dataset.ann_file",
                       "val_dataloader.dataset.img_folder", "val_dataloader.dataset.ann_file"),
}


def configuration_differences(rgb: dict, hsv: dict, framework: str) -> dict:
    """Compare all resolved fields, excluding only explicit representation/run paths."""
    def controlled(config):
        config = copy.deepcopy(config)
        for field in ALLOWED_CONFIG_DIFFERENCES[framework]:
            parent = config
            parts = field.split(".")
            for part in parts[:-1]:
                parent = parent.get(part, {})
            parent.pop(parts[-1], None)
        return config

    def diff(a, b, path=""):
        if isinstance(a, dict) and isinstance(b, dict):
            result = {}
            for key in a.keys() | b.keys():
                result.update(diff(a.get(key, "<missing>"), b.get(key, "<missing>"), f"{path}.{key}".strip(".")))
            return result
        return {} if a == b else {path: {"rgb": a, "hsv": b}}
    return diff(controlled(rgb), controlled(hsv))


def yolo_resolved_config(root: Path, task: str) -> dict:
    from ultralytics.cfg import get_cfg
    options = TrainingOptions(yolo=final_yolo_options(task))
    args = resolved_yolo_arguments(options, root / "data.yaml", task, root / "preflight", "preflight")
    args["model"] = YOLO_FINAL_CHECKPOINTS[task]
    return {**vars(get_cfg(overrides=args)), "training_policy": policy_metadata(
        options.epochs, options.yolo.min_epochs_before_early_stop, options.yolo.patience,
        "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)")}


def yolo_training_batch(root: Path, task: str) -> dict:
    """One actual training dataset/collate/preprocess batch, using fixed seed/geometry."""
    from ultralytics.cfg import get_cfg
    from ultralytics.models.yolo.detect import DetectionTrainer
    from .yolo_training import MaskSelectionTrainer
    seed_training()
    trainer_type = DetectionTrainer if task == "detect" else MaskSelectionTrainer
    trainer = trainer_type.__new__(trainer_type)
    config = yolo_resolved_config(root, task)
    config.pop("training_policy")
    trainer.args = get_cfg(overrides=config)
    names = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))["names"]
    trainer.data = {"names": names, "nc": len(names), "channels": 3}
    trainer.model = torch.nn.Module()
    trainer.model.stride = torch.tensor([32])
    trainer.device = torch.device("cpu")
    trainer.stride = 32
    directory = dataset_identity(root / "data.yaml")["splits"]["train"]["path"]
    dataset = trainer.build_dataset(directory, mode="train", batch=1)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=dataset.collate_fn)
    batch = trainer.preprocess_batch(next(iter(loader)))
    tensor = batch["img"]
    return {"tensor_shape": list(tensor.shape), "tensor_dtype": str(tensor.dtype),
            "tensor_finite": bool(tensor.isfinite().all()), "tensor_min": float(tensor.min()),
            "tensor_max": float(tensor.max()), "image_paths": batch["im_file"],
            "target_classes": batch["cls"].tolist(), "target_boxes": batch["bboxes"].tolist(),
            "target_boxes_finite": bool(batch["bboxes"].isfinite().all()),
            "target_masks_sha256": hashlib.sha256(batch["masks"].numpy().tobytes()).hexdigest() if "masks" in batch else None,
            "tensor_centre_channels": tensor[0, :, tensor.shape[-2] // 2, tensor.shape[-1] // 2].tolist()}


def verify_batch(batch: dict) -> None:
    if batch["tensor_shape"] != [1, 3, 640, 640]:
        raise ValueError(f"Invalid tensor shape: {batch['tensor_shape']}")
    if batch["tensor_dtype"] != "torch.float32" or not batch["tensor_finite"]:
        raise ValueError("Model input must be finite float32")
    if not (0 <= batch["tensor_min"] <= batch["tensor_max"] <= 1.000001):
        raise ValueError("Model input must use /255 scaling in [0,1]")
    if not batch["target_boxes_finite"]:
        raise ValueError("Nonfinite label boxes")


def verify_batch_pair(rgb: dict, hsv: dict) -> None:
    verify_batch(rgb)
    verify_batch(hsv)
    for key in ("target_classes", "target_boxes", "target_masks_sha256"):
        if rgb.get(key) != hsv.get(key):
            raise ValueError(f"RGB/HSV real loader labels differ: {key}")
    if [Path(p).stem for p in rgb["image_paths"]] != [Path(p).stem for p in hsv["image_paths"]]:
        raise ValueError("RGB/HSV batch source identities differ")


def project_dfine_training_batch(config: Path, upstream: Path) -> dict:
    runner = Path(__file__).resolve().parents[2] / "scripts" / "dfine_seg_runner.py"
    completed = subprocess.run([sys.executable, str(runner), "--upstream", str(upstream), "--config", str(config),
                                "--action", "smoke"], cwd=upstream, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=180)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("MODEL_TRAINER_RESULT=")), "")
    if not line:
        raise RuntimeError("Project D-FINE did not emit a batch result: " + completed.stdout + completed.stderr)
    return json.loads(line.split("=", 1)[1])


def verify_duration_config(config: dict, framework: str, task: str) -> dict:
    if framework == "yolo":
        policy = config["training_policy"]
        values = (config["epochs"], policy["min_epochs_before_early_stop"], config["patience"], config["workers"])
        metric = "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)"
        assert policy["monitored_metric"] == metric and config["val"] and config["split"] == "val"
        assert policy["validation_frequency"] == 1 and policy["monitor_split"] == "val"
    else:
        train = config["train"]
        values = (train["epochs"], train["min_epochs_before_early_stop"], train["early_stopping"], train["num_workers"])
        metric = "mAP_50_95" if task == "detect" else "mAP_50_95_mask"
        assert train["decision_metrics"] == [metric]
    assert values == (300, 100, 40, 4), f"Final duration/worker policy differs: {values}"
    return {"max_epochs": 300, "min_epochs_before_early_stop": 100, "patience": 40, "workers": 4,
            "monitored_metric": metric, "monitor_split": "val", "validation_frequency": 1,
            "min_delta": 0.0, "improvement_rule": "strict_greater"}


def validation_only_route_contract(upstream: Path) -> dict:
    """Fail if the supported native project loop starts consulting held-out test."""
    import ast
    tree = ast.parse((upstream / "src/dl/train.py").read_text(encoding="utf-8"))
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Trainer")
    train = next(node for node in trainer.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    assert not any(isinstance(node, ast.Attribute) and node.attr == "test_loader" for node in ast.walk(train)), "Test loader entered training/stopping loop"
    evaluations = [node for node in ast.walk(train) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "evaluate"]
    assert len(evaluations) == 1
    loader = next(keyword.value for keyword in evaluations[0].keywords if keyword.arg == "val_loader")
    assert isinstance(loader, ast.Attribute) and loader.attr == "val_loader"
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    checkpoints = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "load_state_dict"]
    test_calls = [node.lineno for node in ast.walk(main) if isinstance(node, ast.Call)
                  and any(isinstance(keyword.value, ast.Attribute) and keyword.value.attr == "test_loader" for keyword in node.keywords)]
    assert checkpoints and test_calls and min(checkpoints) < min(test_calls)
    from ultralytics.utils.metrics import Metric
    import numpy as np
    metric = Metric()
    metric.p, metric.r = np.array([0.99]), np.array([0.98])
    metric.all_ap = np.array([[0.9] + [0.2] * 9])
    assert math.isclose(metric.fitness(), metric.map), "YOLO detection fitness changed from box mAP50-95"
    return {"project_training_loader": "val", "test_loader_in_training_loop": False,
            "project_test_evaluation_after_best_load": True, "yolo_detection_fitness": "box mAP50-95",
            "yolo_segmentation_fitness": "MaskSelectionValidator: mask mAP50-95",
            "test_selection_regression": "test_yolo_records_validation_selection_before_separate_test_evaluation"}


def yolo_scheduler_diagnostics(config: dict, training_images: int) -> dict:
    from ultralytics.models.yolo.detect import DetectionTrainer
    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.args = SimpleNamespace(cos_lr=config["cos_lr"], lrf=config["lrf"])
    trainer.epochs = config["epochs"]
    trainer.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=config["lr0"])
    trainer._setup_scheduler()
    assert trainer.epochs == 300 and not config["cos_lr"]
    batches = math.ceil(training_images / config["batch"])
    warmup_batches = max(round(config["warmup_epochs"] * batches), 100)
    assert warmup_batches == 3 * batches, "Final dataset must resolve the intended three-epoch YOLO warmup"
    samples = [{"epoch": epoch, "native_zero_based_epoch": epoch - 1,
                "lr_factor_during_epoch": trainer.lf(epoch - 1),
                "scheduled_lr": config["lr0"] * trainer.lf(epoch - 1),
                "lr_at_completed_epoch_boundary": config["lr0"] * trainer.lf(epoch)} for epoch in DIAGNOSTIC_EPOCHS]
    assert trainer.lf(0) == 1 and math.isclose(trainer.lf(300), 0.01, abs_tol=1e-12)
    assert all(math.isfinite(trainer.lf(i)) and trainer.lf(i) >= 0 for i in range(301))
    return {"family": "linear", "horizon_epochs": 300, "initial_lr": config["lr0"],
            "warmup_epochs": config["warmup_epochs"], "warmup_batches": warmup_batches,
            "warmup_start_lr": 0.0, "terminal_factor": trainer.lf(300),
            "terminal_lr": config["lr0"] * trainer.lf(300), "samples": samples,
            "note": "Native YOLO epochs are 0..299; lf(300)=0.01 at the completed horizon boundary. Warmup interpolates from zero over three epochs."}


def onecycle_diagnostics(scheduler, steps_per_epoch: int, epochs: int) -> dict:
    total = steps_per_epoch * epochs
    assert epochs == 300 and scheduler.total_steps == total
    saved_epoch = scheduler.last_epoch
    def rates(step):
        scheduler.last_epoch = step
        scheduler._get_lr_called_within_step = True
        try:
            return scheduler.get_lr()
        finally:
            scheduler._get_lr_called_within_step = False
    try:
        samples = [{"epoch": epoch, "optimizer_update": steps_per_epoch * epoch,
                    "lr_used_for_last_update": rates(steps_per_epoch * epoch - 1)} for epoch in DIAGNOSTIC_EPOCHS]
        all_rates = [rates(step) for step in range(total + 1)]
        assert all(math.isfinite(rate) and rate >= 0 for row in all_rates for rate in row)
        peaks = [g["max_lr"] for g in scheduler.optimizer.param_groups]
        minima = [g["min_lr"] for g in scheduler.optimizer.param_groups]
        assert all(math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-16) for a, b in zip(rates(total - 1), minima))
        peak_step = scheduler._schedule_phases[0]["end_step"]
        nearby_peak = rates(round(peak_step))
        assert all(math.isclose(a, b, rel_tol=1e-4) for a, b in zip(nearby_peak, peaks))
        assert all(a < b for a, b in zip(rates(0), nearby_peak))
        return {"family": "OneCycleLR", "horizon_epochs": epochs, "optimizer_steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total, "initial_lrs": rates(0), "peak_lrs": peaks,
                "peak_optimizer_step": peak_step, "minimum_lrs": minima, "samples": samples,
                "post_final_step_lrs": rates(total), "all_rates_finite_nonnegative": True,
                "note": "PyTorch initializes schedule position 0; final update uses position total_steps-1 (minimum). Native post-update step total_steps has a tiny cosine rebound."}
    finally:
        scheduler.last_epoch = saved_epoch


def project_scheduler_diagnostics(config: dict, training_batches: int) -> dict:
    from scripts.dfine_seg_runner import _install_detection_lr_groups
    train = config["train"]
    # Instantiate the same runner adapter on a scalar optimizer; never update a model.
    class DiagnosticTrainer:
        def __init__(self, cfg):
            self.b_accum_steps = cfg.train.b_accum_steps
            self.train_loader = range(training_batches)
            if cfg.train.use_muon:
                aux = cfg.train.base_lr * 2 * (cfg.train.get("adan_lr_mult", 5.0) if cfg.train.aux_optimizer == "adan" else 1)
                peaks = [aux] * 4 + [cfg.train.base_lr * 10 * 2]
            else:
                peaks = [cfg.train.base_lr * 2] * 4
            self.optimizer = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.zeros(1))]} for _ in peaks])
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=peaks,
                epochs=cfg.train.epochs, steps_per_epoch=max(1, training_batches // self.b_accum_steps),
                pct_start=cfg.train.cycler_pct_start, cycle_momentum=False)
    from omegaconf import OmegaConf
    module = SimpleNamespace(Trainer=DiagnosticTrainer, OneCycleLR=torch.optim.lr_scheduler.OneCycleLR)
    _install_detection_lr_groups(module)
    trainer = module.Trainer(OmegaConf.create(config))
    result = onecycle_diagnostics(trainer.scheduler, trainer.optimizer_steps_per_epoch, train["epochs"])
    result.update(training_batches=training_batches, gradient_accumulation=train["b_accum_steps"],
                  warmup_fraction=train["cycler_pct_start"], partial_window_update="included")
    return result
