"""One fresh final detector run, then validation confidence selection and test."""
from __future__ import annotations

import json
import os
import runpy
import sys
from functools import partial
from pathlib import Path

import yaml

from .detection_data import detection_ground_truth
from .detection_metrics import evaluate_test, select_confidence
from .detector_experiment import POLICY, verify_experiment, write_json
from .experiment_tracking import update_metadata
from .runtime import seed_training


def protocol_for(manifest, job, directory):
    recipe = manifest["recipes"][job["model"]]
    training = recipe["training"]
    flip = training.get("fliplr", training.get("horizontal_flip", 0.))
    return {"rgb_yaml": manifest["paired_dataset"]["rgb"], "colour": job["colour"],
            "batch": training["batch"], "workers": training["workers"], "seed": job["seed"],
            "metadata": str(directory / "run_metadata.json"),
            "augmentation": {"colour": "disabled", "horizontal_flip": flip, "vertical_flip": 0.,
                             "rotation": 0., "perspective": 0., "scale_jitter": 0., "translation": 0.,
                             "mosaic": 0., "mixup": 0., "cutmix": 0.,
                             "geometry": "640-square aspect-preserving letterbox, RGB geometry before representation"}}


def official_config(manifest, job, directory, protocol):
    from .datasets import prepare_dfine_seg_dataset
    from .train import TrainingOptions, _dfine_config
    training = manifest["recipes"]["dfine_n"]["training"]
    prepared = prepare_dfine_seg_dataset(Path(protocol["rgb_yaml"]).parent, "detect", directory / "prepared_dataset")
    options = TrainingOptions(epochs=300, imgsz=640, batch=training["batch"], workers=training["workers"],
                              seed=job["seed"], amp=training["amp"], dfine_base_lr=training["head_lr"],
                              dfine_backbone_lr=training["backbone_lr"], dfine_weight_decay=training["weight_decay"],
                              dfine_clip_max_norm=training["clip_max_norm"], dfine_warmup_steps=training["warmup_steps"])
    path = _dfine_config(Path(manifest["dfine_root"]), directory, prepared, options)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config.update(device="cuda:0", use_wandb=False, checkpoint_freq=training["checkpoint_frequency"], lr_scheduler=training["lr_scheduler"],
                  ema={"type": "ModelEMA", **training["ema"]}, HGNetv2={"pretrained": False},
                  controlled_detection=protocol, controlled_policy=POLICY)
    for split in ("train", "val"):
        loader = config[f"{split}_dataloader"]
        loader.update(drop_last=False, shuffle=split == "train")
        ops = [{"type": "ControlledLetterbox", "size": 640, "colour": job["colour"],
                "flip_probability": training["horizontal_flip"] if split == "train" else 0.},
               {"type": "ConvertPILImage", "dtype": "float32", "scale": True}]
        if split == "train":
            ops.append({"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True})
        loader["dataset"]["transforms"] = {"type": "Compose", "ops": ops, "policy": {"name": "default"}}
    config["train_dataloader"]["collate_fn"].update(base_size=640, base_size_repeat=None,
                                                    stop_epoch=training["stage_transition_epoch"],
                                                    ema_restart_decay=training["ema"]["decay"])
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def train_yolo(manifest, job, directory, protocol):
    from ultralytics import YOLO
    from .yolo_detection import ControlledDetectionTrainer, collect_yolo_predictions
    recipe = manifest["recipes"]["yolo26n"]
    arguments = {key: value for key, value in recipe["training"].items() if value is not None}
    arguments.update(data=protocol["rgb_yaml"], seed=job["seed"], device="0",
                     project=str(directory.parent), name=directory.name, exist_ok=True,
                     patience=40, val=True, plots=False, save=True, auto_augment=None)
    model = YOLO(recipe["checkpoint"])
    model.train(trainer=partial(ControlledDetectionTrainer, protocol=protocol), **arguments)
    checkpoint = directory / "weights/best_controlled.pt"
    model = YOLO(checkpoint)
    update_metadata(Path(protocol["metadata"]), checkpoint_paths={"best": str(checkpoint), "last": str(directory / "weights/last.pt")})
    return lambda split: collect_yolo_predictions(model, protocol, split)


def train_official(manifest, job, directory, protocol):
    from .official_detection import install_official_geometry, install_controlled_solver, predict_official
    upstream = Path(manifest["dfine_root"])
    sys.path.insert(0, str(upstream))
    install_official_geometry()
    config = official_config(manifest, job, directory, protocol)
    install_controlled_solver(protocol, Path(protocol["metadata"]))
    original_argv, original_directory = sys.argv[:], Path.cwd()
    try:
        os.chdir(upstream)
        sys.argv = [str(upstream / "train.py"), "-c", str(config), "--seed", str(job["seed"]),
                    "--tuning", manifest["recipes"]["dfine_n"]["checkpoint"], "--use-amp"]
        runpy.run_path(str(upstream / "train.py"), run_name="__main__")
    finally:
        sys.argv = original_argv
        os.chdir(original_directory)
    from src.core import YAMLConfig
    from src.solver.det_solver import DetSolver
    checkpoint = directory / "best.pth"
    cfg = YAMLConfig(str(config), resume=str(checkpoint))
    solver = DetSolver(cfg)
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    update_metadata(Path(protocol["metadata"]), checkpoint_paths={"best": str(checkpoint), "last": str(directory / "last.pth")})
    def collect(split):
        truth = detection_ground_truth(protocol["rgb_yaml"], split)
        return truth, predict_official(model, solver.postprocessor, truth, job["colour"], solver.device, protocol["batch"])
    return collect


def final_evaluation(collect, metadata_path):
    """Persist the validation-only decision before opening test images."""
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("test_evaluation"):
        raise FileExistsError("Held-out test evaluation was already started; preserving the once-only result")
    if not 100 <= metadata.get("completed_epochs", 0) <= 300:
        raise ValueError("Final test requires a completed controlled training run")
    truth, predictions = collect("val")
    selection = select_confidence(truth, predictions, split="val")
    write_json(metadata_path.parent / "confidence_selection.json", selection)
    update_metadata(metadata_path, validation_confidence_selection=selection,
                    test_evaluation={"status": "started", "inference_count": 1, "selection_frozen_before_test": True})
    truth, predictions = collect("test")
    result = evaluate_test(truth, predictions, selection)
    write_json(metadata_path.parent / "test_predictions.json", predictions)
    write_json(metadata_path.parent / "test_metrics.json", result)
    update_metadata(metadata_path, final_test_metrics=result, status="complete",
                    test_evaluation={"status": "complete", "inference_count": 1, "selection_frozen_before_test": True})
    return result


def run_one(manifest_path, run_id):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    verify_experiment(manifest)
    import torch
    if not torch.cuda.is_available() or "RTX 4060 Laptop" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Controlled final training requires the RTX 4060 Laptop GPU")
    jobs = [j for j in manifest["matrix"] if j["run_id"] == run_id]
    if len(jobs) != 1:
        raise ValueError("Run is not uniquely present in the frozen matrix")
    job = jobs[0]
    directory = Path(manifest["output_dir"]) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    if any(p.name != "trainer_console.log" for p in directory.iterdir()):
        raise FileExistsError(f"Preserving prior run contents: {directory}")
    protocol = protocol_for(manifest, job, directory)
    recipe = manifest["recipes"][job["model"]]
    from .train import _runtime
    write_json(Path(protocol["metadata"]), {"status": "training", **job, "architecture": recipe["architecture"],
               "model_size": "Nano", "colour_representation": job["colour"],
               "manifest_sha256": manifest["manifest_sha256"], "pretrained_checkpoint": recipe["checkpoint"],
               "pretrained_checkpoint_sha256": recipe["checkpoint_sha256"], "frozen_recipe": recipe,
               "dataset": manifest["paired_dataset"], "image_size": 640, "policy": POLICY,
               "max_epochs": 300, "min_epochs": 100, "patience": 40, "completed_epochs": 0,
               "evaluation_policy": manifest["evaluation"], "tuning": manifest["tuning"],
               "fewer_seeds_reason": manifest["fewer_seeds_reason"], "runtime": _runtime(),
               "library_versions": manifest["library_versions"], "augmentation": protocol["augmentation"]})
    seed_training(job["seed"], deterministic=True)
    try:
        collect = (train_yolo if job["model"] == "yolo26n" else train_official)(manifest, job, directory, protocol)
        return final_evaluation(collect, protocol["metadata"])
    except BaseException as error:
        update_metadata(Path(protocol["metadata"]), status="manual_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                        training_error=str(error))
        raise
