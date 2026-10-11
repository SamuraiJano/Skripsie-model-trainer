"""One real forward/backward batch only: never train, step, or save weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("yolo", "project_dfine"), required=True)
    parser.add_argument("--task", choices=("detect", "segment"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--upstream", type=Path)
    args = parser.parse_args()
    import torch
    from naartjie_cv.runtime import seed_training
    seed_training(0)
    result = {"framework": args.framework, "task": args.task, "status": "FAIL",
              "representation": "RGB", "optimizer_updates": 0, "training_epochs": 0}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for physical batch memory validation")
        torch.cuda.reset_peak_memory_stats()
        if args.framework == "yolo":
            from naartjie_cv.final_validation import yolo_resolved_config
            from naartjie_cv.train import YOLO_FINAL_CHECKPOINTS, _resolved_yolo_data_yaml
            from ultralytics.models.yolo.detect import DetectionTrainer
            from naartjie_cv.yolo_training import MaskSelectionTrainer
            config = yolo_resolved_config(args.data, args.task)
            config.pop("training_policy")
            # Filter fields which native get_cfg adds for display only.
            config.pop("save_dir", None)
            config["data"] = str(_resolved_yolo_data_yaml(args.data / "data.yaml", args.output.parent, args.task))
            config.update(model=str(PROJECT / YOLO_FINAL_CHECKPOINTS[args.task]),
                          project=str(args.output.parent), name=args.output.stem, plots=False)
            trainer_type = DetectionTrainer if args.task == "detect" else MaskSelectionTrainer
            trainer = trainer_type(overrides=config)
            trainer._setup_train()
            trainer._model_train()
            batch = trainer.preprocess_batch(next(iter(trainer.train_loader)))
            with torch.autocast("cuda", dtype=torch.float16, enabled=bool(trainer.amp)):
                loss, _ = trainer.model(batch)
                loss = loss.sum()
            # This is a memory/integration diagnostic, with no optimizer update.
            # Use the raw loss for backward so the native FP16 scaler's initial
            # overflow/recovery does not masquerade as a memory failure.
            loss.backward()
            result["native_grad_scaler_enabled"] = trainer.scaler.is_enabled()
            result["native_initial_loss_scale"] = trainer.scaler.get_scale()
            result["forward_amp_dtype"] = "float16" if trainer.amp else "float32"
            physical = trainer.batch_size
            accumulation = trainer.accumulate
            shape = list(batch["img"].shape)
            optimizer_name = type(trainer.optimizer).__name__
            result["actual_scheduler"] = {"horizon_epochs": trainer.epochs, "terminal_factor": trainer.lf(300),
                                          "warmup_epochs": trainer.args.warmup_epochs}
        else:
            from omegaconf import OmegaConf
            os.chdir(args.upstream)
            sys.path.insert(0, str(args.upstream))
            from src.dl import dataset as dataset_module
            from scripts.dfine_seg_runner import (_install_consistent_letterbox, _install_reproducible_transforms,
                                                   _install_detection_lr_groups)
            _install_consistent_letterbox(dataset_module)
            _install_reproducible_transforms(dataset_module)
            from src.dl import train as train_module
            train_module.set_seeds = seed_training
            _install_detection_lr_groups(train_module)
            config = OmegaConf.load(args.config)
            trainer = train_module.Trainer(config)
            trainer.model.train()
            inputs, targets, _ = next(iter(trainer.train_loader))
            inputs = inputs.to(trainer.device)
            targets = [{k: v.to(trainer.device) if v is not None and hasattr(v, "to") else v
                        for k, v in target.items()} for target in targets]
            with torch.autocast("cuda", dtype=trainer.amp_dtype, enabled=trainer.amp_enabled):
                outputs = trainer.model(inputs, targets=targets)
            with torch.autocast("cuda", enabled=False):
                losses = trainer.loss_fn(outputs, targets)
                loss = sum(losses.values()) / trainer.b_accum_steps
            loss.backward()
            result["native_grad_scaler_enabled"] = trainer.scaler.is_enabled() if trainer.amp_enabled else False
            result["native_initial_loss_scale"] = trainer.scaler.get_scale() if trainer.amp_enabled else 1.0
            result["forward_amp_dtype"] = str(trainer.amp_dtype) if trainer.amp_enabled else "float32"
            physical, accumulation = config.train.batch_size, trainer.b_accum_steps
            shape = list(inputs.shape)
            optimizer_name = "Muon + " + config.train.aux_optimizer.title() if config.train.use_muon else "AdamW"
            from naartjie_cv.final_validation import onecycle_diagnostics
            result["actual_scheduler"] = onecycle_diagnostics(trainer.scheduler, trainer.optimizer_steps_per_epoch, config.train.epochs)
        torch.cuda.synchronize()
        assert shape == [physical, 3, 640, 640]
        assert trainer.train_loader.num_workers == 4
        assert torch.isfinite(loss).all(), "Nonfinite loss"
        gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert gradients and all(bool(g.isfinite().all()) for g in gradients), "Nonfinite or missing gradients"
        result.update(status="PASS", physical_batch=physical, gradient_accumulation=accumulation,
                      backward_loss_scaling="unscaled memory diagnostic; native training GradScaler is unchanged",
                      effective_batch=physical * accumulation, tensor_shape=shape, workers=trainer.train_loader.num_workers,
                      loss=float(loss.detach()), optimizer=optimizer_name,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      device=torch.cuda.get_device_name())
    except Exception as error:
        result.update(error=str(error), error_type=type(error).__name__,
                      oom=isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("MODEL_TRAINER_RESULT=" + json.dumps(result))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
