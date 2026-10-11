from __future__ import annotations

import argparse
from functools import wraps
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Run a generated Model Trainer config inside an ArgoHA/D-FINE-seg checkout."
    )
    result.add_argument("--upstream", required=True)
    result.add_argument("--config", required=True)
    result.add_argument("--action", choices=("train", "smoke"), default="train")
    return result


def _install_initial_best_checkpoint_fallback(train_module: ModuleType) -> None:
    """Ensure D-FINE-seg always has a checkpoint for final evaluation.

    Upstream only creates ``model.pt`` when its decision metric is strictly
    greater than zero.  A valid run whose selected metrics remain at zero still
    creates ``last.pt``, then crashes during final evaluation because it assumes
    that ``model.pt`` exists.  Promote the first completed checkpoint to the
    initial best; later genuine improvements continue to replace it upstream.
    """
    original_save_model = train_module.Trainer.save_model
    if getattr(original_save_model, "_model_trainer_best_fallback", False):
        return

    @wraps(original_save_model)
    def save_model_with_initial_best(self, metrics, best_metric):
        updated_best_metric = original_save_model(self, metrics, best_metric)
        checkpoint_dir = Path(self.path_to_save)
        best_checkpoint = checkpoint_dir / "model.pt"
        if not best_checkpoint.is_file():
            last_checkpoint = checkpoint_dir / "last.pt"
            if not last_checkpoint.is_file():
                raise RuntimeError(
                    "D-FINE-seg completed checkpoint saving without producing "
                    f"either {best_checkpoint} or {last_checkpoint}"
                )
            shutil.copy2(last_checkpoint, best_checkpoint)
            self.early_stopping_steps = 0
            logger = getattr(train_module, "logger", None)
            if logger is not None:
                logger.info(
                    "Established the first completed epoch as the initial best model"
                )
        return updated_best_metric

    save_model_with_initial_best._model_trainer_best_fallback = True
    train_module.Trainer.save_model = save_model_with_initial_best


def _install_consistent_letterbox(dataset_module: ModuleType) -> None:
    """Match validation/test resizing to training and inverse box/mask geometry.

    Upstream's instance loader disables scaleup outside training, but its box
    and mask postprocessors always invert a scaleup-enabled letterbox. Small
    validation images consequently produce misplaced masks against the
    original-resolution polygons. Keep the upstream checkout untouched and
    correct the transforms used by this project's runner and evaluator.
    """
    original_init = dataset_module.CustomDataset._init_augs
    if getattr(original_init, "_model_trainer_consistent_letterbox", False):
        return

    @wraps(original_init)
    def init_with_consistent_letterbox(self, cfg):
        original_init(self, cfg)
        if self.keep_ratio and self.mode != "train":
            for transform in self.transform.transforms:
                if isinstance(transform, dataset_module.LetterboxRect):
                    transform.scaleup = True

    init_with_consistent_letterbox._model_trainer_consistent_letterbox = True
    dataset_module.CustomDataset._init_augs = init_with_consistent_letterbox


def _install_reproducible_transforms(dataset_module: ModuleType) -> None:
    from naartjie_cv.runtime import seed_dataset_transforms, seed_dfine_worker

    original_init = dataset_module.CustomDataset._init_augs
    if getattr(original_init, "_model_trainer_seeded", False):
        return

    @wraps(original_init)
    def seeded_init(self, cfg):
        original_init(self, cfg)
        seed_dataset_transforms(self, int(cfg.train.seed))

    seeded_init._model_trainer_seeded = True
    dataset_module.CustomDataset._init_augs = seeded_init
    dataset_module.seed_worker = seed_dfine_worker


def _install_memory_efficient_loaders(dataset_module: ModuleType) -> None:
    """Avoid overlapping Windows worker pools and excessive prefetched batches."""
    if os.name != "nt":
        return
    original_build = dataset_module.Loader._build_dataloader_impl
    if getattr(original_build, "_model_trainer_memory_efficient", False):
        return

    @wraps(original_build)
    def build_with_bounded_memory(self, dataset, *args, **kwargs):
        workers = self.num_workers
        # Windows spawns a complete Python/PyTorch runtime for every worker.
        # Train workers persist through validation; evaluate in the main process
        # to avoid adding another pool alongside them every epoch.
        if dataset.mode != "train":
            self.num_workers = 0
        try:
            loader = original_build(self, dataset, *args, **kwargs)
        finally:
            self.num_workers = workers
        if loader.num_workers > 0:
            # PyTorch reads this when the iterator starts, after construction.
            loader.prefetch_factor = 1
        return loader

    build_with_bounded_memory._model_trainer_memory_efficient = True
    dataset_module.Loader._build_dataloader_impl = build_with_bounded_memory


def _install_detection_lr_groups(train_module: ModuleType) -> None:
    """Preserve native peaks and count every accumulated optimizer update."""
    from naartjie_cv.training_policy import optimizer_steps_per_epoch
    original_init = train_module.Trainer.__init__
    if getattr(original_init, "_model_trainer_lr_groups", False):
        return

    @wraps(original_init)
    def init_with_lr_groups(self, cfg):
        original_init(self, cfg)
        self._model_trainer_optimizer_steps = 0
        self.optimizer_steps_per_epoch = optimizer_steps_per_epoch(len(self.train_loader), self.b_accum_steps)
        if cfg.train.use_scheduler:
            peaks = [group["max_lr"] for group in self.optimizer.param_groups]
            if (cfg.task == "detect" and not cfg.train.use_muon
                    and cfg.train.get("preserve_lr_groups", False)):
                if len(self.optimizer.param_groups) != 4:
                    raise RuntimeError("D-FINE AdamW parameter groups have changed; check backbone/head scheduling")
                peaks = [cfg.train.backbone_lr * 2] * 2 + [cfg.train.base_lr * 2] * 2
            self.scheduler = train_module.OneCycleLR(
                self.optimizer,
                max_lr=peaks,
                epochs=cfg.train.epochs,
                steps_per_epoch=self.optimizer_steps_per_epoch,
                pct_start=cfg.train.cycler_pct_start,
                cycle_momentum=False,
            )
        scheduler_step = self.scheduler.step if self.scheduler is not None else None
        if self.scheduler is not None:
            # The upstream loop calls step only on complete accumulation windows.
            # Advance from the optimizer hook instead, including the trailing update
            # and excluding updates skipped by FP16 GradScaler.
            self.scheduler.step = lambda *args, **kwargs: None
        def after_optimizer_step(*_):
            self._model_trainer_optimizer_steps += 1
            if scheduler_step is not None:
                scheduler_step()
        self.optimizer.register_step_post_hook(after_optimizer_step)

    init_with_lr_groups._model_trainer_lr_groups = True
    train_module.Trainer.__init__ = init_with_lr_groups


def _install_experiment_tracking(train_module: ModuleType, metadata_path: Path) -> None:
    from naartjie_cv.experiment_tracking import record_validation, update_metadata
    from naartjie_cv.training_policy import ValidationStoppingState, policy_metadata, MIN_EPOCHS, PATIENCE

    original_init = train_module.Trainer.__init__
    original_save = train_module.Trainer.save_model

    @wraps(original_init)
    def tracked_init(self, cfg):
        original_init(self, cfg)
        import platform
        self._model_trainer_epoch = 0
        self._model_trainer_stopping = ValidationStoppingState(
            int(cfg.train.epochs), int(cfg.train.get("min_epochs_before_early_stop", MIN_EPOCHS)),
            int(cfg.train.get("early_stopping", PATIENCE)))
        physical = cfg.train.batch_size
        optimisation = {
            "optimizer": "Muon + " + cfg.train.aux_optimizer.title() if cfg.train.use_muon else "AdamW",
            "physical_batch_size": physical, "gradient_accumulation": self.b_accum_steps,
            "effective_batch_size": physical * self.b_accum_steps, "amp": self.amp_enabled,
            "amp_dtype": str(self.amp_dtype), "ema": bool(cfg.train.use_ema),
            "scheduler": "OneCycleLR" if self.scheduler is not None else None,
            "scheduler_epochs": cfg.train.epochs,
            "optimizer_steps_per_epoch": getattr(self, "optimizer_steps_per_epoch", None),
            "total_optimizer_steps": self.scheduler.total_steps if self.scheduler is not None else None,
            "cycler_pct_start": cfg.train.get("cycler_pct_start"),
            "parameter_groups": [{"initial_lr": group["lr"], "peak_lr": group.get("max_lr"),
                                  "weight_decay": group.get("weight_decay", 0),
                                  **{key: group[key] for key in ("betas", "momentum", "eps") if key in group}}
                                 for group in self.optimizer.param_groups],
        }
        data_loading = {}
        for split in ("train", "val", "test"):
            loader = getattr(self, f"{split}_loader", None)
            if loader is not None:
                data_loading[split] = {
                    "workers": loader.num_workers,
                    "prefetch_factor": loader.prefetch_factor,
                    "persistent_workers": loader.persistent_workers,
                    "pin_memory": loader.pin_memory,
                }
        torch = train_module.torch
        from naartjie_cv.colour import file_sha256
        from naartjie_cv.runtime import library_versions
        from omegaconf import OmegaConf
        checkpoint = Path(cfg.train.get("pretrained_model_path", ""))
        update_metadata(metadata_path, **policy_metadata(cfg.train.epochs,
            self._model_trainer_stopping.min_epochs_before_early_stop, self._model_trainer_stopping.patience,
            " + ".join(self.decision_metrics)), optimisation=optimisation, data_loading=data_loading, runtime={
            "python": platform.python_version(), "torch": str(torch.__version__),
            "cuda_available": torch.cuda.is_available(), "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "libraries": library_versions(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "device": str(cfg.train.get("device", "cuda")),
        }, effective_training_config=OmegaConf.to_container(cfg, resolve=True), pretrained_checkpoint=str(checkpoint),
            pretrained_checkpoint_sha256=file_sha256(checkpoint) if checkpoint.is_file() else None)
        train_module.logger.info(
            f"Physical batch size: {physical}; accumulation: {self.b_accum_steps}; "
            f"effective batch size: {physical * self.b_accum_steps}; AMP: {self.amp_enabled}"
        )
        train_module.logger.info(f"Data loading: {data_loading}")

    @wraps(original_save)
    def tracked_save(self, metrics, best_metric):
        # This hook is called once per completed validation epoch, before the
        # separate final test evaluation. Missing selection metrics must fail.
        values = [float(metrics[name]) for name in self.decision_metrics]
        result = original_save(self, metrics, best_metric)
        self._model_trainer_epoch += 1
        self._model_trainer_stopping.observe(self._model_trainer_epoch, sum(values) / len(values))
        self.early_stopping_steps = self._model_trainer_stopping.epochs_since_best
        record_validation(
            metadata_path, epoch=self._model_trainer_epoch, metric=" + ".join(self.decision_metrics),
            value=sum(values) / len(values), checkpoint=Path(self.path_to_save) / "model.pt",
            metrics={name: float(value) for name, value in metrics.items()},
        )
        update_metadata(metadata_path, actual_optimizer_steps=getattr(self, "_model_trainer_optimizer_steps", 0),
                        early_stopping_state=vars(self._model_trainer_stopping))
        return result

    train_module.Trainer.__init__ = tracked_init
    train_module.Trainer.save_model = tracked_save
    # Upstream uses one-based epochs and checks this property after save_model.
    # Keep its native loop/broadcast; explicitly suppress patience before the gate.
    def gated_patience(self):
        state = getattr(self, "_model_trainer_stopping", None)
        return state.patience if state is not None and state.current_epoch >= state.min_epochs_before_early_stop else 0
    train_module.Trainer.early_stopping = property(gated_patience, lambda self, value: setattr(self, "_configured_patience", value))
    original_train = getattr(train_module.Trainer, "train", None)
    if original_train is not None:
        @wraps(original_train)
        def tracked_train(self, *args, **kwargs):
            try:
                result = original_train(self, *args, **kwargs)
            except BaseException as error:
                update_metadata(metadata_path, actual_stop_epoch=self._model_trainer_epoch,
                                stop_reason="manual_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                                actual_optimizer_steps=getattr(self, "_model_trainer_optimizer_steps", 0), training_error=str(error))
                raise
            reason = self._model_trainer_stopping.stop_reason
            update_metadata(metadata_path, actual_stop_epoch=self._model_trainer_epoch,
                            stop_reason=reason or "manual_interrupt",
                            actual_optimizer_steps=getattr(self, "_model_trainer_optimizer_steps", 0))
            return result
        train_module.Trainer.train = tracked_train


def main() -> None:
    args = parser().parse_args()
    upstream = Path(args.upstream).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not (upstream / "src" / "dl" / "train.py").is_file():
        raise FileNotFoundError(f"Not an ArgoHA/D-FINE-seg checkout: {upstream}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Generated D-FINE-seg config does not exist: {config_path}")

    os.chdir(upstream)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(upstream))
    from omegaconf import OmegaConf
    from src.dl import dataset as dataset_module

    _install_consistent_letterbox(dataset_module)
    _install_reproducible_transforms(dataset_module)
    _install_memory_efficient_loaders(dataset_module)
    config = OmegaConf.load(config_path)
    from naartjie_cv.runtime import seed_training
    seed_training(int(config.train.seed), bool(config.train.get("deterministic", True)))
    print("Experiment configuration:\n" + OmegaConf.to_yaml(config, resolve=True))
    if args.action == "train":
        from src.dl import train as train_module

        # Upstream calls this before model/loader creation; seed all frameworks
        # again with the same configured seed and deterministic CUDA settings.
        train_module.set_seeds = seed_training

        _install_initial_best_checkpoint_fallback(train_module)
        _install_detection_lr_groups(train_module)
        _install_experiment_tracking(train_module, config_path.parent / "run_metadata.json")
        train_main = train_module.main
        wrapped = getattr(train_main, "__wrapped__", None)
        if wrapped is None:
            raise RuntimeError("D-FINE-seg training entry point is incompatible with this runner")
        metadata_path = config_path.parent / "run_metadata.json"
        from naartjie_cv.experiment_tracking import update_metadata
        try:
            wrapped(config)
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if recorded.get("stop_reason") == "error":
                raise RuntimeError(recorded.get("training_error", "D-FINE training failed"))
        except BaseException as error:
            recorded = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
            update_metadata(metadata_path, stop_reason="manual_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                            actual_stop_epoch=recorded.get("actual_stop_epoch", recorded.get("completed_epochs", 0)),
                            training_error=str(error))
            raise
        return

    from src.dl.dataset import Loader

    configured_loader = Loader(root_path=Path(config.train.data_path), img_size=tuple(config.train.img_size),
        batch_size=config.train.batch_size, num_workers=config.train.num_workers, cfg=config, debug_img_processing=False)
    configured_train, _, _ = configured_loader.build_dataloaders(distributed=False)
    configured_batches = len(configured_train)
    configured_workers = configured_train.num_workers
    configured_batch_size = configured_train.batch_size
    config.train.batch_size = 1
    config.train.num_workers = 0
    config.train.debug_img_processing = False
    config.train.mosaic_augs.mosaic_prob = 0.0
    loader = Loader(
        root_path=Path(config.train.data_path),
        img_size=tuple(config.train.img_size),
        batch_size=1,
        num_workers=0,
        cfg=config,
        debug_img_processing=False,
    )
    train_loader, _, _ = loader.build_dataloaders(distributed=False)
    batch = next(iter(train_loader))
    result = {
        "task": str(config.task),
        "configured_training_batches": configured_batches,
        "configured_workers": configured_workers,
        "configured_physical_batch": configured_batch_size,
        "images": len(train_loader.dataset),
        "batch_items": len(batch) if batch is not None else 0,
        "format": "COCO",
        "tensor_shape": list(batch[0].shape),
        "tensor_dtype": str(batch[0].dtype),
        "tensor_finite": bool(batch[0].isfinite().all()),
        "tensor_min": float(batch[0].min()),
        "tensor_max": float(batch[0].max()),
        "image_paths": [str(p) for p in batch[2]],
        "target_counts": [len(t["labels"]) for t in batch[1]],
        "target_classes": [t["labels"].tolist() for t in batch[1]],
        "target_boxes": [t["boxes"].tolist() for t in batch[1]],
        "target_boxes_finite": all(bool(t["boxes"].isfinite().all()) for t in batch[1]),
        "target_masks": [list(t["masks"].shape) for t in batch[1]] if str(config.task) == "segment" else [],
        "tensor_centre_channels": batch[0][0, :, batch[0].shape[-2] // 2, batch[0].shape[-1] // 2].tolist(),
    }
    import hashlib
    result["target_masks_sha256"] = hashlib.sha256(b"".join(t["masks"].numpy().tobytes() for t in batch[1])).hexdigest() if str(config.task) == "segment" else None
    print("MODEL_TRAINER_RESULT=" + json.dumps(result))


if __name__ == "__main__":
    main()
