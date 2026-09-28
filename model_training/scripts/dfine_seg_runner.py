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
    """Match validation/test resizing to training and inverse mask geometry.

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
        if str(cfg.task).lower() == "segment" and self.keep_ratio and self.mode != "train":
            for transform in self.transform.transforms:
                if isinstance(transform, dataset_module.LetterboxRect):
                    transform.scaleup = True

    init_with_consistent_letterbox._model_trainer_consistent_letterbox = True
    dataset_module.CustomDataset._init_augs = init_with_consistent_letterbox


def main() -> None:
    args = parser().parse_args()
    upstream = Path(args.upstream).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    if not (upstream / "src" / "dl" / "train.py").is_file():
        raise FileNotFoundError(f"Not an ArgoHA/D-FINE-seg checkout: {upstream}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Generated D-FINE-seg config does not exist: {config_path}")

    os.chdir(upstream)
    sys.path.insert(0, str(upstream))
    from omegaconf import OmegaConf
    from src.dl import dataset as dataset_module

    _install_consistent_letterbox(dataset_module)
    config = OmegaConf.load(config_path)
    if args.action == "train":
        from src.dl import train as train_module

        _install_initial_best_checkpoint_fallback(train_module)
        train_main = train_module.main
        wrapped = getattr(train_main, "__wrapped__", None)
        if wrapped is None:
            raise RuntimeError("D-FINE-seg training entry point is incompatible with this runner")
        wrapped(config)
        return

    from src.dl.dataset import Loader

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
        "images": len(train_loader.dataset),
        "batch_items": len(batch) if batch is not None else 0,
        "format": "COCO",
    }
    print("MODEL_TRAINER_RESULT=" + json.dumps(result))


if __name__ == "__main__":
    main()
