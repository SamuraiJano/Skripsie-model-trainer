"""Apply reproducibility before launching the unchanged official D-FINE entry point."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys
from functools import wraps


def _install_tracking(metadata_path: Path) -> None:
    from src.solver._solver import BaseSolver
    from naartjie_cv.train import _runtime
    from naartjie_cv.colour import file_sha256
    from naartjie_cv.experiment_tracking import update_metadata
    import torch
    original = BaseSolver.train

    @wraps(original)
    def tracked_train(self):
        result = original(self)
        checkpoint = str(self.cfg.tuning or self.cfg.resume or "")
        path = Path(checkpoint)
        if checkpoint.startswith("https://"):
            path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint.rsplit("/", 1)[-1]
        update_metadata(metadata_path, runtime=_runtime(), effective_training_config=self.cfg.yaml_cfg,
                        pretrained_checkpoint=checkpoint,
                        pretrained_checkpoint_sha256=file_sha256(path) if path.is_file() else None,
                        optimisation={"optimizer": type(self.optimizer).__name__,
                                      "physical_batch_size": self.cfg.train_dataloader.batch_size,
                                      "gradient_accumulation": 1,
                                      "effective_batch_size": self.cfg.train_dataloader.batch_size,
                                      "amp": bool(self.cfg.use_amp), "ema": bool(self.cfg.use_ema),
                                      "parameter_groups": [{key: group[key] for key in
                                          ("lr", "initial_lr", "weight_decay", "betas", "momentum", "eps") if key in group}
                                          for group in self.optimizer.param_groups]})
        return result
    BaseSolver.train = tracked_train


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args, remaining = parser.parse_known_args()
    upstream = Path(args.upstream).resolve()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    sys.path.insert(0, str(upstream))
    from naartjie_cv.runtime import seed_training
    from naartjie_cv.experiment_tracking import update_metadata
    from src.core import YAMLConfig
    cfg = YAMLConfig(args.config)
    seed_training(args.seed, bool(cfg.yaml_cfg.get("deterministic", True)))
    update_metadata(Path(args.config).parent / "run_metadata.json", effective_training_config=cfg.yaml_cfg)
    _install_tracking(Path(args.config).parent / "run_metadata.json")
    print("Experiment configuration:", cfg.yaml_cfg)
    os.chdir(upstream)
    sys.argv = [str(upstream / "train.py"), "-c", args.config, "--seed", str(args.seed), *remaining]
    runpy.run_path(str(upstream / "train.py"), run_name="__main__")


if __name__ == "__main__":
    main()
