"""Compare saved segmentation checkpoints using the original validation split."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.dfine_seg_model_tools import (
    _checkpoint_hash,
    _enter_upstream,
    _load_run,
    _write_report,
)
from scripts.dfine_seg_runner import _install_consistent_letterbox


def diagnose(run_dir: Path, device: str) -> Path:
    import torch

    run_dir, _, upstream, config, last_checkpoint = _load_run(run_dir, "last.pt")
    output_dir = run_dir / "validation_diagnosis" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output_dir.mkdir(parents=True, exist_ok=False)
    config.train.path_to_save = str(output_dir / "evaluation")
    config.train.debug_img_path = str(output_dir / "debug_images")
    config.train.eval_preds_path = str(output_dir / "preview_images")
    config.train.device = device
    config.train.num_workers = 0
    config.train.pretrained_model_path = None
    config.train.use_wandb = False
    config.train.use_ema = False
    config.train.use_scheduler = False
    config.train.use_muon = False
    config.train.aux_optimizer = "adamw"
    config.train.to_visualize_eval = False
    config.train.debug_img_processing = False
    if device == "cpu":
        config.train.amp_enabled = False

    _enter_upstream(upstream)
    from src.dl import dataset as dataset_module
    from src.dl.train import Trainer

    # Build the original loader first to reproduce the run's reported scores.
    trainer = Trainer(config)
    report = {
        "run": run_dir.name,
        "split": "val",
        "images": len(trainer.val_loader.dataset),
        "confidence_threshold": float(trainer.conf_thresh),
        "iou_threshold": float(trainer.iou_thresh),
        "cases": {},
    }
    report_path = output_dir / "diagnosis.json"
    for case, checkpoint, corrected in (
        ("last_original_geometry", last_checkpoint, False),
        ("last_corrected_geometry", last_checkpoint, True),
        ("best_corrected_geometry", run_dir / "model" / "model.pt", True),
    ):
        if corrected:
            _install_consistent_letterbox(dataset_module)
            trainer.val_loader.dataset._init_augs(config)
        trainer.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        print(f"Evaluating {case}: {checkpoint.name}", flush=True)
        metrics = trainer.evaluate(
            val_loader=trainer.val_loader,
            conf_thresh=trainer.conf_thresh,
            iou_thresh=trainer.iou_thresh,
            path_to_save=None,
            extended=True,
            mode="val",
        )
        report["cases"][case] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _checkpoint_hash(checkpoint),
            "metrics": metrics,
        }
        _write_report(report_path, report)
        print({key: value for key, value in metrics.items() if key != "extended_metrics"}, flush=True)
    return report_path


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    diagnose(args.run_dir, args.device)
