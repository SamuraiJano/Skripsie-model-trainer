"""Evaluate an official D-FINE checkpoint on a held-out COCO test split."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


COCO_BBOX_METRICS = (
    "mAP50-95", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--images", required=True, type=int)
    args = parser.parse_args()

    upstream = Path(args.upstream).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"D-FINE test checkpoint does not exist: {checkpoint}")
    os.chdir(upstream)
    sys.path.insert(0, str(upstream))

    from src.core import YAMLConfig
    from src.misc import dist_utils
    from src.solver import TASKS
    from src.solver.det_engine import evaluate

    dist_utils.setup_distributed()
    config = YAMLConfig(str(config_path), resume=str(checkpoint))
    if "HGNetv2" in config.yaml_cfg:
        config.yaml_cfg["HGNetv2"]["pretrained"] = False
    solver = TASKS[config.yaml_cfg["task"]](config)
    solver.eval()
    module = solver.ema.module if solver.ema else solver.model
    stats, _ = evaluate(
        module,
        solver.criterion,
        solver.postprocessor,
        solver.val_dataloader,
        solver.evaluator,
        solver.device,
        epoch=-1,
        use_wandb=False,
    )
    values = stats.get("coco_eval_bbox")
    if not values:
        raise RuntimeError("Official D-FINE test evaluation returned no COCO box metrics")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": "test",
                "images": args.images,
                "checkpoint": str(checkpoint),
                "metrics": dict(zip(COCO_BBOX_METRICS, values)),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"Held-out test metrics saved to {output}")
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
