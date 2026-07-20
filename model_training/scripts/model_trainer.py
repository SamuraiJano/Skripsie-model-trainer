from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from naartjie_cv.datasets import (
    dfine_loader_smoke,
    ultralytics_loader_smoke,
    validate_training_export,
    validate_yolo_training_dataset,
)
from naartjie_cv.train import TrainingOptions, synthetic_asset_summary, train_dfine_export, train_yolo_export


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Validate a Data Labeller export and train YOLO26-N or D-FINE-N.")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("export_root")
    validate.add_argument("--task", choices=("detect", "segment", "all"), default="detect")
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("export_root")
    smoke = commands.add_parser("smoke-loaders")
    smoke.add_argument("export_root")
    smoke.add_argument("--dfine-root", required=True)
    for name in ("yolo-detect", "yolo-segment", "dfine-detect"):
        command = commands.add_parser(name)
        command.add_argument("export_root")
        command.add_argument("--epochs", type=int, default=100)
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument("--batch", type=int, default=16)
        command.add_argument("--workers", type=int, default=8)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default="0")
        command.add_argument("--output-dir", default="runs")
        command.add_argument("--run-name")
        command.add_argument("--resume")
        command.add_argument("--no-amp", action="store_true")
        if name == "dfine-detect":
            command.add_argument("--dfine-root", required=True)
            command.add_argument("--dry-run", action="store_true")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate":
        result = (
            validate_training_export(args.export_root)
            if args.task == "all"
            else validate_yolo_training_dataset(args.export_root, args.task)
        )
    elif args.command == "synthetic":
        result = synthetic_asset_summary(args.export_root)
    elif args.command == "smoke-loaders":
        result = {
            "ultralytics_detection": ultralytics_loader_smoke(args.export_root, "detect"),
            "ultralytics_segmentation": ultralytics_loader_smoke(args.export_root, "segment"),
            "dfine_detection": dfine_loader_smoke(args.export_root, args.dfine_root),
        }
    else:
        options = TrainingOptions(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            device=args.device,
            amp=not args.no_amp,
            output_dir=args.output_dir,
            run_name=args.run_name,
            resume=args.resume,
        )
        if args.command == "yolo-detect":
            result = train_yolo_export(args.export_root, "detect", options)
        elif args.command == "yolo-segment":
            result = train_yolo_export(args.export_root, "segment", options)
        else:
            result = train_dfine_export(args.export_root, args.dfine_root, options, execute=not args.dry_run)
    print(json.dumps(result, indent=2, default=str))
    if isinstance(result, dict) and result.get("ok") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
