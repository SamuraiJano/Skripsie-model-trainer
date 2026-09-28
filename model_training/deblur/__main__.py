from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .dataset import validate_paired_dataset
from .evaluation import DisagreementConfig, evaluate_deblurring
from .inference import export_generator, infer_folder, infer_image, load_generator_checkpoint
from .models import DeblurModelConfig
from .routing import calibrate_blur_routing
from .trainer import DeblurTrainer, DeblurTrainingConfig


def _print(value: object) -> None:
    print(json.dumps(value, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Ghost-DeblurGAN training, inference, export, and safety evaluation.")
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("dataset_root")
    train = commands.add_parser("train")
    train.add_argument("dataset_root"); train.add_argument("--architecture", choices=("GhostDeblurGAN", "AGGDeblurGANReproduction"), default="GhostDeblurGAN")
    train.add_argument("--output-dir", default="runs/deblur"); train.add_argument("--run-name", default="ghost_deblurgan")
    train.add_argument("--image-size", type=int, default=256); train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--batch-size", type=int, default=4); train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--workers", type=int, default=0); train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); train.add_argument("--no-amp", action="store_true")
    train.add_argument("--resume-checkpoint"); train.add_argument("--auto-resume", action=argparse.BooleanOptionalAction, default=True)
    train.add_argument("--checkpoint-interval-steps", type=int, default=100); train.add_argument("--vgg19-weights-path"); train.add_argument("--require-vgg19", action="store_true")
    inference = commands.add_parser("infer")
    inference.add_argument("checkpoint"); inference.add_argument("input"); inference.add_argument("output")
    inference.add_argument("--image-size", type=int, default=256); inference.add_argument("--device", default="cpu")
    export = commands.add_parser("export")
    export.add_argument("checkpoint"); export.add_argument("output_dir"); export.add_argument("--image-size", type=int, default=256); export.add_argument("--device", default="cpu")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("checkpoint"); evaluate.add_argument("dataset_dir"); evaluate.add_argument("output_dir")
    evaluate.add_argument("--segmentation-model"); evaluate.add_argument("--split", default="test"); evaluate.add_argument("--image-size", type=int, default=256); evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--class-risk-order", nargs="+", default=["Class 1", "Class 2", "Class 3"])
    calibrate = commands.add_parser("calibrate-blur")
    calibrate.add_argument("acceptable_dir"); calibrate.add_argument("unacceptable_dir"); calibrate.add_argument("output_dir")
    calibrate.add_argument("--target-recall", type=float, default=0.95); calibrate.add_argument("--size", type=int, default=256)
    calibrate.add_argument("--inner-crop", type=float, default=0.85); calibrate.add_argument("--kernel-size", type=int, default=3)
    dashboard = commands.add_parser("dashboard")
    dashboard.add_argument("--host", default="127.0.0.1"); dashboard.add_argument("--port", type=int, default=8876)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "validate":
        report = validate_paired_dataset(args.dataset_root); _print(report)
        if not report["valid"]: raise SystemExit(1)
    elif args.command == "train":
        config = DeblurTrainingConfig(
            dataset_root=args.dataset_root, output_dir=args.output_dir, run_name=args.run_name,
            model=DeblurModelConfig(architecture=args.architecture), image_size=args.image_size, epochs=args.epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate, workers=args.workers, seed=args.seed,
            device=args.device, amp=not args.no_amp, resume_checkpoint=args.resume_checkpoint,
            auto_resume=args.auto_resume, checkpoint_interval_steps=args.checkpoint_interval_steps,
            vgg19_weights_path=args.vgg19_weights_path, allow_perceptual_fallback=not args.require_vgg19,
        )
        _print(DeblurTrainer(config, log_callback=lambda row: print(json.dumps(row))).fit())
    elif args.command == "infer":
        generator, _ = load_generator_checkpoint(args.checkpoint, args.device)
        source = Path(args.input)
        _print(infer_folder(generator, source, args.output, args.image_size, args.device) if source.is_dir() else {"output": infer_image(generator, source, args.output, args.image_size, args.device)})
    elif args.command == "export":
        _print(export_generator(args.checkpoint, args.output_dir, args.image_size, args.device))
    elif args.command == "evaluate":
        _print(evaluate_deblurring(args.checkpoint, args.dataset_dir, args.output_dir, segmentation_model=args.segmentation_model, split=args.split, image_size=args.image_size, device=args.device, config=DisagreementConfig(class_risk_order=tuple(args.class_risk_order))))
    elif args.command == "calibrate-blur":
        _print(calibrate_blur_routing(args.acceptable_dir, args.unacceptable_dir, args.output_dir, target_unacceptable_recall=args.target_recall, size=args.size, inner_crop_fraction=args.inner_crop, kernel_size=args.kernel_size))
    elif args.command == "dashboard":
        import uvicorn
        from .api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
