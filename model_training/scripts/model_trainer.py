from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
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
from naartjie_cv.train import (
    DFineSegOptions,
    TrainingOptions,
    YoloTrainingOptions,
    final_yolo_options,
    final_dfine_seg_options,
    dfine_seg_loader_smoke,
    synthetic_asset_summary,
    train_dfine_export,
    train_dfine_seg_export,
    train_yolo_export,
)


YOLO_BOOLEAN_ARGUMENTS = (
    "save",
    "pretrained",
    "cls_remap",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "profile",
    "overlap_mask",
    "val",
    "save_json",
    "agnostic_nms",
    "plots",
    "show_labels",
)
YOLO_INTEGER_ARGUMENTS = ("patience", "min_epochs_before_early_stop", "save_period", "close_mosaic", "mask_ratio", "max_det", "line_width", "nbs")
YOLO_FLOAT_ARGUMENTS = (
    "fraction",
    "multi_scale",
    "dropout",
    "iou",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_epochs",
    "warmup_momentum",
    "warmup_bias_lr",
    "box",
    "cls",
    "cls_pw",
    "dfl",
    "rle",
    "dis",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "erasing",
)
YOLO_TEXT_ARGUMENTS = ("optimizer", "split", "copy_paste_mode", "auto_augment", "distill_model")


def _int_or_float(value: str) -> int | float:
    number = float(value)
    return int(number) if number.is_integer() else number


def _integer_selection(value: str) -> int | list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected an integer or comma-separated integers")
    return values[0] if len(values) == 1 else values


def _json_object(value: str) -> dict[str, object]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return payload


def _add_yolo_arguments(command: argparse.ArgumentParser, task: str) -> None:
    defaults = final_yolo_options(task)
    command.add_argument("--time", type=float)
    command.add_argument("--conf", type=float)
    command.add_argument("--cache", nargs="?", const="ram", choices=("ram", "disk"), default=defaults.cache)
    command.add_argument(
        "--compile",
        nargs="?",
        const="default",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default=defaults.compile,
    )
    command.add_argument("--freeze", type=_integer_selection)
    command.add_argument("--classes", type=_integer_selection)
    for name in YOLO_BOOLEAN_ARGUMENTS:
        command.add_argument(
            f"--{name.replace('_', '-')}",
            action=argparse.BooleanOptionalAction,
            default=getattr(defaults, name),
        )
    for name in YOLO_INTEGER_ARGUMENTS:
        command.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in YOLO_FLOAT_ARGUMENTS:
        command.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    for name in YOLO_TEXT_ARGUMENTS:
        command.add_argument(f"--{name.replace('_', '-')}", default=getattr(defaults, name))
    command.add_argument(
        "--yolo-args-json",
        type=_json_object,
        default=defaults.extra_args,
        metavar="JSON",
        help="Additional or version-specific Ultralytics train arguments as a JSON object.",
    )


def _native_optional_float(value: str) -> float | None:
    return None if value.lower() in {"none", "default"} else float(value)


def _add_dfine_seg_arguments(command: argparse.ArgumentParser, task: str = "segment") -> None:
    defaults = final_dfine_seg_options(task)
    command.add_argument("export_root")
    command.add_argument("--dfine-seg-root", required=True)
    command.add_argument("--python-executable", default=sys.executable)
    command.add_argument("--model-size", choices=("n", "s", "m", "l", "x"), default=defaults.model_size)
    command.add_argument("--epochs", type=int, default=defaults.epochs)
    command.add_argument("--imgsz", type=int, default=defaults.imgsz)
    command.add_argument("--batch", type=int, default=defaults.batch)
    command.add_argument("--workers", type=int, default=defaults.workers)
    command.add_argument("--seed", type=int, default=defaults.seed)
    command.add_argument("--colour-space", choices=("auto", "rgb", "hsv"), default="auto")
    command.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    command.add_argument("--device", default=defaults.device)
    command.add_argument("--amp", action=argparse.BooleanOptionalAction, default=defaults.amp)
    command.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default=defaults.amp_dtype)
    command.add_argument("--early-stopping", type=int, default=defaults.early_stopping)
    command.add_argument("--min-epochs-before-early-stop", type=int, default=defaults.min_epochs_before_early_stop)
    command.add_argument("--grad-accumulation", type=int, default=defaults.grad_accumulation)
    command.add_argument("--clip-max-norm", type=float, default=defaults.clip_max_norm)
    command.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=defaults.use_wandb)
    command.add_argument("--ema", action=argparse.BooleanOptionalAction, default=defaults.use_ema)
    command.add_argument("--scheduler", action=argparse.BooleanOptionalAction, default=defaults.use_scheduler)
    command.add_argument("--muon", action=argparse.BooleanOptionalAction, default=defaults.use_muon)
    command.add_argument("--aux-optimizer", choices=("adamw", "adan"), default=defaults.aux_optimizer)
    command.add_argument("--base-lr", type=_native_optional_float, default=defaults.base_lr)
    command.add_argument("--backbone-lr", type=_native_optional_float, default=defaults.backbone_lr)
    command.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    command.add_argument("--mosaic-probability", type=_native_optional_float, default=defaults.mosaic_probability)
    command.add_argument("--no-mosaic-epochs", type=int, default=defaults.no_mosaic_epochs)
    command.add_argument("--conf-threshold", type=float, default=defaults.conf_threshold)
    command.add_argument("--iou-threshold", type=float, default=defaults.iou_threshold)
    command.add_argument(
        "--pretrained-dataset",
        choices=("coco", "obj2coco"),
        default=defaults.pretrained_dataset,
    )
    command.add_argument("--pretrained-model")
    command.add_argument(
        "--visualize-validation",
        action=argparse.BooleanOptionalAction,
        default=defaults.visualize_validation,
    )
    command.add_argument("--output-dir", default=defaults.output_dir)
    command.add_argument("--run-name")
    command.add_argument("--extra-config-json", type=_json_object, default={})
    command.add_argument("--dry-run", action="store_true")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Validate a Data Labeller export and train YOLO26, official D-FINE, or D-FINE-seg."
    )
    commands = root.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("export_root")
    validate.add_argument("--task", choices=("detect", "segment", "all"), default="detect")
    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("export_root")
    smoke = commands.add_parser("smoke-loaders")
    smoke.add_argument("export_root")
    smoke.add_argument("--dfine-root", required=True)
    smoke.add_argument("--dfine-python", default=sys.executable)
    for name in ("yolo-detect", "yolo-segment", "dfine-detect"):
        command = commands.add_parser(name)
        command.add_argument("export_root")
        command.add_argument("--epochs", type=int, default=TrainingOptions().epochs)
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument(
            "--batch",
            type=int if name == "dfine-detect" else _int_or_float,
            default=8 if name == "dfine-detect" else 16,
        )
        command.add_argument("--workers", type=int, default=TrainingOptions().workers)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--colour-space", choices=("auto", "rgb", "hsv"), default="auto")
        command.add_argument("--device", default="0")
        command.add_argument("--output-dir", default="runs")
        command.add_argument("--run-name")
        command.add_argument("--resume")
        command.add_argument("--no-amp", action="store_true")
        if name == "dfine-detect":
            command.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
            command.add_argument("--dfine-root", required=True)
            command.add_argument("--python-executable", default=sys.executable)
            command.add_argument("--dfine-base-lr", type=float)
            command.add_argument("--dfine-backbone-lr", type=float)
            command.add_argument("--dfine-weight-decay", type=float)
            command.add_argument("--dfine-clip-max-norm", type=float)
            command.add_argument("--dfine-warmup-steps", type=int)
            command.add_argument("--dfine-ema", action=argparse.BooleanOptionalAction, default=True)
            command.add_argument("--dfine-config-json", type=_json_object, default={})
            command.add_argument("--dry-run", action="store_true")
        else:
            _add_yolo_arguments(command, "segment" if name == "yolo-segment" else "detect")
    for name in ("dfine-seg-detect", "dfine-seg-segment"):
        _add_dfine_seg_arguments(commands.add_parser(name), "segment" if name.endswith("segment") else "detect")
    dfine_seg_smoke = commands.add_parser("smoke-dfine-seg")
    dfine_seg_smoke.add_argument("--task", choices=("detect", "segment"), required=True)
    _add_dfine_seg_arguments(dfine_seg_smoke)
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
            "dfine_detection": dfine_loader_smoke(
                args.export_root,
                args.dfine_root,
                args.dfine_python,
            ),
        }
    elif args.command.startswith("dfine-seg-") or args.command == "smoke-dfine-seg":
        task = (
            args.task
            if args.command == "smoke-dfine-seg"
            else ("segment" if args.command == "dfine-seg-segment" else "detect")
        )
        options = DFineSegOptions(
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            colour_space=args.colour_space,
            deterministic=args.deterministic,
            device=args.device,
            amp=args.amp,
            amp_dtype=args.amp_dtype,
            early_stopping=args.early_stopping,
            min_epochs_before_early_stop=args.min_epochs_before_early_stop,
            grad_accumulation=args.grad_accumulation,
            clip_max_norm=args.clip_max_norm,
            use_wandb=args.wandb,
            use_ema=args.ema,
            use_scheduler=args.scheduler,
            use_muon=args.muon,
            aux_optimizer=args.aux_optimizer,
            base_lr=args.base_lr,
            backbone_lr=args.backbone_lr,
            weight_decay=args.weight_decay,
            mosaic_probability=args.mosaic_probability,
            no_mosaic_epochs=args.no_mosaic_epochs,
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            pretrained_dataset=args.pretrained_dataset,
            pretrained_model=args.pretrained_model,
            visualize_validation=args.visualize_validation,
            output_dir=args.output_dir,
            run_name=args.run_name,
            extra_config=args.extra_config_json,
        )
        if args.command == "smoke-dfine-seg":
            result = dfine_seg_loader_smoke(
                args.export_root,
                task,
                args.dfine_seg_root,
                options,
                python_executable=args.python_executable,
            )
        else:
            result = train_dfine_seg_export(
                args.export_root,
                task,
                args.dfine_seg_root,
                options,
                python_executable=args.python_executable,
                execute=not args.dry_run,
            )
    else:
        yolo = YoloTrainingOptions(deterministic=args.deterministic)
        if args.command.startswith("yolo-"):
            yolo_values = {
                item.name: getattr(args, item.name)
                for item in fields(YoloTrainingOptions)
                if item.name != "extra_args"
            }
            yolo_values["extra_args"] = args.yolo_args_json
            yolo = YoloTrainingOptions(**yolo_values)
        options = TrainingOptions(
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            seed=args.seed,
            colour_space=args.colour_space,
            device=args.device,
            amp=not args.no_amp,
            output_dir=args.output_dir,
            run_name=args.run_name,
            resume=args.resume,
            yolo=yolo,
            dfine_base_lr=getattr(args, "dfine_base_lr", None),
            dfine_backbone_lr=getattr(args, "dfine_backbone_lr", None),
            dfine_weight_decay=getattr(args, "dfine_weight_decay", None),
            dfine_clip_max_norm=getattr(args, "dfine_clip_max_norm", None),
            dfine_warmup_steps=getattr(args, "dfine_warmup_steps", None),
            dfine_use_ema=getattr(args, "dfine_ema", True),
            dfine_extra_config=getattr(args, "dfine_config_json", {}),
        )
        if args.command == "yolo-detect":
            result = train_yolo_export(args.export_root, "detect", options)
        elif args.command == "yolo-segment":
            result = train_yolo_export(args.export_root, "segment", options)
        else:
            result = train_dfine_export(
                args.export_root,
                args.dfine_root,
                options,
                python_executable=args.python_executable,
                execute=not args.dry_run,
            )
    print(json.dumps(result, indent=2, default=str))
    if isinstance(result, dict) and result.get("ok") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
