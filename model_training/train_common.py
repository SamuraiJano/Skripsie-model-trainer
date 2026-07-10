from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def training_root() -> Path:
    return Path(__file__).resolve().parent


def add_training_args(parser: argparse.ArgumentParser, default_config: Path, default_name: str) -> None:
    parser.add_argument("--data", type=Path, default=default_config, help="Ultralytics dataset YAML.")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="Base YOLO segmentation checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--device", default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", type=Path, default=training_root() / "runs" / "segment")
    parser.add_argument("--name", default=default_name)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt when available.")


def train(args: argparse.Namespace, disable_color_aug: bool) -> Any:
    from ultralytics import YOLO

    data_path = args.data.resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    project_dir = args.project.resolve()
    last_pt = project_dir / args.name / "weights" / "last.pt"

    if args.resume and last_pt.exists():
        model = YOLO(str(last_pt))
        return model.train(resume=True)

    model = YOLO(args.model)
    train_kwargs: dict[str, Any] = {
        "data": str(data_path),
        "task": "segment",
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": str(project_dir),
        "name": args.name,
        "patience": args.patience,
        "exist_ok": True,
        "plots": True,
        "optimizer": "auto",
        "cos_lr": True,
        "amp": True,
    }
    if disable_color_aug:
        train_kwargs.update({"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0})

    return model.train(**train_kwargs)
