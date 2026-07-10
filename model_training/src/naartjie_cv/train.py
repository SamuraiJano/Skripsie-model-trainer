from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config, resolve_path


def variant_paths(cfg: dict[str, Any], variant: str) -> tuple[Path, str]:
    if variant == "rgb":
        data_yaml = resolve_path(cfg, cfg["paths"]["synthetic_dir"]) / "data.yaml"
        run_name = "naartjie_rgb"
    elif variant == "rgb_hsv":
        data_yaml = resolve_path(cfg, cfg["paths"]["synthetic_multispectral_dir"]) / "data.yaml"
        run_name = "naartjie_rgb_hsv"
    else:
        raise ValueError("variant must be 'rgb' or 'rgb_hsv'")
    return data_yaml, run_name


def train_yolo(config_path: str | Path = "config/project.yaml", variant: str = "rgb") -> Any:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    train_cfg = cfg["training"]
    data_yaml, run_name = variant_paths(cfg, variant)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML does not exist: {data_yaml}")

    project_dir = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "segment"
    last_pt = project_dir / run_name / "weights" / "last.pt"

    if bool(train_cfg.get("resume", True)) and last_pt.exists():
        model = YOLO(str(last_pt))
        return model.train(resume=True)

    model_name = train_cfg["model_rgb_hsv"] if variant == "rgb_hsv" else train_cfg["model_rgb"]
    model = YOLO(model_name)
    return model.train(
        data=str(data_yaml),
        task="segment",
        epochs=int(train_cfg["epochs"]),
        imgsz=int(train_cfg["imgsz"]),
        batch=train_cfg["batch"],
        workers=int(train_cfg["workers"]),
        device=train_cfg["device"],
        patience=int(train_cfg["patience"]),
        save_period=int(train_cfg["save_period"]),
        amp=bool(train_cfg["amp"]),
        cos_lr=bool(train_cfg["cos_lr"]),
        optimizer=train_cfg["optimizer"],
        plots=bool(train_cfg["plots"]),
        hsv_h=float(train_cfg["hsv_h"]),
        hsv_s=float(train_cfg["hsv_s"]),
        hsv_v=float(train_cfg["hsv_v"]),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

