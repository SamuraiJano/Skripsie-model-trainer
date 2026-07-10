from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_config, resolve_path


def export_ncnn(
    config_path: str | Path,
    weights: str | Path,
    variant: str = "rgb",
    imgsz: int | None = None,
    half: bool | None = None,
    int8: bool | None = None,
    simplify: bool | None = None,
) -> Any:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    export_cfg = cfg["export"]
    model = YOLO(str(weights))
    data_key = "synthetic_multispectral_dir" if variant == "rgb_hsv" else "synthetic_dir"
    data_yaml = resolve_path(cfg, cfg["paths"][data_key]) / "data.yaml"
    return model.export(
        format="ncnn",
        imgsz=int(imgsz if imgsz is not None else export_cfg["imgsz"]),
        half=bool(half if half is not None else export_cfg["half"]),
        int8=bool(int8 if int8 is not None else export_cfg["int8"]),
        simplify=bool(simplify if simplify is not None else export_cfg["simplify"]),
        data=str(data_yaml),
    )
