from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .config import class_names, load_config, resolve_path
from .synthetic import prepare_output_dirs, write_dataset_yaml


def rgb_image_to_rgbhsv_chw(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hsv_scaled = hsv.copy()
    hsv_scaled[:, :, 0] = np.round(hsv_scaled[:, :, 0].astype(np.float32) * (255.0 / 179.0)).clip(0, 255).astype(np.uint8)
    six = np.concatenate([rgb, hsv_scaled], axis=2)
    return np.transpose(six, (2, 0, 1)).astype(np.uint8)


def write_multichannel_tiff(path: Path, chw: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [chw[i, :, :] for i in range(chw.shape[0])]
    ok = cv2.imwritemulti(str(path), frames)
    if not ok:
        raise IOError(f"Failed to write multi-channel TIFF: {path}")


def create_rgb_hsv_dataset(config_path: str | Path = "config/project.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    rgb_root = resolve_path(cfg, cfg["paths"]["synthetic_dir"])
    out_root = resolve_path(cfg, cfg["paths"]["synthetic_multispectral_dir"])
    prepare_output_dirs(out_root)
    (out_root / "coco").mkdir(parents=True, exist_ok=True)

    summary = {"root": str(out_root), "images": 0, "labels": 0}
    for split in ("train", "val", "test"):
        src_images = sorted((rgb_root / "images" / split).glob("*"))
        for image_path in tqdm(src_images, desc=f"RGB+HSV TIFF {split}"):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Could not read RGB image {image_path}")
            chw = rgb_image_to_rgbhsv_chw(image)
            out_path = out_root / "images" / split / f"{image_path.stem}.tiff"
            write_multichannel_tiff(out_path, chw)
            summary["images"] += 1

        src_labels = sorted((rgb_root / "labels" / split).glob("*.txt"))
        for label_path in src_labels:
            shutil.copy2(label_path, out_root / "labels" / split / label_path.name)
            summary["labels"] += 1

        src_coco = rgb_root / "coco" / f"instances_{split}.json"
        if src_coco.exists():
            shutil.copy2(src_coco, out_root / "coco" / src_coco.name)

    yaml_path = write_dataset_yaml(out_root, class_names(cfg), channels=6)
    summary["data_yaml"] = str(yaml_path)
    return summary

