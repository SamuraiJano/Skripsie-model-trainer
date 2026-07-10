from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .coco import add_image, empty_coco, make_annotation, write_yolo_segmentation_label
from .config import class_names, load_config, resolve_path, write_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# Asset discovery -----------------------------------------------------------
@dataclass(frozen=True)
class ForegroundAsset:
    path: Path
    class_id: int


def list_images(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_foreground_assets(cfg: dict[str, Any]) -> dict[int, list[ForegroundAsset]]:
    fg_dir = resolve_path(cfg, cfg["paths"]["foreground_dir"])
    names = class_names(cfg)
    assets: dict[int, list[ForegroundAsset]] = {}
    for class_id, name in names.items():
        class_dir = fg_dir / name
        files = [p for p in sorted(class_dir.glob("*.png")) if p.is_file()]
        if not files:
            raise FileNotFoundError(f"No transparent PNG foregrounds found in {class_dir}")
        assets[class_id] = [ForegroundAsset(path=p, class_id=class_id) for p in files]
    return assets


def load_backgrounds(cfg: dict[str, Any]) -> list[Path]:
    bg_dir = resolve_path(cfg, cfg["paths"]["background_dir"])
    files = list_images(bg_dir)
    if not files:
        raise FileNotFoundError(f"No background images found in {bg_dir}")
    return files


# Dataset planning ----------------------------------------------------------
def balanced_class_plan(total: int, class_ids: list[int], seed: int) -> list[int]:
    plan: list[int] = []
    per_class = total // len(class_ids)
    remainder = total % len(class_ids)
    for i, class_id in enumerate(class_ids):
        plan.extend([class_id] * (per_class + (1 if i < remainder else 0)))
    rng = random.Random(seed)
    rng.shuffle(plan)
    return plan


def split_name_for_index(index: int, total: int, split_cfg: dict[str, float]) -> str:
    train_end = int(total * float(split_cfg["train"]))
    val_end = train_end + int(total * float(split_cfg["val"]))
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def balanced_split_plan(total: int, class_ids: list[int], split_cfg: dict[str, float], seed: int) -> list[tuple[str, int]]:
    train_count = int(total * float(split_cfg["train"]))
    val_count = int(total * float(split_cfg["val"]))
    counts = {
        "train": train_count,
        "val": val_count,
        "test": total - train_count - val_count,
    }
    plan: list[tuple[str, int]] = []
    for offset, (split, count) in enumerate(counts.items()):
        class_plan = balanced_class_plan(count, class_ids, seed + offset)
        plan.extend((split, class_id) for class_id in class_plan)
    return plan


# Image loading and mask preparation ----------------------------------------
def read_background(path: Path, target_size: tuple[int, int] | None) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read background image {path}")
    if target_size:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return image


def read_foreground(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read foreground image {path}")
    if image.ndim != 3 or image.shape[2] != 4:
        raise ValueError(f"Foreground image must be a transparent PNG with alpha channel: {path}")
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
    if int((alpha > 0).sum()) == 0:
        raise ValueError(f"Foreground alpha mask is empty: {path}")
    return bgr, alpha


def estimate_fruit_mask(background_bgr: np.ndarray, mask_cfg: dict[str, Any]) -> np.ndarray:
    hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(mask_cfg["hsv_lower"], dtype=np.uint8)
    upper = np.array(mask_cfg["hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = background_bgr.shape[0] * background_bgr.shape[1] * float(mask_cfg.get("min_area_fraction", 0.05))
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    clean = np.zeros(mask.shape, dtype=np.uint8)
    if valid:
        cv2.drawContours(clean, [max(valid, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)
        return clean

    if mask_cfg.get("fallback", "ellipse") == "ellipse":
        h, w = mask.shape
        center = (w // 2, h // 2)
        axes = (int(w * 0.38), int(h * 0.38))
        cv2.ellipse(clean, center, axes, 0, 0, 360, 255, thickness=cv2.FILLED)
        return clean
    clean[:, :] = 255
    return clean


# Foreground augmentation ----------------------------------------------------
def transform_foreground(
    bgr: np.ndarray,
    alpha: np.ndarray,
    bg_shape: tuple[int, int, int],
    synth_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    min_bg_dim = min(bg_shape[0], bg_shape[1])
    scale_low, scale_high = synth_cfg["defect_scale_range"]
    target_max_dim = float(rng.uniform(scale_low, scale_high)) * min_bg_dim
    scale = max(0.02, target_max_dim / max(h, w))
    resized_size = (max(2, int(w * scale)), max(2, int(h * scale)))
    bgr = cv2.resize(bgr, resized_size, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, resized_size, interpolation=cv2.INTER_AREA)

    jitter = synth_cfg.get("color_jitter", {})
    bgr = apply_color_jitter(bgr, jitter, rng)

    angle = float(rng.uniform(*synth_cfg["rotation_degrees"]))
    bgr, alpha = rotate_rgba_like(bgr, alpha, angle)

    opacity = float(rng.uniform(*synth_cfg["opacity_range"]))
    alpha = np.clip(alpha.astype(np.float32) * opacity, 0, 255).astype(np.uint8)

    if rng.random() < float(synth_cfg.get("blur_probability", 0.0)):
        choices = synth_cfg.get("blur_kernel_choices", [3])
        k = int(rng.choice(choices))
        if k % 2 == 0:
            k += 1
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return bgr, alpha


def apply_color_jitter(bgr: np.ndarray, jitter: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    brightness = rng.uniform(*jitter.get("brightness", [1.0, 1.0]))
    saturation = rng.uniform(*jitter.get("saturation", [1.0, 1.0]))
    hue_degrees = rng.uniform(*jitter.get("hue_degrees", [0.0, 0.0]))
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_degrees / 2.0) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotate_rgba_like(bgr: np.ndarray, alpha: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = alpha.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2.0) - center[0]
    matrix[1, 2] += (new_h / 2.0) - center[1]
    rotated_bgr = cv2.warpAffine(bgr, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    rotated_alpha = cv2.warpAffine(alpha, matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    return rotated_bgr, rotated_alpha


# Compositing ---------------------------------------------------------------
def choose_location(
    fruit_mask: np.ndarray,
    alpha: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> tuple[int, int]:
    bg_h, bg_w = fruit_mask.shape
    fg_h, fg_w = alpha.shape
    if fg_h >= bg_h or fg_w >= bg_w:
        return max(0, (bg_w - fg_w) // 2), max(0, (bg_h - fg_h) // 2)

    ys, xs = np.where(fruit_mask > 0)
    if len(xs) == 0:
        return int(rng.integers(0, bg_w - fg_w)), int(rng.integers(0, bg_h - fg_h))

    alpha_binary = alpha > 10
    alpha_area = max(1, int(alpha_binary.sum()))
    for _ in range(max_attempts):
        idx = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[idx]), int(ys[idx])
        x = int(np.clip(cx - fg_w // 2, 0, bg_w - fg_w))
        y = int(np.clip(cy - fg_h // 2, 0, bg_h - fg_h))
        roi = fruit_mask[y : y + fg_h, x : x + fg_w]
        if roi.shape != alpha_binary.shape:
            continue
        outside = alpha_binary & (roi == 0)
        if outside.sum() / alpha_area < 0.02:
            return x, y
    return int(rng.integers(0, bg_w - fg_w)), int(rng.integers(0, bg_h - fg_h))


def paste_foreground(
    background: np.ndarray,
    fg_bgr: np.ndarray,
    alpha: np.ndarray,
    x: int,
    y: int,
) -> tuple[np.ndarray, np.ndarray]:
    out = background.copy()
    fg_h, fg_w = alpha.shape
    bg_h, bg_w = background.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + fg_w), min(bg_h, y + fg_h)
    if x1 >= x2 or y1 >= y2:
        return out, np.zeros((bg_h, bg_w), dtype=np.uint8)

    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    patch_alpha = alpha[fy1:fy2, fx1:fx2].astype(np.float32) / 255.0
    patch_alpha_3 = patch_alpha[:, :, None]
    roi = out[y1:y2, x1:x2].astype(np.float32)
    fg = fg_bgr[fy1:fy2, fx1:fx2].astype(np.float32)
    out[y1:y2, x1:x2] = np.clip(fg * patch_alpha_3 + roi * (1.0 - patch_alpha_3), 0, 255).astype(np.uint8)
    full_mask = np.zeros((bg_h, bg_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = (alpha[fy1:fy2, fx1:fx2] > 10).astype(np.uint8) * 255
    return out, full_mask


# Output writing -------------------------------------------------------------
def prepare_output_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (root / "coco").mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(root: Path, names: dict[int, str], channels: int = 3) -> Path:
    data: dict[str, Any] = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    if channels != 3:
        data["channels"] = channels
    yaml_path = root / "data.yaml"
    write_yaml(yaml_path, data)
    return yaml_path


# Main generation pipeline ---------------------------------------------------
def generate_synthetic_dataset(config_path: str | Path = "config/project.yaml") -> dict[str, Any]:
    cfg = load_config(config_path)
    synth_cfg = cfg["synthetic"]
    seed = int(cfg["project"].get("seed", 42))
    random.seed(seed)
    rng = np.random.default_rng(seed)

    names = class_names(cfg)
    fg_assets = load_foreground_assets(cfg)
    bg_paths = load_backgrounds(cfg)
    out_root = resolve_path(cfg, cfg["paths"]["synthetic_dir"])
    prepare_output_dirs(out_root)

    total = int(synth_cfg["total_images"])
    plan = balanced_split_plan(total, sorted(names), synth_cfg["split"], seed)
    target_size_cfg = synth_cfg.get("target_size")
    target_size = tuple(map(int, target_size_cfg)) if target_size_cfg else None
    image_ext = str(synth_cfg.get("image_ext", ".jpg"))
    jpeg_quality = int(synth_cfg.get("jpeg_quality", 95))
    instances_min, instances_max = map(int, synth_cfg.get("instances_per_image", [1, 1]))

    coco_by_split = {split: empty_coco(names) for split in ("train", "val", "test")}
    image_ids = {split: 1 for split in ("train", "val", "test")}
    annotation_ids = {split: 1 for split in ("train", "val", "test")}
    summary = {split: {"images": 0, "annotations": 0} for split in ("train", "val", "test")}

    for split, class_id in tqdm(plan, desc="Generating synthetic images"):
        image_id = image_ids[split]
        image_ids[split] += 1

        bg = read_background(Path(rng.choice(bg_paths)), target_size=target_size)
        fruit_mask = estimate_fruit_mask(bg, synth_cfg["fruit_mask"])
        image = bg.copy()
        annotations_for_image: list[dict[str, Any]] = []

        n_instances = int(rng.integers(instances_min, instances_max + 1))
        for _ in range(n_instances):
            asset = rng.choice(fg_assets[class_id])
            fg_bgr, fg_alpha = read_foreground(asset.path)
            fg_bgr, fg_alpha = transform_foreground(fg_bgr, fg_alpha, image.shape, synth_cfg, rng)
            x, y = choose_location(fruit_mask, fg_alpha, rng)
            image, mask = paste_foreground(image, fg_bgr, fg_alpha, x, y)
            annotation = make_annotation(annotation_ids[split], image_id, class_id, mask)
            if annotation is None:
                continue
            annotations_for_image.append(annotation)
            annotation_ids[split] += 1

        if not annotations_for_image:
            continue

        file_stem = f"synth_{split}_{image_id:06d}"
        file_name = f"{file_stem}{image_ext}"
        image_rel = Path("images") / split / file_name
        image_path = out_root / image_rel
        if image_ext.lower() in {".jpg", ".jpeg"}:
            cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        else:
            cv2.imwrite(str(image_path), image)

        h, w = image.shape[:2]
        add_image(coco_by_split[split], image_id, str(image_rel), w, h)
        coco_by_split[split]["annotations"].extend(annotations_for_image)
        write_yolo_segmentation_label(out_root / "labels" / split / f"{file_stem}.txt", annotations_for_image, w, h)
        summary[split]["images"] += 1
        summary[split]["annotations"] += len(annotations_for_image)

    for split, coco in coco_by_split.items():
        path = out_root / "coco" / f"instances_{split}.json"
        path.write_text(json.dumps(coco, indent=2), encoding="utf-8")

    yaml_path = write_dataset_yaml(out_root, names, channels=3)
    summary["data_yaml"] = str(yaml_path)
    summary["root"] = str(out_root)
    return summary
