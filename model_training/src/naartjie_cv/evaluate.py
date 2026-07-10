from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .config import class_names, load_config, resolve_path


def _safe_metric(obj: Any, attr: str) -> float | None:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_ultralytics_metrics(results: Any) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for prefix in ("box", "seg", "mask"):
        obj = getattr(results, prefix, None)
        if obj is None:
            continue
        for attr in ("mp", "mr", "map50", "map75", "map"):
            value = _safe_metric(obj, attr)
            if value is not None:
                metrics[f"{prefix}_{attr}"] = value

    precision = metrics.get("seg_mp", metrics.get("mask_mp", metrics.get("box_mp", 0.0)))
    recall = metrics.get("seg_mr", metrics.get("mask_mr", metrics.get("box_mr", 0.0)))
    metrics["f1"] = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return metrics


def load_dataset_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def image_paths_for_split(data_yaml: Path, split: str) -> list[Path]:
    data = load_dataset_yaml(data_yaml)
    root = Path(data["path"])
    split_value = data.get(split)
    if split_value is None:
        return []
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path
    return sorted(p for p in split_path.glob("*") if p.is_file())


def label_path_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    except ValueError:
        return image_path.parent.parent / "labels" / image_path.parent.name / f"{image_path.stem}.txt"


def load_yolo_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    boxes: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [float(x) for x in line.split()]
        class_id = int(parts[0])
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        xs = coords[:, 0] * width
        ys = coords[:, 1] * height
        boxes.append(
            {
                "class_id": class_id,
                "xyxy": np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
            }
        )
    return boxes


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def greedy_match_counts(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float,
    num_classes: int,
) -> dict[str, Any]:
    matched_gt: set[int] = set()
    tp = 0
    fp = 0
    per_class = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(num_classes)}

    predictions = sorted(predictions, key=lambda x: float(x["conf"]), reverse=True)
    for pred in predictions:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(ground_truth):
            if idx in matched_gt or gt["class_id"] != pred["class_id"]:
                continue
            iou = box_iou(pred["xyxy"], gt["xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
            per_class[pred["class_id"]]["tp"] += 1
        else:
            fp += 1
            per_class[pred["class_id"]]["fp"] += 1

    fn = len(ground_truth) - len(matched_gt)
    for idx, gt in enumerate(ground_truth):
        if idx not in matched_gt:
            per_class[gt["class_id"]]["fn"] += 1
    return {"tp": tp, "fp": fp, "fn": fn, "per_class": per_class}


def predict_boxes(result: Any) -> list[dict[str, Any]]:
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or boxes_obj.xyxy is None:
        return []
    xyxy = boxes_obj.xyxy.cpu().numpy()
    cls = boxes_obj.cls.cpu().numpy().astype(int)
    conf = boxes_obj.conf.cpu().numpy()
    return [{"xyxy": xyxy[i], "class_id": int(cls[i]), "conf": float(conf[i])} for i in range(len(cls))]


def detection_accuracy_metrics(
    model: Any,
    data_yaml: Path,
    split: str,
    class_count: int,
    imgsz: int,
    conf: float,
    iou: float,
    max_det: int,
) -> dict[str, Any]:
    image_paths = image_paths_for_split(data_yaml, split)
    counts = {"tp": 0, "fp": 0, "fn": 0, "per_class": {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(class_count)}}
    speed_ms: list[float] = []
    start = time.perf_counter()

    for result in model.predict(
        source=[str(p) for p in image_paths],
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        max_det=max_det,
        stream=True,
        verbose=False,
    ):
        image_path = Path(result.path)
        height, width = result.orig_shape
        gt = load_yolo_boxes(label_path_for_image(image_path), width, height)
        pred = predict_boxes(result)
        batch_counts = greedy_match_counts(pred, gt, iou, class_count)
        counts["tp"] += batch_counts["tp"]
        counts["fp"] += batch_counts["fp"]
        counts["fn"] += batch_counts["fn"]
        for cid, values in batch_counts["per_class"].items():
            for key in ("tp", "fp", "fn"):
                counts["per_class"][cid][key] += values[key]
        speed = getattr(result, "speed", None)
        if isinstance(speed, dict):
            speed_ms.append(float(sum(speed.values())))

    elapsed = time.perf_counter() - start
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    accuracy = 0.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)
    counts.update(
        {
            "images": len(image_paths),
            "precision_iou50": precision,
            "recall_iou50": recall,
            "f1_iou50": f1,
            "detection_accuracy_iou50": accuracy,
            "elapsed_seconds": elapsed,
            "avg_speed_ms": float(np.mean(speed_ms)) if speed_ms else None,
        }
    )
    return counts


def evaluate_model(
    config_path: str | Path,
    weights: str | Path,
    variant: str = "rgb",
    split: str = "test",
) -> dict[str, Any]:
    from ultralytics import YOLO

    cfg = load_config(config_path)
    data_root_key = "synthetic_multispectral_dir" if variant == "rgb_hsv" else "synthetic_dir"
    data_yaml = resolve_path(cfg, cfg["paths"][data_root_key]) / "data.yaml"
    eval_cfg = cfg["evaluation"]
    model = YOLO(str(weights))

    val_results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=int(cfg["training"]["imgsz"]),
        conf=float(eval_cfg["conf"]),
        iou=float(eval_cfg["iou"]),
        max_det=int(eval_cfg["max_det"]),
        plots=True,
        verbose=False,
    )
    metrics = extract_ultralytics_metrics(val_results)
    detection_metrics = detection_accuracy_metrics(
        model=model,
        data_yaml=data_yaml,
        split=split,
        class_count=len(class_names(cfg)),
        imgsz=int(cfg["training"]["imgsz"]),
        conf=float(eval_cfg["conf"]),
        iou=float(eval_cfg["iou"]),
        max_det=int(eval_cfg["max_det"]),
    )
    report = {
        "weights": str(weights),
        "variant": variant,
        "split": split,
        "ultralytics": metrics,
        "detection_accuracy": detection_metrics,
    }
    out_dir = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "evaluations"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{Path(weights).stem}_{variant}_{split}"
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            writer.writerow([f"ultralytics.{key}", value])
        for key, value in detection_metrics.items():
            if key != "per_class":
                writer.writerow([f"detection_accuracy.{key}", value])
    report["json_path"] = str(json_path)
    report["csv_path"] = str(csv_path)
    return report

