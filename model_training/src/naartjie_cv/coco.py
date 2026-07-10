from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, epsilon: float = 1.5, min_area: float = 12.0) -> list[list[float]]:
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[float]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        poly = approx.reshape(-1, 2).astype(float).flatten().tolist()
        if len(poly) >= 6:
            polygons.append(poly)
    polygons.sort(key=lambda p: cv2.contourArea(np.array(p, dtype=np.float32).reshape(-1, 2)), reverse=True)
    return polygons


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())
    return [x1, y1, x2 - x1 + 1.0, y2 - y1 + 1.0]


def polygon_to_yolo_line(class_id: int, polygon: list[float], width: int, height: int) -> str:
    values: list[str] = [str(class_id)]
    for i, coord in enumerate(polygon):
        denom = width if i % 2 == 0 else height
        values.append(f"{max(0.0, min(1.0, coord / denom)):.6f}")
    return " ".join(values)


def write_yolo_segmentation_label(
    path: str | Path,
    annotations: list[dict[str, Any]],
    width: int,
    height: int,
) -> None:
    lines: list[str] = []
    for ann in annotations:
        polygons = ann.get("segmentation", [])
        if not polygons:
            continue
        largest = max(polygons, key=len)
        lines.append(polygon_to_yolo_line(int(ann["category_id"]), largest, width, height))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def empty_coco(class_names: dict[int, str]) -> dict[str, Any]:
    return {
        "info": {
            "description": "Synthetic naartjie defect instance segmentation dataset",
            "version": "0.1.0",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": int(i), "name": name, "supercategory": "defect"} for i, name in class_names.items()],
    }


def add_image(coco: dict[str, Any], image_id: int, file_name: str, width: int, height: int) -> None:
    coco["images"].append(
        {
            "id": int(image_id),
            "file_name": file_name.replace("\\", "/"),
            "width": int(width),
            "height": int(height),
        }
    )


def make_annotation(
    annotation_id: int,
    image_id: int,
    class_id: int,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    polygons = mask_to_polygons(mask)
    if not polygons:
        return None
    bbox = bbox_from_mask(mask)
    area = float((mask > 0).sum())
    if area <= 0:
        return None
    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(class_id),
        "segmentation": polygons,
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }

