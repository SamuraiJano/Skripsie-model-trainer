from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from .inference import load_generator_checkpoint
from .metrics import psnr, ssim
from .routing import laplacian_blur_score


@dataclass
class SegmentationPrediction:
    class_name: str = "none"
    confidence: float = 0.0
    mask: np.ndarray | None = None


class Segmenter(Protocol):
    def predict(self, image: np.ndarray) -> SegmentationPrediction: ...


class NullSegmenter:
    def predict(self, image: np.ndarray) -> SegmentationPrediction:
        return SegmentationPrediction(mask=np.zeros(image.shape[:2], dtype=bool))


class YoloSegmentationAdapter:
    def __init__(self, model_path: str | Path, device: str = "cpu", confidence: float = 0.25) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("YOLO disagreement evaluation requires the installed ultralytics package.") from exc
        self.model = YOLO(str(Path(model_path).expanduser().resolve()))
        self.device = device
        self.confidence = confidence

    def predict(self, image: np.ndarray) -> SegmentationPrediction:
        result = self.model.predict(source=image, device=self.device, conf=self.confidence, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return SegmentationPrediction(mask=np.zeros(image.shape[:2], dtype=bool))
        confidences = result.boxes.conf.detach().cpu().numpy()
        best = int(np.argmax(confidences))
        class_id = int(result.boxes.cls[best].item())
        names = result.names
        name = str(names[class_id] if isinstance(names, dict) else names[class_id])
        mask = np.zeros(image.shape[:2], dtype=bool)
        if result.masks is not None and len(result.masks.data):
            all_masks = result.masks.data.detach().cpu().numpy()
            selected = all_masks[np.asarray(result.boxes.cls.detach().cpu().numpy(), dtype=int) == class_id]
            if selected.size:
                union = np.any(selected > 0.5, axis=0).astype(np.uint8)
                mask = cv2.resize(union, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        return SegmentationPrediction(name, float(confidences[best]), mask)


@dataclass(frozen=True)
class DisagreementConfig:
    class_risk_order: tuple[str, ...] = ("Class 1", "Class 2", "Class 3")
    accepting_classes: tuple[str, ...] = ("Class 1",)
    mask_decrease_threshold: float = 0.30
    color_shift_threshold: float = 0.08
    edge_change_threshold: float = 0.05
    confidence_increase_threshold: float = 0.20
    high_confidence_threshold: float = 0.80
    dark_disappearance_threshold: float = 0.20


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _save_image(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".png", dir=path.parent)
    os.close(descriptor)
    try:
        Image.fromarray(array.astype(np.uint8), "RGB").save(temporary, "PNG")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_rgb(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as opened:
        image = ImageOps.fit(ImageOps.exif_transpose(opened).convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def _tensor(image: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def _mask(prediction: SegmentationPrediction, shape: tuple[int, int]) -> np.ndarray:
    if prediction.mask is None:
        return np.zeros(shape, dtype=bool)
    value = np.asarray(prediction.mask, dtype=np.uint8)
    return cv2.resize(value, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def _overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.astype(np.float32).copy()
    colour = np.zeros_like(output); colour[..., 0] = 255; colour[..., 1] = 75
    output[mask] = output[mask] * 0.55 + colour[mask] * 0.45
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (255, 220, 30), 1)
    return np.clip(output, 0, 255).astype(np.uint8)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 1.0


def _new_edge_fraction(before: np.ndarray, after: np.ndarray, sharp: np.ndarray | None) -> float:
    before_edges = cv2.Canny(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    after_edges = cv2.Canny(cv2.cvtColor(after, cv2.COLOR_RGB2GRAY), 80, 160) > 0
    unsupported = after_edges & ~cv2.dilate(before_edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if sharp is not None:
        sharp_edges = cv2.Canny(cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY), 80, 160) > 0
        unsupported &= ~cv2.dilate(sharp_edges.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return float(unsupported.mean())


def _dark_disappearance(before: np.ndarray, after: np.ndarray, sharp: np.ndarray | None) -> float:
    if sharp is None:
        return 0.0
    before_gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    sharp_gray = cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)
    genuine_dark = (before_gray < 65) & (sharp_gray < 65)
    if not genuine_dark.any():
        return 0.0
    return float((genuine_dark & (after_gray > 95)).sum() / genuine_dark.sum())


def _risk_index(name: str, config: DisagreementConfig) -> int | None:
    lookup = {value.casefold(): index for index, value in enumerate(config.class_risk_order)}
    return lookup.get(name.casefold())


def _discover_inputs(dataset_dir: Path, split: str) -> tuple[Path, Path | None]:
    if (dataset_dir / split / "blur").is_dir():
        blur, sharp = dataset_dir / split / "blur", dataset_dir / split / "sharp"
    elif (dataset_dir / "blur").is_dir():
        blur, sharp = dataset_dir / "blur", dataset_dir / "sharp"
    else:
        blur, sharp = dataset_dir, None
    return blur, sharp if sharp and sharp.is_dir() else None


@torch.no_grad()
def evaluate_deblurring(
    checkpoint_path: str | Path,
    dataset_dir: str | Path,
    output_dir: str | Path,
    *,
    segmentation_model: str | Path | None = None,
    segmenter: Segmenter | None = None,
    split: str = "test",
    image_size: int = 256,
    device: str = "cpu",
    config: DisagreementConfig | None = None,
) -> dict[str, Any]:
    config = config or DisagreementConfig()
    generator, checkpoint = load_generator_checkpoint(checkpoint_path, device)
    segmenter = segmenter or (YoloSegmentationAdapter(segmentation_model, device) if segmentation_model else NullSegmenter())
    input_root = Path(dataset_dir).expanduser().resolve()
    blur_root, sharp_root = _discover_inputs(input_root, split)
    output_root = Path(output_dir).expanduser().resolve() / "deblur_evaluation"
    image_paths = sorted((path for path in blur_root.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}), key=lambda path: path.name.casefold())
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for blurred_path in image_paths:
        try:
            blurred = _load_rgb(blurred_path, image_size)
            restored_tensor = generator(_tensor(blurred, device)).clamp(0, 1)
            restored = (restored_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            sharp_path = sharp_root / blurred_path.name if sharp_root and (sharp_root / blurred_path.name).is_file() else None
            sharp = _load_rgb(sharp_path, image_size) if sharp_path else None
            before_prediction, after_prediction = segmenter.predict(blurred), segmenter.predict(restored)
            before_mask, after_mask = _mask(before_prediction, blurred.shape[:2]), _mask(after_prediction, blurred.shape[:2])
            before_area, after_area = float(before_mask.mean()), float(after_mask.mean())
            relative_area_change = (after_area - before_area) / max(before_area, 1.0 / before_mask.size)
            colour_shift = float(np.abs(restored.astype(np.float32) - blurred.astype(np.float32)).mean() / 255.0)
            edge_change = _new_edge_fraction(blurred, restored, sharp)
            dark_disappearance = _dark_disappearance(blurred, restored, sharp)
            before_risk, after_risk = _risk_index(before_prediction.class_name, config), _risk_index(after_prediction.class_name, config)
            class_changed = before_prediction.class_name != after_prediction.class_name
            accepting = after_prediction.class_name.casefold() in {name.casefold() for name in config.accepting_classes}
            reasons: list[str] = []
            if before_risk is not None and after_risk is not None and after_risk < before_risk:
                reasons.append("less_severe_class_transition")
            if before_area > 0 and relative_area_change < -config.mask_decrease_threshold:
                reasons.append("large_defect_mask_decrease")
            if dark_disappearance > config.dark_disappearance_threshold:
                reasons.append("genuine_dark_feature_disappearance")
            if edge_change > config.edge_change_threshold:
                reasons.append("new_artificial_edge_regions")
            if colour_shift > config.color_shift_threshold:
                reasons.append("excessive_colour_shift")
            if class_changed and after_prediction.confidence - before_prediction.confidence > config.confidence_increase_threshold:
                reasons.append("confidence_increase_with_prediction_disagreement")
            if accepting and after_prediction.confidence >= config.high_confidence_threshold and before_prediction.class_name.casefold() not in {name.casefold() for name in config.accepting_classes}:
                reasons.append("high_confidence_accept_after_original_defect")
            reference_metrics = {"psnr": None, "ssim": None}
            if sharp is not None:
                sharp_tensor = _tensor(sharp, device)
                reference_metrics = {"psnr": float(psnr(restored_tensor, sharp_tensor)), "ssim": float(ssim(restored_tensor, sharp_tensor))}
            file_name = blurred_path.stem + ".png"
            _save_image(blurred, output_root / "images" / "blurred" / file_name)
            _save_image(restored, output_root / "images" / "restored" / file_name)
            if sharp is not None:
                _save_image(sharp, output_root / "images" / "sharp" / file_name)
            _save_image(_overlay(blurred, before_mask), output_root / "images" / "overlays_before" / file_name)
            _save_image(_overlay(restored, after_mask), output_root / "images" / "overlays_after" / file_name)
            rows.append({
                "pair_id": blurred_path.stem, "architecture": checkpoint["model_config"]["architecture"],
                "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
                "laplacian_before": laplacian_blur_score(blurred, size=image_size), "laplacian_after": laplacian_blur_score(restored, size=image_size),
                **reference_metrics, "predicted_class_before": before_prediction.class_name, "predicted_class_after": after_prediction.class_name,
                "confidence_before": before_prediction.confidence, "confidence_after": after_prediction.confidence,
                "class_transition": f"{before_prediction.class_name} -> {after_prediction.class_name}",
                "mask_iou": _iou(before_mask, after_mask), "defect_area_before": before_area, "defect_area_after": after_area,
                "relative_defect_area_change": relative_area_change, "colour_shift": colour_shift, "edge_map_change": edge_change,
                "dark_feature_disappearance": dark_disappearance, "flagged": bool(reasons), "flag_reasons": reasons,
            })
        except Exception as exc:
            failures.append({"image": str(blurred_path), "error": str(exc)})
    reports = output_root / "reports"
    serializable_rows = [{**row, "flag_reasons": ";".join(row["flag_reasons"])} for row in rows]
    fields = list(serializable_rows[0]) if serializable_rows else ["pair_id", "flagged", "flag_reasons"]
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(serializable_rows)
    _atomic_text(reports / "disagreements.csv", buffer.getvalue())
    _atomic_text(reports / "disagreements.json", json.dumps({"rows": rows, "failures": failures}, indent=2) + "\n")
    summary = {
        "sample_count": len(rows), "flagged_count": sum(bool(row["flagged"]) for row in rows),
        "failure_count": len(failures), "architecture": checkpoint["model_config"]["architecture"],
        "checkpoint": str(Path(checkpoint_path).expanduser().resolve()), "class_risk_order": list(config.class_risk_order),
        "warning": "A sharper-looking restoration is not evidence of safer or more accurate defect segmentation; inspect before/after disagreement exports.",
    }
    _atomic_text(reports / "summary.json", json.dumps(summary, indent=2) + "\n")
    return {**summary, "output_dir": str(output_root), "failures": failures}
