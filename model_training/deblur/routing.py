from __future__ import annotations

import csv
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class BlurRoutingConfig:
    size: int = 256
    inner_crop_fraction: float = 0.85
    kernel_size: int = 3
    low_threshold: float = 80.0
    high_threshold: float = 140.0

    def validate(self) -> None:
        if self.size < 16 or not 0 < self.inner_crop_fraction <= 1:
            raise ValueError("Routing size must be >= 16 and inner crop fraction must be in (0, 1].")
        if self.kernel_size not in {1, 3, 5, 7}:
            raise ValueError("OpenCV Laplacian kernel size must be 1, 3, 5, or 7.")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("Low threshold must be below high threshold.")


def _image_array(image: str | Path | Image.Image | np.ndarray, size: int) -> np.ndarray:
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            pil_image = ImageOps.exif_transpose(opened).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            return cv2.resize(array.astype(np.uint8), (size, size), interpolation=cv2.INTER_AREA)
        pil_image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    pil_image = ImageOps.fit(pil_image, (size, size), method=Image.Resampling.BILINEAR)
    return np.asarray(pil_image, dtype=np.uint8)


def laplacian_blur_score(
    image: str | Path | Image.Image | np.ndarray,
    *,
    size: int = 256,
    inner_crop_fraction: float = 0.85,
    fruit_mask: str | Path | Image.Image | np.ndarray | None = None,
    kernel_size: int = 3,
) -> float:
    config = BlurRoutingConfig(size, inner_crop_fraction, kernel_size, 0.0, 1.0)
    if size < 16 or not 0 < inner_crop_fraction <= 1 or kernel_size not in {1, 3, 5, 7}:
        raise ValueError("Invalid Laplacian preprocessing configuration.")
    rgb = _image_array(image, size)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    margin = round(size * (1.0 - inner_crop_fraction) / 2.0)
    selection = np.ones((size, size), dtype=bool)
    if margin:
        selection[:] = False
        selection[margin : size - margin, margin : size - margin] = True
    if fruit_mask is not None:
        mask = _image_array(fruit_mask, size)
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        selection &= mask > 0
    response = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
    values = response[selection]
    if values.size < 2:
        raise ValueError("Fruit mask/inner crop selects too few pixels for a blur score.")
    return float(np.var(values))


def route_blur(score: float, low_threshold: float, high_threshold: float) -> str:
    if low_threshold >= high_threshold:
        raise ValueError("Low threshold must be below high threshold.")
    if score < low_threshold:
        return "deblur"
    if score < high_threshold:
        return "uncertain_select_sharpest_from_burst"
    return "bypass_deblurring"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def calibrate_blur_routing(
    acceptable_dir: str | Path,
    unacceptable_dir: str | Path,
    output_dir: str | Path,
    *,
    target_unacceptable_recall: float = 0.95,
    size: int = 256,
    inner_crop_fraction: float = 0.85,
    kernel_size: int = 3,
) -> dict[str, Any]:
    if not 0 < target_unacceptable_recall <= 1:
        raise ValueError("Target unacceptable-blur recall must be in (0, 1].")
    rows: list[dict[str, Any]] = []
    for label, directory in (("acceptable", acceptable_dir), ("unacceptable", unacceptable_dir)):
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Calibration folder does not exist: {root}")
        for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
                try:
                    rows.append({"path": str(path), "label": label, "score": laplacian_blur_score(path, size=size, inner_crop_fraction=inner_crop_fraction, kernel_size=kernel_size), "error": ""})
                except Exception as exc:
                    rows.append({"path": str(path), "label": label, "score": None, "error": str(exc)})
    unacceptable = np.asarray([row["score"] for row in rows if row["label"] == "unacceptable" and row["score"] is not None], dtype=float)
    acceptable = np.asarray([row["score"] for row in rows if row["label"] == "acceptable" and row["score"] is not None], dtype=float)
    if unacceptable.size == 0 or acceptable.size == 0:
        raise ValueError("Calibration needs at least one readable acceptable and unacceptable crop.")
    low = float(np.quantile(unacceptable, target_unacceptable_recall, method="higher"))
    high_candidate = float(np.quantile(acceptable, 0.10, method="lower"))
    high = max(low + max(1e-6, abs(low) * 0.01), high_candidate)
    output_root = Path(output_dir).expanduser().resolve()
    stream = []
    header = ["path", "label", "score", "error"]
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=header, lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    _atomic_text(output_root / "blur_calibration.csv", buffer.getvalue())
    summary = {
        "preprocessing": {"size": size, "inner_crop_fraction": inner_crop_fraction, "kernel_size": kernel_size},
        "target_unacceptable_recall": target_unacceptable_recall,
        "suggested_low_threshold": low, "suggested_high_threshold": high,
        "unacceptable": {"count": int(unacceptable.size), "min": float(unacceptable.min()), "median": float(np.median(unacceptable)), "max": float(unacceptable.max())},
        "acceptable": {"count": int(acceptable.size), "min": float(acceptable.min()), "median": float(np.median(acceptable)), "max": float(acceptable.max())},
        "warning": "Thresholds are dataset/crop/preprocessing specific and must be validated on real conveyor captures.",
    }
    _atomic_text(output_root / "blur_calibration.json", json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    return summary
