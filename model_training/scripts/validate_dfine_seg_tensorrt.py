"""Compare D-FINE-seg TensorRT and ONNX detections/masks on real images."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preprocess(path: Path, height: int, width: int) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode test image: {path}")
    old_height, old_width = image.shape[:2]
    scale = min(height / old_height, width / old_width)
    resized_width = round(old_width * scale)
    resized_height = round(old_height * scale)
    image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    width_padding = width - resized_width
    height_padding = height - resized_height
    image = cv2.copyMakeBorder(
        image,
        height_padding // 2,
        height_padding - height_padding // 2,
        width_padding // 2,
        width_padding - width_padding // 2,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.0


def _select(outputs: dict[str, np.ndarray], threshold: float) -> dict[str, np.ndarray]:
    logits = outputs["logits"][0]
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    flat = scores.reshape(-1)
    top_k = min(300, flat.size)
    top = np.argpartition(flat, -top_k)[-top_k:]
    top = top[np.argsort(flat[top])[::-1]]
    top = top[flat[top] >= threshold]
    query_indices = top // logits.shape[1]
    return {
        "classes": top % logits.shape[1],
        "confidence": flat[top],
        "boxes": outputs["boxes"][0, query_indices],
        "masks": outputs["masks"][0, query_indices],
    }


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax, ay, aw, ah = map(float, first)
    bx, by, bw, bh = map(float, second)
    left = max(ax - aw / 2, bx - bw / 2)
    top = max(ay - ah / 2, by - bh / 2)
    right = min(ax + aw / 2, bx + bw / 2)
    bottom = min(ay + ah / 2, by + bh / 2)
    intersection = max(right - left, 0.0) * max(bottom - top, 0.0)
    return intersection / (aw * ah + bw * bh - intersection + 1e-9)


def _compare(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], summary: dict, image_name: str) -> None:
    count_reference = len(reference["classes"])
    count_candidate = len(candidate["classes"])
    for class_id in reference["classes"]:
        summary["by_class"].setdefault(str(int(class_id)), {"onnx": 0, "tensorrt": 0, "matched": 0})["onnx"] += 1
    for class_id in candidate["classes"]:
        summary["by_class"].setdefault(str(int(class_id)), {"onnx": 0, "tensorrt": 0, "matched": 0})["tensorrt"] += 1
    summary["onnx_detections"] += count_reference
    summary["tensorrt_detections"] += count_candidate
    if count_reference == count_candidate:
        summary["images_same_detection_count"] += 1
    else:
        summary["count_mismatch_images"].append(image_name)

    possible = sorted(
        (
            (_box_iou(reference["boxes"][i], candidate["boxes"][j]), i, j)
            for i in range(count_reference)
            for j in range(count_candidate)
            if reference["classes"][i] == candidate["classes"][j]
        ),
        reverse=True,
    )
    matched_reference: set[int] = set()
    matched_candidate: set[int] = set()
    for box_iou, i, j in possible:
        if box_iou < 0.5 or i in matched_reference or j in matched_candidate:
            continue
        matched_reference.add(i)
        matched_candidate.add(j)
        summary["by_class"][str(int(reference["classes"][i]))]["matched"] += 1
        first_mask = reference["masks"][i] >= 0.5
        second_mask = candidate["masks"][j] >= 0.5
        union = int(np.count_nonzero(first_mask | second_mask))
        mask_iou = int(np.count_nonzero(first_mask & second_mask)) / union if union else 1.0
        summary["box_ious"].append(box_iou)
        summary["mask_ious"].append(mask_iou)
        summary["score_absolute_differences"].append(
            abs(float(reference["confidence"][i]) - float(candidate["confidence"][j]))
        )
    summary["matched_detections"] += len(matched_reference)
    summary["unmatched_onnx_detections"] += count_reference - len(matched_reference)
    summary["unmatched_tensorrt_detections"] += count_candidate - len(matched_candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--tensorrt-site", type=Path, help="Optional separate TensorRT site-packages directory")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.25, 0.5])
    args = parser.parse_args()

    if args.tensorrt_site:
        sys.path.insert(0, str(args.tensorrt_site.resolve()))
    import tensorrt as trt

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not load {args.engine}")
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    if set(names) != {"input", "logits", "boxes", "masks"}:
        raise ValueError(f"Unexpected TensorRT tensors: {names}")
    input_shape = tuple(context.get_tensor_shape("input"))
    if input_shape[0] != 1 or input_shape[1] != 3:
        raise ValueError(f"Expected one RGB image, got {input_shape}")
    buffers = {
        name: torch.empty(tuple(context.get_tensor_shape(name)), device="cuda", dtype=torch.float32)
        for name in names
    }
    for name, buffer in buffers.items():
        context.set_tensor_address(name, buffer.data_ptr())
    stream = torch.cuda.Stream()

    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not image_paths:
        raise FileNotFoundError(f"No test images in {args.images}")
    summaries = {
        threshold: {
            "images": len(image_paths),
            "images_same_detection_count": 0,
            "count_mismatch_images": [],
            "onnx_detections": 0,
            "tensorrt_detections": 0,
            "matched_detections": 0,
            "unmatched_onnx_detections": 0,
            "unmatched_tensorrt_detections": 0,
            "by_class": {},
            "box_ious": [],
            "mask_ious": [],
            "score_absolute_differences": [],
        }
        for threshold in args.thresholds
    }

    for index, image_path in enumerate(image_paths, 1):
        sample = _preprocess(image_path, input_shape[2], input_shape[3])
        reference = dict(zip((item.name for item in session.get_outputs()), session.run(None, {"input": sample})))
        buffers["input"].copy_(torch.from_numpy(sample))
        stream.wait_stream(torch.cuda.current_stream())
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT inference failed for {image_path}")
        stream.synchronize()
        candidate = {name: buffers[name].cpu().numpy() for name in names if name != "input"}
        for threshold, summary in summaries.items():
            _compare(_select(reference, threshold), _select(candidate, threshold), summary, image_path.name)
        if index % 15 == 0:
            print(f"Compared {index}/{len(image_paths)} images", flush=True)

    result = {
        "onnx": str(args.onnx.resolve()),
        "onnx_sha256": _sha256(args.onnx),
        "engine": str(args.engine.resolve()),
        "engine_sha256": _sha256(args.engine),
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "thresholds": {},
    }
    for threshold, summary in summaries.items():
        for metric in ("box_ious", "mask_ious", "score_absolute_differences"):
            values = summary.pop(metric)
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_min"] = float(np.min(values)) if values else None
            summary[f"{metric}_max"] = float(np.max(values)) if values else None
        result["thresholds"][str(threshold)] = summary
    output = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
        print(f"Report: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
