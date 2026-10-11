"""One COCO bounding-box evaluator and validation-only operating-point rule.

Prediction records use absolute xywh in source pixels and zero-based classes.
"""
from __future__ import annotations

import contextlib
import copy
import io
from collections import defaultdict

import numpy as np

EVALUATION_POLICY = {
    "implementation": "faster_coco_eval.COCOeval_faster",
    "iou_thresholds": [round(x, 2) for x in np.arange(0.5, 0.951, 0.05)],
    "recall_thresholds": [round(x, 2) for x in np.linspace(0, 1, 101)],
    "max_detections": 100, "score_floor": 0.001,
    "operating_iou": 0.5, "confidence_selection_split": "val",
    "confidence_selection": "maximum validation micro-F1; highest threshold wins ties",
}


def clean_predictions(ground_truth, predictions):
    images = {i["id"]: i for i in ground_truth["images"]}
    classes = {c["id"] for c in ground_truth["categories"]}
    grouped = defaultdict(list)
    for pred in predictions:
        image_id, category = int(pred["image_id"]), int(pred["category_id"])
        if image_id not in images or category not in classes:
            raise ValueError("Prediction refers to an unknown image or category")
        score, box = float(pred["score"]), np.asarray(pred["bbox"], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid prediction score or xywh box")
        if score < EVALUATION_POLICY["score_floor"]:
            continue
        x, y, w, h = box
        if w < 0 or h < 0:
            raise ValueError("Prediction box has negative width or height")
        image = images[image_id]
        x1, y1 = max(0., x), max(0., y)
        x2, y2 = min(float(image["width"]), x + w), min(float(image["height"]), y + h)
        if x2 > x1 and y2 > y1:
            grouped[image_id].append({"image_id": image_id, "category_id": category, "score": score,
                                      "bbox": [x1, y1, x2 - x1, y2 - y1]})
    return [p for image_id in sorted(grouped) for p in sorted(grouped[image_id], key=lambda p: -p["score"])[:100]]


def coco_ap(ground_truth, predictions):
    from faster_coco_eval import COCO, COCOeval_faster
    predictions = clean_predictions(ground_truth, predictions)
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(ground_truth)
        truth.dataset.setdefault("info", {})
        truth.createIndex()
        if predictions:
            detected = truth.loadRes(predictions)
        else:
            detected = COCO()
            detected.dataset = {**copy.deepcopy(truth.dataset), "annotations": []}
            detected.createIndex()
        evaluator = COCOeval_faster(truth, detected, "bbox")
        evaluator.params.imgIds = sorted(i["id"] for i in truth.dataset["images"])
        evaluator.params.iouThrs = np.asarray(EVALUATION_POLICY["iou_thresholds"])
        evaluator.params.recThrs = np.asarray(EVALUATION_POLICY["recall_thresholds"])
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {"mAP50-95": max(0., float(evaluator.stats[0])), "mAP50": max(0., float(evaluator.stats[1])),
            "coco_stats": [float(v) for v in evaluator.stats]}


def _iou(box, boxes):
    if not len(boxes):
        return np.empty(0)
    a, b = np.asarray(box, dtype=float), np.asarray(boxes, dtype=float)
    intersection = np.maximum(0, np.minimum(a[:2] + a[2:], b[:, :2] + b[:, 2:]) - np.maximum(a[:2], b[:, :2])).prod(axis=1)
    return intersection / np.maximum(a[2:].prod() + b[:, 2:].prod(axis=1) - intersection, 1e-12)


def _operating_curve(ground_truth, predictions):
    truth = defaultdict(list)
    for a in ground_truth["annotations"]:
        if a.get("iscrowd", 0):
            raise ValueError("Operating F1 requires explicit crowd handling; exported boxes must have iscrowd=0")
        truth[a["image_id"], a["category_id"]].append(a["bbox"])
    matched = {k: np.zeros(len(v), dtype=bool) for k, v in truth.items()}
    events = []
    for p in sorted(clean_predictions(ground_truth, predictions), key=lambda p: -p["score"]):
        key = p["image_id"], p["category_id"]
        ious = _iou(p["bbox"], truth[key])
        if len(ious):
            ious[matched[key]] = -1
        hit = bool(len(ious) and ious.max() >= EVALUATION_POLICY["operating_iou"])
        if hit:
            matched[key][int(ious.argmax())] = True
        events.append((p["score"], hit))
    return events, len(ground_truth["annotations"])


def operating_metrics(ground_truth, predictions, threshold):
    if not 0.001 <= threshold <= 1:
        raise ValueError("Operating confidence must be in [0.001, 1]")
    events, total = _operating_curve(ground_truth, predictions)
    selected = [hit for score, hit in events if score >= threshold]
    tp, fp = int(sum(selected)), int(len(selected) - sum(selected))
    fn = total - tp
    return {"precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / total if total else 0.,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0., "TP": tp, "FP": fp, "FN": fn}


def select_confidence(ground_truth, predictions, *, split="val"):
    if split != "val":
        raise ValueError("Operating confidence selection is allowed only on validation data")
    events, total = _operating_curve(ground_truth, predictions)
    tp = fp = 0
    best_score, best_f1 = 1., 0.
    for i, (score, hit) in enumerate(events):
        tp += int(hit)
        fp += int(not hit)
        if i + 1 < len(events) and events[i + 1][0] == score:
            continue
        f1 = 2 * tp / (total + tp + fp) if total + tp + fp else 0.
        if f1 > best_f1:
            best_f1, best_score = f1, score
    return {"split": "val", "confidence": best_score, "rule": EVALUATION_POLICY["confidence_selection"],
            "metrics": operating_metrics(ground_truth, predictions, best_score)}


def evaluate_test(ground_truth, predictions, selection):
    if selection.get("split") != "val":
        raise ValueError("Test evaluation requires an operating threshold frozen on validation")
    threshold = float(selection["confidence"])
    return {**coco_ap(ground_truth, predictions), **operating_metrics(ground_truth, predictions, threshold),
            "selected_confidence": threshold, "evaluation_policy": EVALUATION_POLICY}
