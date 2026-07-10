from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .config import class_names, load_config, resolve_path
from .evaluate import box_iou, greedy_match_counts, image_paths_for_split, label_path_for_image, load_yolo_boxes
from .synthetic import estimate_fruit_mask


# Data structures -----------------------------------------------------------
@dataclass
class ClassicalPrediction:
    class_id: int
    conf: float
    xyxy: np.ndarray
    mask: np.ndarray | None
    method: str

    def as_metric_dict(self) -> dict[str, Any]:
        return {"class_id": self.class_id, "conf": self.conf, "xyxy": self.xyxy}


# Label and image scaling helpers ------------------------------------------
def load_yolo_instances(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    if not label_path.exists():
        return []
    instances: list[dict[str, Any]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = [float(x) for x in line.split()]
        class_id = int(parts[0])
        polygon = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        polygon[:, 0] *= width
        polygon[:, 1] *= height
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        xs = polygon[:, 0]
        ys = polygon[:, 1]
        instances.append(
            {
                "class_id": class_id,
                "polygon": polygon,
                "mask": mask,
                "xyxy": np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32),
            }
        )
    return instances


def resize_for_baseline(image: np.ndarray, cfg: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    size = cfg["classical_baselines"].get("input_size")
    if not size:
        return image, 1.0, 1.0
    target_w, target_h = map(int, size)
    resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    scale_x = image.shape[1] / target_w
    scale_y = image.shape[0] / target_h
    return resized, scale_x, scale_y


def scale_predictions(predictions: list[ClassicalPrediction], scale_x: float, scale_y: float) -> list[ClassicalPrediction]:
    scaled: list[ClassicalPrediction] = []
    for pred in predictions:
        box = pred.xyxy.copy().astype(np.float32)
        box[[0, 2]] *= scale_x
        box[[1, 3]] *= scale_y
        scaled.append(ClassicalPrediction(pred.class_id, pred.conf, box, pred.mask, pred.method))
    return scaled


def clean_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


# Classical candidate generators -------------------------------------------
def contour_predictions(
    mask: np.ndarray,
    class_id: int,
    conf: float,
    method: str,
    min_area: float,
) -> list[ClassicalPrediction]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    predictions: list[ClassicalPrediction] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        instance_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(instance_mask, [contour], -1, 255, thickness=cv2.FILLED)
        predictions.append(
            ClassicalPrediction(
                class_id=class_id,
                conf=float(conf),
                xyxy=np.array([x, y, x + w, y + h], dtype=np.float32),
                mask=instance_mask,
                method=method,
            )
        )
    return predictions


def hsv_thresholding_predict(image_bgr: np.ndarray, cfg: dict[str, Any]) -> list[ClassicalPrediction]:
    baseline_cfg = cfg["classical_baselines"]
    hsv_cfg = baseline_cfg["hsv_thresholding"]["class_ranges"]
    names = class_names(cfg)
    name_to_id = {name: class_id for class_id, name in names.items()}
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    fruit_mask = estimate_fruit_mask(image_bgr, baseline_cfg["fruit_mask"])
    kernel_size = int(baseline_cfg["morphology_kernel"])
    min_area = float(baseline_cfg["min_contour_area"])

    predictions: list[ClassicalPrediction] = []
    for class_name, ranges in hsv_cfg.items():
        if class_name not in name_to_id:
            continue
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for item in ranges:
            lower = np.array(item["lower"], dtype=np.uint8)
            upper = np.array(item["upper"], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        mask = cv2.bitwise_and(mask, fruit_mask)
        mask = clean_binary_mask(mask, kernel_size)
        predictions.extend(
            contour_predictions(
                mask=mask,
                class_id=name_to_id[class_name],
                conf=0.60,
                method="hsv_thresholding",
                min_area=min_area,
            )
        )
    return predictions


def lab_mean_for_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    pixels = lab[mask > 0]
    if len(pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return pixels.mean(axis=0).astype(np.float32)


def learn_class_colour_prototypes(cfg: dict[str, Any], split: str = "train") -> dict[int, np.ndarray]:
    data_yaml = resolve_path(cfg, cfg["paths"]["synthetic_dir"]) / "data.yaml"
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    for image_path in image_paths_for_split(data_yaml, split):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        instances = load_yolo_instances(label_path_for_image(image_path), image.shape[1], image.shape[0])
        for instance in instances:
            class_id = int(instance["class_id"])
            mean = lab_mean_for_mask(image, instance["mask"])
            sums[class_id] = sums.get(class_id, np.zeros(3, dtype=np.float32)) + mean
            counts[class_id] = counts.get(class_id, 0) + 1
    if not sums:
        raise RuntimeError("Could not learn colour prototypes. Generate the synthetic dataset first.")
    return {class_id: sums[class_id] / counts[class_id] for class_id in sums}


def nearest_prototype_class(lab_center: np.ndarray, prototypes: dict[int, np.ndarray]) -> tuple[int, float]:
    best_class = min(prototypes, key=lambda cid: float(np.linalg.norm(lab_center - prototypes[cid])))
    dist = float(np.linalg.norm(lab_center - prototypes[best_class]))
    conf = float(1.0 / (1.0 + dist / 50.0))
    return int(best_class), conf


def lab_kmeans_candidate_masks(image_bgr: np.ndarray, cfg: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    baseline_cfg = cfg["classical_baselines"]
    lab_cfg = baseline_cfg["lab_kmeans"]
    fruit_mask = estimate_fruit_mask(image_bgr, baseline_cfg["fruit_mask"])
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    fruit_pixels = lab[fruit_mask > 0]
    if len(fruit_pixels) == 0:
        return []

    rng = np.random.default_rng(int(lab_cfg.get("random_state", 42)))
    sample_count = min(int(lab_cfg["sample_pixels"]), len(fruit_pixels))
    cluster_count = min(int(lab_cfg["n_clusters"]), sample_count)
    if cluster_count < 1:
        return []
    sample_idx = rng.choice(len(fruit_pixels), size=sample_count, replace=False)
    sample = np.float32(fruit_pixels[sample_idx])

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.5)
    _, _, centers = cv2.kmeans(
        sample,
        cluster_count,
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )

    lab_float = lab.reshape(-1, 3).astype(np.float32)
    distances = np.linalg.norm(lab_float[:, None, :] - centers[None, :, :], axis=2)
    cluster_map = distances.argmin(axis=1).reshape(lab.shape[:2])
    fruit_mean = fruit_pixels.mean(axis=0).astype(np.float32)
    max_fraction = float(lab_cfg.get("max_cluster_fraction", 0.45))
    distance_threshold = float(lab_cfg.get("defect_distance_threshold", 18.0))
    kernel_size = int(baseline_cfg["morphology_kernel"])

    masks: list[tuple[np.ndarray, np.ndarray]] = []
    fruit_area = max(1, int((fruit_mask > 0).sum()))
    for cluster_id, center in enumerate(centers):
        mask = ((cluster_map == cluster_id) & (fruit_mask > 0)).astype(np.uint8) * 255
        fraction = float((mask > 0).sum()) / fruit_area
        distance = float(np.linalg.norm(center - fruit_mean))
        if fraction > max_fraction or distance < distance_threshold:
            continue
        masks.append((clean_binary_mask(mask, kernel_size), center.astype(np.float32)))
    return masks


def lab_kmeans_predict(
    image_bgr: np.ndarray,
    cfg: dict[str, Any],
    prototypes: dict[int, np.ndarray],
) -> list[ClassicalPrediction]:
    min_area = float(cfg["classical_baselines"]["min_contour_area"])
    predictions: list[ClassicalPrediction] = []
    for mask, lab_center in lab_kmeans_candidate_masks(image_bgr, cfg):
        class_id, conf = nearest_prototype_class(lab_center, prototypes)
        predictions.extend(contour_predictions(mask, class_id, conf, "lab_kmeans", min_area))
    return predictions


def candidate_masks_hsv_and_lab(image_bgr: np.ndarray, cfg: dict[str, Any]) -> list[np.ndarray]:
    baseline_cfg = cfg["classical_baselines"]
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    fruit_mask = estimate_fruit_mask(image_bgr, baseline_cfg["fruit_mask"])
    union = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for ranges in baseline_cfg["hsv_thresholding"]["class_ranges"].values():
        for item in ranges:
            union = cv2.bitwise_or(
                union,
                cv2.inRange(hsv, np.array(item["lower"], dtype=np.uint8), np.array(item["upper"], dtype=np.uint8)),
            )
    for mask, _ in lab_kmeans_candidate_masks(image_bgr, cfg):
        union = cv2.bitwise_or(union, mask)
    union = cv2.bitwise_and(union, fruit_mask)
    union = clean_binary_mask(union, int(baseline_cfg["morphology_kernel"]))

    masks: list[np.ndarray] = []
    contours, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < float(baseline_cfg["min_contour_area"]):
            continue
        mask = np.zeros(union.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        masks.append(mask)
    return masks


# Region features and negative samples --------------------------------------
def bbox_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros(4, dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def overlap_with_instances(mask: np.ndarray, instances: list[dict[str, Any]]) -> bool:
    box = bbox_from_mask(mask)
    for instance in instances:
        if box_iou(box, instance["xyxy"]) > 0.05:
            return True
    return False


def extract_region_features(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int((mask > 0).sum()) == 0:
        return np.zeros(56, dtype=np.float32)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pixels_bgr = image_bgr[mask > 0]
    pixels_hsv = hsv[mask > 0]
    pixels_lab = lab[mask > 0]
    pixels_gray = gray[mask > 0]

    features: list[float] = []
    for pixels in (pixels_bgr, pixels_hsv, pixels_lab):
        features.extend(pixels.mean(axis=0).astype(float).tolist())
        features.extend(pixels.std(axis=0).astype(float).tolist())

    h_hist = cv2.calcHist([hsv], [0], mask, [12], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask, [8], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask, [8], [0, 256]).flatten()
    for hist in (h_hist, s_hist, v_hist):
        denom = float(hist.sum()) or 1.0
        features.extend((hist / denom).astype(float).tolist())

    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(sobel_x, sobel_y)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    for texture in (pixels_gray, gradient[mask > 0], laplacian[mask > 0]):
        features.append(float(np.mean(texture)))
        features.append(float(np.std(texture)))

    box = bbox_from_mask(mask)
    area = float((mask > 0).sum())
    box_area = max(1.0, float((box[2] - box[0] + 1) * (box[3] - box[1] + 1)))
    features.extend(
        [
            area / max(1.0, float(mask.shape[0] * mask.shape[1])),
            area / box_area,
            float((box[2] - box[0] + 1) / max(1.0, box[3] - box[1] + 1)),
            float(mask.shape[1] / max(1.0, mask.shape[0])),
        ]
    )
    return np.nan_to_num(np.array(features, dtype=np.float32))


def random_negative_masks(
    fruit_mask: np.ndarray,
    instances: list[dict[str, Any]],
    count: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    negatives: list[np.ndarray] = []
    ys, xs = np.where(fruit_mask > 0)
    if len(xs) == 0:
        return negatives
    for _ in range(count * 20):
        if len(negatives) >= count:
            break
        idx = int(rng.integers(0, len(xs)))
        cx, cy = int(xs[idx]), int(ys[idx])
        radius = int(rng.integers(12, 45))
        mask = np.zeros(fruit_mask.shape, dtype=np.uint8)
        cv2.circle(mask, (cx, cy), radius, 255, thickness=cv2.FILLED)
        mask = cv2.bitwise_and(mask, fruit_mask)
        if int((mask > 0).sum()) < 80 or overlap_with_instances(mask, instances):
            continue
        negatives.append(mask)
    return negatives


# Train and run the optional texture classifier -----------------------------
def train_defect_classifier(cfg: dict[str, Any]) -> Path:
    from joblib import dump
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    clf_cfg = cfg["classical_baselines"]["defect_classifier"]
    data_yaml = resolve_path(cfg, cfg["paths"]["synthetic_dir"]) / "data.yaml"
    image_paths = image_paths_for_split(data_yaml, "train")
    if not image_paths:
        raise RuntimeError("No training images found for the classical defect classifier.")

    rng = np.random.default_rng(int(cfg["project"].get("seed", 42)))
    max_per_class = int(clf_cfg["max_train_samples_per_class"])
    positive_counts = {cid: 0 for cid in class_names(cfg)}
    negative_count = 0
    x_rows: list[np.ndarray] = []
    y_rows: list[int] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        image, _, _ = resize_for_baseline(image, cfg)
        instances = load_yolo_instances(label_path_for_image(image_path), image.shape[1], image.shape[0])
        for instance in instances:
            cid = int(instance["class_id"])
            if positive_counts[cid] >= max_per_class:
                continue
            x_rows.append(extract_region_features(image, instance["mask"]))
            y_rows.append(cid)
            positive_counts[cid] += 1

        fruit_mask = estimate_fruit_mask(image, cfg["classical_baselines"]["fruit_mask"])
        for mask in random_negative_masks(fruit_mask, instances, int(clf_cfg["negative_samples_per_image"]), rng):
            if negative_count >= max_per_class:
                break
            x_rows.append(extract_region_features(image, mask))
            y_rows.append(-1)
            negative_count += 1

        if all(count >= max_per_class for count in positive_counts.values()) and negative_count >= max_per_class:
            break

    if not x_rows:
        raise RuntimeError("No samples were extracted for the classical defect classifier.")

    classifier_name = str(clf_cfg.get("classifier", "svm")).lower()
    if classifier_name == "knn":
        params = clf_cfg["knn"]
        estimator = KNeighborsClassifier(n_neighbors=int(params["n_neighbors"]), weights=params["weights"])
    elif classifier_name == "svm":
        params = clf_cfg["svm"]
        estimator = SVC(C=float(params["c"]), gamma=params["gamma"], kernel=params["kernel"], probability=True)
    else:
        raise ValueError("classical_baselines.defect_classifier.classifier must be 'knn' or 'svm'")

    model = make_pipeline(StandardScaler(), estimator)
    model.fit(np.vstack(x_rows), np.array(y_rows, dtype=np.int32))

    out_dir = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "classical" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"defect_classifier_{classifier_name}.joblib"
    dump(
        {
            "model": model,
            "classifier": classifier_name,
            "classes": class_names(cfg),
            "positive_counts": positive_counts,
            "negative_count": negative_count,
        },
        model_path,
    )
    return model_path


def classifier_predict(image_bgr: np.ndarray, cfg: dict[str, Any], model_path: Path | None = None) -> list[ClassicalPrediction]:
    from joblib import load

    if model_path is None:
        classifier_name = cfg["classical_baselines"]["defect_classifier"].get("classifier", "svm")
        model_path = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "classical" / "models" / f"defect_classifier_{classifier_name}.joblib"
    if not model_path.exists():
        model_path = train_defect_classifier(cfg)

    package = load(model_path)
    model = package["model"]
    threshold = float(cfg["classical_baselines"]["defect_classifier"].get("min_confidence", 0.0))
    predictions: list[ClassicalPrediction] = []
    for mask in candidate_masks_hsv_and_lab(image_bgr, cfg):
        features = extract_region_features(image_bgr, mask).reshape(1, -1)
        pred_class = int(model.predict(features)[0])
        if pred_class < 0:
            continue
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            labels = list(model.classes_)
            conf = float(proba[labels.index(pred_class)]) if pred_class in labels else 0.50
        else:
            conf = 0.50
        if conf < threshold:
            continue
        predictions.append(
            ClassicalPrediction(
                class_id=pred_class,
                conf=conf,
                xyxy=bbox_from_mask(mask),
                mask=mask,
                method="hsv_texture_classifier",
            )
        )
    return predictions


# Reporting and evaluation --------------------------------------------------
def annotate_predictions(image: np.ndarray, predictions: list[ClassicalPrediction], names: dict[int, str]) -> np.ndarray:
    out = image.copy()
    palette = [(40, 180, 255), (70, 220, 90), (220, 80, 180)]
    for pred in predictions:
        color = palette[pred.class_id % len(palette)]
        if pred.mask is not None:
            overlay = out.copy()
            overlay[pred.mask > 0] = color
            out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
        x1, y1, x2, y2 = [int(round(v)) for v in pred.xyxy]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(pred.class_id, str(pred.class_id))} {pred.conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def write_classical_report(report: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "classical"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{report['method']}_{report['split']}"
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in report.items():
            if key not in {"per_class", "notes"}:
                writer.writerow([key, value])
    report["json_path"] = str(json_path)
    report["csv_path"] = str(csv_path)
    return report


def evaluate_classical_method(
    cfg: dict[str, Any],
    split: str,
    method: str,
    predictor: Callable[[np.ndarray], list[ClassicalPrediction]],
) -> dict[str, Any]:
    data_yaml = resolve_path(cfg, cfg["paths"]["synthetic_dir"]) / "data.yaml"
    image_paths = image_paths_for_split(data_yaml, split)
    if not image_paths:
        raise RuntimeError(f"No images found for split {split}. Generate the synthetic dataset first.")

    names = class_names(cfg)
    counts = {"tp": 0, "fp": 0, "fn": 0, "per_class": {i: {"tp": 0, "fp": 0, "fn": 0} for i in names}}
    latencies: list[float] = []
    annotated_limit = int(cfg["classical_baselines"].get("save_annotated", 25))
    annotated_dir = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "classical" / method / split / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    for index, image_path in enumerate(image_paths):
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if original is None:
            continue
        image, scale_x, scale_y = resize_for_baseline(original, cfg)
        t0 = time.perf_counter()
        raw_predictions = predictor(image)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        scaled_predictions = scale_predictions(raw_predictions, scale_x, scale_y)
        gt = load_yolo_boxes(label_path_for_image(image_path), original.shape[1], original.shape[0])
        batch_counts = greedy_match_counts(
            [pred.as_metric_dict() for pred in scaled_predictions],
            gt,
            float(cfg["evaluation"]["iou"]),
            len(names),
        )
        counts["tp"] += batch_counts["tp"]
        counts["fp"] += batch_counts["fp"]
        counts["fn"] += batch_counts["fn"]
        for cid, values in batch_counts["per_class"].items():
            for key in ("tp", "fp", "fn"):
                counts["per_class"][cid][key] += values[key]

        if index < annotated_limit:
            annotated = annotate_predictions(image, raw_predictions, names)
            cv2.imwrite(str(annotated_dir / image_path.name), annotated)

    elapsed = time.perf_counter() - start
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    accuracy = 0.0 if tp + fp + fn == 0 else tp / (tp + fp + fn)

    report = {
        "method": method,
        "split": split,
        "images": len(image_paths),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_iou50": precision,
        "recall_iou50": recall,
        "f1_iou50": f1,
        "detection_accuracy_iou50": accuracy,
        "elapsed_seconds": elapsed,
        "avg_latency_ms": float(np.mean(latencies)) if latencies else None,
        "per_class": counts["per_class"],
        "annotated_dir": str(annotated_dir),
    }
    return write_classical_report(report, cfg)


def run_classical_baselines(
    config_path: str | Path = "config/project.yaml",
    split: str = "test",
    methods: list[str] | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    methods = methods or ["hsv_thresholding", "lab_kmeans", "hsv_texture_classifier"]
    reports: dict[str, Any] = {}

    prototypes: dict[int, np.ndarray] | None = None
    classifier_model_path: Path | None = None
    for method in methods:
        if method == "hsv_thresholding":
            predictor = lambda image: hsv_thresholding_predict(image, cfg)
        elif method == "lab_kmeans":
            if prototypes is None:
                prototypes = learn_class_colour_prototypes(cfg)
            predictor = lambda image, p=prototypes: lab_kmeans_predict(image, cfg, p)
        elif method == "hsv_texture_classifier":
            if classifier_model_path is None:
                classifier_model_path = train_defect_classifier(cfg)
            predictor = lambda image, p=classifier_model_path: classifier_predict(image, cfg, p)
        else:
            raise ValueError(f"Unknown classical method: {method}")
        reports[method] = evaluate_classical_method(cfg, split, method, predictor)
    return reports
