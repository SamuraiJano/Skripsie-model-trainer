from __future__ import annotations

import contextlib
import csv
import io
import json
import shutil
from pathlib import Path
from typing import Any

from .config import class_names, load_config, resolve_path


def _require_fiftyone() -> Any:
    try:
        import fiftyone as fo
    except Exception as exc:  # pragma: no cover - depends on conda env
        raise RuntimeError(
            "FiftyOne is not installed. Recreate/update the conda environment with "
            "`conda env update -f environment.yml --prune`."
        ) from exc
    return fo


def _require_view_field() -> Any:
    try:
        from fiftyone import ViewField as F
    except Exception as exc:  # pragma: no cover - depends on conda env
        raise RuntimeError("Could not import FiftyOne ViewField.") from exc
    return F


def fiftyone_dataset_name(cfg: dict[str, Any], split: str) -> str:
    return f"{cfg['fiftyone']['dataset_prefix']}_{split}"


def load_fiftyone_dataset(
    config_path: str | Path = "config/project.yaml",
    split: str = "test",
    dataset_name: str | None = None,
    overwrite: bool = False,
) -> Any:
    """Loads one synthetic COCO split into FiftyOne."""
    fo = _require_fiftyone()
    cfg = load_config(config_path)
    root = resolve_path(cfg, cfg["paths"]["synthetic_dir"])
    labels_path = root / "coco" / f"instances_{split}.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"COCO labels not found: {labels_path}")

    name = dataset_name or fiftyone_dataset_name(cfg, split)
    if name in fo.list_datasets():
        if overwrite:
            fo.delete_dataset(name)
        else:
            return fo.load_dataset(name)

    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=str(root),
        labels_path=str(labels_path),
        label_types=cfg["fiftyone"].get("label_types", "segmentations"),
        label_field=cfg["fiftyone"].get("label_field", "ground_truth"),
        include_id=True,
        name=name,
        overwrite=True,
    )
    dataset.persistent = True
    dataset.default_classes = [name for _, name in sorted(class_names(cfg).items())]
    dataset.info["source"] = "synthetic_naartjie_coco"
    dataset.info["split"] = split
    dataset.info["config_path"] = str(Path(config_path).resolve())
    dataset.tag_samples(split)
    return dataset


def apply_ultralytics_predictions(
    dataset: Any,
    weights: str | Path,
    prediction_field: str = "predictions",
    confidence: float = 0.25,
    batch_size: int | None = None,
) -> Any:
    """Applies a trained Ultralytics model to a FiftyOne dataset."""
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - depends on conda env
        raise RuntimeError("Ultralytics is not installed in the active environment.") from exc

    model = YOLO(str(weights))
    kwargs: dict[str, Any] = {
        "label_field": prediction_field,
        "confidence_thresh": confidence,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    dataset.apply_model(model, **kwargs)
    return dataset


def evaluate_fiftyone_predictions(
    dataset: Any,
    cfg: dict[str, Any],
    prediction_field: str | None = None,
    label_field: str | None = None,
    eval_key: str | None = None,
    iou: float | None = None,
    use_masks: bool = True,
) -> dict[str, Any]:
    """Runs FiftyOne COCO-style evaluation and writes a compact report."""
    prediction_field = prediction_field or cfg["fiftyone"]["prediction_field"]
    label_field = label_field or cfg["fiftyone"]["label_field"]
    eval_key = eval_key or cfg["fiftyone"]["eval_key"]
    iou = float(cfg["evaluation"]["iou"] if iou is None else iou)

    if eval_key in dataset.list_evaluations():
        dataset.delete_evaluation(eval_key)

    results = dataset.evaluate_detections(
        prediction_field,
        gt_field=label_field,
        eval_key=eval_key,
        method="coco",
        iou=iou,
        classwise=True,
        use_masks=use_masks,
        compute_mAP=True,
    )

    classes = [name for _, name in sorted(class_names(cfg).items())]
    report_text = ""
    with contextlib.suppress(Exception):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            results.print_report(classes=classes)
        report_text = buffer.getvalue()

    patch_counts: dict[str, int] = {}
    with contextlib.suppress(Exception):
        patch_counts = dataset.to_evaluation_patches(eval_key).count_values("type")

    metrics = {
        "dataset": dataset.name,
        "samples": len(dataset),
        "prediction_field": prediction_field,
        "label_field": label_field,
        "eval_key": eval_key,
        "iou": iou,
        "use_masks": use_masks,
        "tp": int(dataset.sum(f"{eval_key}_tp") or 0),
        "fp": int(dataset.sum(f"{eval_key}_fp") or 0),
        "fn": int(dataset.sum(f"{eval_key}_fn") or 0),
        "patch_counts": patch_counts,
    }
    with contextlib.suppress(Exception):
        metrics["mAP"] = float(results.mAP())
    with contextlib.suppress(Exception):
        metrics["mAR"] = float(results.mAR())

    out_dir = resolve_path(cfg, cfg["paths"]["runs_dir"]) / "fiftyone"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_base = out_dir / f"{dataset.name}_{eval_key}"
    (report_base.with_suffix(".json")).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if report_text:
        (report_base.with_suffix(".txt")).write_text(report_text, encoding="utf-8")
    metrics["json_path"] = str(report_base.with_suffix(".json"))
    if report_text:
        metrics["text_report_path"] = str(report_base.with_suffix(".txt"))
    return metrics


def _sample_field(sample: Any, field: str, default: Any = None) -> Any:
    try:
        return sample[field]
    except Exception:
        return default


def export_hard_examples(
    dataset: Any,
    cfg: dict[str, Any],
    eval_key: str | None = None,
    label_field: str | None = None,
) -> dict[str, Any]:
    """Copies hard images and writes a manifest for manual inspection/curation."""
    F = _require_view_field()
    fo = _require_fiftyone()
    eval_key = eval_key or cfg["fiftyone"]["eval_key"]
    label_field = label_field or cfg["fiftyone"]["label_field"]
    hard_cfg = cfg["fiftyone"]["hard_examples"]

    max_samples = int(hard_cfg["max_samples"])
    min_fp = int(hard_cfg["min_false_positives"])
    min_fn = int(hard_cfg["min_false_negatives"])
    export_dir = resolve_path(cfg, hard_cfg["export_dir"]) / dataset.name
    images_dir = export_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    hard_view = (
        dataset.match((F(f"{eval_key}_fp") >= min_fp) | (F(f"{eval_key}_fn") >= min_fn))
        .sort_by(f"{eval_key}_fp", reverse=True)
        .limit(max_samples)
    )

    manifest_path = export_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "source_path", "copied_path", "tp", "fp", "fn"])
        for sample in hard_view:
            src = Path(sample.filepath)
            dst = images_dir / src.name
            if src.exists():
                shutil.copy2(src, dst)
            writer.writerow(
                [
                    sample.id,
                    str(src),
                    str(dst),
                    _sample_field(sample, f"{eval_key}_tp", 0),
                    _sample_field(sample, f"{eval_key}_fp", 0),
                    _sample_field(sample, f"{eval_key}_fn", 0),
                ]
            )

    coco_export_dir = export_dir / "coco"
    coco_export_status = "not_exported"
    with contextlib.suppress(Exception):
        hard_view.export(
            export_dir=str(coco_export_dir),
            dataset_type=fo.types.COCODetectionDataset,
            label_field=label_field,
        )
        coco_export_status = "exported"

    return {
        "dataset": dataset.name,
        "hard_samples": len(hard_view),
        "export_dir": str(export_dir),
        "manifest_path": str(manifest_path),
        "coco_export_status": coco_export_status,
    }


def run_fiftyone_review(
    config_path: str | Path = "config/project.yaml",
    split: str = "test",
    weights: str | Path | None = None,
    dataset_name: str | None = None,
    overwrite: bool = False,
    launch: bool = True,
    export_hard: bool = True,
    wait: bool = False,
) -> dict[str, Any]:
    """Loads data, optionally applies/evaluates a model, exports hard examples, and launches the app."""
    fo = _require_fiftyone()
    cfg = load_config(config_path)
    dataset = load_fiftyone_dataset(config_path, split=split, dataset_name=dataset_name, overwrite=overwrite)
    report: dict[str, Any] = {"dataset": dataset.name, "split": split, "samples": len(dataset)}

    if weights is not None:
        apply_ultralytics_predictions(
            dataset,
            weights=weights,
            prediction_field=cfg["fiftyone"]["prediction_field"],
            confidence=float(cfg["evaluation"]["conf"]),
        )
        report["evaluation"] = evaluate_fiftyone_predictions(dataset, cfg)
        if export_hard:
            report["hard_examples"] = export_hard_examples(dataset, cfg)

    if launch:
        session = fo.launch_app(dataset, port=int(cfg["fiftyone"]["launch_port"]))
        report["app_url"] = session.url
        if wait:
            session.wait()
    return report

