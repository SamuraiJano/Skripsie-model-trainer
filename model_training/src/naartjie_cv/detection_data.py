"""Paired detector representations over RGB storage and reversible geometry."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from .colour import HSV_ENCODING_METADATA, file_sha256, verify_paired_datasets
from .datasets import _matching_label_path, require_valid_yolo_training_dataset, YOLO_IMAGE_SUFFIXES
from .experiment_tracking import dataset_identity, dataset_split_integrity, image_identity


def letterbox_rgb(rgb: np.ndarray, size: int = 640):
    h, w = rgb.shape[:2]
    gain = min(size / h, size / w)
    resized_w, resized_h = round(w * gain), round(h * gain)
    left, top = (size - resized_w) // 2, (size - resized_h) // 2
    resized = cv2.resize(rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    output = np.full((size, size, 3), 114, dtype=np.uint8)
    output[top:top + resized_h, left:left + resized_w] = resized
    # Separate exact axis gains account for rounding to integer resized dimensions.
    return output, (resized_w / w, resized_h / h, left, top)


def to_representation(rgb: np.ndarray, colour: str) -> np.ndarray:
    if colour == "rgb":
        return np.ascontiguousarray(rgb)
    if colour == "hsv":
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV_FULL)
    raise ValueError("Detector representation must be rgb or hsv")


def load_canonical_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode RGB source: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def map_boxes(boxes, geometry, *, inverse=False):
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    gx, gy, left, top = geometry
    if inverse:
        result[:, [0, 2]] = (result[:, [0, 2]] - left) / gx
        result[:, [1, 3]] = (result[:, [1, 3]] - top) / gy
    else:
        result[:, [0, 2]] = result[:, [0, 2]] * gx + left
        result[:, [1, 3]] = result[:, [1, 3]] * gy + top
    return result


def detector_split_integrity(data_yaml):
    """Check every available grouping field; never infer fruit IDs from UUIDs."""
    base = dataset_split_integrity(data_yaml)
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    splits = dataset_identity(data_yaml)["splits"]
    groups = {key: payload.get(key, {}) for key in
              ("fruit_identity", "track_identity", "sequence_identity", "recording_sequence_identity")}
    manifest_path = next((p / "export_manifest.json" for p in (data_yaml.parent, data_yaml.parent.parent)
                          if (p / "export_manifest.json").is_file()), None)
    if manifest_path:
        exported = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in exported.get("images", []):
            split, filename = item.get("split"), item.get("file_name") or item.get("image_uuid")
            if not split or not filename:
                continue
            relative = Path(filename)
            if relative.parts[:2] == ("images", split):
                relative = Path(*relative.parts[2:])
            identity = f"{split}/{image_identity(relative)}"
            for key, fields in (("fruit_identity", ("fruit_id", "naartjie_id")), ("track_identity", ("track_id",)),
                                ("sequence_identity", ("sequence_id", "recording_id", "recording_sequence_id"))):
                value = next((item.get(f) for f in fields if item.get(f) is not None), None)
                if value is not None:
                    groups[key] = {**groups[key], identity: value}
    coverage = {}
    expected = sum(s["images"] for s in splits.values())
    for field, mapping in groups.items():
        if not isinstance(mapping, dict):
            raise ValueError(f"{field} must map split/sample identities to group IDs")
        seen, covered = {}, 0
        for split, info in splits.items():
            for identity in info["image_identities"]:
                value = mapping.get(f"{split}/{identity}")
                if value is None:
                    continue
                covered += 1
                value = str(value)
                if value in seen and seen[value] != split:
                    raise ValueError(f"{field} group crosses splits: {value}")
                seen[value] = split
        coverage[field] = {"covered_images": covered, "total_images": expected,
                           "status": "PASS" if covered == expected else "NOT PROVABLE"}
    base["grouping_checks"] = coverage
    base["group_metadata_sha256"] = hashlib.sha256(json.dumps(groups, sort_keys=True).encode()).hexdigest()
    if not any(c["covered_images"] == expected for c in coverage.values()):
        base["warning"] = "No complete fruit, track or recording-sequence grouping metadata; related-frame leakage cannot be ruled out. Fixed exported splits are preserved."
    return base


def canonical_label_bytes(label):
    if not label.is_file():
        return None, 0
    content = label.read_bytes()
    rows = [r for r in content.decode("utf-8").splitlines() if r.strip()]
    if not rows:
        return content, 0
    values = np.array([r.split() for r in rows], dtype=np.float32)
    _, indices = np.unique(values, axis=0, return_index=True)
    removed = len(rows) - len(indices)
    if not removed:
        return content, 0
    # Match native YOLO duplicate semantics once, for both detector families.
    return ("\n".join(rows[int(i)] for i in indices) + "\n").encode(), removed


def prepare_detector_pairs(source: str | Path, cache: Path) -> dict:
    source_yaml = require_valid_yolo_training_dataset(source, "detect")
    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if str(payload.get("colour_space", "rgb")).lower() != "rgb" or source_yaml.parent.name.endswith("_hsv"):
        raise ValueError("The controlled detector experiment requires the original RGB source")
    integrity = detector_split_integrity(source_yaml)
    identity = dataset_identity(source_yaml)
    if set(identity["splits"]) != {"train", "val", "test"} or any(not s["images"] for s in identity["splits"].values()):
        raise ValueError("A controlled final experiment requires populated fixed train/val/test splits")
    root = Path(payload.get("path") or source_yaml.parent)
    if not root.is_absolute():
        root = (source_yaml.parent / root).resolve()
    records = []
    digest = hashlib.sha256(b"controlled-detector-runtime-pairs-v3-shared-label-deduplication")
    digest.update(source_yaml.read_bytes())
    digest.update(integrity["group_metadata_sha256"].encode())
    for split, info in identity["splits"].items():
        directory = Path(info["path"])
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in YOLO_IMAGE_SUFFIXES:
                continue
            relative = path.relative_to(directory)
            label = _matching_label_path(path, root, split)
            digest.update(f"{split}/{relative}:{file_sha256(path)}".encode())
            digest.update(label.read_bytes() if label.is_file() else b"<missing>")
            records.append((split, path, relative, label))
    destination = cache.resolve() / digest.hexdigest()[:24]
    report_path = destination / "pair_verification.json"
    rgb_root = destination / "rgb"
    duplicate_counts = {s: 0 for s in ("train", "val", "test")}
    for split, path, relative, label in records:
        image_target = rgb_root / "images" / split / relative
        label_target = rgb_root / "labels" / split / relative.with_suffix(".txt")
        canonical, removed = canonical_label_bytes(label)
        duplicate_counts[split] += removed
        if report_path.is_file():
            if not image_target.is_file() or file_sha256(image_target) != file_sha256(path):
                raise ValueError("Canonical RGB source image bytes changed")
            actual_label = label_target.read_bytes() if label_target.is_file() else None
            if actual_label != canonical:
                raise ValueError("Canonical shared labels changed")
        else:
            image_target.parent.mkdir(parents=True, exist_ok=True)
            if not image_target.is_file():
                # Block JPEGs a native loader might repair in place before sharing
                # their bytes via a hard link. No source image may be rewritten.
                if path.suffix.lower() in {".jpg", ".jpeg"}:
                    with path.open("rb") as original:
                        original.seek(-2, os.SEEK_END)
                        if original.read() != b"\xff\xd9":
                            raise ValueError(f"JPEG needs repair before freezing: {path}")
                try:
                    os.link(path, image_target)
                except OSError:
                    shutil.copy2(path, image_target)
            if canonical is not None:
                label_target.parent.mkdir(parents=True, exist_ok=True)
                label_target.write_bytes(canonical)
    configurations = {}
    for colour in ("rgb", "hsv"):
        view = destination / colour
        view.mkdir(parents=True, exist_ok=True)
        # Logical paired datasets share immutable RGB storage. Exact uint8 HSV
        # arrays are generated after geometry, without lossy image compression.
        config = {**payload, "path": str(rgb_root), **{s: str(rgb_root / "images" / s) for s in ("train", "val", "test")},
                  "colour_space": colour, "source_data_yaml": str(source_yaml), "stored_colour_space": "rgb",
                  "representation_mode": "controlled_runtime_conversion", "controlled_detector_only": True}
        if colour == "hsv":
            config.update(HSV_ENCODING_METADATA)
            config["hsv_storage_order"] = "generated_at_load"
        target = view / "data.yaml"
        if report_path.is_file() and yaml.safe_load(target.read_text(encoding="utf-8")) != config:
            raise ValueError("Frozen paired dataset configuration differs from original source")
        if not report_path.is_file():
            target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configurations[colour] = file_sha256(target)
    rgb_yaml, hsv_yaml = destination / "rgb/data.yaml", destination / "hsv/data.yaml"
    verify_paired_datasets(rgb_yaml, hsv_yaml)
    prepared_identity = dataset_identity(rgb_yaml)
    for split in ("train", "val", "test"):
        for key in ("identity_sha256",):
            if prepared_identity["splits"][split][key] != identity["splits"][split][key]:
                raise ValueError(f"Paired cache differs from original source {split} membership or labels")
    # Pixel and dimension checks are essential: equal labels alone cannot prove pairing.
    pixels = hashlib.sha256()
    hsv_pixels = hashlib.sha256()
    samples = []
    for split, path, relative, label in records:
        a = load_canonical_rgb(rgb_root / "images" / split / relative)
        b = to_representation(a, "hsv")
        with Image.open(path) as original:
            dimensions = original.size
        if a.shape != b.shape or a.dtype != np.uint8 or b.dtype != np.uint8 or dimensions != (a.shape[1], a.shape[0]):
            raise ValueError(f"RGB/HSV source dimensions or exact HSV_FULL pixels differ: {split}/{relative}")
        pixels.update(a.tobytes())
        hsv_pixels.update(b.tobytes())
        samples.append({"split": split, "sample_identity": image_identity(relative), "source_file": str(path.resolve()),
                        "dimensions": [a.shape[1], a.shape[0]], "rgb_pixel_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
                        "hsv_full_pixel_sha256": hashlib.sha256(b.tobytes()).hexdigest(),
                        "label_sha256": file_sha256(label) if label.is_file() else None,
                        "canonical_label_sha256": file_sha256(rgb_root / "labels" / split / relative.with_suffix(".txt")) if label.is_file() else None})
    audit = destination / "paired_samples.json"
    if report_path.is_file() and json.loads(audit.read_text(encoding="utf-8")) != samples:
        raise ValueError("Frozen RGB/HSV pixel, dimension or annotation audit changed")
    if not report_path.is_file():
        audit.write_text(json.dumps(samples, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PASS", "source": str(source_yaml), "rgb": str(rgb_yaml), "hsv": str(hsv_yaml),
              "source_identity": identity, "dataset_identity": prepared_identity, "split_integrity": integrity,
              "source_pixel_sha256": pixels.hexdigest(), "pair_fingerprint": digest.hexdigest(),
              "HSV_FULL_pixel_sha256": hsv_pixels.hexdigest(), "paired_config_sha256": configurations,
              "paired_sample_audit": str(audit), "paired_sample_audit_sha256": file_sha256(audit),
              "duplicate_annotations_removed": duplicate_counts,
              "storage": "logical RGB/HSV views over original RGB files; exact uint8 HSV generated at load",
              "dimensions_identical": True, "labels_identical": True, "HSV_FULL_pixels_verified": True,
              "runtime_conversion": "geometry in canonical RGB, then RGB2HSV_FULL; float /255 once"}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def detection_ground_truth(data_yaml: str | Path, split: str) -> dict:
    data_yaml = Path(data_yaml)
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(payload.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    directory = Path(payload[split])
    if not directory.is_absolute():
        directory = root / directory
    names = payload["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    result = {"info": {}, "images": [], "annotations": [], "categories": [{"id": int(k), "name": v} for k, v in names.items()]}
    for path in sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in YOLO_IMAGE_SUFFIXES):
        with Image.open(path) as image:
            w, h = image.size
        image_id = len(result["images"]) + 1
        result["images"].append({"id": image_id, "width": w, "height": h, "file_name": str(path.resolve()),
                                 "sample_identity": image_identity(path.relative_to(directory))})
        label = _matching_label_path(path, root, split)
        for row in label.read_text().splitlines() if label.is_file() else []:
            if not row.strip():
                continue
            cls, xc, yc, bw, bh = map(float, row.split())
            bbox = [(xc - bw / 2) * w, (yc - bh / 2) * h, bw * w, bh * h]
            result["annotations"].append({"id": len(result["annotations"]) + 1, "image_id": image_id,
                                           "category_id": int(cls), "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
    return result
