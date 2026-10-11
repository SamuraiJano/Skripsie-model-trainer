"""Lossless, split-preserving HSV preparation for the framework's RGB readers.

PNG file channels are H,S,V (PIL RGB slots), never HSV converted back to RGB.
OpenCV consequently reads V,S,H; the frameworks' usual reversal restores H,S,V.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from .datasets import YOLO_IMAGE_SUFFIXES, _matching_label_path, require_valid_yolo_training_dataset
from .experiment_tracking import dataset_identity, dataset_split_integrity, image_identity

HSV_ENCODING_METADATA = {
    "colour_space": "HSV", "hsv_encoding": "HSV_FULL",
    "hue_range": [0, 255], "saturation_range": [0, 255], "value_range": [0, 255],
    "tensor_scaling": "divide_by_255", "hsv_storage_order": "HSV", "tensor_channel_order": "HSV",
}
HSV_CACHE_VERSION = "hsv-full-v2"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def colour_metadata(data_yaml: Path) -> dict:
    if dataset_colour(data_yaml) == "hsv":
        return dict(HSV_ENCODING_METADATA)
    return {"colour_space": "RGB", "tensor_scaling": "divide_by_255", "tensor_channel_order": "RGB"}


def dataset_colour(data_yaml: Path) -> str:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    value = payload.get("colour_space")
    if value is None:
        value = "hsv" if any(p.name.lower().endswith("_hsv") for p in data_yaml.parents) else "rgb"
    value = str(value).lower()
    if value not in {"rgb", "hsv"}:
        raise ValueError("Dataset colour_space must be rgb or hsv")
    return value


def verify_paired_datasets(rgb_yaml: Path, hsv_yaml: Path) -> None:
    """Reject different classes, split membership or annotations before training."""
    rgb = dataset_identity(rgb_yaml)
    hsv = dataset_identity(hsv_yaml)
    a = yaml.safe_load(rgb_yaml.read_text(encoding="utf-8"))
    b = yaml.safe_load(hsv_yaml.read_text(encoding="utf-8"))
    if a["names"] != b["names"] or rgb["splits"].keys() != hsv["splits"].keys():
        raise ValueError("RGB/HSV classes or configured splits differ")
    for split in rgb["splits"]:
        for key in ("identity_sha256", "annotation_sha256"):
            if rgb["splits"][split][key] != hsv["splits"][split][key]:
                raise ValueError(f"RGB/HSV {split} {key} differs; use the same source images and labels")


def prepare_colour_dataset(root: str | Path, task: str, cache: Path, colour_space: str = "auto") -> Path:
    """Return a dataset root; materialize canonical HSV only when necessary.

    Existing HSV with a sibling *_rgb dataset is checked against that source.
    Historical and generated HSV must have a verifiable paired RGB source.
    Encoding/order declarations alone cannot establish source provenance.
    """
    if colour_space not in {"auto", "rgb", "hsv"}:
        raise ValueError("colour_space must be auto, rgb or hsv")
    source_yaml = require_valid_yolo_training_dataset(root, task)
    integrity = dataset_split_integrity(source_yaml)
    if integrity["warning"]:
        from logging import getLogger
        getLogger(__name__).warning(integrity["warning"])
    source_colour = dataset_colour(source_yaml)
    selected = source_colour if colour_space == "auto" else colour_space
    if selected == "rgb":
        if source_colour != "rgb":
            raise ValueError("Select an RGB source dataset for RGB training")
        return Path(root).expanduser().resolve()

    input_yaml = source_yaml
    stored_yaml = None
    if source_colour == "hsv":
        payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
        sibling = source_yaml.parent.with_name(source_yaml.parent.name.removesuffix("_hsv") + "_rgb")
        provenance = payload.get("source_data_yaml")
        report_path = source_yaml.parent / "colour_preparation.json"
        if not provenance and report_path.is_file():
            provenance = json.loads(report_path.read_text(encoding="utf-8")).get("source_data_yaml")
        paired_path = Path(str(provenance)) if provenance else sibling / "data.yaml"
        if not paired_path.is_absolute():
            paired_path = source_yaml.parent / paired_path
        if not paired_path.is_file():
            raise ValueError("Stored HSV requires RGB source provenance (paired *_rgb dataset or source_data_yaml); explicit hsv_storage_order alone is insufficient")
        paired_yaml = require_valid_yolo_training_dataset(paired_path.parent, task)
        if dataset_colour(paired_yaml) != "rgb":
            raise ValueError("HSV provenance must point to the original RGB source")
        verify_paired_datasets(paired_yaml, source_yaml)
        stored_yaml, source_yaml = source_yaml, paired_yaml
        integrity = dataset_split_integrity(source_yaml)

    payload = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    identity = dataset_identity(source_yaml)
    stored = dataset_identity(stored_yaml) if stored_yaml else None
    source_root = Path(str(payload.get("path") or "."))
    if not source_root.is_absolute():
        source_root = (source_yaml.parent / source_root).resolve()
    records = []
    fingerprint = hashlib.sha256(f"{HSV_CACHE_VERSION}:HSV_FULL:{source_yaml}:{input_yaml}".encode())
    fingerprint.update(json.dumps(HSV_ENCODING_METADATA, sort_keys=True).encode())
    image_provenance = {}
    fingerprint.update(source_yaml.read_bytes())
    if stored_yaml:
        fingerprint.update(stored_yaml.read_bytes())
    for split, info in identity["splits"].items():
        paths = sorted((p for p in Path(info["path"]).rglob("*") if p.suffix.lower() in YOLO_IMAGE_SUFFIXES),
                       key=lambda p: image_identity(p.relative_to(Path(info["path"]))))
        paired = {}
        if stored:
            directory = Path(stored["splits"][split]["path"])
            paired = {image_identity(p.relative_to(directory)): p for p in directory.rglob("*") if p.suffix.lower() in YOLO_IMAGE_SUFFIXES}
        seen = set()
        for path in paths:
            relative = path.relative_to(Path(info["path"]))
            source_id = image_identity(relative)
            if source_id in seen:
                raise ValueError(f"Duplicate source image identity: {split}/{source_id}")
            seen.add(source_id)
            label = _matching_label_path(path, source_root, split)
            fingerprint.update(f"{split}/{relative}:{path.stat().st_size}:{path.stat().st_mtime_ns}".encode())
            source_hash = file_sha256(path)
            fingerprint.update(source_hash.encode())
            image_provenance[f"{split}/{relative.with_suffix('.png').as_posix()}"] = {
                "source_identity": source_id, "source_path": str(path), "source_sha256": source_hash,
            }
            fingerprint.update(label.read_bytes() if label.is_file() else b"<missing>")
            partner = paired.get(source_id)
            if partner:
                fingerprint.update(f"{partner.stat().st_size}:{partner.stat().st_mtime_ns}".encode())
            records.append((split, path, relative.with_suffix(".png"), label, partner))
    destination = cache / fingerprint.hexdigest()[:24]
    report_path = destination / "colour_preparation.json"
    if (destination / "data.yaml").is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        metadata = yaml.safe_load((destination / "data.yaml").read_text(encoding="utf-8"))
        valid_encoding = all(report.get(key) == value and metadata.get(key) == value for key, value in HSV_ENCODING_METADATA.items())
        valid_encoding = valid_encoding and report.get("cache_version") == HSV_CACHE_VERSION
        valid_images = valid_encoding and all(
            (destination / "images" / relative).is_file()
            and file_sha256(destination / "images" / relative) == item.get("prepared_sha256")
            for relative, item in report.get("image_provenance", {}).items()
        ) and report.get("image_provenance", {}).keys() == image_provenance.keys()
        if valid_images:
            verify_paired_datasets(source_yaml, destination / "data.yaml")
            return destination
    # A cache with an old encoding, missing metadata or altered pixels is stale.
    (destination / "data.yaml").unlink(missing_ok=True)
    orders = set()
    encodings = set()
    for split, path, relative, label, partner in records:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot read RGB source image: {path}")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        if partner:
            with Image.open(partner) as opened:
                disk = np.array(opened.convert("RGB"))
            matched = False
            for encoding, reference in (("HSV_FULL", hsv), ("HSV_180", cv2.cvtColor(image, cv2.COLOR_BGR2HSV))):
                for order, candidate in (("HSV", disk), ("VSH", disk[..., ::-1])):
                    if np.array_equal(candidate, reference):
                        orders.add(order)
                        encodings.add(encoding)
                        matched = True
                        break
                if matched:
                    break
            if not matched:
                raise ValueError(f"Stored HSV does not match its RGB source: {partner}")
        target = destination / "images" / split / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(hsv).save(target, compress_level=1)
        image_provenance[f"{split}/{relative.as_posix()}"]["prepared_sha256"] = file_sha256(target)
        if label.is_file():
            target_label = destination / "labels" / split / relative.with_suffix(".txt")
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(label, target_label)
    report = {"input_data_yaml": str(input_yaml), "source_data_yaml": str(source_yaml), "split_integrity": integrity,
              **HSV_ENCODING_METADATA, "cache_version": HSV_CACHE_VERSION,
              "source_storage_orders": sorted(orders), "source_encodings": sorted(encodings),
              "conversion": "OpenCV COLOR_BGR2HSV_FULL; H,S,V each 0..255; model tensor H,S,V /255",
              "paired_source_verified": True, "image_provenance": image_provenance}
    (destination / "colour_preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    data = {"path": ".", "names": payload["names"], "source_data_yaml": str(source_yaml), **HSV_ENCODING_METADATA}
    data.update({split: f"images/{split}" for split in identity["splits"]})
    # Commit the cache only after every image and label has been written.
    (destination / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    verify_paired_datasets(source_yaml, destination / "data.yaml")
    return destination
