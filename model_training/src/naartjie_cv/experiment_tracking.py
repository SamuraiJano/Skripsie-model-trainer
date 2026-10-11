"""Persist the final experiment inputs and validation selection independently of tests."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .datasets import YOLO_IMAGE_SUFFIXES, _matching_label_path


def image_identity(path: Path) -> str:
    """RGB JPG and converted HSV PNG names refer to the same source image."""
    identity = path.with_suffix("")
    if identity.name.endswith("_hsv"):
        identity = identity.with_name(identity.name.removesuffix("_hsv")).with_suffix("")
    return identity.as_posix()


def dataset_identity(data_yaml: Path) -> dict[str, Any]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(payload.get("path") or ".")).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    representation = next(
        (colour.upper() for parent in (data_yaml.parent, *data_yaml.parents)
         for colour in ("rgb", "hsv") if parent.name.lower().endswith("_" + colour)),
        "unspecified",
    )
    splits = {}
    for split in ("train", "val", "test"):
        configured = payload.get(split)
        if not configured:
            continue
        directory = Path(str(configured)).expanduser()
        if not directory.is_absolute():
            directory = root / directory
        images = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in YOLO_IMAGE_SUFFIXES)
        files = [p.relative_to(directory).as_posix() for p in images]
        identities = [image_identity(p.relative_to(directory)) for p in images]
        labels = hashlib.sha256()
        for identity, path in zip(identities, images):
            label = _matching_label_path(path, root, split)
            labels.update(identity.encode("utf-8") + b"\0")
            labels.update(label.read_bytes() if label.is_file() else b"<missing>")
            labels.update(b"\0")
        splits[split] = {
            "path": str(directory.resolve()), "images": len(images), "image_identities": identities, "image_files": files,
            "identity_sha256": hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
            "annotation_sha256": labels.hexdigest(),
        }
    return {"colour_representation": str(payload.get("colour_space") or representation).upper(), "splits": splits}


def dataset_split_integrity(data_yaml: Path) -> dict[str, Any]:
    """Check fixed image splits and optional physical fruit/track identities.

    Identity mappings may be supplied in data.yaml as fruit_identity or
    track_identity, keyed by split/relative-image-name (without extension).
    An export_manifest.json may instead contain images with split, file_name
    and fruit_id/track_id. Missing coverage cannot prove physical fruit isolation.
    This function never changes split membership.
    """
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    splits = dataset_identity(data_yaml)["splits"]
    seen = {}
    for split, info in splits.items():
        for identity in info["image_identities"]:
            if identity in seen:
                raise ValueError(f"Source image identity crosses splits: {identity} ({seen[identity]}, {split})")
            seen[identity] = split
    mapping = payload.get("fruit_identity") or payload.get("track_identity") or {}
    if not isinstance(mapping, dict):
        raise ValueError("fruit_identity/track_identity must be an image-to-identity mapping")
    manifest_path = next((parent / "export_manifest.json" for parent in (data_yaml.parent, data_yaml.parent.parent)
                          if (parent / "export_manifest.json").is_file()), None)
    if not mapping and manifest_path:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest.get("images", []):
            physical = next((item.get(field) for field in ("fruit_id", "track_id", "naartjie_id")
                             if item.get(field) is not None), None)
            filename = item.get("file_name") or item.get("image_uuid")
            split = item.get("split")
            if physical is not None and split and filename:
                relative = Path(filename)
                if relative.parts[:2] == ("images", split):
                    relative = Path(*relative.parts[2:])
                mapping[f"{split}/{image_identity(relative)}"] = physical
    fruits = {}
    missing = []
    for split, info in splits.items():
        for identity in info["image_identities"]:
            key = f"{split}/{identity}"
            physical = mapping.get(key)
            if physical is None:
                missing.append(key)
                continue
            physical = str(physical)
            if physical in fruits and fruits[physical] != split:
                raise ValueError(f"Fruit/track identity crosses splits: {physical} ({fruits[physical]}, {split})")
            fruits[physical] = split
    return {"image_split_identity": "PASS", "fruit_track_split_integrity": "NOT PROVABLE" if missing else "PASS",
            "identity_metadata_images": sum(info["images"] for info in splits.values()) - len(missing),
            "warning": "No complete physical fruit/track identity metadata; different views of one fruit may cross splits." if missing else None}


def update_metadata(path: Path, **updates: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    payload.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Windows indexers can briefly deny atomic replacement while reading JSON.
    # Retain atomic writes and fail after a bounded retry for genuine permissions.
    for attempt in range(8):
        try:
            temporary.replace(path)
            break
        except PermissionError as error:
            if getattr(error, "winerror", None) not in {5, 32} or attempt == 7:
                raise
            time.sleep(0.01 * 2 ** attempt)


def record_validation(path: Path, *, epoch: int, metric: str, value: float,
                      checkpoint: Path, metrics: dict[str, Any], replace_equal: bool = False) -> None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    selection = payload.get("selection") or {}
    previous = selection.get("best_validation_metric")
    if previous is None or value > previous or (replace_equal and value == previous):
        selection.update({
            "split": "val", "metric": metric, "best_epoch": epoch,
            "best_validation_metric": value, "best_checkpoint_path": str(checkpoint.resolve()),
        })
    update_metadata(path, selection=selection, completed_epochs=epoch,
                    best_metric=selection.get("best_validation_metric"), best_epoch=selection.get("best_epoch"))
    history = path.parent / "validation_history.jsonl"
    with history.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"epoch": epoch, "split": "val", "metric": metric,
                                 "value": value, "metrics": metrics}) + "\n")
