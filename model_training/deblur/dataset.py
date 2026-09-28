from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class PairedDatasetConfig:
    root: str
    image_size: int = 256
    augment: bool = True
    horizontal_flip_probability: float = 0.5
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _files(path: Path) -> dict[str, Path]:
    if not path.is_dir():
        return {}
    return {item.name: item for item in sorted(path.iterdir(), key=lambda value: value.name.casefold()) if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS}


def _manifest_rows(root: Path) -> list[dict[str, Any]]:
    path = root / "manifests" / "pairs.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else payload.get("pairs", [])
    if not isinstance(rows, list):
        raise ValueError("pairs.json must contain a list under 'pairs'.")
    return list(rows)


def validate_paired_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    split_counts: dict[str, int] = {}
    names_by_split: dict[str, set[str]] = {}
    dimensions: dict[tuple[str, str], tuple[int, int]] = {}
    for split in SPLITS:
        blur = _files(dataset_root / split / "blur")
        sharp = _files(dataset_root / split / "sharp")
        names_by_split[split] = set(blur) & set(sharp)
        split_counts[split] = len(names_by_split[split])
        if set(blur) != set(sharp):
            missing_sharp = sorted(set(blur) - set(sharp))
            missing_blur = sorted(set(sharp) - set(blur))
            if missing_sharp:
                errors.append(f"{split}: missing sharp matches for {missing_sharp[:10]}")
            if missing_blur:
                errors.append(f"{split}: missing blur matches for {missing_blur[:10]}")
        for name in sorted(names_by_split[split], key=str.casefold):
            pair_sizes = []
            for member, path in (("blur", blur[name]), ("sharp", sharp[name])):
                try:
                    with Image.open(path) as image:
                        image.load()
                        if image.mode != "RGB":
                            errors.append(f"{path} is {image.mode}, expected RGB.")
                        pair_sizes.append(image.size)
                        dimensions[(split, f"{member}/{name}")] = image.size
                except Exception as exc:
                    errors.append(f"Could not read {path}: {exc}")
            if len(pair_sizes) == 2 and pair_sizes[0] != pair_sizes[1]:
                errors.append(f"{split}/{name}: blur dimensions {pair_sizes[0]} do not match sharp {pair_sizes[1]}.")
    duplicate_names = (names_by_split["train"] & names_by_split["val"]) | (names_by_split["train"] & names_by_split["test"]) | (names_by_split["val"] & names_by_split["test"])
    if duplicate_names:
        errors.append(f"Pair filenames occur in multiple splits: {sorted(duplicate_names)[:10]}")

    try:
        rows = _manifest_rows(dataset_root)
    except Exception as exc:
        errors.append(f"Invalid pairs manifest: {exc}")
        rows = []
    if rows:
        pair_ids: set[str] = set()
        manifest_names: dict[str, set[str]] = {split: set() for split in SPLITS}
        identities: dict[tuple[str, str], set[str]] = {}
        for row in rows:
            pair_id = str(row.get("pair_id") or "")
            split = str(row.get("split") or "")
            name = str(row.get("file_name") or "")
            if not pair_id:
                errors.append("Manifest row has no pair_id.")
            elif pair_id in pair_ids:
                errors.append(f"Duplicate pair ID in manifest: {pair_id}")
            pair_ids.add(pair_id)
            if split not in SPLITS:
                errors.append(f"Manifest pair {pair_id} has invalid split {split!r}.")
                continue
            manifest_names[split].add(name)
            for key in ("source_hash", "perceptual_hash", "capture_group", "fruit_id", "tree_id", "session_id"):
                value = str(row.get(key) or "").casefold()
                if value:
                    identities.setdefault((key, value), set()).add(split)
        for split in SPLITS:
            if manifest_names[split] != names_by_split[split]:
                errors.append(f"{split}: pairs manifest does not agree with files on disk.")
        for identity, splits in identities.items():
            if len(splits) > 1:
                errors.append(f"Leakage: {identity[0]}={identity[1]} occurs in {sorted(splits)}.")
    else:
        warnings.append("No pairs.json manifest was found; source/group leakage cannot be fully verified.")
    if split_counts["train"] == 0:
        errors.append("Training split contains no valid pairs.")
    if split_counts["val"] == 0:
        warnings.append("Validation split contains no pairs.")
    return {
        "valid": not errors, "dataset_root": str(dataset_root), "split_counts": split_counts,
        "pair_count": sum(split_counts.values()), "errors": errors, "warnings": warnings,
    }


def require_valid_paired_dataset(root: str | Path) -> dict[str, Any]:
    report = validate_paired_dataset(root)
    if not report["valid"]:
        raise ValueError("Invalid paired deblurring dataset:\n- " + "\n- ".join(report["errors"]))
    return report


def _to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


class PairedAugment:
    def __init__(self, image_size: int, flip_probability: float = 0.5) -> None:
        self.image_size = image_size
        self.flip_probability = flip_probability

    def __call__(self, blurred: Image.Image, sharp: Image.Image, rng: random.Random) -> tuple[Image.Image, Image.Image]:
        if blurred.size != sharp.size:
            raise ValueError("Paired augmentation requires matching source dimensions.")
        width, height = blurred.size
        if min(width, height) < self.image_size:
            scale = self.image_size / min(width, height)
            target_size = (max(self.image_size, round(width * scale)), max(self.image_size, round(height * scale)))
            blurred = blurred.resize(target_size, Image.Resampling.BICUBIC)
            sharp = sharp.resize(target_size, Image.Resampling.BICUBIC)
            width, height = target_size
        left = rng.randint(0, width - self.image_size) if width > self.image_size else 0
        top = rng.randint(0, height - self.image_size) if height > self.image_size else 0
        box = (left, top, left + self.image_size, top + self.image_size)
        blurred, sharp = blurred.crop(box), sharp.crop(box)
        if rng.random() < self.flip_probability:
            blurred, sharp = ImageOps.mirror(blurred), ImageOps.mirror(sharp)
        return blurred, sharp


class PairedDeblurDataset(Dataset[dict[str, Any]]):
    def __init__(self, config: PairedDatasetConfig, split: str) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.config = config
        self.split = split
        root = Path(config.root).expanduser().resolve()
        blur_files, sharp_files = _files(root / split / "blur"), _files(root / split / "sharp")
        if set(blur_files) != set(sharp_files):
            raise ValueError(f"{split} blur/sharp filenames do not match; validate the dataset first.")
        self.pairs = [(name, blur_files[name], sharp_files[name]) for name in sorted(blur_files, key=str.casefold)]
        self.epoch = 0
        self.augment = PairedAugment(config.image_size, config.horizontal_flip_probability)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name, blur_path, sharp_path = self.pairs[index]
        with Image.open(blur_path) as image:
            blurred = image.convert("RGB")
        with Image.open(sharp_path) as image:
            sharp = image.convert("RGB")
        if blurred.size != sharp.size:
            raise ValueError(f"Pair {name} has mismatched dimensions.")
        if self.config.augment and self.split == "train":
            digest = hashlib.sha256(f"{self.config.seed}:{self.epoch}:{name}".encode()).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            blurred, sharp = self.augment(blurred, sharp, rng)
        else:
            blurred = ImageOps.fit(blurred, (self.config.image_size, self.config.image_size), method=Image.Resampling.BICUBIC)
            sharp = ImageOps.fit(sharp, (self.config.image_size, self.config.image_size), method=Image.Resampling.BICUBIC)
        return {"blurred": _to_tensor(blurred), "sharp": _to_tensor(sharp), "pair_id": Path(name).stem, "file_name": name}
