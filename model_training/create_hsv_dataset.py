from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def default_training_root() -> Path:
    return Path(__file__).resolve().parent


def iter_images(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def convert_image_to_hsv(src: Path, dst: Path) -> None:
    import cv2

    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {src}")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(dst), hsv):
        raise IOError(f"Could not write HSV image: {dst}")


def copy_labels_unchanged(src_root: Path, dst_root: Path) -> int:
    copied = 0
    for label_path in sorted(src_root.rglob("*")):
        if not label_path.is_file():
            continue
        rel = label_path.relative_to(src_root)
        out_path = dst_root / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(label_path, out_path)
        copied += 1
    return copied


def create_hsv_dataset(rgb_root: Path, hsv_root: Path) -> dict[str, object]:
    rgb_root = rgb_root.resolve()
    hsv_root = hsv_root.resolve()

    summary: dict[str, object] = {
        "rgb_root": str(rgb_root),
        "hsv_root": str(hsv_root),
        "splits": {},
    }

    for split in SPLITS:
        src_images = rgb_root / "images" / split
        src_labels = rgb_root / "labels" / split
        if not src_images.exists():
            raise FileNotFoundError(f"Missing RGB image split: {src_images}")
        if not src_labels.exists():
            raise FileNotFoundError(f"Missing RGB label split: {src_labels}")

        image_count = 0
        for image_path in iter_images(src_images):
            rel = image_path.relative_to(src_images)
            out_path = hsv_root / "images" / split / rel
            convert_image_to_hsv(image_path, out_path)
            image_count += 1

        label_count = copy_labels_unchanged(src_labels, hsv_root / "labels" / split)
        summary["splits"][split] = {"images": image_count, "labels": label_count}

    return summary


def parse_args() -> argparse.Namespace:
    training_root = default_training_root()
    parser = argparse.ArgumentParser(
        description="Create an HSV YOLO segmentation dataset from a pre-split RGB dataset."
    )
    parser.add_argument(
        "--rgb-root",
        type=Path,
        default=training_root / "dataset_rgb",
        help="Path to the source RGB dataset root.",
    )
    parser.add_argument(
        "--hsv-root",
        type=Path,
        default=training_root / "dataset_hsv",
        help="Path to the destination HSV dataset root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_hsv_dataset(args.rgb_root, args.hsv_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
