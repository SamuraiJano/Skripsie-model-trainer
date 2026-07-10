from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate basic COCO instance segmentation structure.")
    parser.add_argument("json_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.json_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    image_ids = {img["id"] for img in data["images"]}
    category_ids = {cat["id"] for cat in data["categories"]}
    errors = []
    for ann in data["annotations"]:
        if ann["image_id"] not in image_ids:
            errors.append(f"annotation {ann['id']} references missing image_id {ann['image_id']}")
        if ann["category_id"] not in category_ids:
            errors.append(f"annotation {ann['id']} references missing category_id {ann['category_id']}")
        if not ann.get("segmentation"):
            errors.append(f"annotation {ann['id']} has no segmentation")
        if ann.get("area", 0) <= 0:
            errors.append(f"annotation {ann['id']} has non-positive area")
    if errors:
        raise SystemExit("\n".join(errors[:50]))
    print(f"OK: {len(data['images'])} images, {len(data['annotations'])} annotations")


if __name__ == "__main__":
    main()

