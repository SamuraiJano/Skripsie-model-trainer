from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

TRAINING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING_ROOT))

from create_hsv_dataset import create_hsv_dataset  # noqa: E402


def write_split(root: Path, split: str, value: int) -> None:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image = np.array(
        [
            [[value, 20, 200], [30, value, 80]],
            [[90, 120, value], [255 - value, 40, 30]],
        ],
        dtype=np.uint8,
    )
    cv2.imwrite(str(image_dir / "sample.png"), image)
    (label_dir / "sample.txt").write_text("1 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")


def test_create_hsv_dataset_converts_images_and_copies_labels(tmp_path: Path) -> None:
    rgb_root = tmp_path / "dataset_rgb"
    hsv_root = tmp_path / "dataset_hsv"
    for index, split in enumerate(("train", "val", "test"), start=1):
        write_split(rgb_root, split, index * 40)

    summary = create_hsv_dataset(rgb_root, hsv_root)

    for split in ("train", "val", "test"):
        rgb = cv2.imread(str(rgb_root / "images" / split / "sample.png"), cv2.IMREAD_COLOR)
        hsv = cv2.imread(str(hsv_root / "images" / split / "sample.png"), cv2.IMREAD_COLOR)
        expected = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        assert np.array_equal(hsv, expected)
        assert (
            hsv_root / "labels" / split / "sample.txt"
        ).read_bytes() == (rgb_root / "labels" / split / "sample.txt").read_bytes()
        assert summary["splits"][split]["images"] == 1
        assert summary["splits"][split]["labels"] == 1


def test_dataset_configs_share_classes_and_splits() -> None:
    rgb = yaml.safe_load((TRAINING_ROOT / "configs" / "naartjie_rgb.yaml").read_text(encoding="utf-8"))
    hsv = yaml.safe_load((TRAINING_ROOT / "configs" / "naartjie_hsv.yaml").read_text(encoding="utf-8"))

    assert rgb["names"] == hsv["names"] == {0: "safe", 1: "minor_defect", 2: "major_defect"}
    assert rgb["train"] == hsv["train"] == "images/train"
    assert rgb["val"] == hsv["val"] == "images/val"
    assert rgb["test"] == hsv["test"] == "images/test"


def test_training_scripts_expose_help() -> None:
    for script in ("create_hsv_dataset.py", "train_rgb.py", "train_hsv.py", "export_models.py"):
        result = subprocess.run(
            [sys.executable, str(TRAINING_ROOT / script), "--help"],
            cwd=TRAINING_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
