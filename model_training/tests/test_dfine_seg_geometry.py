"""Exercise upstream preprocessing and inverse geometry in an isolated process.

The official D-FINE checkout also owns a top-level ``src`` package, so importing
both upstreams into pytest's process would make test order affect the result.
"""

from pathlib import Path
import subprocess
import sys

import pytest


def test_segmentation_letterbox_round_trip() -> None:
    project = Path(__file__).resolve().parents[1]
    upstream = project / "third_party" / "D-FINE-seg"
    if not (upstream / "src" / "dl" / "dataset.py").is_file():
        pytest.skip("Optional D-FINE-seg checkout is not installed")
    for dependency in ("torch", "albumentations", "omegaconf"):
        pytest.importorskip(dependency)

    result = subprocess.run(
        [sys.executable, "-c", r'''
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

project = Path(sys.argv[1])
upstream = project / "third_party" / "D-FINE-seg"
sys.path.insert(0, str(project))
sys.path.insert(0, str(upstream))
from scripts.dfine_seg_runner import _install_consistent_letterbox
from src.dl import dataset as module
from src.dl.utils import process_boxes, process_masks, abs_xyxy_to_norm_xywh

cfg = OmegaConf.load(upstream / "config.yaml")
cfg.task = "segment"
cfg.train.in_channels = 3
cfg.train.mosaic_augs.mosaic_prob = 0.0

def round_trip(h, w, mode, keep_ratio):
    cfg.train.keep_ratio = keep_ratio
    x1, y1, x2, y2 = w // 10, h // 5, w // 3, h // 2
    polygon = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    dataset = module.CustomDataset(
        img_size=(640, 640), root_path=project, split=pd.DataFrame(["unused.jpg"]),
        debug_img_processing=False, mode=mode, cfg=cfg,
        coco_annotations=[{"polys_abs": [[polygon]]}],
    )
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    transformed = dataset.transform(
        image=np.zeros((h, w, 3), dtype=np.uint8),
        bboxes=[[x1, y1, x2, y2]], class_labels=[0], box_indices=[0], masks=[mask],
    )
    boxes = torch.tensor(abs_xyxy_to_norm_xywh(np.asarray(transformed["bboxes"]), 640, 640))
    original_size = torch.tensor([[h, w]])
    restored_boxes = process_boxes(boxes[None], (640, 640), original_size, keep_ratio, "cpu")
    restored_mask = process_masks(
        transformed["masks"][0][None, None].float(), (640, 640), original_size, keep_ratio,
    )[0][0].numpy() >= 0.5
    iou = np.logical_and(restored_mask, mask).sum() / np.logical_or(restored_mask, mask).sum()
    error = np.max(np.abs(restored_boxes.numpy()[0, 0] - [x1, y1, x2, y2]))
    return iou, error

# The old path must reproduce the small-image failure before applying the fix.
broken_iou, _ = round_trip(200, 200, "val", True)
assert broken_iou < 0.2, broken_iou
_install_consistent_letterbox(module)
_install_consistent_letterbox(module)
for task in ("detect", "segment"):
    cfg.task = task
    for mode in ("val", "test", "bench"):
        for h, w in ((200, 200), (207, 170), (171, 209), (800, 1000)):
            for keep_ratio in (True, False):
                iou, error = round_trip(h, w, mode, keep_ratio)
                assert iou > 0.95, (task, mode, h, w, keep_ratio, iou)
                # Upstream rounds boxes to integer input pixels before undoing
                # resizing; a pixel corresponds to >1 original pixel when reduced.
                tolerance = max(h / 640, w / 640, 1.0) + 0.1
                assert error < tolerance, (task, mode, h, w, keep_ratio, error)
print("All mask and box geometry round trips passed")
''', str(project)],
        cwd=project,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
