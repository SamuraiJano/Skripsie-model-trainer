from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from naartjie_cv.coco import bbox_from_mask, mask_to_polygons
from naartjie_cv.classical import clean_binary_mask, contour_predictions, extract_region_features
from naartjie_cv.multispectral import rgb_image_to_rgbhsv_chw
from naartjie_cv.synthetic import balanced_class_plan, split_name_for_index


def test_mask_to_polygons_and_bbox() -> None:
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[20:60, 30:90] = 255

    polygons = mask_to_polygons(mask)
    bbox = bbox_from_mask(mask)

    assert polygons
    assert bbox == [30.0, 20.0, 60.0, 40.0]


def test_balanced_class_plan_counts() -> None:
    plan = balanced_class_plan(total=10, class_ids=[0, 1, 2], seed=1)
    counts = {class_id: plan.count(class_id) for class_id in [0, 1, 2]}

    assert sum(counts.values()) == 10
    assert max(counts.values()) - min(counts.values()) <= 1


def test_split_name_for_index() -> None:
    split_cfg = {"train": 0.8, "val": 0.1, "test": 0.1}

    assert split_name_for_index(0, 100, split_cfg) == "train"
    assert split_name_for_index(80, 100, split_cfg) == "val"
    assert split_name_for_index(90, 100, split_cfg) == "test"


def test_rgb_hsv_conversion_is_six_channel_chw() -> None:
    image = np.zeros((8, 10, 3), dtype=np.uint8)
    image[:, :, 1] = 128

    chw = rgb_image_to_rgbhsv_chw(image)

    assert chw.shape == (6, 8, 10)
    assert chw.dtype == np.uint8


def test_classical_feature_vector_and_contours() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:40, 24:44] = (10, 120, 220)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:40, 24:44] = 255

    features = extract_region_features(image, mask)
    cleaned = clean_binary_mask(mask, 5)
    predictions = contour_predictions(cleaned, class_id=1, conf=0.5, method="test", min_area=20)

    assert features.shape == (56,)
    assert predictions
    assert predictions[0].class_id == 1
