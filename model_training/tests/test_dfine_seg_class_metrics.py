from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from scripts.dfine_seg_model_tools import _per_class_average_precision


def test_per_class_ap_uses_class_ids_and_ignores_invalid_recall_points() -> None:
    precision = torch.tensor(
        [
            [
                [[[1.0]], [[0.0]]],
                [[[0.5]], [[0.2]]],
                [[[-1.0]], [[0.4]]],
            ]
        ]
    )
    result = {
        "classes": torch.tensor([2, 0]),
        "map_per_class": torch.tensor([0.4, 0.2]),
        "precision": precision,
    }

    by_class = _per_class_average_precision(result)

    assert by_class[2]["ap50"] == pytest.approx(0.75)
    assert by_class[2]["ap50_95"] == pytest.approx(0.4)
    assert by_class[0]["ap50"] == pytest.approx(0.2)
    assert by_class[0]["ap50_95"] == pytest.approx(0.2)
