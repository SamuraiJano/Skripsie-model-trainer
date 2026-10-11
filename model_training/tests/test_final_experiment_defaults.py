from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from naartjie_cv.train import _dfine_seg_config, final_dfine_seg_options, final_yolo_options
from naartjie_cv.experiment_tracking import record_validation
from naartjie_cv.yolo_training import MaskSelectionTrainer, MaskSelectionValidator, ValidationEarlyStopping
from scripts.dfine_seg_runner import _install_detection_lr_groups
from scripts.model_trainer import parser


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_active_dfine_config_overrides_old_upstream_defaults(tmp_path, task):
    root = Path(__file__).resolve().parents[1]
    upstream = root / "third_party" / "D-FINE-seg"
    if not (upstream / "config.yaml").is_file():
        pytest.skip("Local active ArgoHA checkout is unavailable")
    options = final_dfine_seg_options(task)
    prepared = {"root": str(tmp_path / "dataset"), "classes": {0: "naartjie"}}
    config = yaml.safe_load(_dfine_seg_config(upstream, tmp_path / task, prepared, task, options).read_text())
    train = config["train"]
    assert config["model_name"] == "n"
    assert train["img_size"] == [640, 640]
    assert train["epochs"] == 300 and train["early_stopping"] == 40
    assert train["min_epochs_before_early_stop"] == 100 and train["num_workers"] == 0
    assert train["batch_size"] * train["b_accum_steps"] == 16
    assert train["batch_size"] > 0 and train["amp_enabled"] is True
    assert train["decision_metrics"] == ["mAP_50_95" if task == "detect" else "mAP_50_95_mask"]
    assert train["mosaic_augs"]["mosaic_prob"] == (0.5 if task == "detect" else 0)
    assert train["use_muon"] is (task == "segment")
    assert train["use_ema"] and train["use_scheduler"]
    assert train["max_walltime_min"] is None
    prefix = "dfine_n_coco.pt" if task == "detect" else "dfine_seg_n_coco.pt"
    assert Path(train["pretrained_model_path"]).name == prefix
    assert train["base_lr"] == pytest.approx(0.0008)
    assert train["backbone_lr"] == pytest.approx(0.0004)
    args = parser().parse_args(["dfine-seg-" + task, "dataset", "--dfine-seg-root", str(upstream)])
    for name in ("model_size", "epochs", "imgsz", "batch", "workers", "early_stopping", "grad_accumulation",
                 "base_lr", "backbone_lr", "weight_decay", "mosaic_probability"):
        assert getattr(args, name) == getattr(options, name)
    assert args.muon is options.use_muon


def test_segmentation_selection_prefers_masks_when_boxes_disagree(monkeypatch):
    from ultralytics.models.yolo.segment import SegmentationValidator

    values = {"metrics/mAP50-95(B)": 0.9, "metrics/mAP50-95(M)": 0.2, "fitness": 1.1}
    monkeypatch.setattr(SegmentationValidator, "get_stats", lambda _: values.copy())
    validator = MaskSelectionValidator.__new__(MaskSelectionValidator)
    first = validator.get_stats()
    values.update({"metrics/mAP50-95(B)": 0.1, "metrics/mAP50-95(M)": 0.6, "fitness": 0.7})
    second = validator.get_stats()
    assert second["fitness"] > first["fitness"]
    assert first["metrics/mAP50-95(B)"] == 0.9  # reporting is unchanged
    assert second["fitness"] == second["metrics/mAP50-95(M)"] == 0.6


def test_segmentation_trainer_installs_mask_selection_validator(tmp_path):
    from ultralytics.cfg import get_cfg

    trainer = MaskSelectionTrainer.__new__(MaskSelectionTrainer)
    trainer.test_loader = None
    trainer.save_dir = tmp_path
    trainer.args = get_cfg(overrides={"task": "segment"})
    trainer.callbacks = None
    assert isinstance(trainer.get_validator(), MaskSelectionValidator)


def test_yolo_linear_scheduler_tracks_300_epoch_ceiling():
    from ultralytics.models.yolo.detect import DetectionTrainer

    trainer = DetectionTrainer.__new__(DetectionTrainer)
    trainer.args = SimpleNamespace(cos_lr=False, lrf=final_yolo_options().lrf)
    trainer.epochs = 300
    trainer.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.001)
    trainer._setup_scheduler()
    assert trainer.lf(0) == 1
    assert trainer.lf(150) == pytest.approx(0.505)
    assert trainer.lf(300) == pytest.approx(0.01)


def test_dfine_scheduler_retains_backbone_head_ratio_for_300_epochs():
    from torch.optim.lr_scheduler import OneCycleLR

    class FakeTrainer:
        def __init__(self, cfg):
            self.b_accum_steps = cfg.train.b_accum_steps
            self.train_loader = range(211)
            groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr}
                      for lr in (0.0002, 0.0002, 0.0004, 0.0004)]
            self.optimizer = torch.optim.AdamW(groups)
            self.scheduler = OneCycleLR(self.optimizer, max_lr=0.0008, total_steps=300 * 52)

    module = SimpleNamespace(Trainer=FakeTrainer, OneCycleLR=OneCycleLR)
    _install_detection_lr_groups(module)
    cfg = OmegaConf.create({"task": "detect", "train": {
        "use_muon": False, "use_scheduler": True, "preserve_lr_groups": True,
        "base_lr": 0.0004, "backbone_lr": 0.0002, "b_accum_steps": 4,
        "epochs": 300, "cycler_pct_start": 0.1,
    }})
    trainer = module.Trainer(cfg)
    assert trainer.scheduler.total_steps == 300 * 53
    assert [group["max_lr"] for group in trainer.optimizer.param_groups] == pytest.approx([0.0004] * 2 + [0.0008] * 2)
    for step in (0, 1589, 7950, 15899):
        trainer.scheduler.last_epoch = step
        with pytest.warns(UserWarning, match="get_last_lr"):
            rates = trainer.scheduler.get_lr()
        assert rates[2] == pytest.approx(2 * rates[0])
        assert all(0 <= rate <= 0.0008 for rate in rates)


def test_best_epoch_metadata_keeps_best_validation_score(tmp_path):
    path = tmp_path / "run_metadata.json"
    for epoch, score in enumerate((0.2, 0.4, 0.3), start=1):
        record_validation(path, epoch=epoch, metric="mask mAP50-95", value=score,
                          checkpoint=tmp_path / "best.pt", metrics={"mask mAP50-95": score})
    data = json.loads(path.read_text())
    assert data["selection"]["best_epoch"] == 2
    assert data["selection"]["best_validation_metric"] == 0.4
    assert data["selection"]["split"] == "val"
    assert data["completed_epochs"] == 3


@pytest.mark.parametrize("score", [0.0, 0.4])
def test_validation_plateau_stops_at_patience_40(score):
    stopper = ValidationEarlyStopping(40)
    for epoch in range(1, 100):
        assert not stopper(epoch, score)
    assert stopper(100, score)
    assert stopper.best_epoch == 1


def test_hsv_segmentation_augmentation_skips_colour_conversion(monkeypatch):
    import cv2
    import numpy as np
    from ultralytics.data.augment import RandomHSV

    options = final_yolo_options("segment")
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    def forbidden_conversion(*args, **kwargs):
        raise AssertionError("Stored HSV data must not be converted to HSV again")
    monkeypatch.setattr(cv2, "cvtColor", forbidden_conversion)
    result = RandomHSV(options.hsv_h, options.hsv_s, options.hsv_v).apply_image({"img": image.copy()})
    np.testing.assert_array_equal(result["img"], image)


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_yolo_records_validation_selection_before_separate_test_evaluation(tmp_path, monkeypatch, task):
    import ultralytics
    from naartjie_cv import train as train_module
    from naartjie_cv.train import TrainingOptions, train_yolo_export
    from test_training_export import make_yolo_dataset, add_test_split

    root = make_yolo_dataset(tmp_path, task)
    add_test_split(root, task)
    metric = "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)"
    run_dir = tmp_path / "runs" / "yolo26" / "controlled"
    models = []
    class FakeYOLO:
        def __init__(self, model_name):
            self.model_name = model_name
            self.callbacks = {}
            self.trainer = None
            models.append(self)
        def add_callback(self, name, callback):
            self.callbacks[name] = callback
        def train(self, trainer=None, **arguments):
            if task == "segment":
                assert trainer is MaskSelectionTrainer
            (run_dir / "weights").mkdir(parents=True)
            (run_dir / "weights" / "best.pt").touch()
            optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.001)
            self.trainer = SimpleNamespace(
                args=SimpleNamespace(**arguments), save_dir=run_dir, batch_size=16, accumulate=1,
                amp=True, optimizer=optimizer, epochs=300, metrics={}, epoch=0,
            )
            self.callbacks["on_pretrain_routine_end"](self.trainer)
            for epoch, score in enumerate((0.3, 0.6, 0.5)):
                self.trainer.epoch = epoch
                self.trainer.metrics = {metric: score}
                self.callbacks["on_fit_epoch_end"](self.trainer)
            # Final evaluation repeats the callback on the same epoch, and
            # must not replace the selected training-validation epoch.
            self.trainer.metrics = {metric: 0.99}
            self.callbacks["on_fit_epoch_end"](self.trainer)
            return SimpleNamespace(results_dict={metric: 0.6})
        def val(self, **arguments):
            assert arguments["split"] == "test"
            assert self.model_name == str(run_dir / "weights" / "best.pt")
            recorded = json.loads((run_dir / "run_metadata.json").read_text())
            assert recorded["selection"]["best_epoch"] == 2
            return SimpleNamespace(results_dict={metric: 0.99})
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(train_module, "_runtime", lambda: {"python": "test"})
    result = train_yolo_export(root, task, TrainingOptions(
        device="cpu", workers=0, output_dir=str(tmp_path / "runs"), run_name="controlled",
        yolo=final_yolo_options(task),
    ))
    metadata = json.loads(Path(result["metadata"]).read_text())
    assert metadata["selection"]["best_epoch"] == 2
    assert metadata["selection"]["best_validation_metric"] == 0.6
    assert metadata["optimisation"]["effective_batch_size"] == 16
    assert metadata["dataset"]["splits"]["test"]["images"] == 1
    assert len(models) == 2
