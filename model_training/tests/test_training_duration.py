from types import SimpleNamespace
import json

import pytest
import torch
from omegaconf import OmegaConf

from naartjie_cv.training_policy import ValidationStoppingState, optimizer_steps_per_epoch
from naartjie_cv.yolo_training import ValidationEarlyStopping
from scripts.dfine_seg_runner import _install_detection_lr_groups, _install_experiment_tracking


@pytest.mark.parametrize("best_epoch,expected_stop", [(1, 100), (95, 135), (120, 160)])
@pytest.mark.parametrize("colour", ["rgb", "hsv"])
def test_minimum_and_patience_histories(best_epoch, expected_stop, colour):
    stopper = ValidationEarlyStopping(40, 100, 300)
    stopped = None
    for epoch in range(1, 301):
        if stopper(epoch, 0.8 if epoch >= best_epoch else 0.5 * epoch / best_epoch):
            stopped = epoch
            break
    assert stopped == expected_stop
    assert stopper.state.best_epoch == best_epoch
    assert stopper.state.epochs_since_best == expected_stop - best_epoch


def test_improvement_one_epoch_before_expiry_resets_patience():
    state = ValidationStoppingState()
    for epoch in range(1, 174):
        value = 0.1 if epoch < 95 else 0.2 if epoch < 134 else 0.3
        assert not state.observe(epoch, value)
    assert state.observe(174, 0.3)
    assert state.best_epoch == 134


def test_continuous_improvement_reaches_maximum():
    state = ValidationStoppingState()
    for epoch in range(1, 301):
        assert not state.observe(epoch, epoch / 300)
    assert state.stop_reason == "max_epochs"


def test_ties_do_not_reset_patience_and_tiny_strict_improvements_do():
    state = ValidationStoppingState()
    state.observe(1, 0.0)
    state.observe(2, 0.0)
    assert state.best_epoch == 1
    state.observe(3, 1e-12)
    assert state.best_epoch == 3
    with pytest.raises(ValueError, match="finite"):
        state.observe(4, float("nan"))


@pytest.mark.parametrize("task,accumulation,batches", [("detect", 4, 211), ("segment", 8, 356), ("detect", 4, 212), ("segment", 8, 3)])
def test_scheduler_advances_once_per_actual_update_including_partial_windows(task, accumulation, batches):
    class FakeTrainer:
        def __init__(self, cfg):
            self.b_accum_steps = cfg.train.b_accum_steps
            self.train_loader = range(batches)
            peaks = [0.0008] * 4 if task == "detect" else [0.008] * 4 + [0.016]
            self.optimizer = torch.optim.AdamW([{"params": [torch.nn.Parameter(torch.zeros(1))]} for _ in peaks])
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, peaks, epochs=300, steps_per_epoch=max(1, batches // accumulation), cycle_momentum=False)
    module = SimpleNamespace(Trainer=FakeTrainer, OneCycleLR=torch.optim.lr_scheduler.OneCycleLR)
    _install_detection_lr_groups(module)
    cfg = OmegaConf.create({"task": task, "train": {"use_muon": task == "segment", "use_scheduler": True,
        "preserve_lr_groups": True, "base_lr": 0.0008, "backbone_lr": 0.0004,
        "b_accum_steps": accumulation, "epochs": 300, "cycler_pct_start": 0.1}})
    trainer = module.Trainer(cfg)
    trainer.scheduler.step()  # Simulate a GradScaler-skipped native boundary.
    assert trainer.scheduler.last_epoch == trainer._model_trainer_optimizer_steps == 0
    steps = optimizer_steps_per_epoch(batches, accumulation)
    assert trainer.scheduler.total_steps == 300 * steps
    assert [g["max_lr"] for g in trainer.optimizer.param_groups] == pytest.approx(
        [0.0008] * 2 + [0.0016] * 2 if task == "detect" else [0.008] * 4 + [0.016])
    for epoch in range(300):
        for update in range(steps):
            trainer.optimizer.step()
            if update < batches // accumulation:
                trainer.scheduler.step()  # Native boundary call must not double-step.
    assert trainer.scheduler.last_epoch == trainer._model_trainer_optimizer_steps == steps * 300
    assert all(g["lr"] >= 0 for g in trainer.optimizer.param_groups)


@pytest.mark.parametrize("reason", ["early_stopping", "max_epochs", "manual_interrupt", "error"])
def test_dfine_runtime_gate_and_stop_metadata(tmp_path, reason):
    class FakeTrainer:
        def __init__(self, cfg):
            self.b_accum_steps = 4
            self.amp_enabled, self.amp_dtype = True, torch.bfloat16
            self.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
            self.scheduler = None
            self.decision_metrics = ["mAP_50_95"]
            self.path_to_save = tmp_path
            self.early_stopping = cfg.train.early_stopping
        def save_model(self, metrics, best):
            return max(best, metrics["mAP_50_95"])
        def train(self):
            for epoch in range(1, 301):
                self.save_model({"mAP_50_95": epoch / 300 if reason == "max_epochs" else 0.5}, 0)
                if reason == "manual_interrupt" and epoch == 10:
                    raise KeyboardInterrupt()
                if reason == "error" and epoch == 10:
                    raise RuntimeError("artificial error")
                if self.early_stopping and self.early_stopping_steps >= self.early_stopping:
                    assert epoch >= 100
                    break
    module = SimpleNamespace(Trainer=FakeTrainer, torch=torch, logger=SimpleNamespace(info=lambda _: None))
    path = tmp_path / "run_metadata.json"
    _install_experiment_tracking(module, path)
    trainer = module.Trainer(OmegaConf.create({"train": {"batch_size": 4, "b_accum_steps": 4, "epochs": 300,
        "early_stopping": 40, "min_epochs_before_early_stop": 100, "use_muon": False, "use_ema": True}}))
    if reason in ("manual_interrupt", "error"):
        with pytest.raises(KeyboardInterrupt if reason == "manual_interrupt" else RuntimeError):
            trainer.train()
    else:
        trainer.train()
    metadata = json.loads(path.read_text())
    assert metadata["stop_reason"] == reason
    assert metadata["actual_stop_epoch"] == {"early_stopping": 100, "max_epochs": 300, "manual_interrupt": 10, "error": 10}[reason]
    assert metadata["monitor_split"] == metadata["selection"]["split"] == "val"
    assert metadata["max_epochs"] == 300 and metadata["patience"] == 40


@pytest.mark.parametrize("framework", ["yolo", "project_dfine"])
@pytest.mark.parametrize("task", ["detect", "segment"])
def test_duration_validator_rejects_every_policy_reversion(framework, task):
    from copy import deepcopy
    from naartjie_cv.final_validation import verify_duration_config, configuration_differences
    from naartjie_cv.training_policy import policy_metadata
    if framework == "yolo":
        config = {"epochs": 300, "patience": 40, "workers": 4, "val": True, "split": "val",
            "training_policy": policy_metadata(300, 100, 40,
                "metrics/mAP50-95(B)" if task == "detect" else "metrics/mAP50-95(M)")}
        fields = (("epochs", 100), ("patience", 0), ("workers", 0),
                  ("training_policy.min_epochs_before_early_stop", 0), ("split", "test"))
    else:
        config = {"train": {"epochs": 300, "min_epochs_before_early_stop": 100, "early_stopping": 40,
                            "num_workers": 4, "decision_metrics": ["mAP_50_95" if task == "detect" else "mAP_50_95_mask"]}}
        fields = (("train.epochs", 100), ("train.min_epochs_before_early_stop", 0),
                  ("train.early_stopping", 0), ("train.num_workers", 0), ("train.decision_metrics", ["test_mAP"]))
    verify_duration_config(config, framework, task)
    for field, value in fields:
        altered = deepcopy(config)
        parts = field.split(".")
        parent = altered
        for part in parts[:-1]:
            parent = parent[part]
        parent[parts[-1]] = value
        with pytest.raises(AssertionError):
            verify_duration_config(altered, framework, task)
        assert configuration_differences(config, altered, framework)


@pytest.mark.parametrize("replace_equal,expected_best", [(False, 1), (True, 2)])
def test_checkpoint_ties_are_independent_of_stopping_patience(tmp_path, replace_equal, expected_best):
    from naartjie_cv.experiment_tracking import record_validation
    state = ValidationStoppingState()
    path = tmp_path / "run_metadata.json"
    for epoch in (1, 2):
        state.observe(epoch, 0.5)
        record_validation(path, epoch=epoch, metric="validation box mAP50-95", value=0.5,
                          checkpoint=tmp_path / "best.pt", metrics={}, replace_equal=replace_equal)
    assert json.loads(path.read_text())["best_epoch"] == expected_best
    assert state.best_epoch == 1 and state.epochs_since_best == 1


@pytest.mark.parametrize("reason", ["early_stopping", "max_epochs", "manual_interrupt", "error"])
def test_yolo_runtime_gate_and_stop_metadata(tmp_path, monkeypatch, reason):
    import ultralytics
    from naartjie_cv import train as train_module
    from naartjie_cv.train import TrainingOptions, train_yolo_export
    from test_training_export import make_yolo_dataset
    root = make_yolo_dataset(tmp_path, "detect")
    metric = "metrics/mAP50-95(B)"
    directory = tmp_path / "runs/yolo26/controlled"
    class FakeYOLO:
        def __init__(self, _):
            self.callbacks = {}
        def add_callback(self, name, callback):
            self.callbacks[name] = callback
        def train(self, **arguments):
            (directory / "weights").mkdir(parents=True)
            (directory / "weights/best.pt").touch()
            self.trainer = SimpleNamespace(args=SimpleNamespace(**arguments), save_dir=directory,
                batch_size=16, accumulate=1, amp=True, epochs=300, epoch=0, metrics={},
                optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.001))
            self.callbacks["on_pretrain_routine_end"](self.trainer)
            for epoch in range(1, 301):
                self.trainer.epoch = epoch - 1
                self.callbacks["on_train_epoch_start"](self.trainer)
                assert self.trainer._oom_retries == 3  # OOM must never halve this pair's physical batch.
                score = epoch / 300 if reason == "max_epochs" else 0.5
                stop = self.trainer.stopper(epoch, score)
                self.trainer.metrics = {metric: score}
                self.callbacks["on_fit_epoch_end"](self.trainer)
                if reason == "manual_interrupt" and epoch == 10:
                    raise KeyboardInterrupt()
                if reason == "error" and epoch == 10:
                    raise RuntimeError("artificial error")
                if stop:
                    break
            return SimpleNamespace(results_dict={metric: score})
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(train_module, "_runtime", lambda: {"python": "test"})
    options = TrainingOptions(device="cpu", output_dir=str(tmp_path / "runs"), run_name="controlled")
    if reason in ("manual_interrupt", "error"):
        with pytest.raises(KeyboardInterrupt if reason == "manual_interrupt" else RuntimeError):
            train_yolo_export(root, "detect", options)
    else:
        train_yolo_export(root, "detect", options)
    metadata = json.loads((directory / "run_metadata.json").read_text())
    assert metadata["stop_reason"] == reason
    assert metadata["actual_stop_epoch"] == {"early_stopping": 100, "max_epochs": 300, "manual_interrupt": 10, "error": 10}[reason]
    assert metadata["best_metric"] is not None and metadata["best_epoch"] is not None
    assert metadata["monitor_split"] == metadata["selection"]["split"] == "val"
    assert metadata["max_epochs"] == 300 and metadata["min_epochs_before_early_stop"] == 100
    assert metadata["patience"] == 40 and metadata["training_arguments"]["workers"] == 4


@pytest.mark.parametrize("exception,reason", [(KeyboardInterrupt, "manual_interrupt"), (RuntimeError, "error")])
def test_project_launcher_records_subprocess_termination_without_losing_best(tmp_path, monkeypatch, exception, reason):
    from pathlib import Path
    from naartjie_cv import train as train_module
    from naartjie_cv.experiment_tracking import record_validation
    from test_training_export import make_yolo_dataset
    root = make_yolo_dataset(tmp_path, "detect")
    upstream = Path(__file__).resolve().parents[1] / "third_party/D-FINE-seg"
    monkeypatch.setattr(train_module, "_ensure_dfine_seg_pretrained", lambda *args, **kwargs: None)
    def interrupted(command, **kwargs):
        path = Path(command[command.index("--config") + 1]).parent / "run_metadata.json"
        record_validation(path, epoch=5, metric="mAP_50_95", value=0.5, checkpoint=path.parent / "model/model.pt", metrics={})
        raise exception()
    monkeypatch.setattr(train_module, "_run_training_process", interrupted)
    options = train_module.final_dfine_seg_options("detect")
    options.output_dir, options.run_name, options.device = str(tmp_path / "runs"), "terminated", "cpu"
    with pytest.raises(exception):
        train_module.train_dfine_seg_export(root, "detect", upstream, options)
    path = tmp_path / "runs/dfine-seg/terminated/run_metadata.json"
    data = json.loads(path.read_text())
    assert data["stop_reason"] == reason and data["actual_stop_epoch"] == 5
    assert data["best_epoch"] == 5 and data["best_metric"] == 0.5
    assert data["max_epochs"] == 300 and data["min_epochs_before_early_stop"] == 100 and data["patience"] == 40


def test_metadata_atomic_replace_retries_transient_windows_lock(tmp_path, monkeypatch):
    from pathlib import Path
    from naartjie_cv import experiment_tracking as tracking
    original = Path.replace
    calls = []
    waits = []
    def locked_once(self, target):
        calls.append(target)
        if len(calls) == 1:
            error = PermissionError("Transient Windows sharing lock")
            error.winerror = 32
            raise error
        return original(self, target)
    monkeypatch.setattr(Path, "replace", locked_once)
    monkeypatch.setattr(tracking.time, "sleep", waits.append)
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"dataset": "frozen"}))
    tracking.update_metadata(path, stop_reason="max_epochs", actual_stop_epoch=300)
    assert json.loads(path.read_text()) == {"dataset": "frozen", "stop_reason": "max_epochs", "actual_stop_epoch": 300}
    assert len(calls) == 2 and waits == [0.01]
