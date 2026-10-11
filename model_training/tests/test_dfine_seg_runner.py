from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.dfine_seg_runner import _install_initial_best_checkpoint_fallback


def test_zero_score_run_promotes_first_completed_checkpoint(tmp_path) -> None:
    messages: list[str] = []

    class FakeTrainer:
        def __init__(self) -> None:
            self.path_to_save = tmp_path

        def save_model(self, metrics, best_metric):
            # Reproduce upstream behavior: last.pt is always saved, while a
            # decision score equal to the zero baseline does not create model.pt.
            (self.path_to_save / "last.pt").write_bytes(b"epoch-one")
            return best_metric

    train_module = SimpleNamespace(
        Trainer=FakeTrainer,
        logger=SimpleNamespace(info=messages.append),
    )
    _install_initial_best_checkpoint_fallback(train_module)
    _install_initial_best_checkpoint_fallback(train_module)

    trainer = train_module.Trainer()
    best_metric = trainer.save_model(
        {"f1": 0.0, "mAP_50_mask": 0.0, "iou": 0.0},
        0.0,
    )

    assert best_metric == 0.0
    assert (tmp_path / "model.pt").read_bytes() == b"epoch-one"
    assert trainer.early_stopping_steps == 0
    assert messages == ["Established the first completed epoch as the initial best model"]


def test_runner_records_actual_batch_and_best_validation_epoch(tmp_path) -> None:
    import json
    import torch
    from omegaconf import OmegaConf
    from scripts.dfine_seg_runner import _install_experiment_tracking

    checkpoint_dir = tmp_path / "model"
    checkpoint_dir.mkdir()
    class FakeTrainer:
        def __init__(self, cfg):
            self.b_accum_steps = cfg.train.b_accum_steps
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
            self.scheduler = None
            self.optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0008)
            self.decision_metrics = ["mAP_50_95_mask"]
            self.path_to_save = checkpoint_dir
            self.saved_epochs = 0
            self.train_loader = SimpleNamespace(num_workers=4, prefetch_factor=1, persistent_workers=True, pin_memory=True)
            self.val_loader = SimpleNamespace(num_workers=0, prefetch_factor=None, persistent_workers=False, pin_memory=True)
            self.test_loader = None
        def save_model(self, metrics, best_metric):
            self.saved_epochs += 1
            (checkpoint_dir / "last.pt").write_text(str(self.saved_epochs))
            score = metrics["mAP_50_95_mask"]
            if score > best_metric:
                (checkpoint_dir / "model.pt").write_text(str(self.saved_epochs))
                return score
            return best_metric

    module = SimpleNamespace(Trainer=FakeTrainer, torch=torch, logger=SimpleNamespace(info=lambda _: None))
    _install_initial_best_checkpoint_fallback(module)
    path = tmp_path / "run_metadata.json"
    path.write_text(json.dumps({"dataset": {"source_root": "unchanged"}}))
    _install_experiment_tracking(module, path)
    trainer = module.Trainer(OmegaConf.create({"train": {
        "batch_size": 2, "b_accum_steps": 8, "epochs": 300,
        "use_muon": True, "aux_optimizer": "adan", "use_ema": True,
    }}))
    best = 0
    for score in (0.0, 0.5, 0.2):
        best = trainer.save_model({"mAP_50_95_mask": score}, best)
    recorded = json.loads(path.read_text())
    assert recorded["optimisation"]["physical_batch_size"] == 2
    assert recorded["optimisation"]["gradient_accumulation"] == 8
    assert recorded["optimisation"]["effective_batch_size"] == 16
    assert recorded["selection"]["best_epoch"] == 2
    assert recorded["selection"]["best_validation_metric"] == 0.5
    assert recorded["selection"]["split"] == "val"
    assert (checkpoint_dir / "model.pt").read_text() == "2"
    assert recorded["dataset"]["source_root"] == "unchanged"
    assert recorded["data_loading"]["train"]["workers"] == 4
    assert recorded["data_loading"]["train"]["prefetch_factor"] == 1
    assert recorded["data_loading"]["val"]["workers"] == 0
    assert "test" not in recorded["data_loading"]
    assert len((tmp_path / "validation_history.jsonl").read_text().splitlines()) == 3


@pytest.mark.parametrize("platform", ["nt", "posix"])
@pytest.mark.parametrize("workers", [0, 4])
@pytest.mark.parametrize("mode", ["train", "val", "test"])
def test_memory_efficient_loaders_preserve_training_and_bound_windows_evaluation(monkeypatch, platform, workers, mode):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from scripts import dfine_seg_runner as runner

    class Loader:
        def __init__(self):
            self.num_workers = workers

        def _build_dataloader_impl(self, dataset, shuffle=False, distributed=False):
            options = {"num_workers": self.num_workers, "shuffle": shuffle}
            if self.num_workers > 0:
                options.update(prefetch_factor=2, persistent_workers=dataset.mode == "train")
            return DataLoader(dataset, batch_size=2, **options)

    module = SimpleNamespace(Loader=Loader)
    monkeypatch.setattr(runner, "os", SimpleNamespace(name=platform))
    runner._install_memory_efficient_loaders(module)
    runner._install_memory_efficient_loaders(module)
    dataset = TensorDataset(torch.arange(4))
    dataset.mode = mode
    builder = module.Loader()
    loader = builder._build_dataloader_impl(dataset, shuffle=False)
    expected_workers = 0 if platform == "nt" and mode != "train" else workers
    assert loader.num_workers == expected_workers
    assert builder.num_workers == workers
    expected_prefetch = (1 if platform == "nt" else 2) if expected_workers else None
    assert loader.prefetch_factor == expected_prefetch
    assert loader.persistent_workers == (expected_workers > 0 and mode == "train")
    if expected_workers == 0:
        assert torch.cat([batch[0] for batch in loader]).tolist() == [0, 1, 2, 3]


def test_memory_efficient_loaders_restore_worker_setting_after_build_failure(monkeypatch):
    from scripts import dfine_seg_runner as runner

    class Loader:
        num_workers = 4

        def _build_dataloader_impl(self, dataset):
            assert self.num_workers == 0
            raise ValueError("invalid dataset")

    module = SimpleNamespace(Loader=Loader)
    monkeypatch.setattr(runner, "os", SimpleNamespace(name="nt"))
    runner._install_memory_efficient_loaders(module)
    builder = module.Loader()
    with pytest.raises(ValueError, match="invalid dataset"):
        builder._build_dataloader_impl(SimpleNamespace(mode="val"))
    assert builder.num_workers == 4
