from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


DFINE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "D-FINE"
sys.path.insert(0, str(DFINE_ROOT))

from src.solver import det_solver as det_solver_module  # noqa: E402


def test_stage_two_checkpoints_keep_the_actual_epoch(
    tmp_path: Path, monkeypatch
) -> None:
    solver = object.__new__(det_solver_module.DetSolver)
    solver.cfg = SimpleNamespace(
        epochs=202,
        checkpoint_freq=12,
        clip_max_norm=0.1,
        print_freq=10,
        use_wandb=False,
    )
    solver.train = lambda: None
    solver.last_epoch = 199
    solver.output_dir = tmp_path
    solver.writer = None
    solver.use_wandb = False
    solver.device = "cpu"
    solver.model = object()
    solver.criterion = object()
    solver.postprocessor = object()
    solver.val_dataloader = object()
    solver.evaluator = object()
    solver.optimizer = object()
    solver.scaler = None
    solver.lr_warmup_scheduler = None
    solver.lr_scheduler = SimpleNamespace(step=lambda: None)
    solver.ema = SimpleNamespace(module=object(), decay=0.999)
    solver.train_dataloader = SimpleNamespace(
        set_epoch=lambda epoch: None,
        sampler=object(),
        collate_fn=SimpleNamespace(stop_epoch=200, ema_restart_decay=0.9),
    )
    solver.state_dict = lambda: {"last_epoch": solver.last_epoch}

    loaded_epochs: list[int] = []

    def fake_load_resume_state(checkpoint: str) -> None:
        loaded_epochs.append(solver.last_epoch)
        solver.last_epoch = 150

    solver.load_resume_state = fake_load_resume_state

    scores = {199: 0.5, 200: 0.6, 201: 0.4}

    def fake_evaluate(*args, **kwargs):
        epoch = args[6]
        return {"coco_eval_bbox": [scores[epoch]]}, None

    saved: list[tuple[str, int]] = []

    def fake_save_on_master(state, checkpoint: Path) -> None:
        saved.append((checkpoint.name, state["last_epoch"]))

    monkeypatch.setattr(det_solver_module, "stats", lambda cfg: (0, "model stats"))
    monkeypatch.setattr(det_solver_module, "evaluate", fake_evaluate)
    monkeypatch.setattr(det_solver_module, "train_one_epoch", lambda *a, **k: {})
    monkeypatch.setattr(
        det_solver_module.dist_utils,
        "is_dist_available_and_initialized",
        lambda: False,
    )
    monkeypatch.setattr(det_solver_module.dist_utils, "is_main_process", lambda: True)
    monkeypatch.setattr(det_solver_module.dist_utils, "save_on_master", fake_save_on_master)

    solver.fit()

    assert loaded_epochs == [199, 201]
    assert solver.last_epoch == 201
    assert ("best_stg2.pth", 200) in saved
    assert [(name, epoch) for name, epoch in saved if name == "last.pth"] == [
        ("last.pth", 200),
        ("last.pth", 201),
    ]
