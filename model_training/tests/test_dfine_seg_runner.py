from __future__ import annotations

from types import SimpleNamespace

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
