"""One-based validation duration policy shared by the final RGB/HSV trainers."""
from __future__ import annotations

from dataclasses import dataclass
import math


MAX_EPOCHS = 300
MIN_EPOCHS = 100
PATIENCE = 40
WORKERS = 4
DIAGNOSTIC_EPOCHS = (1, 3, 10, 50, 100, 150, 200, 250, 300)


@dataclass
class ValidationStoppingState:
    max_epochs: int = MAX_EPOCHS
    min_epochs_before_early_stop: int = MIN_EPOCHS
    patience: int = PATIENCE
    current_epoch: int = 0
    best_metric: float | None = None
    best_epoch: int = 0
    epochs_since_best: int = 0

    def observe(self, epoch: int, value: float) -> bool:
        """Strict > improvement, min_delta=0; ties do not reset patience."""
        if epoch < 1 or epoch <= self.current_epoch:
            raise ValueError("Validation epochs must increase and use one-based numbering")
        if not math.isfinite(value):
            raise ValueError("Validation stopping metric must be finite")
        self.current_epoch = epoch
        if self.best_metric is None or value > self.best_metric:
            self.best_metric, self.best_epoch = value, epoch
        self.epochs_since_best = epoch - self.best_epoch
        return self.should_stop

    @property
    def should_stop(self) -> bool:
        return bool(self.patience and self.current_epoch >= self.min_epochs_before_early_stop
                    and self.epochs_since_best >= self.patience)

    @property
    def stop_reason(self) -> str | None:
        if self.current_epoch >= self.max_epochs:
            return "max_epochs"
        return "early_stopping" if self.should_stop else None


def optimizer_steps_per_epoch(training_batches: int, accumulation: int) -> int:
    """Project D-FINE flushes the trailing partial accumulation window."""
    if training_batches <= 0 or accumulation <= 0:
        raise ValueError("Training batches and accumulation must be positive")
    return math.ceil(training_batches / accumulation)


def policy_metadata(max_epochs: int, min_epochs: int, patience: int, metric: str) -> dict:
    return {"max_epochs": max_epochs, "min_epochs_before_early_stop": min_epochs,
            "patience": patience, "monitored_metric": metric, "monitor_split": "val",
            "validation_frequency": 1, "improvement_rule": "strict_greater", "min_delta": 0.0,
            "epoch_numbering": "one_based", "best_metric": None, "best_epoch": None,
            "actual_stop_epoch": None, "stop_reason": None}
