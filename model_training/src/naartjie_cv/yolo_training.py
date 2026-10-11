"""Validation-only checkpoint selection without changing reported YOLO metrics."""
from copy import copy

from ultralytics.models.yolo.segment import SegmentationTrainer, SegmentationValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import EarlyStopping
from .training_policy import MAX_EPOCHS, MIN_EPOCHS, PATIENCE, ValidationStoppingState


class ValidationEarlyStopping(EarlyStopping):
    """Count a flat zero-score plateau as no improvement, like positive scores."""
    def __init__(self, patience=PATIENCE, min_epochs=MIN_EPOCHS, max_epochs=MAX_EPOCHS):
        super().__init__(patience)
        self.best_fitness = float("-inf")
        self.state = ValidationStoppingState(max_epochs, min_epochs, patience)

    def __call__(self, epoch, fitness):
        if fitness is None:
            return False
        stop = self.state.observe(epoch, float(fitness))
        self.best_epoch = self.state.best_epoch
        self.best_fitness = self.state.best_metric
        self.possible_stop = bool(self.state.patience and epoch + 1 >= self.state.min_epochs_before_early_stop
                                  and self.state.epochs_since_best >= self.state.patience - 1)
        if stop:
            LOGGER.info(f"Early stopping: validation did not improve for {self.patience} epochs; best weights retained.")
        return stop


class MaskSelectionValidator(SegmentationValidator):
    def get_stats(self):
        stats = super().get_stats()
        # Native segmentation fitness adds box and mask mAP. The final crop
        # experiment selects and early-stops exclusively on validation masks.
        stats["fitness"] = stats["metrics/mAP50-95(M)"]
        return stats


class MaskSelectionTrainer(SegmentationTrainer):
    def get_validator(self):
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss"
        return MaskSelectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
