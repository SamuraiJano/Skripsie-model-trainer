"""Controlled detection subclasses; segmentation and legacy trainers are untouched."""
from __future__ import annotations

import shutil
import time
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import ops

from .detection_data import detection_ground_truth
from .detection_metrics import coco_ap
from .experiment_tracking import record_validation, update_metadata
from .yolo_training import ValidationEarlyStopping


def convert_batch_representation(batch: dict, colour: str) -> dict:
    """Ultralytics Format has already produced uint8 RGB, after geometry."""
    if colour == "hsv":
        image = batch["img"]
        if image.dtype != torch.uint8:
            raise ValueError("Colour conversion must happen before float scaling")
        converted = [cv2.cvtColor(x.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2HSV_FULL).transpose(2, 0, 1)
                     for x in image]
        batch = {**batch, "img": torch.from_numpy(np.ascontiguousarray(np.stack(converted)))}
    elif colour != "rgb":
        raise ValueError("Expected rgb or hsv")
    return batch


class CommonDetectionValidator(DetectionValidator):
    def __init__(self, *args, protocol: dict, **kwargs):
        self.protocol = protocol
        super().__init__(*args, **kwargs)

    def preprocess(self, batch):
        return super().preprocess(convert_batch_representation(batch, self.protocol["colour"]))

    def init_metrics(self, model):
        super().init_metrics(model)
        split = self.args.split
        if split not in {"val", "test"}:
            raise ValueError("Controlled detection validates only val or final test")
        self.common_ground_truth = detection_ground_truth(self.protocol["rgb_yaml"], split)
        self.common_image_ids = {str(Path(i["file_name"]).resolve()).casefold(): i["id"] for i in self.common_ground_truth["images"]}
        self.prediction_records = []
        self.common_seen = set()

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for index, prediction in enumerate(preds):
            key = str(Path(batch["im_file"][index]).resolve()).casefold()
            image_id = self.common_image_ids[key]
            if image_id in self.common_seen:
                raise ValueError("Repeated validation/test sample")
            self.common_seen.add(image_id)
            prepared = self._prepare_batch(index, batch)
            boxes = ops.scale_boxes(prepared["imgsz"], prediction["bboxes"].clone(),
                                    prepared["ori_shape"], ratio_pad=prepared["ratio_pad"]).cpu().numpy()
            for box, score, cls in zip(boxes, prediction["conf"].cpu().tolist(), prediction["cls"].cpu().tolist()):
                x1, y1, x2, y2 = map(float, box)
                self.prediction_records.append({"image_id": image_id, "category_id": int(cls), "score": score,
                                                "bbox": [x1, y1, max(0., x2 - x1), max(0., y2 - y1)]})

    def get_stats(self):
        stats = super().get_stats()
        if self.common_seen != set(self.common_image_ids.values()):
            raise ValueError("YOLO did not evaluate the complete fixed split")
        if self.protocol.get("collection_only"):
            return stats  # Final AP is computed once by the common final phase.
        self.common_result = coco_ap(self.common_ground_truth, self.prediction_records)
        from ultralytics.utils import LOGGER
        LOGGER.info(f"Controlled COCO box mAP50-95={self.common_result['mAP50-95']:.6f}; mAP50={self.common_result['mAP50']:.6f}")
        stats.update({"metrics/mAP50-95(B)": self.common_result["mAP50-95"],
                      "metrics/mAP50(B)": self.common_result["mAP50"], "fitness": self.common_result["mAP50-95"]})
        return stats


class ControlledDetectionTrainer(DetectionTrainer):
    def __init__(self, *args, protocol: dict, **kwargs):
        self.protocol = protocol
        self.metadata_path = Path(protocol["metadata"])
        self.observed_accumulation = set()
        self.actual_updates = 0
        self.started = time.monotonic()
        super().__init__(*args, **kwargs)
        self.add_callback("on_train_epoch_start", self._prevent_batch_change)

    @staticmethod
    def _prevent_batch_change(trainer):
        trainer._oom_retries = 3  # Fail rather than silently changing paired recipes.

    def _setup_train(self):
        super()._setup_train()
        self.stopper = ValidationEarlyStopping(40, 100, 300)
        if self.train_loader.drop_last:
            raise ValueError("Controlled detection must not drop training images")
        self.observed_accumulation.add(int(self.accumulate))
        self.optimizer.register_step_post_hook(lambda *_: setattr(self, "actual_updates", self.actual_updates + 1))
        from .runtime import library_versions
        from .train import _runtime
        update_metadata(self.metadata_path, effective_training_config=vars(self.args),
                        runtime={**_runtime(), "libraries": library_versions()}, augmentation=self.protocol["augmentation"],
                        optimisation={"optimizer": type(self.optimizer).__name__, "physical_batch_size": self.batch_size,
                                      "gradient_accumulation": int(self.accumulate), "effective_batch_size": self.batch_size * int(self.accumulate),
                                      "effective_batch_is_nominal": True, "last_physical_batch": len(self.train_loader.dataset) % self.batch_size or self.batch_size,
                                      "accumulation_policy": "native Ultralytics; actual warmup observations recorded",
                                      "workers": self.train_loader.num_workers, "drop_last": self.train_loader.drop_last,
                                      "amp": bool(self.amp), "scheduler": "cosine" if self.args.cos_lr else "linear",
                                      "amp_dtype": "float16" if self.amp else "float32", "ema_checkpoint_dtype": "float16",
                                      "warmup_epochs": self.args.warmup_epochs, "ema": "native Ultralytics ModelEMA",
                                      "gradient_clipping": 10., "weight_decay": self.args.weight_decay,
                                      "parameter_groups": [{k: g[k] for k in ("lr", "initial_lr", "weight_decay", "betas", "eps") if k in g}
                                                           for g in self.optimizer.param_groups]})

    def preprocess_batch(self, batch):
        self.observed_accumulation.add(int(self.accumulate))
        return super().preprocess_batch(convert_batch_representation(batch, self.protocol["colour"]))

    def get_validator(self):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        return CommonDetectionValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args),
                                        _callbacks=self.callbacks, protocol=self.protocol)

    def build_dataset(self, img_path, mode="train", batch=None):
        from ultralytics.data import build_yolo_dataset
        from ultralytics.utils.torch_utils import unwrap_model
        stride = max(int(unwrap_model(self.model).stride.max()) if self.model else 0, 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=stride)

    def save_model(self):
        result = super().save_model()
        if not result:
            return result
        value = float(self.metrics["metrics/mAP50-95(B)"])
        previous = self.protocol.get("best_validation")
        checkpoint = Path(self.save_dir) / "weights/best_controlled.pt"
        if previous is None or value > previous:
            shutil.copy2(self.last, checkpoint)
            self.protocol["best_validation"] = value
        record_validation(self.metadata_path, epoch=self.epoch + 1, metric="box COCO mAP50-95", value=value,
                          checkpoint=checkpoint, metrics={k: float(v) for k, v in self.metrics.items()})
        update_metadata(self.metadata_path, actual_optimizer_steps=self.actual_updates,
                        final_learning_rates=[float(g["lr"]) for g in self.optimizer.param_groups],
                        early_stopping_state=vars(self.stopper.state),
                        observed_accumulation_steps=sorted(self.observed_accumulation),
                        observed_nominal_effective_batches=[self.batch_size * k for k in sorted(self.observed_accumulation)])
        return result

    def final_eval(self):
        # The experiment worker performs validation confidence selection and the
        # held-out test phase using this same evaluator after training completes.
        # Avoid a duplicate final native summary overwriting selection metadata.
        state = self.stopper.state
        if state.stop_reason not in {"early_stopping", "max_epochs"} or not 100 <= state.current_epoch <= 300:
            raise RuntimeError("YOLO ended outside the controlled stopping policy")
        update_metadata(self.metadata_path, completed_epochs=state.current_epoch, actual_stop_epoch=state.current_epoch,
                        stop_reason=state.stop_reason, training_seconds=time.monotonic() - self.started,
                        actual_optimizer_steps=self.actual_updates)


def collect_yolo_predictions(model, protocol: dict, split: str):
    validators = []
    def factory(*args, **kwargs):
        validator = CommonDetectionValidator(*args, protocol={**protocol, "collection_only": True}, **kwargs)
        validators.append(validator)
        return validator
    model.val(validator=factory, data=protocol["rgb_yaml"], split=split, imgsz=640,
              batch=protocol["batch"], workers=protocol["workers"], device="0", half=False,
              conf=0.001, max_det=100, rect=False, plots=False,
              project=str(Path(protocol["metadata"]).parent / "evaluation"), name=split, exist_ok=False)
    validator = validators[0]
    return validator.common_ground_truth, validator.prediction_records
