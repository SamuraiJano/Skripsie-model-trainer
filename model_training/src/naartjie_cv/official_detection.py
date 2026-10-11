"""Detection-only adapters around Peterande's native solver and train loop."""
from __future__ import annotations

import json
import random
import time
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from .detection_data import letterbox_rgb, map_boxes, to_representation, detection_ground_truth, load_canonical_rgb
from .detection_metrics import coco_ap
from .experiment_tracking import record_validation, update_metadata
from .training_policy import ValidationStoppingState, policy_metadata


class ControlledLetterbox(torch.nn.Module):
    """Keep native loss box conventions, with invertible letterbox geometry."""
    def __init__(self, size=640, colour="rgb", flip_probability=0.):
        super().__init__()
        self.size, self.colour, self.flip_probability = size, colour, flip_probability

    def forward(self, sample):
        image, original, dataset = sample
        target = dict(original)
        rgb = load_canonical_rgb(target["image_path"]) if target.get("image_path") else np.array(image.convert("RGB"))
        h, w = rgb.shape[:2]
        boxes = np.asarray(target["boxes"].cpu(), dtype=np.float32).reshape(-1, 4).copy()
        if random.random() < self.flip_probability:
            rgb = np.ascontiguousarray(rgb[:, ::-1])
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        rgb, geometry = letterbox_rgb(rgb, self.size)
        target["boxes"] = tv_tensors.BoundingBoxes(torch.from_numpy(map_boxes(boxes, geometry)),
                                                   format="XYXY", canvas_size=(self.size, self.size))
        target["letterbox_geometry"] = torch.tensor(geometry, dtype=torch.float32)
        target["size"] = torch.tensor([self.size, self.size])
        if "area" in target:
            target["area"] = target["area"] * geometry[0] * geometry[1]
        return Image.fromarray(to_representation(rgb, self.colour)), target, dataset


def install_official_geometry():
    from src.core import register, GLOBAL_CONFIG
    if "ControlledLetterbox" not in GLOBAL_CONFIG:
        register()(ControlledLetterbox)


def predictions_from_output(outputs, image_ids, geometries):
    result = []
    for output, image_id, geometry in zip(outputs, image_ids, geometries):
        boxes = map_boxes(output["boxes"].detach().cpu().numpy(), geometry, inverse=True)
        for box, score, cls in zip(boxes, output["scores"].detach().cpu().tolist(), output["labels"].detach().cpu().tolist()):
            x1, y1, x2, y2 = map(float, box)
            result.append({"image_id": image_id, "category_id": int(cls), "score": score,
                           "bbox": [x1, y1, max(0., x2 - x1), max(0., y2 - y1)]})
    return result


@torch.no_grad()
def predict_official(model, postprocessor, ground_truth, colour, device, batch_size=8):
    predictions = []
    model.eval()
    for start in range(0, len(ground_truth["images"]), batch_size):
        images = ground_truth["images"][start:start + batch_size]
        tensors, geometries = [], []
        for image in images:
            rgb = load_canonical_rgb(image["file_name"])
            rgb, geometry = letterbox_rgb(rgb)
            tensor = torch.from_numpy(to_representation(rgb, colour).transpose(2, 0, 1).copy()).float() / 255
            tensors.append(tensor)
            geometries.append(geometry)
        batch = torch.stack(tensors).to(device)
        outputs = postprocessor(model(batch), torch.full((len(images), 2), 640, device=device))
        predictions.extend(predictions_from_output(outputs, [i["id"] for i in images], geometries))
    return predictions


class _ControlledStop(Exception):
    pass


def install_controlled_solver(protocol: dict, metadata_path: Path):
    """Retain native optimisation/EMA/stage behaviour; replace selection metric."""
    from src.solver._solver import BaseSolver
    from src.solver import det_solver
    from .runtime import library_versions
    from .train import _runtime

    active = {}
    state = ValidationStoppingState(300, 100, 40)
    ground_truth = detection_ground_truth(protocol["rgb_yaml"], "val")
    source_ids = {image["sample_identity"]: image["id"] for image in ground_truth["images"]}
    original_train = BaseSolver.train
    original_epoch = det_solver.train_one_epoch
    original_fit = det_solver.DetSolver.fit

    @wraps(original_train)
    def train(self):
        result = original_train(self)
        active["solver"] = self
        active["updates"] = 0
        def counted(*_):
            active["updates"] += 1
        self.optimizer.register_step_post_hook(counted)
        groups = [{key: group[key] for key in ("lr", "initial_lr", "weight_decay", "betas", "eps") if key in group}
                  for group in self.optimizer.param_groups]
        cfg = self.cfg.yaml_cfg
        update_metadata(metadata_path, runtime={**_runtime(), "libraries": library_versions()},
                        effective_training_config=cfg, augmentation=protocol["augmentation"],
                        optimisation={"optimizer": type(self.optimizer).__name__, "physical_batch_size": self.train_dataloader.batch_size,
                                      "gradient_accumulation": 1, "effective_batch_size": self.train_dataloader.batch_size,
                                      "accumulation_policy": "native official D-FINE; one update per batch",
                                      "workers": self.train_dataloader.num_workers, "drop_last": self.train_dataloader.drop_last,
                                      "scheduler": cfg["lr_scheduler"], "warmup": cfg.get("lr_warmup_scheduler"),
                                      "ema": cfg.get("ema"), "gradient_clipping": cfg.get("clip_max_norm"),
                                      "amp": bool(self.scaler is not None), "parameter_groups": groups})
        update_metadata(metadata_path, precision={"amp_enabled": bool(self.scaler is not None and self.scaler.is_enabled()),
                                                 "training_dtype": str(torch.get_autocast_dtype(torch.device(self.device).type)) if self.scaler is not None else "torch.float32",
                                                 "validation_dtype": "torch.float32"})
        dataset = getattr(self.train_dataloader, "dataset", None)
        if dataset is not None:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            update_metadata(metadata_path, optimisation={**payload["optimisation"], "effective_batch_is_nominal": True,
                            "last_physical_batch": len(dataset) % self.train_dataloader.batch_size or self.train_dataloader.batch_size})
        return result

    @wraps(original_epoch)
    def train_epoch(*args, **kwargs):
        if state.should_stop:
            raise _ControlledStop()
        return original_epoch(*args, **kwargs)

    @torch.no_grad()
    def evaluate(model, criterion, postprocessor, data_loader, evaluator, device, epoch, use_wandb, **kwargs):
        # This hook is validation-only. Final test inference is a separate phase.
        solver = active["solver"]
        if data_loader is not solver.val_dataloader:
            raise ValueError("Checkpoint selection attempted to use a non-validation loader")
        model.eval()
        criterion.eval()
        predictions = []
        seen = set()
        from .experiment_tracking import image_identity
        for images, targets in data_loader:
            images = images.to(device)
            outputs = postprocessor(model(images), torch.full((len(targets), 2), 640, device=device))
            ids, geometries = [], []
            for target in targets:
                relative = Path(target["image_path"]).relative_to(data_loader.dataset.img_folder)
                # COCO staging keeps the original split as the first directory.
                identity = image_identity(Path(*relative.parts[1:]))
                image_id = source_ids[identity]
                if image_id in seen:
                    raise ValueError("Duplicate validation image")
                seen.add(image_id)
                ids.append(image_id)
                geometries.append(target["letterbox_geometry"].tolist())
            predictions.extend(predictions_from_output(outputs, ids, geometries))
        if seen != set(source_ids.values()):
            raise ValueError("Official D-FINE did not evaluate the complete fixed validation split")
        metrics = coco_ap(ground_truth, predictions)
        value = metrics["mAP50-95"]
        previous = state.best_metric
        state.observe(epoch + 1, value)
        checkpoint = metadata_path.parent / "best.pth"
        if previous is None or value > previous:
            torch.save(solver.state_dict(), checkpoint)
        record_validation(metadata_path, epoch=epoch + 1, metric="box COCO mAP50-95", value=value,
                          checkpoint=checkpoint, metrics=metrics)
        update_metadata(metadata_path, early_stopping_state=vars(state), actual_optimizer_steps=active["updates"])
        update_metadata(metadata_path, final_learning_rates=solver.lr_scheduler.get_last_lr(),
                        observed_accumulation_steps=[1], observed_nominal_effective_batches=[solver.train_dataloader.batch_size],
                        actual_ema_decay=float(solver.ema.decay) if solver.ema else None)
        print(f"Controlled validation epoch {epoch + 1}: box COCO mAP50-95={value:.6f}")
        return {"coco_eval_bbox": metrics["coco_stats"]}, None

    @wraps(original_fit)
    def fit(self):
        started = time.monotonic()
        try:
            try:
                original_fit(self)
            except _ControlledStop:
                pass  # The preceding epoch already saved its last checkpoint.
            if not 100 <= state.current_epoch <= 300 or state.stop_reason not in {"early_stopping", "max_epochs"}:
                raise RuntimeError("Official D-FINE ended outside the controlled stopping policy")
            update_metadata(metadata_path, actual_stop_epoch=state.current_epoch, completed_epochs=state.current_epoch,
                            stop_reason=state.stop_reason, training_seconds=time.monotonic() - started,
                            actual_optimizer_steps=active.get("updates", 0))
        except BaseException as error:
            update_metadata(metadata_path, actual_stop_epoch=state.current_epoch, completed_epochs=state.current_epoch,
                            stop_reason="manual_interrupt" if isinstance(error, KeyboardInterrupt) else "error",
                            training_seconds=time.monotonic() - started, training_error=str(error))
            raise

    BaseSolver.train = train
    det_solver.train_one_epoch = train_epoch
    det_solver.evaluate = evaluate
    det_solver.DetSolver.fit = fit
