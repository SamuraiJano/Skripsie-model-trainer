from __future__ import annotations

import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from .dataset import PairedDatasetConfig, PairedDeblurDataset, require_valid_paired_dataset
from .losses import GeneratorLoss, RelativisticLeastSquaresLoss, VGG19PerceptualLoss
from .metrics import ValidationMetrics, average_metric_rows
from .models import DeblurModelConfig, DoubleScaleDiscriminator, create_generator

LogCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


@dataclass(frozen=True)
class DeblurTrainingConfig:
    dataset_root: str
    output_dir: str = "runs/deblur"
    run_name: str = "ghost_deblurgan"
    model: DeblurModelConfig = field(default_factory=DeblurModelConfig)
    image_size: int = 256
    epochs: int = 200
    batch_size: int = 4
    learning_rate: float = 1e-4
    final_learning_rate: float = 1e-6
    weight_decay: float = 0.0
    workers: int = 0
    seed: int = 42
    device: str = "cuda"
    amp: bool = True
    generator_updates: int = 1
    discriminator_updates: int = 1
    linear_decay_start_epoch: int = 10
    perceptual_weight: float = 0.7
    l1_weight: float = 0.3
    adversarial_weight: float = 0.005
    vgg19_weights_path: str | None = None
    allow_perceptual_fallback: bool = True
    lpips: bool = False
    resume_checkpoint: str | None = None
    auto_resume: bool = True
    checkpoint_interval_steps: int = 100

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DeblurTrainingConfig":
        values = dict(values)
        values["model"] = DeblurModelConfig.from_dict(values.get("model"))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model"] = self.model.to_dict()
        return payload

    def validate(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.image_size < 32:
            raise ValueError("Epochs/batch size must be positive and image size must be at least 32.")
        if self.learning_rate <= 0 or self.final_learning_rate < 0 or self.final_learning_rate > self.learning_rate:
            raise ValueError("Learning rates must satisfy 0 <= final <= initial and initial > 0.")
        if not 0 <= self.linear_decay_start_epoch < self.epochs:
            raise ValueError("Linear decay start must be between zero and epochs - 1.")
        if self.generator_updates < 1 or self.discriminator_updates < 1:
            raise ValueError("Generator and discriminator update counts must be positive.")
        if self.checkpoint_interval_steps < 1:
            raise ValueError("Checkpoint interval must be at least one training step.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _scheduler(optimizer: torch.optim.Optimizer, config: DeblurTrainingConfig) -> LambdaLR:
    minimum_factor = config.final_learning_rate / config.learning_rate
    denominator = max(1, config.epochs - 1 - config.linear_decay_start_epoch)

    def factor(epoch: int) -> float:
        if epoch <= config.linear_decay_start_epoch:
            return 1.0
        progress = min(1.0, (epoch - config.linear_decay_start_epoch) / denominator)
        return 1.0 + progress * (minimum_factor - 1.0)

    return LambdaLR(optimizer, factor)


def _scaler(enabled: bool) -> Any:
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _random_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_random_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


class DeblurTrainer:
    def __init__(self, config: DeblurTrainingConfig, log_callback: LogCallback | None = None, cancel_check: CancelCheck | None = None) -> None:
        config.validate()
        self.config = config
        self.log_callback = log_callback
        self.cancel_check = cancel_check or (lambda: False)
        seed_everything(config.seed)
        require_valid_paired_dataset(config.dataset_root)
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but is not available. Select CPU or install a compatible CUDA PyTorch build.")
        self.device = torch.device(config.device)
        self.run_dir = Path(config.output_dir).expanduser().resolve() / config.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        dataset_config = PairedDatasetConfig(config.dataset_root, config.image_size, augment=True, seed=config.seed)
        self.dataset_config = dataset_config
        self.train_dataset = PairedDeblurDataset(dataset_config, "train")
        self.val_dataset = PairedDeblurDataset(PairedDatasetConfig(config.dataset_root, config.image_size, augment=False, seed=config.seed), "val")
        self.loader_generator = torch.Generator().manual_seed(config.seed)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
            drop_last=False, generator=self.loader_generator, pin_memory=self.device.type == "cuda",
        )
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)
        self.generator = create_generator(config.model).to(self.device)
        self.discriminator = DoubleScaleDiscriminator().to(self.device)
        self.optimizer_g = Adam(self.generator.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.optimizer_d = Adam(self.discriminator.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler_g = _scheduler(self.optimizer_g, config)
        self.scheduler_d = _scheduler(self.optimizer_d, config)
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        self.scaler = _scaler(self.amp_enabled)
        self.perceptual = VGG19PerceptualLoss(config.vgg19_weights_path, config.allow_perceptual_fallback).to(self.device)
        self.generator_loss = GeneratorLoss(self.perceptual, config.perceptual_weight, config.l1_weight, config.adversarial_weight)
        self.adversarial_loss = RelativisticLeastSquaresLoss()
        self.metrics = ValidationMetrics(self.perceptual, config.lpips)
        self.epoch = 0
        self.global_step = 0
        self.next_batch_index = 0
        self._epoch_loader_generator_state: Tensor | None = None
        self._epoch_loss_sums: dict[str, float] = {}
        self._epoch_loss_counts: dict[str, int] = {}
        self.best_metrics: dict[str, Any] = {"selection_score": float("inf")}
        self.history: list[dict[str, Any]] = []
        automatic_checkpoint = self.run_dir / "last_checkpoint.pt"
        resume_checkpoint = config.resume_checkpoint
        resumed_automatically = False
        if resume_checkpoint is None and config.auto_resume and automatic_checkpoint.is_file():
            resume_checkpoint = str(automatic_checkpoint)
            resumed_automatically = True
        if resume_checkpoint:
            self.load_checkpoint(resume_checkpoint)
            if resumed_automatically:
                self._log("info", f"Automatically selected recovery checkpoint: {automatic_checkpoint}")
        _atomic_json({"training": config.to_dict(), "dataset": dataset_config.to_dict()}, self.run_dir / "resolved_config.json")
        if self.perceptual.using_fallback:
            self._log("warning", "Using the deterministic offline perceptual fallback. Configure a local torchvision VGG19 state_dict for research training.")

    def _log(self, level: str, message: str, **values: Any) -> None:
        payload = {"level": level, "message": message, **values}
        if self.log_callback:
            self.log_callback(payload)

    def _move(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        return batch["blurred"].to(self.device, non_blocking=True), batch["sharp"].to(self.device, non_blocking=True)

    def train_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        self.generator.train()
        self.discriminator.train()
        blurred, sharp = self._move(batch)
        losses: dict[str, float] = {}
        for _ in range(self.config.discriminator_updates):
            self.optimizer_d.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                with torch.no_grad():
                    restored_for_d = self.generator(blurred)
                real_prediction = self.discriminator(sharp)
                fake_prediction = self.discriminator(restored_for_d.detach())
                discriminator_loss = self.adversarial_loss.discriminator(real_prediction, fake_prediction)
            self.scaler.scale(discriminator_loss).backward()
            self.scaler.step(self.optimizer_d)
            self.scaler.update()
            losses["discriminator"] = float(discriminator_loss.detach())
        for _ in range(self.config.generator_updates):
            self.optimizer_g.zero_grad(set_to_none=True)
            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(False)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                restored = self.generator(blurred)
                fake_prediction = self.discriminator(restored)
                with torch.no_grad():
                    real_prediction = self.discriminator(sharp)
                adversarial = self.adversarial_loss.generator(real_prediction, fake_prediction)
                total, components = self.generator_loss(restored, sharp, adversarial)
            self.scaler.scale(total).backward()
            self.scaler.step(self.optimizer_g)
            self.scaler.update()
            for parameter in self.discriminator.parameters():
                parameter.requires_grad_(True)
            losses.update({name: float(value.detach()) for name, value in components.items()})
        self.global_step += 1
        return losses

    @torch.no_grad()
    def validate(self, save_preview: bool = True) -> dict[str, float | None]:
        self.generator.eval()
        rows: list[dict[str, float | None]] = []
        preview: tuple[Tensor, Tensor, Tensor] | None = None
        for batch in self.val_loader:
            blurred, sharp = self._move(batch)
            restored = self.generator(blurred)
            rows.append(self.metrics(blurred, restored, sharp))
            if preview is None:
                preview = (blurred[0], restored[0], sharp[0])
        result = average_metric_rows(rows)
        if result:
            laplacian_ratio = abs(float(result["laplacian_after"] or 0) - float(result["laplacian_before"] or 0)) / max(float(result["laplacian_before"] or 0), 1e-6)
            result["selection_score"] = 0.7 * float(result["perceptual"] or 0) + 0.3 * float(result["l1"] or 0) + 0.001 * laplacian_ratio
        if preview and save_preview:
            self._save_preview(*preview)
        return result

    def _save_preview(self, blurred: Tensor, restored: Tensor, sharp: Tensor) -> None:
        arrays = []
        for tensor in (blurred, restored.clamp(0, 1), sharp):
            arrays.append((tensor.detach().cpu().permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))
        destination = self.run_dir / "previews" / f"epoch-{self.epoch:04d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(np.concatenate(arrays, axis=1), "RGB")
        descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".png", dir=destination.parent)
        os.close(descriptor)
        try:
            image.save(temporary, "PNG")
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    def checkpoint_payload(
        self,
        *,
        resume_epoch: int | None = None,
        resume_batch_index: int = 0,
        epoch_loader_generator_state: Tensor | None = None,
    ) -> dict[str, Any]:
        return {
            "format": "deblur-training-checkpoint-v2", "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(), "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(), "scheduler_g": self.scheduler_g.state_dict(),
            "scheduler_d": self.scheduler_d.state_dict(), "amp_scaler": self.scaler.state_dict(),
            "epoch": self.epoch, "global_step": self.global_step, "best_metrics": self.best_metrics,
            "resume_epoch": self.epoch + 1 if resume_epoch is None else int(resume_epoch),
            "resume_batch_index": int(resume_batch_index),
            "epoch_loader_generator_state": epoch_loader_generator_state,
            "epoch_loss_sums": self._epoch_loss_sums,
            "epoch_loss_counts": self._epoch_loss_counts,
            "model_config": self.config.model.to_dict(), "training_config": self.config.to_dict(),
            "dataset_config": self.dataset_config.to_dict(), "seed": self.config.seed, "random_states": _random_state(),
            "data_loader_generator_state": self.loader_generator.get_state(),
            "history": self.history,
        }

    def save_checkpoint(
        self,
        best: bool = False,
        *,
        resume_epoch: int | None = None,
        resume_batch_index: int = 0,
        epoch_loader_generator_state: Tensor | None = None,
    ) -> dict[str, str]:
        last = self.run_dir / "last_checkpoint.pt"
        _atomic_torch_save(
            self.checkpoint_payload(
                resume_epoch=resume_epoch,
                resume_batch_index=resume_batch_index,
                epoch_loader_generator_state=epoch_loader_generator_state,
            ),
            last,
        )
        result = {"last_checkpoint": str(last)}
        if best:
            best_path = self.run_dir / "best_generator.pt"
            _atomic_torch_save({
                "format": "deblur-generator-v1", "architecture": self.config.model.architecture,
                "model_config": self.config.model.to_dict(), "generator": self.generator.state_dict(),
                "best_metrics": self.best_metrics, "input_range": [0.0, 1.0], "output_clamp": "preview/export only",
            }, best_path)
            result["best_generator"] = str(best_path)
        return result

    def load_checkpoint(self, path: str | Path) -> None:
        checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=self.device, weights_only=False)
        checkpoint_format = checkpoint.get("format")
        if checkpoint_format not in {"deblur-training-checkpoint-v1", "deblur-training-checkpoint-v2"}:
            raise ValueError("Resume requires a full deblurring training checkpoint.")
        if checkpoint.get("model_config") != self.config.model.to_dict():
            raise ValueError("Resume checkpoint model configuration does not match the selected architecture/config.")
        if checkpoint.get("dataset_config") != self.dataset_config.to_dict():
            raise ValueError("Resume checkpoint dataset or image configuration does not match this run.")
        self.generator.load_state_dict(checkpoint["generator"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
        self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])
        self.scheduler_g.load_state_dict(checkpoint["scheduler_g"])
        self.scheduler_d.load_state_dict(checkpoint["scheduler_d"])
        self.scaler.load_state_dict(checkpoint["amp_scaler"])
        if checkpoint_format == "deblur-training-checkpoint-v2":
            self.epoch = int(checkpoint["resume_epoch"])
            self.next_batch_index = int(checkpoint.get("resume_batch_index", 0))
            saved_training = checkpoint.get("training_config", {})
            if self.next_batch_index and saved_training.get("batch_size") != self.config.batch_size:
                raise ValueError("Batch size cannot change while resuming part-way through an epoch.")
            self._epoch_loss_sums = {
                str(name): float(value) for name, value in checkpoint.get("epoch_loss_sums", {}).items()
            }
            self._epoch_loss_counts = {
                str(name): int(value) for name, value in checkpoint.get("epoch_loss_counts", {}).items()
            }
        else:
            self.epoch = int(checkpoint["epoch"]) + 1
            self.next_batch_index = 0
        self.global_step = int(checkpoint["global_step"])
        self.best_metrics = dict(checkpoint.get("best_metrics", self.best_metrics))
        self.history = list(checkpoint.get("history", []))
        if checkpoint.get("random_states"):
            _restore_random_state(checkpoint["random_states"])
        loader_state = (
            checkpoint.get("epoch_loader_generator_state")
            if self.next_batch_index
            else checkpoint.get("data_loader_generator_state")
        )
        if loader_state is not None:
            self.loader_generator.set_state(loader_state)
        location = f"epoch {self.epoch + 1}, batch {self.next_batch_index + 1}" if self.next_batch_index else f"epoch {self.epoch + 1}"
        self._log("info", f"Resumed from {location} and global step {self.global_step}.")

    def fit(self, max_batches_per_epoch: int | None = None) -> dict[str, Any]:
        paths: dict[str, str] = {}
        start_epoch = self.epoch
        cancelled = False
        active_epoch: int | None = None
        try:
            for epoch in range(start_epoch, self.config.epochs):
                if self.cancel_check():
                    cancelled = True
                    break
                active_epoch = epoch
                self.epoch = epoch
                self.train_dataset.set_epoch(epoch)
                start_batch = self.next_batch_index if epoch == start_epoch else 0
                if start_batch > len(self.train_loader):
                    raise ValueError("Resume checkpoint batch position is beyond the end of this dataset.")
                if start_batch == 0:
                    self._epoch_loss_sums = {}
                    self._epoch_loss_counts = {}
                self.next_batch_index = start_batch
                self._epoch_loader_generator_state = self.loader_generator.get_state().clone()
                for batch_index, batch in enumerate(self.train_loader):
                    if batch_index < start_batch:
                        continue
                    if self.cancel_check():
                        cancelled = True
                        break
                    row = self.train_batch(batch)
                    for name, value in row.items():
                        self._epoch_loss_sums[name] = self._epoch_loss_sums.get(name, 0.0) + value
                        self._epoch_loss_counts[name] = self._epoch_loss_counts.get(name, 0) + 1
                    self.next_batch_index = batch_index + 1
                    self._log("progress", f"Epoch {epoch + 1}/{self.config.epochs}, batch {batch_index + 1}/{len(self.train_loader)}", epoch=epoch + 1, batch=batch_index + 1, batches=len(self.train_loader), losses=row)
                    if self.global_step % self.config.checkpoint_interval_steps == 0:
                        paths = self.save_checkpoint(
                            resume_epoch=epoch,
                            resume_batch_index=self.next_batch_index,
                            epoch_loader_generator_state=self._epoch_loader_generator_state,
                        )
                        self._log("checkpoint", f"Saved recovery checkpoint at global step {self.global_step}.", checkpoints=paths)
                    if max_batches_per_epoch is not None and self.next_batch_index >= max_batches_per_epoch:
                        break
                if cancelled:
                    paths = self.save_checkpoint(
                        resume_epoch=epoch,
                        resume_batch_index=self.next_batch_index,
                        epoch_loader_generator_state=self._epoch_loader_generator_state,
                    )
                    self._log("checkpoint", "Training was interrupted; saved a recovery checkpoint.", checkpoints=paths)
                    break
                validation = self.validate(save_preview=True)
                means = {
                    name: total / max(1, self._epoch_loss_counts.get(name, 0))
                    for name, total in self._epoch_loss_sums.items()
                }
                history_row = {"epoch": epoch, "global_step": self.global_step, "losses": means, "validation": validation, "lr": self.optimizer_g.param_groups[0]["lr"]}
                self.history.append(history_row)
                score = float(validation.get("selection_score") or float("inf"))
                best = score < float(self.best_metrics.get("selection_score", float("inf")))
                if best:
                    self.best_metrics = dict(validation)
                self.scheduler_g.step()
                self.scheduler_d.step()
                self.next_batch_index = 0
                self._epoch_loss_sums = {}
                self._epoch_loss_counts = {}
                paths = self.save_checkpoint(best=best, resume_epoch=epoch + 1)
                _atomic_json({"history": self.history, "best_metrics": self.best_metrics}, self.run_dir / "metrics.json")
                self._log("epoch", f"Completed epoch {epoch + 1}.", epoch=epoch + 1, losses=means, validation=validation, checkpoints=paths)
        except BaseException:
            if active_epoch is not None:
                try:
                    paths = self.save_checkpoint(
                        resume_epoch=active_epoch,
                        resume_batch_index=self.next_batch_index,
                        epoch_loader_generator_state=self._epoch_loader_generator_state,
                    )
                    self._log("checkpoint", "Training failed; saved an emergency recovery checkpoint.", checkpoints=paths)
                except Exception as checkpoint_error:
                    self._log("error", f"Could not save the emergency checkpoint: {checkpoint_error}")
            raise
        return {
            "state": "cancelled" if cancelled else "complete", "run_dir": str(self.run_dir),
            "epoch": self.epoch, "global_step": self.global_step, "best_metrics": self.best_metrics,
            "last_checkpoint": str(self.run_dir / "last_checkpoint.pt"),
            "best_generator": str(self.run_dir / "best_generator.pt") if (self.run_dir / "best_generator.pt").exists() else None,
        }
