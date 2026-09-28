from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw, ImageFilter

from deblur.dataset import PairedDatasetConfig, PairedDeblurDataset, validate_paired_dataset
from deblur.evaluation import DisagreementConfig, SegmentationPrediction, evaluate_deblurring
from deblur.inference import export_generator, load_generator_checkpoint
from deblur.layers import AttentionGhostModule, GatedHalfInstanceNorm, GhostModule, HalfInstanceNorm
from deblur.losses import GeneratorLoss, OfflinePerceptualFallback, RelativisticLeastSquaresLoss
from deblur.metrics import ValidationMetrics
from deblur.models import DeblurModelConfig, DoubleScaleDiscriminator, create_generator
from deblur.routing import calibrate_blur_routing, laplacian_blur_score, route_blur
from deblur.trainer import DeblurTrainer, DeblurTrainingConfig


TINY_MODEL = DeblurModelConfig(encoder_channels=(4, 6, 8, 12, 16), fpn_width=8, head_width=4)


def _image(path: Path, offset: int = 0) -> None:
    image = Image.new("RGB", (40, 40), (30, 70, 110))
    draw = ImageDraw.Draw(image); draw.rectangle((6 + offset, 8, 27 + offset, 31), fill=(220, 35, 45)); draw.line((0, 3 + offset, 39, 35), fill=(30, 230, 80), width=2)
    image.save(path)


def _dataset(root: Path) -> Path:
    rows = []
    for split, count in (("train", 2), ("val", 1), ("test", 1)):
        (root / split / "blur").mkdir(parents=True); (root / split / "sharp").mkdir(parents=True)
        for index in range(count):
            name = f"{split}-{index}.png"
            _image(root / split / "blur" / name, index); _image(root / split / "sharp" / name, index)
            rows.append({"pair_id": f"{split}-{index}", "file_name": name, "split": split, "source_hash": f"hash-{split}-{index}"})
    (root / "manifests").mkdir(); (root / "manifests/pairs.json").write_text(json.dumps({"pairs": rows}), encoding="utf-8")
    return root


def test_ghost_attention_hin_and_ghin_modules() -> None:
    inputs = torch.rand(2, 7, 16, 16)
    assert GhostModule(7, 9)(inputs).shape == (2, 9, 16, 16)
    assert AttentionGhostModule(7, 9)(inputs).shape == (2, 9, 16, 16)
    for module in (HalfInstanceNorm(7), GatedHalfInstanceNorm(7)):
        output = module(inputs)
        assert output.shape == inputs.shape and torch.isfinite(output).all()
    ghin = GatedHalfInstanceNorm(7)
    assert torch.allclose(ghin(inputs), inputs)


@pytest.mark.parametrize("architecture", ["GhostDeblurGAN", "AGGDeblurGANReproduction"])
def test_generator_forward_shape_and_finite(architecture: str) -> None:
    config = replace(TINY_MODEL, architecture=architecture)
    model = create_generator(config).eval()
    inputs = torch.rand(1, 3, 64, 64)
    with torch.no_grad(): output = model(inputs)
    assert output.shape == inputs.shape and torch.isfinite(output).all()


def test_double_scale_discriminator_and_losses() -> None:
    discriminator = DoubleScaleDiscriminator(base_channels=8).eval()
    real, fake = torch.rand(2, 3, 64, 64), torch.rand(2, 3, 64, 64, requires_grad=True)
    real_prediction, fake_prediction = discriminator(real), discriminator(fake)
    assert len(real_prediction) == 2 and all(item.ndim == 4 for item in real_prediction)
    adversarial = RelativisticLeastSquaresLoss()
    discriminator_loss = adversarial.discriminator(real_prediction, fake_prediction)
    generator_adv = adversarial.generator(real_prediction, fake_prediction)
    total, parts = GeneratorLoss(OfflinePerceptualFallback())(fake, real, generator_adv)
    total.backward()
    assert all(torch.isfinite(value) for value in (discriminator_loss, generator_adv, total, *parts.values()))


def test_dataset_validation_and_paired_augmentation_alignment(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    report = validate_paired_dataset(root)
    assert report["valid"], report["errors"]
    dataset = PairedDeblurDataset(PairedDatasetConfig(str(root), image_size=32, augment=True, seed=42), "train")
    dataset.set_epoch(3)
    item = dataset[0]
    assert torch.equal(item["blurred"], item["sharp"])
    first = item["blurred"].clone(); dataset.set_epoch(3)
    assert torch.equal(first, dataset[0]["blurred"])


def test_validation_metrics_are_deterministic() -> None:
    perceptual = OfflinePerceptualFallback()
    metrics = ValidationMetrics(perceptual)
    blurred, restored, sharp = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    assert metrics(blurred, restored, sharp) == metrics(blurred, restored, sharp)


def test_one_cpu_training_batch_and_checkpoint_resume(tmp_path: Path) -> None:
    root = _dataset(tmp_path / "dataset")
    config = DeblurTrainingConfig(
        dataset_root=str(root), output_dir=str(tmp_path / "runs"), run_name="smoke", model=TINY_MODEL,
        image_size=32, epochs=2, batch_size=1, workers=0, device="cpu", amp=False,
        linear_decay_start_epoch=0, allow_perceptual_fallback=True,
    )
    trainer = DeblurTrainer(config)
    losses = trainer.train_batch(next(iter(trainer.train_loader)))
    assert trainer.global_step == 1 and all(np.isfinite(value) for value in losses.values())
    validation_first = trainer.validate(save_preview=False); validation_second = trainer.validate(save_preview=False)
    assert validation_first == validation_second
    checkpoint = trainer.save_checkpoint(best=True)["last_checkpoint"]
    resumed = DeblurTrainer(replace(config, resume_checkpoint=checkpoint))
    assert resumed.epoch == 1 and resumed.global_step == 1
    assert resumed.best_metrics == trainer.best_metrics
    trainer.save_checkpoint(
        resume_epoch=0,
        resume_batch_index=1,
        epoch_loader_generator_state=trainer.loader_generator.get_state().clone(),
    )
    auto_resumed = DeblurTrainer(config)
    assert auto_resumed.epoch == 0 and auto_resumed.next_batch_index == 1
    assert auto_resumed.global_step == 1


def test_blur_routing_score_and_three_way_routes(tmp_path: Path) -> None:
    sharp = tmp_path / "sharp.png"; _image(sharp)
    array = np.asarray(Image.open(sharp).convert("RGB"))
    blurred = Image.fromarray(np.asarray(Image.fromarray(array).filter(ImageFilter.GaussianBlur(3))))
    blurred_path = tmp_path / "blurred.png"; blurred.save(blurred_path)
    assert laplacian_blur_score(sharp, size=64) > laplacian_blur_score(blurred_path, size=64)
    assert route_blur(10, 20, 40) == "deblur"
    assert route_blur(30, 20, 40) == "uncertain_select_sharpest_from_burst"
    assert route_blur(50, 20, 40) == "bypass_deblurring"


def test_blur_routing_calibration_writes_local_distributions(tmp_path: Path) -> None:
    acceptable, unacceptable = tmp_path / "acceptable", tmp_path / "unacceptable"
    acceptable.mkdir(); unacceptable.mkdir()
    for index in range(3):
        sharp = acceptable / f"sharp-{index}.png"; _image(sharp, index)
        blurred = Image.open(sharp).filter(ImageFilter.GaussianBlur(2 + index * 0.2))
        blurred.save(unacceptable / f"blurred-{index}.png")
    result = calibrate_blur_routing(acceptable, unacceptable, tmp_path / "calibration", size=64)
    assert result["suggested_low_threshold"] < result["suggested_high_threshold"]
    assert (tmp_path / "calibration/blur_calibration.csv").is_file()
    assert (tmp_path / "calibration/blur_calibration.json").is_file()


class _ChangingSegmenter:
    def __init__(self) -> None: self.calls = 0
    def predict(self, image: np.ndarray) -> SegmentationPrediction:
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[5:25, 5:25] = True if self.calls % 2 else False
        if self.calls % 2: return SegmentationPrediction("Class 3", 0.60, mask)
        mask[10:14, 10:14] = True
        return SegmentationPrediction("Class 1", 0.95, mask)


def _generator_checkpoint(path: Path) -> Path:
    model = create_generator(TINY_MODEL)
    torch.save({"format": "deblur-generator-v1", "architecture": "GhostDeblurGAN", "model_config": TINY_MODEL.to_dict(), "generator": model.state_dict(), "input_range": [0.0, 1.0]}, path)
    return path


def test_disagreement_report_and_generator_only_checkpoint(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path / "dataset")
    checkpoint = _generator_checkpoint(tmp_path / "generator.pt")
    loaded, metadata = load_generator_checkpoint(checkpoint)
    assert metadata["format"] == "deblur-generator-v1" and loaded.training is False
    result = evaluate_deblurring(checkpoint, dataset, tmp_path / "reports", split="test", image_size=32, segmenter=_ChangingSegmenter(), config=DisagreementConfig())
    assert result["sample_count"] == 1 and result["flagged_count"] == 1
    report = json.loads((tmp_path / "reports/deblur_evaluation/reports/disagreements.json").read_text())
    assert "high_confidence_accept_after_original_defect" in report["rows"][0]["flag_reasons"]


def test_generator_export_and_optional_onnx(tmp_path: Path) -> None:
    checkpoint = _generator_checkpoint(tmp_path / "generator.pt")
    result = export_generator(checkpoint, tmp_path / "export", image_size=32)
    assert Path(result["pytorch"]).is_file() and Path(result["torchscript"]).is_file()
    assert result["torchscript_max_abs_error"] < 1e-5
    if result["onnx"] and "onnx_verification" not in result:
        assert result["onnx_verified"] and result["onnx_max_abs_error"] < 1e-3
