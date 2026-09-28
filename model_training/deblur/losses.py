from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _predictions(value: Tensor | list[Tensor]) -> list[Tensor]:
    return value if isinstance(value, list) else [value]


class RelativisticLeastSquaresLoss(nn.Module):
    def discriminator(self, real: Tensor | list[Tensor], fake: Tensor | list[Tensor]) -> Tensor:
        losses = []
        for real_scale, fake_scale in zip(_predictions(real), _predictions(fake), strict=True):
            losses.append(0.5 * ((real_scale - fake_scale.mean() - 1.0).square().mean() + (fake_scale - real_scale.mean() + 1.0).square().mean()))
        return torch.stack(losses).mean()

    def generator(self, real: Tensor | list[Tensor], fake: Tensor | list[Tensor]) -> Tensor:
        losses = []
        for real_scale, fake_scale in zip(_predictions(real), _predictions(fake), strict=True):
            losses.append(0.5 * ((real_scale - fake_scale.mean() + 1.0).square().mean() + (fake_scale - real_scale.mean() - 1.0).square().mean()))
        return torch.stack(losses).mean()


class OfflinePerceptualFallback(nn.Module):
    """Deterministic, download-free test fallback; not a substitute for trained VGG19 features."""

    def forward(self, restored: Tensor, target: Tensor) -> Tensor:
        losses = [F.l1_loss(restored, target)]
        for scale in (2, 4):
            losses.append(F.l1_loss(F.avg_pool2d(restored, scale), F.avg_pool2d(target, scale)))
        restored_grad = restored[..., 1:, :] - restored[..., :-1, :]
        target_grad = target[..., 1:, :] - target[..., :-1, :]
        losses.append(F.l1_loss(restored_grad, target_grad))
        return torch.stack(losses).mean()


class VGG19PerceptualLoss(nn.Module):
    """VGG19 feature loss with explicit local weights and no automatic downloads."""

    def __init__(self, weights_path: str | Path | None = None, allow_fallback: bool = True) -> None:
        super().__init__()
        self.using_fallback = False
        if weights_path is None:
            if not allow_fallback:
                raise RuntimeError("VGG19 perceptual loss needs a local torchvision VGG19 state_dict; downloads are intentionally disabled.")
            self.using_fallback = True
            self.features = nn.Identity()
            self.fallback: nn.Module | None = OfflinePerceptualFallback()
            return
        try:
            from torchvision.models import vgg19

            model = vgg19(weights=None)
            state = torch.load(Path(weights_path), map_location="cpu", weights_only=True)
            model.load_state_dict(state.get("state_dict", state), strict=True)
            self.features = nn.Sequential(*list(model.features.children())[:26]).eval()
            for parameter in self.features.parameters():
                parameter.requires_grad = False
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            self.fallback: nn.Module | None = None
        except Exception as exc:
            raise RuntimeError(f"Could not load the configured local VGG19 state_dict: {exc}") from exc

    def forward(self, restored: Tensor, target: Tensor) -> Tensor:
        if self.fallback is not None:
            return self.fallback(restored, target)
        restored_features = self.features((restored - self.mean) / self.std)
        with torch.no_grad():
            target_features = self.features((target - self.mean) / self.std)
        return F.mse_loss(restored_features, target_features)


class GeneratorLoss(nn.Module):
    def __init__(self, perceptual: nn.Module, perceptual_weight: float = 0.7, l1_weight: float = 0.3, adversarial_weight: float = 0.005) -> None:
        super().__init__()
        self.perceptual = perceptual
        self.perceptual_weight = perceptual_weight
        self.l1_weight = l1_weight
        self.adversarial_weight = adversarial_weight

    def content(self, restored: Tensor, target: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        perceptual = self.perceptual(restored, target)
        l1 = F.l1_loss(restored, target)
        return self.perceptual_weight * perceptual + self.l1_weight * l1, perceptual, l1

    def forward(self, restored: Tensor, target: Tensor, adversarial: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        content, perceptual, l1 = self.content(restored, target)
        total = content + self.adversarial_weight * adversarial
        return total, {"generator": total, "content": content, "perceptual": perceptual, "l1": l1, "adversarial": adversarial}
