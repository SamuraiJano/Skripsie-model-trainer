"""Generator/discriminator implementations.

GhostDeblurGAN is an attributed, clean-room adaptation of the public
York-SDCNLab Ghost-DeblurGAN design. AGGDeblurGANReproduction follows the 2025
paper description; it is not an official implementation. Ambiguous details
(exact per-block expansion ratios and fusion ordering) are exposed in config.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import AttentionGhostModule, GatedHalfInstanceNorm, GhostModule, GhostResidualBlock, HalfInstanceNorm


@dataclass(frozen=True)
class DeblurModelConfig:
    architecture: Literal["GhostDeblurGAN", "AGGDeblurGANReproduction"] = "GhostDeblurGAN"
    encoder_channels: tuple[int, int, int, int, int] = (16, 24, 40, 80, 160)
    fpn_width: int = 64
    head_width: int = 32
    ghost_ratio: int = 2
    upsample_mode: Literal["nearest", "bilinear"] = "nearest"
    residual_scale: float = 1.0

    @classmethod
    def from_dict(cls, values: dict[str, object] | None) -> "DeblurModelConfig":
        values = dict(values or {})
        if "encoder_channels" in values:
            values["encoder_channels"] = tuple(int(item) for item in values["encoder_channels"])  # type: ignore[arg-type]
        return cls(**values)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class FusionBlock(nn.Module):
    def __init__(self, channels: int, attention: bool) -> None:
        super().__init__()
        feature = AttentionGhostModule(channels, channels, 3) if attention else GhostModule(channels, channels, 3)
        normalization = GatedHalfInstanceNorm(channels) if attention else HalfInstanceNorm(channels)
        self.layers = nn.Sequential(feature, normalization, nn.ReLU(inplace=True))

    def forward(self, inputs: Tensor) -> Tensor:
        return self.layers(inputs)


class FPNDeblurGenerator(nn.Module):
    def __init__(self, config: DeblurModelConfig, attention: bool) -> None:
        super().__init__()
        channels = config.encoder_channels
        if len(channels) != 5:
            raise ValueError("encoder_channels must contain five hierarchy widths.")
        self.config = config
        self.attention = attention
        self.stem = nn.Sequential(nn.Conv2d(3, channels[0], 3, 1, 1, bias=False), nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))
        self.encoder = nn.ModuleList([
            nn.Sequential(GhostResidualBlock(channels[index - 1], channels[index], stride=2, attention=attention), GhostResidualBlock(channels[index], channels[index], attention=attention))
            for index in range(1, 5)
        ])
        self.lateral = nn.ModuleList([nn.Conv2d(width, config.fpn_width, 1, bias=False) for width in channels])
        self.top_down = nn.ModuleList([FusionBlock(config.fpn_width, attention) for _ in range(4)])
        feature = AttentionGhostModule if attention else GhostModule
        self.heads = nn.ModuleList([
            nn.Sequential(feature(config.fpn_width, config.fpn_width, 3), nn.ReLU(inplace=True), feature(config.fpn_width, config.head_width, 3), nn.ReLU(inplace=True))
            for _ in range(4)
        ])
        concatenated = config.head_width * 4
        self.reconstruct = nn.Sequential(
            feature(concatenated + config.fpn_width, config.fpn_width, 3),
            GatedHalfInstanceNorm(config.fpn_width) if attention else HalfInstanceNorm(config.fpn_width), nn.ReLU(inplace=True),
            feature(config.fpn_width, config.fpn_width, 3),
            GatedHalfInstanceNorm(config.fpn_width) if attention else HalfInstanceNorm(config.fpn_width), nn.ReLU(inplace=True),
            feature(config.fpn_width, 3, 3),
        )

    def _resize(self, inputs: Tensor, size: tuple[int, int]) -> Tensor:
        kwargs = {"align_corners": False} if self.config.upsample_mode == "bilinear" else {}
        return F.interpolate(inputs, size=size, mode=self.config.upsample_mode, **kwargs)

    def forward(self, blurred: Tensor) -> Tensor:
        features = [self.stem(blurred)]
        for stage in self.encoder:
            features.append(stage(features[-1]))
        pyramid: list[Tensor] = [self.lateral[-1](features[-1])]
        for index in range(3, -1, -1):
            fused = self.lateral[index](features[index]) + self._resize(pyramid[-1], features[index].shape[-2:])
            pyramid.append(self.top_down[3 - index](fused))
        pyramid = list(reversed(pyramid))
        full_size = features[0].shape[-2:]
        heads = [self._resize(self.heads[index](pyramid[index + 1]), full_size) for index in range(4)]
        residual = self.reconstruct(torch.cat([*heads, pyramid[0]], dim=1))
        return blurred + torch.tanh(residual) * self.config.residual_scale


class GhostDeblurGAN(FPNDeblurGenerator):
    def __init__(self, config: DeblurModelConfig | None = None) -> None:
        config = config or DeblurModelConfig(architecture="GhostDeblurGAN")
        super().__init__(config, attention=False)


class AGGDeblurGANReproduction(FPNDeblurGenerator):
    """Paper-guided reproduction, not an official AGG-DeblurGAN implementation."""

    def __init__(self, config: DeblurModelConfig | None = None) -> None:
        config = config or DeblurModelConfig(architecture="AGGDeblurGANReproduction")
        super().__init__(config, attention=True)


def create_generator(config: DeblurModelConfig | dict[str, object] | None = None) -> FPNDeblurGenerator:
    resolved = config if isinstance(config, DeblurModelConfig) else DeblurModelConfig.from_dict(config)
    if resolved.architecture == "GhostDeblurGAN":
        return GhostDeblurGAN(resolved)
    if resolved.architecture == "AGGDeblurGANReproduction":
        return AGGDeblurGANReproduction(resolved)
    raise ValueError(f"Unknown deblurring architecture: {resolved.architecture}")


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32, layers: int = 3) -> None:
        super().__init__()
        modules: list[nn.Module] = [nn.Conv2d(3, base_channels, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True)]
        previous = base_channels
        for index in range(1, layers):
            width = min(base_channels * (2**index), base_channels * 8)
            modules.extend([nn.Conv2d(previous, width, 4, 2, 1, bias=False), nn.InstanceNorm2d(width, affine=True), nn.LeakyReLU(0.2, inplace=True)])
            previous = width
        modules.extend([nn.Conv2d(previous, min(previous * 2, base_channels * 8), 3, 1, 1), nn.LeakyReLU(0.2, inplace=True), nn.Conv2d(min(previous * 2, base_channels * 8), 1, 3, 1, 1)])
        self.model = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.model(inputs)


class DoubleScaleDiscriminator(nn.Module):
    """DeblurGAN-v2-style full and half-scale PatchGAN discriminator."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        self.full = PatchDiscriminator(base_channels, layers=3)
        self.half_scale = PatchDiscriminator(base_channels, layers=3)

    def forward(self, inputs: Tensor) -> list[Tensor]:
        half = F.interpolate(inputs, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return [self.full(inputs), self.half_scale(half)]
