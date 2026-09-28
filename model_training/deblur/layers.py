from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GhostModule(nn.Module):
    """Ghost feature generation: primary convolution plus cheap depthwise features."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, ratio: int = 2, stride: int = 1, activation: bool = True) -> None:
        super().__init__()
        if ratio < 2:
            raise ValueError("Ghost ratio must be at least 2.")
        intrinsic = math.ceil(out_channels / ratio)
        cheap = intrinsic * (ratio - 1)
        self.out_channels = out_channels
        self.primary = nn.Conv2d(in_channels, intrinsic, kernel_size, stride, kernel_size // 2, bias=False)
        self.cheap = nn.Conv2d(intrinsic, cheap, 3, 1, 1, groups=intrinsic, bias=False)
        self.activation = nn.ReLU(inplace=True) if activation else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        primary = self.activation(self.primary(inputs))
        cheap = self.activation(self.cheap(primary))
        return torch.cat((primary, cheap), dim=1)[:, : self.out_channels]


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs * self.layers(inputs)


class AttentionGhostModule(nn.Module):
    """Attention-Ghost reproduction: Ghost features, SE attention, then projection."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1, stride: int = 1, ratio: int = 2) -> None:
        super().__init__()
        expanded = max(out_channels, math.ceil(out_channels / ratio) * ratio)
        self.ghost = GhostModule(in_channels, expanded, kernel_size, ratio, stride)
        self.attention = SqueezeExcitation(expanded)
        self.project = nn.Conv2d(expanded, out_channels, 1, bias=False) if expanded != out_channels else nn.Identity()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.project(self.attention(self.ghost(inputs)))


class HalfInstanceNorm(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("HIN needs at least two channels.")
        self.normalized_channels = (channels + 1) // 2
        self.norm = nn.InstanceNorm2d(self.normalized_channels, affine=True)

    def forward(self, inputs: Tensor) -> Tensor:
        normalized, preserved = torch.split(inputs, [self.normalized_channels, inputs.shape[1] - self.normalized_channels], dim=1)
        return torch.cat((self.norm(normalized), preserved), dim=1)


class GatedHalfInstanceNorm(nn.Module):
    """Paper-equation GHIN with safe odd-channel splitting."""

    def __init__(self, channels: int, eps: float = 1e-5) -> None:
        super().__init__()
        if channels < 2:
            raise ValueError("GHIN needs at least two channels.")
        self.normalized_channels = (channels + 1) // 2
        shape = (1, self.normalized_channels, 1, 1)
        self.alpha = nn.Parameter(torch.ones(shape))
        self.gamma = nn.Parameter(torch.zeros(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps

    def forward(self, inputs: Tensor) -> Tensor:
        normalized_part, preserved = torch.split(inputs, [self.normalized_channels, inputs.shape[1] - self.normalized_channels], dim=1)
        variance, mean = torch.var_mean(normalized_part, dim=(2, 3), correction=0, keepdim=True)
        standardized = (normalized_part - mean) * torch.rsqrt(variance + self.eps)
        modulation = self.gamma * self.alpha * standardized + self.beta
        gated = (1.0 + modulation) * normalized_part
        return torch.cat((gated, preserved), dim=1)


class GhostResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, attention: bool = False) -> None:
        super().__init__()
        layer = AttentionGhostModule if attention else GhostModule
        hidden = max(in_channels, out_channels)
        self.body = nn.Sequential(
            layer(in_channels, hidden, kernel_size=3, stride=stride),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(inplace=True),
            layer(hidden, out_channels, kernel_size=1, stride=1),
        )
        self.shortcut = nn.Identity() if stride == 1 and in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return F.relu(self.body(inputs) + self.shortcut(inputs), inplace=True)
