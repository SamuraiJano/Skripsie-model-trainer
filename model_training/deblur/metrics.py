from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F


def psnr(restored: Tensor, target: Tensor, data_range: float = 1.0) -> Tensor:
    mse = F.mse_loss(restored, target)
    return 10.0 * torch.log10(torch.tensor(data_range * data_range, device=mse.device) / mse.clamp_min(1e-12))


def ssim(restored: Tensor, target: Tensor, data_range: float = 1.0, window_size: int = 7) -> Tensor:
    padding = window_size // 2
    mean_x = F.avg_pool2d(restored, window_size, 1, padding)
    mean_y = F.avg_pool2d(target, window_size, 1, padding)
    variance_x = F.avg_pool2d(restored * restored, window_size, 1, padding) - mean_x.square()
    variance_y = F.avg_pool2d(target * target, window_size, 1, padding) - mean_y.square()
    covariance = F.avg_pool2d(restored * target, window_size, 1, padding) - mean_x * mean_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / ((mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2))
    return value.mean()


def laplacian_variance(batch: Tensor) -> Tensor:
    gray = 0.299 * batch[:, 0:1] + 0.587 * batch[:, 1:2] + 0.114 * batch[:, 2:3]
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=batch.device, dtype=batch.dtype).view(1, 1, 3, 3)
    response = F.conv2d(gray, kernel, padding=1)
    return response.flatten(1).var(dim=1, correction=0).mean()


class ValidationMetrics:
    def __init__(self, perceptual: torch.nn.Module, lpips_enabled: bool = False) -> None:
        self.perceptual = perceptual
        self.lpips_model: Any | None = None
        if lpips_enabled:
            try:
                import lpips

                self.lpips_model = lpips.LPIPS(net="alex", pnet_rand=True)
            except ImportError:
                self.lpips_model = None

    @torch.no_grad()
    def __call__(self, blurred: Tensor, restored: Tensor, sharp: Tensor) -> dict[str, float | None]:
        clipped = restored.clamp(0, 1)
        result: dict[str, float | None] = {
            "psnr": float(psnr(clipped, sharp)), "ssim": float(ssim(clipped, sharp)),
            "l1": float(F.l1_loss(clipped, sharp)), "perceptual": float(self.perceptual(clipped, sharp)),
            "laplacian_before": float(laplacian_variance(blurred)), "laplacian_after": float(laplacian_variance(clipped)),
            "lpips": None,
        }
        if self.lpips_model is not None:
            self.lpips_model = self.lpips_model.to(restored.device)
            result["lpips"] = float(self.lpips_model(clipped * 2 - 1, sharp * 2 - 1).mean())
        return result


def average_metric_rows(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not rows:
        return {}
    result: dict[str, float | None] = {}
    for key in rows[0]:
        values = [float(row[key]) for row in rows if row.get(key) is not None and math.isfinite(float(row[key]))]
        result[key] = sum(values) / len(values) if values else None
    return result
