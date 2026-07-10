from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class RuntimeReport:
    conda_env: str | None
    python: str
    platform: str
    torch_version: str
    torch_cuda: str | None
    cuda_available: bool
    device_count: int
    gpu_name: str | None
    ultralytics_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_runtime_report() -> RuntimeReport:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("PyTorch is not importable. Activate the conda environment first.") from exc

    try:
        import ultralytics
    except Exception as exc:  # pragma: no cover - depends on local env
        raise RuntimeError("Ultralytics is not importable. Run `conda env create -f environment.yml`.") from exc

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_available and device_count else None

    return RuntimeReport(
        conda_env=os.environ.get("CONDA_DEFAULT_ENV"),
        python=platform.python_version(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        device_count=device_count,
        gpu_name=gpu_name,
        ultralytics_version=getattr(ultralytics, "__version__", "unknown"),
    )


def assert_training_runtime(expected_env: str = "naartjie-yolo26") -> RuntimeReport:
    report = collect_runtime_report()
    if report.conda_env != expected_env:
        print(
            f"Warning: active conda env is {report.conda_env!r}, expected {expected_env!r}. "
            "Training can still run if dependencies are installed."
        )
    if not report.cuda_available:
        raise RuntimeError("CUDA is not available to PyTorch. Local GPU training requires CUDA.")
    return report


def print_runtime_report(report: RuntimeReport | None = None) -> None:
    report = report or collect_runtime_report()
    for key, value in report.to_dict().items():
        print(f"{key}: {value}")

