from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from .models import DeblurModelConfig, create_generator


def load_generator_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(Path(path).expanduser().resolve(), map_location=device, weights_only=False)
    if checkpoint.get("format") not in {"deblur-generator-v1", "deblur-training-checkpoint-v1", "deblur-training-checkpoint-v2"}:
        raise ValueError("Unsupported checkpoint; expected a generator or full deblurring checkpoint.")
    config = DeblurModelConfig.from_dict(checkpoint.get("model_config"))
    generator = create_generator(config).to(device)
    generator.load_state_dict(checkpoint["generator"], strict=True)
    generator.eval()
    return generator, checkpoint


def _tensor(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        original_size = image.size
        image = ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0), original_size


def _save_tensor(tensor: torch.Tensor, path: Path) -> None:
    array = (tensor.detach().cpu().squeeze(0).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix or ".png", dir=path.parent)
    os.close(descriptor)
    try:
        Image.fromarray(array, "RGB").save(temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@torch.no_grad()
def infer_image(generator: torch.nn.Module, image_path: str | Path, output_path: str | Path, image_size: int = 256, device: str | torch.device = "cpu") -> str:
    inputs, _ = _tensor(Path(image_path), image_size)
    restored = generator(inputs.to(device))
    destination = Path(output_path).expanduser().resolve()
    _save_tensor(restored, destination)
    return str(destination)


@torch.no_grad()
def infer_folder(generator: torch.nn.Module, input_dir: str | Path, output_dir: str | Path, image_size: int = 256, device: str | torch.device = "cpu") -> dict[str, Any]:
    source, destination = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    outputs, failures = [], []
    for path in sorted(source.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            continue
        output = destination / path.relative_to(source).with_suffix(".png")
        try:
            outputs.append(infer_image(generator, path, output, image_size, device))
        except Exception as exc:
            failures.append({"path": str(path), "error": str(exc)})
    return {"count": len(outputs), "outputs": outputs, "failures": failures}


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(descriptor)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def export_generator(checkpoint_path: str | Path, output_dir: str | Path, image_size: int = 256, device: str = "cpu") -> dict[str, Any]:
    checkpoint_source = Path(checkpoint_path).expanduser().resolve()
    generator, checkpoint = load_generator_checkpoint(checkpoint_source, device)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pytorch_path = output_root / "generator_inference.pt"
    if checkpoint.get("format") == "deblur-generator-v1":
        _atomic_copy(checkpoint_source, pytorch_path)
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=".generator.", suffix=".tmp", dir=output_root)
        os.close(descriptor)
        try:
            torch.save({"format": "deblur-generator-v1", "architecture": checkpoint["model_config"]["architecture"], "model_config": checkpoint["model_config"], "generator": checkpoint["generator"], "input_range": [0.0, 1.0]}, temporary)
            os.replace(temporary, pytorch_path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    example = torch.rand(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        expected = generator(example)
    torchscript_path = output_root / "generator.torchscript.pt"
    traced = torch.jit.trace(generator, example, strict=True)
    traced.save(str(torchscript_path))
    with torch.no_grad():
        torchscript_error = float((traced(example) - expected).abs().max())
    result: dict[str, Any] = {
        "pytorch": str(pytorch_path), "torchscript": str(torchscript_path),
        "torchscript_max_abs_error": torchscript_error, "onnx": None, "onnx_verified": False,
    }
    onnx_path = output_root / "generator.onnx"
    try:
        import onnx  # noqa: F401

        torch.onnx.export(
            generator, example, onnx_path, input_names=["blurred"], output_names=["restored"],
            opset_version=17, do_constant_folding=True, dynamic_axes=None,
        )
        result["onnx"] = str(onnx_path)
        try:
            import onnxruntime as ort

            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            actual = session.run(["restored"], {"blurred": example.detach().cpu().numpy()})[0]
            result["onnx_max_abs_error"] = float(np.max(np.abs(actual - expected.detach().cpu().numpy())))
            result["onnx_verified"] = bool(result["onnx_max_abs_error"] < 1e-3)
            if not result["onnx_verified"]:
                raise RuntimeError(f"ONNX numerical verification failed: max abs error {result['onnx_max_abs_error']}")
        except ImportError:
            result["onnx_verification"] = "onnxruntime is not installed; ONNX was exported but not numerically verified."
    except ImportError:
        result["onnx_error"] = "Install optional 'onnx' and 'onnxruntime' packages to export and verify ONNX."
    return result
