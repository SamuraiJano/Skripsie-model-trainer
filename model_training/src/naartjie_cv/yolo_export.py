"""Reusable Ultralytics checkpoint export helpers and notebook controls."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class YoloExportFormat:
    """One export target reported by the installed Ultralytics version."""

    name: str
    argument: str
    suffix: str
    cpu: bool
    gpu: bool
    arguments: tuple[str, ...]
    environment: str


_FORMAT_ALIASES = {
    "trt": "engine",
    "tensorrt": "engine",
    "tflite": "litert",
    "tfjs": "litert",
    "mlmodel": "coreml",
    "mlpackage": "coreml",
}
_COMMON_ARGUMENTS = {"device", "imgsz", "verbose"}
_EXTRA_ARGUMENTS = {
    "engine": {"workspace"},
    # These values are only used when an export target embeds NMS.
    "nms": {"conf", "iou"},
}


def _project_root(project_root: str | Path | None = None) -> Path:
    root = (
        Path(project_root).expanduser().resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[2]
    )
    if not (root / "src" / "naartjie_cv").is_dir():
        raise ValueError(f"Model Trainer project root is invalid: {root}")
    return root


def _normalize_format(export_format: str) -> str:
    value = export_format.strip().lower()
    return _FORMAT_ALIASES.get(value, value)


def available_yolo_export_formats() -> tuple[YoloExportFormat, ...]:
    """Return export targets from the active Ultralytics installation."""
    from ultralytics.engine.exporter import export_formats

    formats = export_formats()
    rows = zip(
        formats["Format"],
        formats["Argument"],
        formats["Suffix"],
        formats["CPU"],
        formats["GPU"],
        formats["Arguments"],
        formats["Env"],
    )
    return tuple(
        YoloExportFormat(
            name=str(name),
            argument=str(argument),
            suffix=str(suffix),
            cpu=bool(cpu),
            gpu=bool(gpu),
            arguments=tuple(str(item) for item in arguments),
            environment=str(environment),
        )
        for name, argument, suffix, cpu, gpu, arguments, environment in rows
        if argument != "-"  # The input checkpoint is already the PyTorch format.
    )


def discover_yolo_checkpoints(project_root: str | Path | None = None) -> tuple[Path, ...]:
    """Find trained and manually stored PyTorch checkpoints, newest first."""
    root = _project_root(project_root)
    candidates = [
        *(root / "runs" / "yolo26").glob("*/weights/*.pt"),
        *(root / "models").glob("*.pt"),
        *root.glob("*.pt"),
    ]
    unique = {path.resolve() for path in candidates if path.is_file()}
    return tuple(sorted(unique, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True))


def resolve_yolo_checkpoint(
    checkpoint: str | Path | None = None,
    project_root: str | Path | None = None,
) -> Path:
    """Resolve an explicit checkpoint or select the newest trained ``best.pt``."""
    root = _project_root(project_root)
    if checkpoint is None or not str(checkpoint).strip():
        discovered = discover_yolo_checkpoints(root)
        best = [path for path in discovered if path.name.lower() == "best.pt"]
        choices = best or list(discovered)
        if not choices:
            raise FileNotFoundError(
                f"No YOLO .pt checkpoints were found under {root / 'runs' / 'yolo26'} or {root / 'models'}"
            )
        return choices[0]

    requested = Path(checkpoint).expanduser()
    candidates = (
        (requested,)
        if requested.is_absolute()
        else (Path.cwd() / requested, root / requested, root.parent / requested)
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            if resolved.suffix.lower() != ".pt":
                raise ValueError(f"YOLO export requires a .pt checkpoint, received: {resolved}")
            return resolved
    attempted = ", ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"YOLO checkpoint was not found. Tried: {attempted}")


def _calibration_data_from_metadata(checkpoint: Path) -> Path | None:
    """Recover the training dataset config for INT8 calibration when available."""
    metadata_path = checkpoint.parent.parent / "run_metadata.json"
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    candidates = (
        metadata.get("artifacts", {}).get("training_data_yaml"),
        metadata.get("training_arguments", {}).get("data"),
        metadata.get("dataset", {}).get("data_yaml"),
    )
    for value in candidates:
        if value and Path(value).expanduser().is_file():
            return Path(value).expanduser().resolve()
    return None


def supported_export_arguments(export_format: str) -> frozenset[str]:
    """Return accepted keyword arguments for an export target."""
    normalized = _normalize_format(export_format)
    formats = {item.argument: item for item in available_yolo_export_formats()}
    if normalized not in formats:
        valid = ", ".join(formats)
        raise ValueError(f"Unknown YOLO export format {export_format!r}. Valid formats: {valid}")
    supported = set(formats[normalized].arguments) | _COMMON_ARGUMENTS
    supported |= _EXTRA_ARGUMENTS.get(normalized, set())
    if "nms" in supported:
        supported |= _EXTRA_ARGUMENTS["nms"]
    return frozenset(supported)


def export_yolo_model(
    checkpoint: str | Path | None = None,
    export_format: str = "engine",
    *,
    project_root: str | Path | None = None,
    **export_args: Any,
) -> Path:
    """Export a YOLO ``.pt`` checkpoint and return the verified output path.

    When ``checkpoint`` is omitted, the newest trained ``best.pt`` is selected.
    Format-specific arguments are validated before Ultralytics starts a build.
    For INT8 exports, the training dataset is recovered from ``run_metadata.json``
    when ``data`` is not supplied explicitly.
    """
    from ultralytics import YOLO

    root = _project_root(project_root)
    weights = resolve_yolo_checkpoint(checkpoint, root)
    normalized = _normalize_format(export_format)
    supported = supported_export_arguments(normalized)
    supplied = {name: value for name, value in export_args.items() if value is not None}
    unsupported = sorted(set(supplied) - supported)
    if unsupported:
        raise ValueError(
            f"Unsupported argument(s) for format={normalized!r}: {', '.join(unsupported)}. "
            f"Supported arguments: {', '.join(sorted(supported))}"
        )

    if supplied.get("quantize") in {8, "w8a16"} and not supplied.get("data"):
        calibration_data = _calibration_data_from_metadata(weights)
        if calibration_data is None:
            raise ValueError(
                "INT8 export needs representative calibration data. Pass data='path/to/data.yaml'."
            )
        supplied["data"] = str(calibration_data)

    if normalized == "engine":
        if importlib.util.find_spec("tensorrt") is None:
            raise RuntimeError(
                "TensorRT is not installed in this kernel. Install the CUDA-matched full TensorRT package, "
                "restart the kernel, and retry. This project uses: python -m pip install tensorrt-cu13"
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT export requires a CUDA-capable NVIDIA GPU, but CUDA is unavailable.")
        supplied.setdefault("device", 0)

    print(f"Exporting checkpoint: {weights}")
    print(f"Export format: {normalized}")
    output = Path(YOLO(str(weights)).export(format=normalized, **supplied)).expanduser().resolve()
    if not output.exists():
        raise RuntimeError(f"Ultralytics returned an export path that does not exist: {output}")
    print(f"Export complete: {output}")
    return output


def _quantization_options(export_format: str) -> tuple[tuple[str, Any], ...]:
    from ultralytics.engine.exporter import (
        FP16_FORMATS,
        FP32_UNSUPPORTED_FORMATS,
        INT8_FORMATS,
        W8A16_FORMATS,
        W8A32_FORMATS,
    )

    options: list[tuple[str, Any]] = []
    if export_format not in FP32_UNSUPPORTED_FORMATS:
        options.append(("FP32 / default", None))
    if export_format in FP16_FORMATS:
        options.append(("FP16", 16))
    if export_format in INT8_FORMATS:
        options.append(("INT8", 8))
    if export_format in W8A16_FORMATS:
        options.append(("W8A16", "w8a16"))
    if export_format in W8A32_FORMATS:
        options.append(("W8A32", "w8a32"))
    return tuple(options)


def create_yolo_export_dashboard(project_root: str | Path | None = None) -> Any:
    """Build notebook controls for selecting and exporting YOLO checkpoints."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError("Install ipywidgets in the active notebook kernel to use the export dashboard.") from exc

    root = _project_root(project_root)
    formats = available_yolo_export_formats()
    format_by_argument = {item.argument: item for item in formats}
    checkpoints = discover_yolo_checkpoints(root)
    checkpoint = widgets.Combobox(
        value=str(resolve_yolo_checkpoint(project_root=root)) if checkpoints else "",
        options=tuple(str(path) for path in checkpoints),
        description="Checkpoint",
        ensure_option=False,
        layout=widgets.Layout(width="85%"),
        style={"description_width": "100px"},
    )
    refresh = widgets.Button(description="Refresh", icon="refresh", layout=widgets.Layout(width="110px"))
    export_format = widgets.Dropdown(
        options=tuple((f"{item.name} ({item.argument})", item.argument) for item in formats),
        value="engine" if "engine" in format_by_argument else formats[0].argument,
        description="Format",
        style={"description_width": "100px"},
    )
    precision = widgets.Dropdown(description="Precision", style={"description_width": "100px"})
    imgsz = widgets.IntText(value=640, description="Image size", style={"description_width": "100px"})
    batch = widgets.IntText(value=1, description="Batch", style={"description_width": "100px"})
    device = widgets.Text(value="0", description="Device", style={"description_width": "100px"})
    dynamic = widgets.Checkbox(value=False, description="Dynamic shapes", indent=False)
    nms = widgets.Checkbox(value=False, description="Embed NMS", indent=False)
    simplify = widgets.Checkbox(value=True, description="Simplify graph", indent=False)
    workspace = widgets.FloatText(value=0.0, description="Workspace GiB", style={"description_width": "100px"})
    data = widgets.Text(
        value="",
        description="INT8 data",
        placeholder="Optional; inferred from run metadata",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "100px"},
    )
    status = widgets.HTML()
    output = widgets.Output(layout=widgets.Layout(border="1px solid #444", padding="8px"))
    run_export = widgets.Button(description="Export model", icon="download", button_style="success")

    def update_controls(*_: Any) -> None:
        item = format_by_argument[export_format.value]
        supported = supported_export_arguments(item.argument)
        previous_precision = precision.value
        precision.options = _quantization_options(item.argument)
        values = [value for _, value in precision.options]
        if previous_precision in values:
            precision.value = previous_precision
        elif item.argument == "engine" and 16 in values:
            precision.value = 16
        else:
            precision.value = values[0]
        batch.disabled = "batch" not in supported
        dynamic.disabled = "dynamic" not in supported
        nms.disabled = "nms" not in supported
        simplify.disabled = "simplify" not in supported
        workspace.disabled = item.argument != "engine"
        data.disabled = "data" not in supported
        if item.gpu and not item.cpu:
            device.value = "0"
            platform_label = "GPU required"
        elif item.cpu and item.gpu:
            platform_label = "CPU or GPU"
        elif item.cpu:
            device.value = "cpu"
            platform_label = "CPU"
        else:
            platform_label = f"special target ({item.environment})"
        status.value = (
            f"<b>{item.name}</b> &rarr; <code>{item.suffix}</code> &middot; {platform_label}. "
            "Some targets require extra vendor tools or a specific operating system."
        )

    def refresh_checkpoints(_: Any) -> None:
        found = discover_yolo_checkpoints(root)
        checkpoint.options = tuple(str(path) for path in found)
        if found and (not checkpoint.value or not Path(checkpoint.value).is_file()):
            checkpoint.value = str(resolve_yolo_checkpoint(project_root=root))

    def export_selected(_: Any) -> None:
        from IPython.display import clear_output

        item = format_by_argument[export_format.value]
        supported = supported_export_arguments(item.argument)
        arguments: dict[str, Any] = {
            "imgsz": imgsz.value,
            "device": device.value.strip() or None,
        }
        optional = {
            "batch": batch.value,
            "dynamic": dynamic.value,
            "nms": nms.value,
            "simplify": simplify.value,
            "quantize": precision.value,
            "data": data.value.strip() or None,
            "workspace": workspace.value or None,
        }
        arguments.update({name: value for name, value in optional.items() if name in supported})
        run_export.disabled = True
        with output:
            clear_output(wait=True)
            try:
                exported = export_yolo_model(
                    checkpoint.value,
                    item.argument,
                    project_root=root,
                    **arguments,
                )
                print(f"Ready to use: {exported}")
            except Exception as exc:
                print(f"Export failed: {type(exc).__name__}: {exc}")
                raise
            finally:
                run_export.disabled = False

    refresh.on_click(refresh_checkpoints)
    export_format.observe(update_controls, names="value")
    run_export.on_click(export_selected)
    update_controls()

    return widgets.VBox(
        [
            widgets.HTML("<h3>YOLO model export</h3>"),
            widgets.HBox([checkpoint, refresh]),
            widgets.HBox([export_format, precision, imgsz, batch]),
            widgets.HBox([device, workspace, dynamic, nms, simplify]),
            data,
            status,
            run_export,
            output,
        ],
        layout=widgets.Layout(width="100%"),
    )
