from __future__ import annotations

import html
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .datasets import ultralytics_loader_smoke, validate_yolo_training_dataset
from .train import TrainingOptions, YoloTrainingOptions, final_yolo_options, train_yolo_export


TRAINING_TASKS = (
    ("Object detection", "detect", "yolo26n.pt"),
    ("Instance segmentation", "segment", "yolo26n-seg.pt"),
)

YOLO_PARAMETER_GROUPS = (
    (
        "Training",
        "Run length, batching, hardware, caching, and core training behaviour.",
        (
            "time",
            "patience",
            "min_epochs_before_early_stop",
            "cache",
            "optimizer",
            "pretrained",
            "deterministic",
            "fraction",
            "single_cls",
            "rect",
            "classes",
            "freeze",
            "multi_scale",
            "compile",
            "close_mosaic",
            "dropout",
            "cls_remap",
        ),
    ),
    (
        "Optimisation & loss",
        "Learning-rate schedule, warm-up, loss weights, distillation, and nominal batch size.",
        (
            "lr0",
            "lrf",
            "momentum",
            "weight_decay",
            "warmup_epochs",
            "warmup_momentum",
            "warmup_bias_lr",
            "cos_lr",
            "box",
            "cls",
            "cls_pw",
            "dfl",
            "rle",
            "nbs",
            "distill_model",
            "dis",
        ),
    ),
    (
        "Augmentation & masks",
        "Image augmentation probabilities and segmentation-specific mask settings.",
        (
            "hsv_h",
            "hsv_s",
            "hsv_v",
            "degrees",
            "translate",
            "scale",
            "shear",
            "perspective",
            "flipud",
            "fliplr",
            "bgr",
            "mosaic",
            "mixup",
            "cutmix",
            "copy_paste",
            "copy_paste_mode",
            "auto_augment",
            "erasing",
            "overlap_mask",
            "mask_ratio",
        ),
    ),
    (
        "Validation & output",
        "Validation thresholds, checkpoint/plot output, profiling, and display choices.",
        (
            "val",
            "split",
            "conf",
            "iou",
            "max_det",
            "agnostic_nms",
            "save_json",
            "save",
            "save_period",
            "plots",
            "show_labels",
            "line_width",
            "verbose",
            "profile",
        ),
    ),
)

YOLO_PARAMETER_LABELS = {
    "time": "Time limit (hours)",
    "patience": "Early-stop patience",
    "min_epochs_before_early_stop": "Minimum training epochs",
    "cache": "Dataset cache",
    "optimizer": "Optimizer",
    "pretrained": "Use pretrained weights",
    "cls_remap": "Remap class head",
    "verbose": "Verbose output",
    "deterministic": "Deterministic mode",
    "single_cls": "Single-class mode",
    "rect": "Rectangular batches",
    "cos_lr": "Cosine learning rate",
    "close_mosaic": "Close mosaic epochs",
    "fraction": "Dataset fraction",
    "profile": "Profile ONNX/TensorRT",
    "freeze": "Freeze layers",
    "multi_scale": "Multi-scale range",
    "compile": "Torch compile",
    "overlap_mask": "Overlap masks",
    "mask_ratio": "Mask downsample ratio",
    "dropout": "Dropout",
    "val": "Validate during training",
    "split": "Validation split",
    "save_json": "Save validation JSON",
    "conf": "Validation confidence",
    "iou": "Validation IoU",
    "max_det": "Maximum detections",
    "agnostic_nms": "Class-agnostic NMS",
    "plots": "Generate plots",
    "classes": "Class IDs",
    "show_labels": "Show plot labels",
    "line_width": "Plot line width",
    "lr0": "Initial learning rate",
    "lrf": "Final LR fraction",
    "momentum": "Momentum / beta1",
    "weight_decay": "Weight decay",
    "warmup_epochs": "Warm-up epochs",
    "warmup_momentum": "Warm-up momentum",
    "warmup_bias_lr": "Warm-up bias LR",
    "box": "Box loss weight",
    "cls": "Class loss weight",
    "cls_pw": "Class positive weight",
    "dfl": "DFL loss weight",
    "rle": "RLE loss weight",
    "nbs": "Nominal batch size",
    "distill_model": "Distillation model",
    "dis": "Distillation loss weight",
    "hsv_h": "HSV hue",
    "hsv_s": "HSV saturation",
    "hsv_v": "HSV value",
    "degrees": "Rotation degrees",
    "translate": "Translation",
    "scale": "Scale",
    "shear": "Shear degrees",
    "perspective": "Perspective",
    "flipud": "Vertical flip",
    "fliplr": "Horizontal flip",
    "bgr": "RGB-to-BGR",
    "mosaic": "Mosaic",
    "mixup": "MixUp",
    "cutmix": "CutMix",
    "copy_paste": "Copy-paste",
    "copy_paste_mode": "Copy-paste mode",
    "auto_augment": "Auto augment policy",
    "erasing": "Random erasing",
    "save": "Save checkpoints",
    "save_period": "Checkpoint period",
}

_OPTIONAL_FLOATS = {"time", "conf"}
_INTEGER_SELECTIONS = {"freeze", "classes"}
_PROBABILITIES = {
    "fraction",
    "dropout",
    "conf",
    "iou",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "cutmix",
    "copy_paste",
    "erasing",
}
_INTEGER_FIELDS = {"patience", "min_epochs_before_early_stop", "save_period", "close_mosaic", "mask_ratio", "max_det", "line_width", "nbs"}


def _project_root(project_root: str | Path | None) -> Path:
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    if not (root / "src" / "naartjie_cv").is_dir():
        raise ValueError(f"Model Trainer project root is invalid: {root}")
    return root


def _formatted_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def _parse_integer_selection(value: str) -> list[int] | None:
    text = value.strip()
    if not text:
        return None
    try:
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Freeze layers and class IDs must be comma-separated integers.") from exc


def _parse_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Additional YOLO arguments are not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Additional YOLO arguments must be a JSON object.")
    return payload


def _batch_value(value: float) -> int | float:
    return int(value) if float(value).is_integer() else float(value)


def _command_line(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _yolo_cli_arguments(options: YoloTrainingOptions) -> list[str]:
    defaults = asdict(final_yolo_options("detect"))
    values = asdict(options)
    extra_args = values.pop("extra_args")
    defaults.pop("extra_args")
    command: list[str] = []
    for name, value in values.items():
        if value == defaults[name] and name not in {
            "patience", "min_epochs_before_early_stop", "optimizer", "lr0", "weight_decay", "nbs", "mosaic",
            "mixup", "hsv_h", "hsv_s", "hsv_v",
        }:
            continue
        flag = f"--{name.replace('_', '-')}"
        if name == "cache":
            if value:
                command.extend([flag, str(value)])
        elif name == "compile":
            if value:
                command.extend([flag, str(value)])
        elif isinstance(value, bool):
            command.append(flag if value else f"--no-{name.replace('_', '-')}")
        elif isinstance(value, list):
            command.extend([flag, ",".join(str(item) for item in value)])
        elif value is not None:
            command.extend([flag, str(value)])
    command.extend(["--yolo-args-json", json.dumps(extra_args, separators=(",", ":"))])
    return command


def _dashboard_yolo_defaults(task: str = "detect") -> YoloTrainingOptions:
    return final_yolo_options(task)


def create_training_dashboard(project_root: str | Path | None = None) -> Any:
    """Build a responsive Jupyter dashboard for YOLO detection and segmentation training."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError(
            "The notebook UI requires ipywidgets in the active Jupyter kernel. Run "
            f"`{sys.executable} -m pip install \"ipywidgets>=8.1\"`, restart the kernel, and rerun the notebook."
        ) from exc

    root = _project_root(project_root)
    field_layout = widgets.Layout(width="100%", min_width="0")
    text_style = {"description_width": "150px"}

    def text_field(**kwargs: Any) -> Any:
        return widgets.Text(layout=field_layout, style=text_style, **kwargs)

    task = widgets.ToggleButtons(
        options=[(label, value) for label, value, _ in TRAINING_TASKS],
        value="detect",
        description="Task",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "90px", "button_width": "220px"},
    )
    dataset_root = text_field(
        value=str(root / "dataset_detect_rgb"),
        description="Dataset",
        placeholder="Standalone YOLO dataset or complete Data Labeller export",
    )
    colour_space = widgets.Dropdown(options=(("Auto (dataset)", "auto"), ("RGB", "rgb"), ("HSV", "hsv")),
                                   value="auto", description="Colour space", layout=field_layout, style=text_style)
    colour_space.tooltip = "HSV creates a lossless H,S,V view from the same RGB images and labels. Training settings stay fixed."
    output_dir = text_field(value=str(root / "runs"), description="Runs directory")
    run_name = text_field(
        value="",
        description="Run name",
        placeholder="Optional; a timestamped name is used when empty",
    )
    resume = text_field(value="", description="Resume checkpoint", placeholder="Optional path to last.pt")

    epochs = widgets.BoundedIntText(
        value=300, min=1, max=10000, description="Epochs", layout=field_layout, style=text_style
    )
    image_size = widgets.BoundedIntText(
        value=640,
        min=32,
        max=4096,
        step=32,
        description="Image size",
        layout=field_layout,
        style=text_style,
    )
    batch = widgets.FloatText(value=16, description="Batch size", layout=field_layout, style=text_style)
    batch.tooltip = "Physical GPU batch: 16 by default. If memory is tight, use 8 or 4 with nominal batch 16; accumulation ramps during warm-up."
    workers = widgets.BoundedIntText(
        value=4,
        min=0,
        max=128,
        description="Data workers",
        layout=field_layout,
        style=text_style,
    )
    seed = widgets.IntText(value=0, description="Random seed", layout=field_layout, style=text_style)
    device = text_field(value="0", description="Device", placeholder="0, 0,1, cpu, or mps")
    amp = widgets.Checkbox(
        value=True,
        description="Automatic mixed precision (AMP)",
        indent=False,
        layout=field_layout,
    )

    yolo_defaults = _dashboard_yolo_defaults()
    yolo_controls: dict[str, Any] = {}
    dropdown_options = {
        "cache": (("Off", False), ("RAM", "ram"), ("Disk", "disk")),
        "compile": (
            ("Off", False),
            ("Default", "default"),
            ("Reduce overhead", "reduce-overhead"),
            ("Max autotune", "max-autotune-no-cudagraphs"),
        ),
        "copy_paste_mode": (("Flip", "flip"), ("MixUp", "mixup")),
        "split": (("Validation (checkpoint selection)", "val"),),
    }

    for _, _, names in YOLO_PARAMETER_GROUPS:
        for name in names:
            if name in yolo_controls:
                continue
            default = getattr(yolo_defaults, name)
            description = YOLO_PARAMETER_LABELS[name]
            if name in dropdown_options:
                control = widgets.Dropdown(
                    options=dropdown_options[name],
                    value=default,
                    description=description,
                    layout=field_layout,
                    style=text_style,
                )
            elif name in {"optimizer", "auto_augment"}:
                options = (
                    ("auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp")
                    if name == "optimizer"
                    else ("none", "randaugment", "autoaugment", "augmix")
                )
                control = widgets.Combobox(
                    options=options,
                    value="none" if default is None else str(default),
                    ensure_option=False,
                    description=description,
                    layout=field_layout,
                    style=text_style,
                )
            elif isinstance(default, bool):
                control = widgets.Checkbox(
                    value=default,
                    description=description,
                    indent=False,
                    layout=field_layout,
                )
                if name in {"val", "save"}:
                    control.disabled = True
            elif name in _INTEGER_SELECTIONS:
                control = text_field(
                    value="",
                    description=description,
                    placeholder="Optional comma-separated IDs",
                )
            elif name == "distill_model":
                control = text_field(
                    value="",
                    description=description,
                    placeholder="Optional model/checkpoint",
                )
            elif name in _OPTIONAL_FLOATS:
                control = text_field(
                    value="",
                    description=description,
                    placeholder="Use Ultralytics default when empty",
                )
            elif name in _INTEGER_FIELDS:
                minimum = -1 if name == "save_period" else 0
                maximum = 100000 if name in {"max_det", "nbs"} else 10000
                control = widgets.BoundedIntText(
                    value=int(default),
                    min=minimum,
                    max=maximum,
                    description=description,
                    layout=field_layout,
                    style=text_style,
                )
            elif name in _PROBABILITIES:
                minimum = 0.000001 if name == "fraction" else 0.0
                control = widgets.BoundedFloatText(
                    value=float(default),
                    min=minimum,
                    max=1.0,
                    step=0.01,
                    description=description,
                    layout=field_layout,
                    style=text_style,
                )
            else:
                control = widgets.FloatText(
                    value=float(default),
                    description=description,
                    layout=field_layout,
                    style=text_style,
                )
            yolo_controls[name] = control
            if name == "split":
                control.tooltip = "Training uses val to select the best checkpoint; test is evaluated only after training."
            elif name == "val":
                control.tooltip = "Validation remains on so test images cannot influence checkpoint selection."
            elif name == "save":
                control.tooltip = "Save the best checkpoint so it can be evaluated on the held-out test split."
            elif name == "nbs":
                control.tooltip = "Target effective batch: 16. Reducing physical batch to 8 or 4 accumulates 2 or 4 batches after warm-up."
            elif name in {"hsv_h", "hsv_s", "hsv_v"}:
                control.tooltip = "Zero for both RGB and HSV to preserve the controlled colour representation."
            elif name == "optimizer":
                control.tooltip = "AdamW is selected explicitly so changing batch size cannot silently switch the optimizer."

    extra_args = widgets.Textarea(
        value=_formatted_json(yolo_defaults.extra_args),
        description="Additional YOLO args",
        placeholder='Optional JSON for version-specific settings, for example {"end2end": true}',
        layout=widgets.Layout(width="100%", min_width="0", height="82px"),
        style=text_style,
    )
    extra_args.tooltip = "The empty augmentations list disables additional default colour transforms. Remove that key to restore them."
    confirmation = widgets.Checkbox(
        value=False,
        description="I understand that training can run for a long time",
        indent=False,
        layout=widgets.Layout(width="100%"),
    )
    validate_button = widgets.Button(
        description="Validate dataset",
        icon="check",
        button_style="info",
        layout=widgets.Layout(width="170px", margin="0 8px 8px 0"),
    )
    smoke_button = widgets.Button(
        description="Smoke-test loader",
        icon="flask",
        layout=widgets.Layout(width="170px", margin="0 8px 8px 0"),
    )
    train_button = widgets.Button(
        description="Start training",
        icon="play",
        button_style="success",
        disabled=True,
        layout=widgets.Layout(width="170px"),
    )
    command_preview = widgets.HTML(layout=widgets.Layout(width="100%"))
    status = widgets.HTML(
        value="<b>Ready.</b> Validate the selected dataset before training.",
        layout=widgets.Layout(width="100%"),
    )
    output = widgets.Output(
        layout=widgets.Layout(
            border="1px solid #cbd5e1",
            max_height="520px",
            overflow="auto",
            padding="8px",
            width="100%",
        )
    )
    busy = False

    def model_name() -> str:
        return next(model for _, value, model in TRAINING_TASKS if value == task.value)

    def selected_yolo_options() -> YoloTrainingOptions:
        values: dict[str, Any] = {}
        for name, control in yolo_controls.items():
            if name in _OPTIONAL_FLOATS:
                values[name] = float(control.value.strip()) if control.value.strip() else None
            elif name in _INTEGER_SELECTIONS:
                values[name] = _parse_integer_selection(control.value)
            elif name == "distill_model":
                values[name] = control.value.strip() or None
            elif name == "auto_augment":
                values[name] = None if control.value.strip().lower() in {"", "none"} else control.value.strip()
            else:
                values[name] = control.value
        values["extra_args"] = _parse_json_object(extra_args.value)
        return YoloTrainingOptions(**values)

    def selected_training_options() -> TrainingOptions:
        return TrainingOptions(
            epochs=epochs.value,
            imgsz=image_size.value,
            batch=_batch_value(batch.value),
            workers=workers.value,
            seed=seed.value,
            colour_space=colour_space.value,
            device=device.value.strip() or "0",
            amp=amp.value,
            output_dir=output_dir.value.strip() or str(root / "runs"),
            run_name=run_name.value.strip() or None,
            resume=resume.value.strip() or None,
            yolo=selected_yolo_options(),
        )

    def update_preview(*_: Any) -> None:
        try:
            options = selected_training_options()
            command = [
                "python",
                "scripts/model_trainer.py",
                "yolo-detect" if task.value == "detect" else "yolo-segment",
                dataset_root.value.strip() or "<dataset>",
                "--epochs",
                str(options.epochs),
                "--imgsz",
                str(options.imgsz),
                "--batch",
                str(options.batch),
                "--workers",
                str(options.workers),
                "--device",
                options.device,
                "--seed",
                str(options.seed),
                "--colour-space",
                options.colour_space,
                "--output-dir",
                options.output_dir,
            ]
            if options.run_name:
                command.extend(["--run-name", options.run_name])
            if options.resume:
                command.extend(["--resume", options.resume])
            if not options.amp:
                command.append("--no-amp")
            command.extend(_yolo_cli_arguments(options.yolo))
            preview = html.escape(_command_line(command))
            nominal_batch = (
                options.batch * max(round(options.yolo.nbs / options.batch), 1)
                if options.batch >= 1
                else "automatic"
            )
            resume_note = (
                "<p>Resume restores the checkpoint's saved training arguments. "
                "Leave resume empty for the final experiment defaults.</p>"
                if options.resume else ""
            )
            command_preview.value = (
                "<div class='model-trainer-preview'>"
                f"<p><b>Base model:</b> <code>{html.escape(model_name())}</code></p>"
                f"<p><b>Nominal effective batch after warm-up:</b> {nominal_batch}</p>"
                f"{resume_note}"
                f"<p><b>Equivalent CLI:</b></p><pre><code>{preview}</code></pre>"
                "</div>"
            )
        except ValueError as exc:
            command_preview.value = (
                "<div class='model-trainer-preview model-trainer-error'>"
                f"<b>Configuration error:</b> {html.escape(str(exc))}</div>"
            )

    def set_busy(value: bool) -> None:
        nonlocal busy
        busy = value
        validate_button.disabled = value
        smoke_button.disabled = value
        train_button.disabled = value or not confirmation.value

    def validate_clicked(_: Any) -> None:
        output.clear_output(wait=True)
        with output:
            report = validate_yolo_training_dataset(dataset_root.value.strip(), task.value)
            print(_formatted_json(report))
        if report["ok"]:
            status.value = "<b style='color:#15803d'>Dataset is valid.</b>"
        else:
            status.value = "<b style='color:#b91c1c'>Dataset validation failed.</b> Review the errors below."

    def smoke_clicked(_: Any) -> None:
        set_busy(True)
        output.clear_output(wait=True)
        status.value = "<b>Loading one real training sample…</b>"
        try:
            with output:
                result = ultralytics_loader_smoke(dataset_root.value.strip(), task.value)
                print(_formatted_json(result))
            status.value = "<b style='color:#15803d'>Loader smoke test passed.</b>"
        except Exception as exc:
            with output:
                print(f"{type(exc).__name__}: {exc}")
            status.value = "<b style='color:#b91c1c'>Loader smoke test failed.</b>"
        finally:
            set_busy(False)

    def train_clicked(_: Any) -> None:
        if not confirmation.value:
            status.value = "<b style='color:#b45309'>Confirm the long-running training action first.</b>"
            return
        try:
            options = selected_training_options()
        except ValueError as exc:
            status.value = f"<b style='color:#b91c1c'>Configuration error:</b> {html.escape(str(exc))}"
            return
        set_busy(True)
        output.clear_output(wait=True)
        status.value = f"<b>Training {html.escape(task.value)} with {html.escape(model_name())}…</b>"
        try:
            with output:
                result = train_yolo_export(dataset_root.value.strip(), task.value, options)
                print("\nTraining complete:\n" + _formatted_json(result))
                metrics_path = Path(result["final_metrics"])
                if metrics_path.is_file():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    print("\nTraining/validation metrics:\n" + _formatted_json(metrics))
                if result.get("test_metrics"):
                    test_metrics = json.loads(Path(result["test_metrics"]).read_text(encoding="utf-8"))
                    print("\nHeld-out test performance:\n" + _formatted_json(test_metrics))
                else:
                    print("\nNo populated test split was configured; held-out test evaluation was skipped.")
                plot_paths = sorted(Path(result["run_dir"]).glob("*.png"))
                if plot_paths:
                    print("\nUltralytics plots:\n" + "\n".join(str(path) for path in plot_paths))
            status.value = "<b style='color:#15803d'>Training completed successfully.</b>"
        except Exception as exc:
            with output:
                print(f"\n{type(exc).__name__}: {exc}")
            status.value = "<b style='color:#b91c1c'>Training failed.</b> Review the output below."
        finally:
            set_busy(False)

    observed_controls = [
        task,
        colour_space,
        dataset_root,
        output_dir,
        run_name,
        resume,
        epochs,
        image_size,
        batch,
        workers,
        seed,
        device,
        amp,
        extra_args,
        *yolo_controls.values(),
    ]
    for control in observed_controls:
        control.observe(update_preview, names="value")

    def task_changed(change: dict[str, Any]) -> None:
        old_task, new_task = change["old"], change["new"]
        old_dataset = root / ("dataset_detect_rgb" if old_task == "detect" else "dataset_seg_rgb")
        if dataset_root.value == str(old_dataset):
            dataset_root.value = str(root / ("dataset_detect_rgb" if new_task == "detect" else "dataset_seg_rgb"))
        old_defaults = _dashboard_yolo_defaults(old_task)
        new_defaults = _dashboard_yolo_defaults(new_task)
        for name in ("mosaic", "hsv_h", "hsv_s", "hsv_v"):
            if yolo_controls[name].value == getattr(old_defaults, name):
                yolo_controls[name].value = getattr(new_defaults, name)

    task.observe(task_changed, names="value")
    confirmation.observe(lambda change: set_busy(busy), names="value")
    validate_button.on_click(validate_clicked)
    smoke_button.on_click(smoke_clicked)
    train_button.on_click(train_clicked)
    update_preview()

    responsive_grid = {
        "grid_template_columns": "repeat(auto-fit, minmax(285px, 1fr))",
        "grid_gap": "10px 14px",
        "width": "100%",
        "overflow": "visible",
    }
    essentials = widgets.GridBox(
        children=[epochs, image_size, batch, workers, seed, device, amp],
        layout=widgets.Layout(**responsive_grid),
    )
    parameter_sections = []
    for title, help_text, names in YOLO_PARAMETER_GROUPS:
        parameter_sections.append(
            widgets.VBox(
                [
                    widgets.HTML(f"<p class='model-trainer-help'>{html.escape(help_text)}</p>"),
                    widgets.GridBox(
                        children=[yolo_controls[name] for name in names],
                        layout=widgets.Layout(**responsive_grid),
                    ),
                ],
                layout=widgets.Layout(width="100%", padding="6px 2px 12px 2px", overflow="visible"),
            )
        )
    parameters = widgets.Accordion(
        children=parameter_sections,
        selected_index=0,
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    for index, (title, _, _) in enumerate(YOLO_PARAMETER_GROUPS):
        parameters.set_title(index, title)

    paths = widgets.VBox(
        [dataset_root, colour_space, output_dir, run_name, resume],
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    button_row = widgets.HBox(
        [validate_button, smoke_button],
        layout=widgets.Layout(width="100%", flex_flow="row wrap"),
    )
    styles = widgets.HTML(
        """
        <style>
        .model-trainer-dashboard { max-width: 100%; overflow: visible; }
        .model-trainer-dashboard .widget-label {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
            line-height: 1.2;
        }
        .model-trainer-dashboard .widget-checkbox { min-width: 0 !important; }
        .model-trainer-dashboard .widget-checkbox .widget-label {
            width: auto !important;
            max-width: calc(100% - 28px) !important;
        }
        .model-trainer-dashboard .widget-toggle-buttons .widget-label { min-width: 90px; }
        .model-trainer-dashboard .widget-toggle-button { white-space: normal; min-height: 32px; }
        .model-trainer-preview { width: 100%; min-width: 0; }
        .model-trainer-preview p { margin: 8px 0 5px; }
        .model-trainer-preview pre {
            box-sizing: border-box;
            width: 100%;
            max-width: 100%;
            margin: 0;
            padding: 9px 11px;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            background: #f8fafc;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .model-trainer-error { color: #b91c1c; padding: 8px 0; }
        .model-trainer-help { margin: 0 0 10px; color: #475569; }
        </style>
        """
    )
    dashboard = widgets.VBox(
        [
            styles,
            widgets.HTML(
                "<h2>YOLO26 Model Trainer</h2>"
                "<p>Validate and train standalone YOLO datasets or complete Data Labeller exports. "
                "Defaults use AdamW, maximum 300 epochs, minimum 100 epochs, patience 40, 4 workers, seed 0 and fixed batch 16. "
                "Colour augmentation is off for detection and segmentation. "
                "RGB and HSV share the same task recipe and starting checkpoint. "
                "Select final settings using validation results.</p>"
            ),
            task,
            widgets.HTML("<h3>Dataset & run</h3>"),
            paths,
            widgets.HTML("<h3>Essentials</h3>"),
            essentials,
            widgets.HTML("<h3>All YOLO parameters</h3>"),
            parameters,
            widgets.HTML("<h4>Version-specific overrides</h4>"),
            extra_args,
            command_preview,
            button_row,
            confirmation,
            train_button,
            status,
            output,
        ],
        layout=widgets.Layout(width="100%", max_width="100%", overflow="visible"),
    )
    dashboard.add_class("model-trainer-dashboard")
    return dashboard
