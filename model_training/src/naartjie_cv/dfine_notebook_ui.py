from __future__ import annotations

import asyncio
import html
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from .datasets import (
    dfine_loader_smoke,
    validate_yolo_training_dataset,
)
from .notebook_ui import _formatted_json, _project_root
from .train import (
    DFineSegOptions,
    TrainingOptions,
    dfine_seg_loader_smoke,
    train_dfine_export,
    train_dfine_seg_export,
)


DFINE_TASKS = (
    ("Official D-FINE detection", "official_detect"),
    ("D-FINE-seg detection", "dfine_seg_detect"),
    ("D-FINE-seg instance segmentation", "dfine_seg_segment"),
)

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_OFFICIAL_PROGRESS = re.compile(
    r"Epoch:\s*\[(\d+)/(\d+)\].*?\[\s*(\d+)/(\d+)\]"
)
_DFINE_SEG_PROGRESS = re.compile(r"Epoch\s+(\d+)/(\d+):\s*(\d+)%")


def _json_object(value: str) -> dict[str, Any]:
    if not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Advanced config is not valid JSON: {exc.msg}.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Advanced config must be a JSON object.")
    return payload


def _optional_float(value: str, label: str) -> float | None:
    if not value.strip():
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be a number or empty.") from exc


def _command_line(command: list[str]) -> str:
    return subprocess.list2cmdline(command)


def _default_dfine_seg_python(checkout: str) -> str:
    upstream = Path(checkout).expanduser()
    candidates = (
        upstream / ".venv" / "Scripts" / "python.exe",
        upstream / ".venv" / "bin" / "python",
    )
    return str(next((path for path in candidates if path.is_file()), Path(sys.executable)))


def _preferred_amp_dtype() -> str:
    """Use the more stable BF16 format when the active GPU supports it."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return "bfloat16"
    except (ImportError, RuntimeError):
        pass
    return "float16"


def create_dfine_training_dashboard(project_root: str | Path | None = None) -> Any:
    """Build the D-FINE/D-FINE-seg detection and instance-segmentation dashboard."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError(
            "The D-FINE notebook UI requires ipywidgets in the active kernel. Run "
            f"`{sys.executable} -m pip install \"ipywidgets>=8.1\"`, restart the kernel, and rerun."
        ) from exc

    root = _project_root(project_root)
    detection_dataset = root / "dataset_detect_rgb"
    segmentation_dataset = root / "dataset_seg_rgb"
    def field_layout() -> Any:
        # Each widget needs its own mutable Layout model. Sharing one causes a
        # task-specific ``display = none`` update to hide every form control.
        return widgets.Layout(width="100%", min_width="0")

    field_style = {"description_width": "175px"}

    def text_field(**kwargs: Any) -> Any:
        return widgets.Text(layout=field_layout(), style=field_style, **kwargs)

    task = widgets.RadioButtons(
        options=DFINE_TASKS,
        value="dfine_seg_segment",
        description="Trainer",
        layout=widgets.Layout(width="100%", min_width="0", overflow="visible"),
        style={"description_width": "70px"},
    )
    dataset_root = text_field(
        value=str(segmentation_dataset),
        description="Dataset/export root",
        placeholder="Complete Data Labeller export or standalone YOLO dataset",
    )
    official_root = text_field(
        value=os.getenv(
            "DFINE_ROOT",
            str(root / "third_party" / "D-FINE")
            if (root / "third_party" / "D-FINE" / "train.py").is_file()
            else "",
        ),
        description="Official D-FINE checkout",
        placeholder=r"Example: C:\src\D-FINE",
    )
    official_python = text_field(
        value=os.getenv("DFINE_PYTHON", sys.executable),
        description="Official D-FINE Python",
        placeholder="Python from the official D-FINE environment",
    )
    dfine_seg_root = text_field(
        value=os.getenv(
            "DFINE_SEG_ROOT",
            str(root / "third_party" / "D-FINE-seg")
            if (root / "third_party" / "D-FINE-seg" / "src" / "dl" / "train.py").is_file()
            else "",
        ),
        description="D-FINE-seg checkout",
        placeholder=r"Example: C:\src\D-FINE-seg",
    )
    dfine_seg_python = text_field(
        value=_default_dfine_seg_python(dfine_seg_root.value),
        description="D-FINE-seg Python",
        placeholder="Python from the D-FINE-seg uv/venv environment",
    )
    output_dir = text_field(value=str(root / "runs"), description="Runs directory")
    run_name = text_field(
        value="",
        description="Run name",
        placeholder="Optional; a timestamped name is used when empty",
    )
    resume = text_field(
        value="",
        description="Resume checkpoint",
        placeholder="Official D-FINE only; optional .pth checkpoint",
    )

    official_epochs = widgets.BoundedIntText(
        value=100,
        min=1,
        max=10000,
        description="Epochs",
        layout=field_layout(),
        style=field_style,
    )
    official_image_size = widgets.BoundedIntText(
        value=640,
        min=32,
        max=4096,
        step=32,
        description="Image size",
        layout=field_layout(),
        style=field_style,
    )
    official_batch = widgets.BoundedIntText(
        value=8,
        min=1,
        max=65536,
        description="Total batch size",
        layout=field_layout(),
        style=field_style,
    )
    official_workers = widgets.BoundedIntText(
        value=0 if os.name == "nt" else 8,
        min=0,
        max=128,
        description="Data workers",
        layout=field_layout(),
        style=field_style,
    )
    official_seed = widgets.IntText(
        value=42,
        description="Random seed",
        layout=field_layout(),
        style=field_style,
    )
    official_device = text_field(value="0", description="CUDA device", placeholder="0 or cuda:0")
    official_amp = widgets.Checkbox(
        value=True,
        description="Automatic mixed precision (AMP)",
        indent=False,
        layout=field_layout(),
    )
    official_base_lr = text_field(value="", description="Base learning rate", placeholder="Upstream default")
    official_backbone_lr = text_field(value="", description="Backbone learning rate", placeholder="Upstream default")
    official_weight_decay = text_field(value="", description="Weight decay", placeholder="Upstream default")
    official_clip_max_norm = text_field(value="", description="Gradient clip norm", placeholder="Upstream default")
    official_warmup_steps = widgets.BoundedIntText(
        value=500,
        min=0,
        max=1000000,
        description="Warmup steps",
        layout=field_layout(),
        style=field_style,
    )
    official_ema = widgets.Checkbox(
        value=True,
        description="Exponential moving average",
        indent=False,
        layout=field_layout(),
    )
    official_extra_config = widgets.Textarea(
        value="",
        description="Advanced YAML overrides",
        placeholder='Optional JSON, e.g. {"train_dataloader":{"dataset":{"transforms":{"policy":{"epoch":70}}}}}',
        layout=widgets.Layout(width="100%", min_width="0", height="92px"),
        style=field_style,
    )

    model_size = widgets.Dropdown(
        options=(("Nano", "n"), ("Small", "s"), ("Medium", "m"), ("Large", "l"), ("X-Large", "x")),
        value="s",
        description="Model size",
        layout=field_layout(),
        style=field_style,
    )
    seg_epochs = widgets.BoundedIntText(
        value=120,
        min=1,
        max=10000,
        description="Epochs",
        layout=field_layout(),
        style=field_style,
    )
    seg_image_size = widgets.BoundedIntText(
        value=640,
        min=32,
        max=4096,
        step=32,
        description="Image size",
        layout=field_layout(),
        style=field_style,
    )
    seg_batch = widgets.IntText(
        value=2,
        description="Physical batch size",
        layout=field_layout(),
        style=field_style,
    )
    seg_workers = widgets.BoundedIntText(
        value=2 if os.name == "nt" else 8,
        min=0,
        max=128,
        description="Data workers",
        layout=field_layout(),
        style=field_style,
    )
    seg_seed = widgets.IntText(
        value=42,
        description="Random seed",
        layout=field_layout(),
        style=field_style,
    )
    seg_device = text_field(value="cuda", description="Torch device", placeholder="cuda or cpu")
    seg_amp = widgets.Checkbox(
        value=True,
        description="Automatic mixed precision (AMP)",
        indent=False,
        layout=field_layout(),
    )
    amp_dtype = widgets.Dropdown(
        options=(("bfloat16", "bfloat16"), ("float16", "float16")),
        value=_preferred_amp_dtype(),
        description="AMP dtype",
        layout=field_layout(),
        style=field_style,
    )
    early_stopping = widgets.BoundedIntText(
        value=20,
        min=0,
        max=10000,
        description="Early-stop patience",
        layout=field_layout(),
        style=field_style,
    )
    grad_accumulation = widgets.BoundedIntText(
        value=4,
        min=1,
        max=10000,
        description="Gradient accumulation",
        layout=field_layout(),
        style=field_style,
    )
    clip_max_norm = widgets.FloatText(
        value=0.1,
        description="Gradient clip norm",
        layout=field_layout(),
        style=field_style,
    )
    use_wandb = widgets.Checkbox(
        value=False,
        description="Weights & Biases logging",
        indent=False,
        layout=field_layout(),
    )
    use_ema = widgets.Checkbox(
        value=True,
        description="Exponential moving average",
        indent=False,
        layout=field_layout(),
    )
    use_scheduler = widgets.Checkbox(
        value=True,
        description="OneCycle scheduler",
        indent=False,
        layout=field_layout(),
    )
    use_muon = widgets.Checkbox(
        value=True,
        description="Muon optimizer routing",
        indent=False,
        layout=field_layout(),
    )
    aux_optimizer = widgets.Dropdown(
        options=(("Adan", "adan"), ("AdamW", "adamw")),
        value="adan",
        description="Auxiliary optimizer",
        layout=field_layout(),
        style=field_style,
    )
    base_lr = text_field(value="", description="Base learning rate", placeholder="Model default")
    backbone_lr = text_field(value="", description="Backbone learning rate", placeholder="Model default")
    weight_decay = widgets.FloatText(
        value=0.000125,
        description="Weight decay",
        layout=field_layout(),
        style=field_style,
    )
    mosaic_probability = text_field(
        value="0",
        description="Mosaic probability",
        placeholder="0 disables mosaic",
    )
    no_mosaic_epochs = widgets.BoundedIntText(
        value=0,
        min=0,
        max=10000,
        description="Close mosaic epochs",
        layout=field_layout(),
        style=field_style,
    )
    conf_threshold = widgets.BoundedFloatText(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description="Validation confidence",
        layout=field_layout(),
        style=field_style,
    )
    iou_threshold = widgets.BoundedFloatText(
        value=0.5,
        min=0,
        max=1,
        step=0.01,
        description="Validation IoU",
        layout=field_layout(),
        style=field_style,
    )
    pretrained_dataset = widgets.Dropdown(
        options=(("COCO", "coco"), ("Objects365 → COCO", "obj2coco")),
        value="coco",
        description="Pretrained dataset",
        layout=field_layout(),
        style=field_style,
    )
    pretrained_model = text_field(
        value="",
        description="Pretrained checkpoint",
        placeholder="Auto-selected when empty",
    )
    visualize_validation = widgets.Checkbox(
        value=True,
        description="Save validation visualizations",
        indent=False,
        layout=field_layout(),
    )
    keep_ratio = widgets.Checkbox(
        value=True,
        description="Preserve image aspect ratio",
        indent=False,
        layout=field_layout(),
    )
    freeze_except_mask = widgets.Checkbox(
        value=False,
        description="Train mask head only",
        indent=False,
        layout=field_layout(),
    )
    mask_batch_size = widgets.BoundedIntText(
        value=16,
        min=1,
        max=100000,
        description="Mask evaluation batch",
        layout=field_layout(),
        style=field_style,
    )
    horizontal_flip = widgets.BoundedFloatText(
        value=0.3,
        min=0,
        max=1,
        step=0.05,
        description="Horizontal flip chance",
        layout=field_layout(),
        style=field_style,
    )
    vertical_flip = widgets.BoundedFloatText(
        value=0.0,
        min=0,
        max=1,
        step=0.05,
        description="Vertical flip chance",
        layout=field_layout(),
        style=field_style,
    )
    rotate_90 = widgets.BoundedFloatText(
        value=0.0,
        min=0,
        max=1,
        step=0.05,
        description="90-degree rotate chance",
        layout=field_layout(),
        style=field_style,
    )
    rotation_degrees = widgets.BoundedFloatText(
        value=10.0,
        min=0,
        max=180,
        step=1,
        description="Max rotation degrees",
        layout=field_layout(),
        style=field_style,
    )
    rotation_probability = widgets.BoundedFloatText(
        value=0.05,
        min=0,
        max=1,
        step=0.05,
        description="Rotation chance",
        layout=field_layout(),
        style=field_style,
    )
    blur_probability = widgets.BoundedFloatText(
        value=0.0,
        min=0,
        max=1,
        step=0.01,
        description="Blur chance",
        layout=field_layout(),
        style=field_style,
    )
    extra_config = widgets.Textarea(
        value="",
        description="Advanced config",
        placeholder='Optional nested JSON, e.g. {"train":{"augs":{"left_right_flip":0.5}}}',
        layout=widgets.Layout(width="100%", min_width="0", height="88px"),
        style=field_style,
    )

    setting_help = (
        (task, "Choose the original box detector, the separate box detector, or per-object mask segmentation."),
        (dataset_root, "Folder containing data.yaml and the train/val images and labels. Segmentation defaults to dataset_seg_rgb."),
        (official_root, "Checkout of the original Peterande D-FINE training code."),
        (official_python, "Python environment used to launch official D-FINE training."),
        (dfine_seg_root, "Checkout of ArgoHA D-FINE-seg, which supports instance masks."),
        (dfine_seg_python, "Python environment used to launch D-FINE-seg training."),
        (output_dir, "Parent folder for run configs, checkpoints, logs, and prepared data."),
        (run_name, "Optional fixed run name; leave blank for a timestamped name."),
        (resume, "Official D-FINE checkpoint to continue from; leave blank for a new run."),
        (official_epochs, "Maximum passes through the official D-FINE training set."),
        (official_image_size, "Resize width and height for official D-FINE. Larger values need more GPU memory."),
        (official_batch, "Images processed together by official D-FINE; larger batches use more GPU memory."),
        (official_workers, "Parallel image-loading processes. More workers use more system RAM."),
        (official_seed, "Controls training randomness so runs can be compared."),
        (official_device, "GPU index such as 0, or cpu for slow CPU training."),
        (official_amp, "Use mixed precision to reduce GPU memory and usually speed up training."),
        (official_base_lr, "Learning rate for most of the official D-FINE model; blank uses its config default."),
        (official_backbone_lr, "Learning rate for the pretrained feature extractor; blank uses its config default."),
        (official_weight_decay, "Regularization on model weights; blank uses the official config default."),
        (official_clip_max_norm, "Limits unusually large gradient updates; blank uses the official config default."),
        (official_warmup_steps, "Optimizer steps used to ramp up the learning rate at the start."),
        (official_ema, "Average recent model weights to make validation and final weights more stable."),
        (official_extra_config, "Optional JSON overrides for other official D-FINE YAML settings."),
        (model_size, "Model capacity. Small balances accuracy and memory on an 8 GB GPU."),
        (seg_epochs, "Maximum passes through the segmentation training set; early stopping may finish sooner."),
        (seg_image_size, "Square input size. 640 preserves this dataset's detail without excessive GPU memory use."),
        (seg_batch, "Images per GPU step. Two leaves headroom on an 8 GB GPU; use one if memory is tight."),
        (seg_workers, "Parallel image-loading processes. Two uses the laptop CPU without many Windows worker copies."),
        (seg_seed, "Controls training randomness so runs can be compared."),
        (seg_device, "PyTorch device, normally cuda for the laptop GPU or cpu for debugging."),
        (seg_amp, "Mixed precision reduces GPU memory and speeds up training."),
        (amp_dtype, "BF16 has a wider numeric range than FP16. The dashboard selects BF16 only on supported GPUs."),
        (early_stopping, "Stop when validation stops improving for this many epochs; zero disables it."),
        (grad_accumulation, "Combine several small GPU batches before one optimizer step. Two times four gives an effective batch of eight."),
        (clip_max_norm, "Cap gradient magnitude to avoid unstable training updates."),
        (use_wandb, "Send experiment metrics to Weights & Biases when configured."),
        (use_ema, "Average recent model weights for more stable validation and final checkpoints."),
        (use_scheduler, "Vary the learning rate during training using the upstream OneCycle schedule."),
        (use_muon, "Use the upstream Muon optimizer for selected matrices; other weights use the auxiliary optimizer."),
        (aux_optimizer, "Optimizer used for weights that are not routed to Muon."),
        (base_lr, "Main learning rate; blank uses the upstream value for the selected model size."),
        (backbone_lr, "Learning rate for the pretrained backbone; blank uses the upstream value."),
        (weight_decay, "Regularization on model weights to limit overfitting."),
        (mosaic_probability, "Chance of combining four training images. Zero avoids distorted instance masks."),
        (no_mosaic_epochs, "Disable mosaic for the final epochs; irrelevant when mosaic probability is zero."),
        (conf_threshold, "Minimum prediction confidence used during validation."),
        (iou_threshold, "Overlap threshold used to match validation predictions with labels."),
        (pretrained_dataset, "Source of pretrained weights. Automatic segmentation weights are available for COCO."),
        (pretrained_model, "Optional custom pretrained checkpoint; blank selects the matching COCO model."),
        (visualize_validation, "Save example validation predictions, using additional disk space."),
        (keep_ratio, "Pad images to keep their original aspect ratio instead of stretching their defect masks."),
        (freeze_except_mask, "Train only the mask head. Leave off when adapting the whole model to new defect classes."),
        (mask_batch_size, "Images processed together for mask metrics; lower values reduce evaluation memory spikes."),
        (horizontal_flip, "Chance of mirroring a training image left to right."),
        (vertical_flip, "Chance of flipping a training image upside down."),
        (rotate_90, "Chance of rotating a training image by 90 degrees."),
        (rotation_degrees, "Largest angle used for small random rotations."),
        (rotation_probability, "Chance of applying a small random rotation."),
        (blur_probability, "Chance of adding blur. Zero keeps small defect details sharp."),
        (extra_config, "Optional nested JSON for upstream D-FINE-seg settings not shown in the form."),
    )
    for control, explanation in setting_help:
        control.tooltip = explanation

    responsive_layout = widgets.Layout(
        grid_template_columns="repeat(auto-fit, minmax(310px, 1fr))",
        grid_gap="10px 14px",
        width="100%",
        overflow="visible",
    )
    official_essentials = widgets.GridBox(
        children=[
            official_epochs,
            official_image_size,
            official_batch,
            official_workers,
            official_seed,
            official_device,
            official_amp,
            resume,
        ],
        layout=responsive_layout,
    )
    official_advanced = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.HTML(
                        "<p class='dfine-help'>Leave learning-rate fields empty to use the official "
                        "Nano config. Advanced JSON overrides additional upstream YAML keys; dataset "
                        "paths, classes, output directory, and batch settings stay managed by this trainer.</p>"
                    ),
                    widgets.GridBox(
                        children=[
                            official_base_lr,
                            official_backbone_lr,
                            official_weight_decay,
                            official_clip_max_norm,
                            official_warmup_steps,
                            official_ema,
                        ],
                        layout=responsive_layout,
                    ),
                    official_extra_config,
                ],
                layout=widgets.Layout(width="100%", overflow="visible"),
            )
        ],
        selected_index=0,
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    official_advanced.set_title(0, "Official D-FINE advanced settings")
    official_settings = widgets.VBox(
        [official_essentials, official_advanced],
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    seg_essentials = widgets.GridBox(
        children=[
            model_size,
            seg_epochs,
            seg_image_size,
            seg_batch,
            seg_workers,
            seg_seed,
            seg_device,
            seg_amp,
            amp_dtype,
            early_stopping,
        ],
        layout=responsive_layout,
    )
    seg_optimisation = widgets.GridBox(
        children=[
            grad_accumulation,
            clip_max_norm,
            use_ema,
            use_scheduler,
            use_muon,
            aux_optimizer,
            base_lr,
            backbone_lr,
            weight_decay,
            mosaic_probability,
            no_mosaic_epochs,
            conf_threshold,
            iou_threshold,
            pretrained_dataset,
            pretrained_model,
            visualize_validation,
            use_wandb,
        ],
        layout=responsive_layout,
    )
    seg_advanced = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.HTML(
                        "<p class='dfine-help'>Optimizers, scheduling, augmentation, validation, "
                        "pretraining, and experiment logging.</p>"
                    ),
                    seg_optimisation,
                    extra_config,
                ],
                layout=widgets.Layout(width="100%", overflow="visible"),
            )
        ],
        selected_index=None,
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    seg_advanced.set_title(0, "Advanced D-FINE-seg settings")
    segment_settings = widgets.Accordion(
        children=[
            widgets.VBox(
                [
                    widgets.HTML(
                        "<p class='dfine-help'>Mask evaluation and image augmentation. "
                        "The defaults match the selected D-FINE-seg checkout. "
                        "Mosaic probability is in Advanced D-FINE-seg settings; set it to 0 "
                        "to disable mosaic for masks. Train mask head only requires a "
                        "suitable pretrained segmentation checkpoint.</p>"
                    ),
                    widgets.GridBox(
                        children=[
                            keep_ratio,
                            freeze_except_mask,
                            mask_batch_size,
                            horizontal_flip,
                            vertical_flip,
                            rotate_90,
                            rotation_degrees,
                            rotation_probability,
                            blur_probability,
                        ],
                        layout=responsive_layout,
                    ),
                ],
                layout=widgets.Layout(width="100%", overflow="visible"),
            )
        ],
        selected_index=None,
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    segment_settings.set_title(0, "Instance segmentation options")
    seg_settings = widgets.VBox(
        [seg_essentials, segment_settings, seg_advanced],
        layout=widgets.Layout(width="100%", overflow="visible"),
    )

    format_note = widgets.HTML(layout=widgets.Layout(width="100%"))
    command_preview = widgets.HTML(layout=widgets.Layout(width="100%"))
    status = widgets.HTML(
        value="<b>Ready.</b> Validate the selected dataset before training.",
        layout=widgets.Layout(width="100%"),
    )
    epoch_progress = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description="Epoch progress",
        bar_style="info",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "110px"},
    )
    progress_detail = widgets.HTML(value="Waiting for trainer output.")
    progress_panel = widgets.VBox(
        [epoch_progress, progress_detail],
        layout=widgets.Layout(width="100%", display="none"),
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
    validate_button = widgets.Button(
        description="Validate dataset",
        icon="check",
        button_style="info",
        layout=widgets.Layout(width="175px", margin="0 8px 8px 0"),
    )
    smoke_button = widgets.Button(
        description="Smoke-test loader",
        icon="flask",
        layout=widgets.Layout(width="175px", margin="0 8px 8px 0"),
    )
    prepare_button = widgets.Button(
        description="Prepare dry run",
        icon="gear",
        layout=widgets.Layout(width="175px", margin="0 8px 8px 0"),
    )
    confirmation = widgets.Checkbox(
        value=False,
        description="I understand that training can run for a long time",
        indent=False,
        layout=widgets.Layout(width="100%"),
        tooltip="Training can run for hours and will write checkpoints and logs under the runs directory.",
    )
    train_button = widgets.Button(
        description="Start training",
        icon="play",
        button_style="success",
        disabled=True,
        layout=widgets.Layout(width="175px"),
    )
    busy = False
    active_training: threading.Thread | None = None
    monitor_tasks: set[asyncio.Task[Any]] = set()

    def is_official() -> bool:
        return task.value == "official_detect"

    def dfine_seg_task() -> str:
        return "segment" if task.value == "dfine_seg_segment" else "detect"

    def official_options() -> TrainingOptions:
        return TrainingOptions(
            epochs=official_epochs.value,
            imgsz=official_image_size.value,
            batch=official_batch.value,
            workers=official_workers.value,
            seed=official_seed.value,
            device=official_device.value.strip() or "0",
            amp=official_amp.value,
            output_dir=output_dir.value.strip() or str(root / "runs"),
            run_name=run_name.value.strip() or None,
            resume=resume.value.strip() or None,
            dfine_base_lr=_optional_float(official_base_lr.value, "Base learning rate"),
            dfine_backbone_lr=_optional_float(official_backbone_lr.value, "Backbone learning rate"),
            dfine_weight_decay=_optional_float(official_weight_decay.value, "Weight decay"),
            dfine_clip_max_norm=_optional_float(official_clip_max_norm.value, "Gradient clip norm"),
            dfine_warmup_steps=official_warmup_steps.value,
            dfine_use_ema=official_ema.value,
            dfine_extra_config=_json_object(official_extra_config.value),
        )

    def seg_options() -> DFineSegOptions:
        config = _json_object(extra_config.value)
        if dfine_seg_task() == "segment":
            train_config = config.setdefault("train", {})
            if not isinstance(train_config, dict):
                raise ValueError("Advanced config 'train' must be a JSON object.")
            train_config.update(
                {
                    "keep_ratio": keep_ratio.value,
                    "freeze_except_mask": freeze_except_mask.value,
                    "mask_batch_size": mask_batch_size.value,
                }
            )
            augmentations = train_config.setdefault("augs", {})
            if not isinstance(augmentations, dict):
                raise ValueError("Advanced config 'train.augs' must be a JSON object.")
            augmentations.update(
                {
                    "left_right_flip": horizontal_flip.value,
                    "up_down_flip": vertical_flip.value,
                    "rotate_90": rotate_90.value,
                    "rotation_degree": rotation_degrees.value,
                    "rotation_p": rotation_probability.value,
                    "blur": blur_probability.value,
                }
            )
        return DFineSegOptions(
            model_size=model_size.value,
            epochs=seg_epochs.value,
            imgsz=seg_image_size.value,
            batch=seg_batch.value,
            workers=seg_workers.value,
            seed=seg_seed.value,
            device=seg_device.value.strip() or "cuda",
            amp=seg_amp.value,
            amp_dtype=amp_dtype.value,
            early_stopping=early_stopping.value,
            grad_accumulation=grad_accumulation.value,
            clip_max_norm=clip_max_norm.value,
            use_wandb=use_wandb.value,
            use_ema=use_ema.value,
            use_scheduler=use_scheduler.value,
            use_muon=use_muon.value,
            aux_optimizer=aux_optimizer.value,
            base_lr=_optional_float(base_lr.value, "Base learning rate"),
            backbone_lr=_optional_float(backbone_lr.value, "Backbone learning rate"),
            weight_decay=weight_decay.value,
            mosaic_probability=_optional_float(mosaic_probability.value, "Mosaic probability"),
            no_mosaic_epochs=no_mosaic_epochs.value,
            conf_threshold=conf_threshold.value,
            iou_threshold=iou_threshold.value,
            pretrained_dataset=pretrained_dataset.value,
            pretrained_model=pretrained_model.value.strip() or None,
            visualize_validation=visualize_validation.value,
            output_dir=output_dir.value.strip() or str(root / "runs"),
            run_name=run_name.value.strip() or None,
            extra_config=config,
        )

    def update_mode(*_: Any) -> None:
        official = is_official()
        if task.value == "dfine_seg_segment":
            if dataset_root.value == str(detection_dataset):
                dataset_root.value = str(segmentation_dataset)
        elif dataset_root.value == str(segmentation_dataset):
            dataset_root.value = str(detection_dataset)
        official_root.layout.display = "" if official else "none"
        official_python.layout.display = "" if official else "none"
        official_settings.layout.display = "" if official else "none"
        dfine_seg_root.layout.display = "none" if official else ""
        dfine_seg_python.layout.display = "none" if official else ""
        seg_settings.layout.display = "none" if official else ""
        segment_settings.layout.display = "" if task.value == "dfine_seg_segment" else "none"
        segment_settings.selected_index = 0 if task.value == "dfine_seg_segment" else None
        if official:
            format_note.value = (
                "<div class='dfine-format'><b>Format:</b> Official Peterande/D-FINE is "
                "<b>object detection only</b>. It does not read YOLO <code>data.yaml</code>; "
                "this workflow validates either a standalone YOLO detection dataset or a complete "
                "Data Labeller export, then stages a non-destructive COCO view for D-FINE.</div>"
            )
        elif dfine_seg_task() == "detect":
            pretrained_dataset.disabled = False
            format_note.value = (
                "<div class='dfine-format'><b>Format:</b> D-FINE-seg detection accepts YOLO "
                "<code>class xc yc width height</code> labels. The notebook reads the detection "
                "<code>data.yaml</code> to preserve its splits, then stages equivalent COCO JSON "
                "without changing the source dataset.</div>"
            )
        else:
            if pretrained_dataset.value != "coco" and not pretrained_model.value.strip():
                pretrained_dataset.value = "coco"
            pretrained_dataset.disabled = not bool(pretrained_model.value.strip())
            format_note.value = (
                "<div class='dfine-format'><b>Format:</b> D-FINE-seg instance segmentation accepts "
                "the same polygon rows as YOLO segmentation: "
                "<code>class x1 y1 x2 y2 ... xN yN</code>, normalized to 0–1 with at least three "
                "points. The notebook reads <code>data.yaml</code> from the selected dataset and stages "
                "COCO JSON containing both <code>bbox</code> and polygon "
                "<code>segmentation</code> fields. The default Small/640 px/batch-2 profile "
                "leaves headroom on an 8 GB GPU, but available memory can change. If training "
                "reports an out-of-memory error, set Physical batch size to 1.</div>"
            )
        update_preview()

    def update_preview(*_: Any) -> None:
        try:
            if is_official():
                options = official_options()
                command = [
                    "python",
                    "scripts/model_trainer.py",
                    "dfine-detect",
                    dataset_root.value.strip() or "<export>",
                    "--dfine-root",
                    official_root.value.strip() or "<D-FINE>",
                    "--python-executable",
                    official_python.value.strip() or sys.executable,
                    "--epochs",
                    str(options.epochs),
                    "--imgsz",
                    str(options.imgsz),
                    "--batch",
                    str(options.batch),
                    "--workers",
                    str(options.workers),
                    "--seed",
                    str(options.seed),
                    "--device",
                    options.device,
                    "--output-dir",
                    options.output_dir,
                ]
                if options.run_name:
                    command.extend(["--run-name", options.run_name])
                if options.resume:
                    command.extend(["--resume", options.resume])
                if not options.amp:
                    command.append("--no-amp")
                if not options.dfine_use_ema:
                    command.append("--no-dfine-ema")
                official_values = (
                    ("dfine-base-lr", options.dfine_base_lr),
                    ("dfine-backbone-lr", options.dfine_backbone_lr),
                    ("dfine-weight-decay", options.dfine_weight_decay),
                    ("dfine-clip-max-norm", options.dfine_clip_max_norm),
                    ("dfine-warmup-steps", options.dfine_warmup_steps),
                )
                for name, value in official_values:
                    if value is not None:
                        command.extend([f"--{name}", str(value)])
                if options.dfine_extra_config:
                    command.extend(
                        ["--dfine-config-json", json.dumps(options.dfine_extra_config, separators=(",", ":"))]
                    )
                model = "Official D-FINE-N detector"
            else:
                options = seg_options()
                command = [
                    "python",
                    "scripts/model_trainer.py",
                    f"dfine-seg-{dfine_seg_task()}",
                    dataset_root.value.strip() or "<dataset>",
                    "--dfine-seg-root",
                    dfine_seg_root.value.strip() or "<D-FINE-seg>",
                    "--python-executable",
                    dfine_seg_python.value.strip() or sys.executable,
                    "--model-size",
                    options.model_size,
                    "--epochs",
                    str(options.epochs),
                    "--imgsz",
                    str(options.imgsz),
                    "--batch",
                    str(options.batch),
                    "--workers",
                    str(options.workers),
                    "--seed",
                    str(options.seed),
                    "--device",
                    options.device,
                    "--amp-dtype",
                    options.amp_dtype,
                    "--early-stopping",
                    str(options.early_stopping),
                    "--grad-accumulation",
                    str(options.grad_accumulation),
                    "--clip-max-norm",
                    str(options.clip_max_norm),
                    "--aux-optimizer",
                    options.aux_optimizer,
                    "--weight-decay",
                    str(options.weight_decay),
                    "--no-mosaic-epochs",
                    str(options.no_mosaic_epochs),
                    "--conf-threshold",
                    str(options.conf_threshold),
                    "--iou-threshold",
                    str(options.iou_threshold),
                    "--pretrained-dataset",
                    options.pretrained_dataset,
                    "--output-dir",
                    options.output_dir,
                ]
                boolean_options = (
                    ("amp", options.amp),
                    ("wandb", options.use_wandb),
                    ("ema", options.use_ema),
                    ("scheduler", options.use_scheduler),
                    ("muon", options.use_muon),
                    ("visualize-validation", options.visualize_validation),
                )
                defaults = {
                    "amp": True,
                    "wandb": False,
                    "ema": True,
                    "scheduler": True,
                    "muon": True,
                    "visualize-validation": True,
                }
                for name, value in boolean_options:
                    if value != defaults[name]:
                        command.append(f"--{name}" if value else f"--no-{name}")
                optional_values = (
                    ("base-lr", options.base_lr),
                    ("backbone-lr", options.backbone_lr),
                    ("mosaic-probability", options.mosaic_probability),
                    ("pretrained-model", options.pretrained_model),
                    ("run-name", options.run_name),
                )
                for name, value in optional_values:
                    if value is not None:
                        command.extend([f"--{name}", str(value)])
                if options.extra_config:
                    command.extend(
                        [
                            "--extra-config-json",
                            json.dumps(options.extra_config, separators=(",", ":")),
                        ]
                    )
                model = f"D-FINE-seg {options.model_size.upper()} ({dfine_seg_task()})"
            command_preview.value = (
                "<div class='dfine-preview'>"
                f"<p><b>Model:</b> <code>{html.escape(model)}</code></p>"
                "<p><b>Equivalent CLI:</b></p>"
                f"<pre><code>{html.escape(_command_line(command))}</code></pre>"
                "</div>"
            )
        except ValueError as exc:
            command_preview.value = (
                "<div class='dfine-error'><b>Configuration error:</b> "
                f"{html.escape(str(exc))}</div>"
            )

    def set_busy(value: bool) -> None:
        nonlocal busy
        busy = value
        validate_button.disabled = value
        smoke_button.disabled = value
        prepare_button.disabled = value
        train_button.disabled = value or not confirmation.value

    def validate_clicked(_: Any) -> None:
        output.clear_output(wait=True)
        report = (
            validate_yolo_training_dataset(dataset_root.value.strip(), "detect")
            if is_official()
            else validate_yolo_training_dataset(dataset_root.value.strip(), dfine_seg_task())
        )
        warnings = report.get("warnings", [])
        shown_report = dict(report)
        shown_report["warning_count"] = len(warnings)
        shown_report["warnings"] = warnings[:12]
        if len(warnings) > 12:
            shown_report["additional_warnings_not_shown"] = len(warnings) - 12
        with output:
            print(_formatted_json(shown_report))
        negative_count = sum("negative image" in warning for warning in warnings)
        status.value = (
            "<b style='color:#15803d'>Dataset is valid for the selected trainer.</b>"
            + (
                f" {negative_count} images have empty or missing labels (background images)."
                if negative_count else ""
            )
            if report["ok"]
            else "<b style='color:#b91c1c'>Dataset validation failed.</b> Review the errors below."
        )

    def smoke_clicked(_: Any) -> None:
        set_busy(True)
        output.clear_output(wait=True)
        status.value = "<b>Loading one real training batch…</b>"
        try:
            if is_official():
                result = dfine_loader_smoke(
                    dataset_root.value.strip(),
                    official_root.value.strip(),
                    official_python.value.strip() or sys.executable,
                )
            else:
                result = dfine_seg_loader_smoke(
                    dataset_root.value.strip(),
                    dfine_seg_task(),
                    dfine_seg_root.value.strip(),
                    seg_options(),
                    python_executable=dfine_seg_python.value.strip() or sys.executable,
                )
            with output:
                print(_formatted_json(result))
            status.value = "<b style='color:#15803d'>Loader smoke test passed.</b>"
        except Exception as exc:
            with output:
                print(f"{type(exc).__name__}: {exc}")
            status.value = "<b style='color:#b91c1c'>Loader smoke test failed.</b>"
        finally:
            set_busy(False)

    def run_selected(execute: bool, output_callback: Any | None = None) -> dict[str, Any]:
        if is_official():
            return train_dfine_export(
                dataset_root.value.strip(),
                official_root.value.strip(),
                official_options(),
                python_executable=official_python.value.strip() or sys.executable,
                execute=execute,
                output_callback=output_callback,
            )
        return train_dfine_seg_export(
            dataset_root.value.strip(),
            dfine_seg_task(),
            dfine_seg_root.value.strip(),
            seg_options(),
            python_executable=dfine_seg_python.value.strip() or sys.executable,
            execute=execute,
            output_callback=output_callback,
        )

    def prepare_clicked(_: Any) -> None:
        set_busy(True)
        output.clear_output(wait=True)
        status.value = "<b>Preparing and validating the training configuration…</b>"
        try:
            result = run_selected(False)
            with output:
                print(_formatted_json(result))
            status.value = (
                "<b style='color:#15803d'>Dry run prepared.</b> "
                "Review the generated config and command below."
            )
        except Exception as exc:
            with output:
                print(f"{type(exc).__name__}: {exc}")
            status.value = "<b style='color:#b91c1c'>Dry-run preparation failed.</b>"
        finally:
            set_busy(False)

    def train_clicked(_: Any) -> None:
        nonlocal active_training
        if not confirmation.value:
            status.value = "<b style='color:#b45309'>Confirm the long-running training action first.</b>"
            return
        if active_training is not None and active_training.is_alive():
            status.value = "<b style='color:#b45309'>A training job is already running.</b>"
            return
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            status.value = (
                "<b style='color:#b91c1c'>The notebook event loop is unavailable.</b> "
                "Restart the kernel and rerun the dashboard cell."
            )
            return
        try:
            selected_dataset = dataset_root.value.strip()
            selected_official = is_official()
            if selected_official:
                selected_upstream = official_root.value.strip()
                selected_python = official_python.value.strip() or sys.executable
                selected_options: TrainingOptions | DFineSegOptions = official_options()
                selected_task = "detect"
            else:
                selected_upstream = dfine_seg_root.value.strip()
                selected_python = dfine_seg_python.value.strip() or sys.executable
                selected_options = seg_options()
                selected_task = dfine_seg_task()
        except Exception as exc:
            status.value = (
                "<b style='color:#b91c1c'>Training configuration is invalid:</b> "
                f"{html.escape(str(exc))}"
            )
            return

        set_busy(True)
        output.clear_output(wait=True)
        started = time.monotonic()
        epoch_progress.value = 0
        epoch_progress.bar_style = "info"
        epoch_progress.description = "Epoch progress"
        progress_detail.value = "Starting trainer…"
        progress_panel.layout.display = ""
        status.value = "<b>Training is starting…</b>"
        events: queue.SimpleQueue[tuple[str, Any]] = queue.SimpleQueue()

        def stream_output(fragment: str) -> None:
            events.put(("output", fragment))

        def training_worker() -> None:
            try:
                if selected_official:
                    result = train_dfine_export(
                        selected_dataset,
                        selected_upstream,
                        selected_options,
                        python_executable=selected_python,
                        execute=True,
                        output_callback=stream_output,
                    )
                else:
                    result = train_dfine_seg_export(
                        selected_dataset,
                        selected_task,
                        selected_upstream,
                        selected_options,
                        python_executable=selected_python,
                        execute=True,
                        output_callback=stream_output,
                    )
                events.put(("result", result))
            except BaseException as exc:
                events.put(("error", (type(exc).__name__, str(exc))))

        async def monitor_training() -> None:
            nonlocal active_training
            terminal: tuple[str, Any] | None = None
            last_status_second = -1
            try:
                while (active_training is not None and active_training.is_alive()) or not events.empty():
                    durable_output: list[str] = []
                    latest_line = ""
                    while True:
                        try:
                            event_kind, payload = events.get_nowait()
                        except queue.Empty:
                            break
                        if event_kind == "output":
                            clean = _ANSI_ESCAPE.sub("", str(payload))
                            line = clean.strip()
                            if line:
                                latest_line = line
                            # tqdm carriage-return frames belong in the compact
                            # panel; newline records are appended in one batch.
                            if not clean.endswith("\r"):
                                durable_output.append(clean)
                        else:
                            terminal = (event_kind, payload)

                    if durable_output:
                        output.append_stdout("".join(durable_output))

                    elapsed = int(time.monotonic() - started)
                    if elapsed != last_status_second:
                        hours, remainder = divmod(elapsed, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        status.value = (
                            "<b>Training is running in the background…</b> "
                            f"Elapsed {hours:02d}:{minutes:02d}:{seconds:02d}"
                        )
                        last_status_second = elapsed

                    if latest_line:
                        official_match = _OFFICIAL_PROGRESS.search(latest_line)
                        seg_match = _DFINE_SEG_PROGRESS.search(latest_line)
                        if official_match:
                            epoch_index, total_epochs, batch_index, total_batches = map(
                                int, official_match.groups()
                            )
                            epoch_progress.value = min(
                                100, round(100 * (batch_index + 1) / max(1, total_batches))
                            )
                            epoch_progress.description = f"Epoch {epoch_index + 1}/{total_epochs}"
                            progress_detail.value = (
                                f"Batch {batch_index + 1}/{total_batches} &mdash; "
                                f"<code>{html.escape(latest_line[-240:])}</code>"
                            )
                        elif seg_match:
                            epoch_index, total_epochs, percentage = map(int, seg_match.groups())
                            epoch_progress.value = min(100, percentage)
                            epoch_progress.description = f"Epoch {epoch_index}/{total_epochs}"
                            progress_detail.value = f"<code>{html.escape(latest_line[-240:])}</code>"
                        else:
                            progress_detail.value = f"<code>{html.escape(latest_line[-240:])}</code>"

                    if terminal is not None and (active_training is None or not active_training.is_alive()):
                        break
                    await asyncio.sleep(0.25)

                if terminal is None:
                    raise RuntimeError("Training worker stopped without returning a result.")
                event_kind, payload = terminal
                if event_kind == "error":
                    error_name, error_message = payload
                    raise RuntimeError(f"{error_name}: {error_message}")

                result = payload
                output.append_stdout("\nTraining complete:\n" + _formatted_json(result) + "\n")
                if result.get("test_metrics"):
                    test_metrics = json.loads(Path(result["test_metrics"]).read_text(encoding="utf-8"))
                    output.append_stdout(
                        "\nHeld-out test performance:\n" + _formatted_json(test_metrics) + "\n"
                    )
                else:
                    output.append_stdout(
                        "\nNo populated test split was configured; held-out test evaluation was skipped.\n"
                    )
                epoch_progress.value = 100
                epoch_progress.bar_style = "success"
                progress_detail.value = "Training and final evaluation completed."
                status.value = "<b style='color:#15803d'>Training completed successfully.</b>"
            except Exception as exc:
                epoch_progress.bar_style = "danger"
                output.append_stdout(f"\n{type(exc).__name__}: {exc}\n")
                status.value = "<b style='color:#b91c1c'>Training failed.</b> Review the output below."
            finally:
                active_training = None
                set_busy(False)

        output.append_stdout("Launching trainer in the background with live output…\n")
        active_training = threading.Thread(
            target=training_worker,
            name="dfine-dashboard-training",
            daemon=True,
        )
        active_training.start()
        monitor_task = event_loop.create_task(monitor_training())
        monitor_tasks.add(monitor_task)
        monitor_task.add_done_callback(monitor_tasks.discard)

    controls = [
        task,
        dataset_root,
        official_root,
        official_python,
        dfine_seg_root,
        dfine_seg_python,
        output_dir,
        run_name,
        official_epochs,
        official_image_size,
        official_batch,
        official_workers,
        official_seed,
        official_device,
        official_amp,
        official_base_lr,
        official_backbone_lr,
        official_weight_decay,
        official_clip_max_norm,
        official_warmup_steps,
        official_ema,
        official_extra_config,
        resume,
        model_size,
        seg_epochs,
        seg_image_size,
        seg_batch,
        seg_workers,
        seg_seed,
        seg_device,
        seg_amp,
        amp_dtype,
        early_stopping,
        grad_accumulation,
        clip_max_norm,
        use_wandb,
        use_ema,
        use_scheduler,
        use_muon,
        aux_optimizer,
        base_lr,
        backbone_lr,
        weight_decay,
        mosaic_probability,
        no_mosaic_epochs,
        conf_threshold,
        iou_threshold,
        pretrained_dataset,
        pretrained_model,
        visualize_validation,
        keep_ratio,
        freeze_except_mask,
        mask_batch_size,
        horizontal_flip,
        vertical_flip,
        rotate_90,
        rotation_degrees,
        rotation_probability,
        blur_probability,
        extra_config,
    ]
    for control in controls:
        control.observe(update_preview, names="value")
    task.observe(update_mode, names="value")

    def checkout_changed(change: dict[str, Any]) -> None:
        current = dfine_seg_python.value
        if current == sys.executable or ".venv" in current:
            dfine_seg_python.value = _default_dfine_seg_python(str(change["new"]))

    dfine_seg_root.observe(checkout_changed, names="value")
    confirmation.observe(lambda change: set_busy(busy), names="value")
    validate_button.on_click(validate_clicked)
    smoke_button.on_click(smoke_clicked)
    prepare_button.on_click(prepare_clicked)
    train_button.on_click(train_clicked)

    styles = widgets.HTML(
        """
        <style>
        .dfine-trainer-dashboard { max-width: 100%; overflow: visible; }
        .dfine-trainer-dashboard .widget-label {
            overflow: visible !important;
            text-overflow: clip !important;
            white-space: normal !important;
            line-height: 1.2;
        }
        .dfine-trainer-dashboard .widget-checkbox { min-width: 0 !important; }
        .dfine-trainer-dashboard .widget-checkbox .widget-label {
            width: auto !important;
            max-width: calc(100% - 28px) !important;
        }
        .dfine-trainer-dashboard .widget-radio-box { min-width: 0 !important; }
        .dfine-trainer-dashboard .widget-radio-box label {
            height: auto !important;
            white-space: normal !important;
            overflow-wrap: anywhere;
            line-height: 1.35;
            padding: 3px 0;
        }
        .dfine-format {
            box-sizing: border-box;
            width: 100%;
            padding: 10px 12px;
            margin: 6px 0 10px;
            border-left: 4px solid #0ea5e9;
            background: #f0f9ff;
            overflow-wrap: anywhere;
        }
        .dfine-preview pre {
            box-sizing: border-box;
            width: 100%;
            max-width: 100%;
            padding: 9px 11px;
            border: 1px solid #cbd5e1;
            border-radius: 4px;
            background: #f8fafc;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            word-break: break-word;
        }
        .dfine-help { color: #475569; margin: 4px 0 10px; }
        .dfine-error { color: #b91c1c; padding: 8px 0; }
        </style>
        """
    )
    paths = widgets.VBox(
        [
            dataset_root,
            official_root,
            official_python,
            dfine_seg_root,
            dfine_seg_python,
            output_dir,
            run_name,
        ],
        layout=widgets.Layout(width="100%", overflow="visible"),
    )
    buttons = widgets.HBox(
        [validate_button, smoke_button, prepare_button],
        layout=widgets.Layout(width="100%", flex_flow="row wrap"),
    )
    dashboard = widgets.VBox(
        [
            styles,
            widgets.HTML(
                "<h2>D-FINE Model Trainer</h2>"
                "<p>Train the official D-FINE detector or the separate D-FINE-seg "
                "detection/instance-segmentation implementation.</p>"
            ),
            task,
            format_note,
            widgets.HTML("<h3>Dataset, checkout & run</h3>"),
            paths,
            widgets.HTML("<h3>Training settings</h3><p class='dfine-help'>Hover over a setting for a short explanation.</p>"),
            official_settings,
            seg_settings,
            command_preview,
            buttons,
            confirmation,
            train_button,
            status,
            progress_panel,
            output,
        ],
        layout=widgets.Layout(width="100%", max_width="100%", overflow="visible"),
    )
    dashboard.add_class("dfine-trainer-dashboard")
    update_mode()
    return dashboard
