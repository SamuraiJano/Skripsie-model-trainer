from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any

from .datasets import ultralytics_loader_smoke, validate_yolo_training_dataset
from .train import TrainingOptions, train_yolo_export


TRAINING_TASKS = (
    ("Object detection", "detect", "yolo26n.pt"),
    ("Instance segmentation", "segment", "yolo26n-seg.pt"),
)


def _project_root(project_root: str | Path | None) -> Path:
    root = Path(project_root or Path.cwd()).expanduser().resolve()
    if not (root / "src" / "naartjie_cv").is_dir():
        raise ValueError(f"Model Trainer project root is invalid: {root}")
    return root


def _formatted_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str, ensure_ascii=False)


def create_training_dashboard(project_root: str | Path | None = None) -> Any:
    """Build a Jupyter widget dashboard for YOLO detection and segmentation training."""
    try:
        import ipywidgets as widgets
    except ImportError as exc:
        raise RuntimeError(
            "The notebook UI requires ipywidgets in the active Jupyter kernel. Run "
            f"`{sys.executable} -m pip install \"ipywidgets>=8.1\"`, restart the kernel, and rerun the notebook."
        ) from exc

    root = _project_root(project_root)
    task = widgets.ToggleButtons(
        options=[(label, value) for label, value, _ in TRAINING_TASKS],
        value="detect",
        description="Task",
        style={"description_width": "80px"},
    )
    dataset_root = widgets.Text(
        value=str(root / "dataset_detect_rgb"),
        description="Dataset",
        placeholder="Standalone YOLO dataset or complete Data Labeller export",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "80px"},
    )
    output_dir = widgets.Text(
        value=str(root / "runs"),
        description="Runs",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "80px"},
    )
    run_name = widgets.Text(
        value="",
        description="Run name",
        placeholder="Optional; a timestamped name is used when empty",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "80px"},
    )
    resume = widgets.Text(
        value="",
        description="Resume",
        placeholder="Optional path to last.pt",
        layout=widgets.Layout(width="100%"),
        style={"description_width": "80px"},
    )
    epochs = widgets.BoundedIntText(value=100, min=1, max=10000, description="Epochs")
    image_size = widgets.BoundedIntText(value=640, min=32, max=4096, step=32, description="Image size")
    batch = widgets.IntText(value=16, description="Batch")
    workers = widgets.BoundedIntText(value=8, min=0, max=128, description="Workers")
    seed = widgets.IntText(value=42, description="Seed")
    device = widgets.Text(value="0", description="Device", placeholder="0, cpu, or mps")
    amp = widgets.Checkbox(value=True, description="Automatic mixed precision (AMP)")
    confirmation = widgets.Checkbox(
        value=False,
        description="I understand that training can run for a long time",
        indent=False,
    )
    validate_button = widgets.Button(description="Validate dataset", icon="check", button_style="info")
    smoke_button = widgets.Button(description="Smoke-test loader", icon="flask")
    train_button = widgets.Button(description="Start training", icon="play", button_style="success", disabled=True)
    command_preview = widgets.HTML()
    status = widgets.HTML(value="<b>Ready.</b> Validate the selected dataset before training.")
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

    def update_preview(*_: Any) -> None:
        command = [
            "python",
            "scripts/model_trainer.py",
            "yolo-detect" if task.value == "detect" else "yolo-segment",
            dataset_root.value.strip() or "<dataset>",
            "--epochs",
            str(epochs.value),
            "--imgsz",
            str(image_size.value),
            "--batch",
            str(batch.value),
            "--workers",
            str(workers.value),
            "--device",
            device.value.strip() or "0",
            "--seed",
            str(seed.value),
            "--output-dir",
            output_dir.value.strip() or "runs",
        ]
        if run_name.value.strip():
            command.extend(["--run-name", run_name.value.strip()])
        if resume.value.strip():
            command.extend(["--resume", resume.value.strip()])
        if not amp.value:
            command.append("--no-amp")
        command_preview.value = (
            f"<p><b>Base model:</b> <code>{html.escape(model_name())}</code></p>"
            f"<p><b>Equivalent CLI:</b> <code>{html.escape(' '.join(command))}</code></p>"
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
        set_busy(True)
        output.clear_output(wait=True)
        status.value = f"<b>Training {html.escape(task.value)} with {html.escape(model_name())}…</b>"
        options = TrainingOptions(
            epochs=epochs.value,
            imgsz=image_size.value,
            batch=batch.value,
            workers=workers.value,
            seed=seed.value,
            device=device.value.strip() or "0",
            amp=amp.value,
            output_dir=output_dir.value.strip() or str(root / "runs"),
            run_name=run_name.value.strip() or None,
            resume=resume.value.strip() or None,
        )
        try:
            with output:
                result = train_yolo_export(dataset_root.value.strip(), task.value, options)
                print("\nTraining complete:\n" + _formatted_json(result))
                metrics_path = Path(result["final_metrics"])
                if metrics_path.is_file():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    print("\nFinal metrics:\n" + _formatted_json(metrics))
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

    for control in (task, dataset_root, output_dir, run_name, resume, epochs, image_size, batch, workers, seed, device, amp):
        control.observe(update_preview, names="value")
    confirmation.observe(lambda change: set_busy(busy), names="value")
    validate_button.on_click(validate_clicked)
    smoke_button.on_click(smoke_clicked)
    train_button.on_click(train_clicked)
    update_preview()

    settings = widgets.GridBox(
        children=[epochs, image_size, batch, workers, seed, device],
        layout=widgets.Layout(grid_template_columns="repeat(3, minmax(220px, 1fr))", grid_gap="8px"),
    )
    dashboard = widgets.VBox(
        [
            widgets.HTML(
                "<h2>YOLO26 Model Trainer</h2>"
                "<p>Validate and train standalone YOLO datasets or complete Data Labeller exports.</p>"
            ),
            task,
            dataset_root,
            settings,
            amp,
            output_dir,
            run_name,
            resume,
            command_preview,
            widgets.HBox([validate_button, smoke_button]),
            confirmation,
            train_button,
            status,
            output,
        ],
        layout=widgets.Layout(width="100%"),
    )
    dashboard.add_class("model-trainer-dashboard")
    return dashboard
