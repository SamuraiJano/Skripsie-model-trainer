"""Dashboard for the frozen detection-only experiment; no segmentation controls."""
from __future__ import annotations

import html
import json
from pathlib import Path
import threading
import time


class ConsoleTail:
    """Bound notebook memory; complete console logs remain in each run folder."""
    def __init__(self, output, limit=60000):
        self.output, self.limit, self.buffer, self.updated = output, limit, "", 0.

    def write(self, text):
        self.buffer = (self.buffer + text)[-self.limit:]
        if time.monotonic() - self.updated >= .25:
            self.flush()

    def flush(self):
        self.output.outputs = ({"output_type": "stream", "name": "stdout", "text": self.buffer},)
        self.updated = time.monotonic()


def create_detector_dashboard(project_root=None):
    import ipywidgets as widgets
    from IPython.display import display
    from .detector_experiment import baseline_recipes, prepare_experiment, verify_experiment, matrix_text, execute_experiment

    project = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    dataset = widgets.Text(value=str(project / "dataset_detect_rgb"), description="RGB source:", layout=widgets.Layout(width="95%"))
    upstream = widgets.Text(value=str(project / "third_party/D-FINE"), description="Official repo:", layout=widgets.Layout(width="95%"))
    output = widgets.Text(value=str(project / "runs/controlled_detection"), description="New results:", layout=widgets.Layout(width="95%"))
    seeds = widgets.Text(value="0,1,2,3,4", description="Paired seeds:")
    reason = widgets.Text(value="", description="Fewer reason:", layout=widgets.Layout(width="95%"))
    freeze = widgets.Button(description="Verify data & freeze matrix", layout=widgets.Layout(width="240px"))
    run = widgets.Button(description="Run frozen detector experiment", disabled=True, button_style="warning", layout=widgets.Layout(width="260px"))
    status = widgets.HTML("<b>Split verification: pending.</b> Prepare the matrix before training.")
    matrix = widgets.HTML()
    log = widgets.Output(layout=widgets.Layout(max_height="360px", overflow="auto"))
    console = ConsoleTail(log)
    state = {"manifest": None, "busy": False}
    recipes = baseline_recipes()
    rows = "".join(f"<tr><td>{html.escape(r['architecture'])}</td><td>RGB and HSV_FULL</td><td>Nano</td>"
                   f"<td>{r['training']['batch']}</td><td>{r['nominal_accumulation']} (native)</td><td>{r['nominal_effective_batch']}</td>"
                   f"<td>{r['training']['optimizer']}</td><td>{r['training'].get('lr0', 'head 0.0008; backbone 0.0004')}</td></tr>"
                   for r in recipes.values())
    settings = widgets.HTML("<h3>Controlled detector comparison</h3>"
                            "<p><b>Max epochs 300 · Minimum 100 · Patience 40 · Image size 640 × 640 · Workers 0 · AMP on</b></p>"
                            "<p>Validation every epoch; common COCO box mAP50–95. Colour augmentation off. "
                            "Geometry: aspect-preserving padding; horizontal/vertical flips, rotation, perspective, scale jitter, Mosaic, MixUp and CutMix off.</p>"
                            "<table><tr><th>Architecture</th><th>Representation</th><th>Size</th><th>Physical batch</th><th>Accumulation</th>"
                            "<th>Nominal effective batch</th><th>Optimiser</th><th>Initial LR</th></tr>" + rows + "</table>"
                            "<p>Native accumulation is recorded during training, including warm-up changes and smaller final batches. "
                            "Each seed starts from its original pretrained checkpoint. Test data is reserved for the final evaluation.</p>")

    def changed(_):
        if not state["busy"]:
            state["manifest"] = None
            run.disabled = True
            status.value = "<b>Settings changed: verify and freeze a new matrix.</b>"
    for control in (dataset, upstream, output, seeds, reason):
        control.observe(changed, names="value")

    def task(action):
        if state["busy"]:
            return
        state["busy"] = True
        freeze.disabled = run.disabled = True
        for control in (dataset, upstream, output, seeds, reason):
            control.disabled = True
        def work():
            try:
                action()
            except BaseException as error:
                state["manifest"] = None
                status.value = "<b style='color:#b91c1c'>WARNING — pair, dataset or run verification failed:</b> " + html.escape(str(error))
                console.write(str(error) + "\n")
            finally:
                console.flush()
                state["busy"] = False
                freeze.disabled = False
                run.disabled = state["manifest"] is None
                for control in (dataset, upstream, output, seeds, reason):
                    control.disabled = False
        threading.Thread(target=work, daemon=True).start()

    def prepare(_):
        # Snapshot form values before the worker starts.
        values = (dataset.value, upstream.value, output.value, seeds.value, reason.value)
        def action():
            path = prepare_experiment(*values[:3], seeds=[int(s.strip()) for s in values[3].split(",")],
                                      fewer_seeds_reason=values[4] or None)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            verification = verify_experiment(manifest)
            state["manifest"] = path
            split = verification["split_integrity"]
            warning = split.get("warning")
            status.value = "<b>RGB/HSV recipes, labels, dimensions, pixels and split hashes: PASS.</b><br>" + \
                ("<b style='color:#b45309'>Grouping warning:</b> " + html.escape(warning) + "<br>" if warning else "") + \
                "Frozen configuration: " + html.escape(str(path))
            matrix.value = "<pre style='white-space:pre-wrap'>" + html.escape(matrix_text(manifest)) + "</pre>"
        task(action)

    def start(_):
        path = state["manifest"]
        if path is None:
            return
        def action():
            result = execute_experiment(path, output_callback=console.write)
            status.value = "<b>Experiment complete.</b> Comparison: " + html.escape(str(result))
        task(action)

    freeze.on_click(prepare)
    run.on_click(start)
    view = widgets.VBox([settings, dataset, upstream, output, widgets.HBox([seeds, reason]),
                         widgets.HBox([freeze, run]), status, matrix, log])
    display(view)
    return view
