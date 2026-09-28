import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { buildTrainingConfig, parseClassList } from "./config.js";
import "./styles.css";

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(data.detail || response.statusText);
  return data;
}

const number = (value) => Number(value);

function LegacyTraining() {
  return <main className="legacy"><section className="card"><p className="eyebrow">Existing workflow</p><h1>Detection & segmentation</h1><p>YOLO26-N and D-FINE-N training remains in the existing notebook dashboard and command-line workflow. Deblurring is isolated in its own package and does not alter RGB, HSV, export, training, or deployment behavior.</p><code>jupyter lab notebooks/model_trainer_ui.ipynb</code></section></main>;
}

function Field({ label, children }) { return <label className="field"><span>{label}</span>{children}</label>; }

function Deblurring() {
  const [health, setHealth] = useState(null);
  const [section, setSection] = useState("train");
  const [form, setForm] = useState({
    architecture: "GhostDeblurGAN", dataset_root: "", output_dir: "runs/deblur", run_name: "ghost_deblurgan",
    image_size: 256, epochs: 200, batch_size: 4, learning_rate: 0.0001, final_learning_rate: 0.000001,
    weight_decay: 0, workers: 0, seed: 42, device: "cpu", amp: true, generator_updates: 1,
    discriminator_updates: 1, linear_decay_start_epoch: 10, perceptual_weight: 0.7, l1_weight: 0.3,
    adversarial_weight: 0.005, vgg19_weights_path: "", resume_checkpoint: "", auto_resume: true,
    checkpoint_interval_steps: 100
  });
  const [validation, setValidation] = useState(null);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [evaluation, setEvaluation] = useState({ checkpoint: "", dataset_dir: "", output_dir: "runs/deblur_evaluation", segmentation_model: "", split: "test", image_size: 256, device: "cpu", class_risk_order: "Class 1, Class 2, Class 3", accepting_classes: "Class 1" });
  const [evaluationJob, setEvaluationJob] = useState(null);
  const [exportForm, setExportForm] = useState({ checkpoint: "", output_dir: "runs/deblur_export", image_size: 256, device: "cpu" });
  const [exportResult, setExportResult] = useState(null);
  const pollRef = useRef(null);
  useEffect(() => { api("/api/health").then((data) => { setHealth(data); setForm((value) => ({ ...value, device: data.cuda_available ? "cuda" : "cpu" })); }).catch((reason) => setError(reason.message)); }, []);
  useEffect(() => () => window.clearTimeout(pollRef.current), []);
  const set = (name, value) => setForm((current) => ({ ...current, [name]: value }));
  const poll = async (id, setter, callback) => {
    try {
      const status = await api(`/api/deblur/jobs/${id}`); setter(status);
      if (["queued", "running", "cancelling"].includes(status.state)) pollRef.current = window.setTimeout(() => poll(id, setter, callback), 500);
      else callback?.(status);
    } catch (reason) { setError(reason.message); }
  };
  const validate = async () => {
    setError("");
    try { setValidation(await api("/api/deblur/validate", { method: "POST", body: JSON.stringify({ dataset_root: form.dataset_root }) })); }
    catch (reason) { setError(reason.message); }
  };
  const start = async () => {
    setError("");
    try { const started = await api("/api/deblur/training/jobs", { method: "POST", body: JSON.stringify({ config: buildTrainingConfig(form) }) }); poll(started.job_id, setJob); }
    catch (reason) { setError(reason.message); }
  };
  const cancel = async () => { if (job?.job_id) await api(`/api/deblur/jobs/${job.job_id}/cancel`, { method: "POST", body: "{}" }); };
  const startEvaluation = async () => {
    setError("");
    try {
      const payload = { ...evaluation, image_size: number(evaluation.image_size), segmentation_model: evaluation.segmentation_model || null, class_risk_order: parseClassList(evaluation.class_risk_order), accepting_classes: parseClassList(evaluation.accepting_classes) };
      const started = await api("/api/deblur/evaluation/jobs", { method: "POST", body: JSON.stringify(payload) }); poll(started.job_id, setEvaluationJob);
    } catch (reason) { setError(reason.message); }
  };
  const runExport = async () => {
    setError(""); setExportResult(null);
    try { setExportResult(await api("/api/deblur/export", { method: "POST", body: JSON.stringify({ ...exportForm, image_size: number(exportForm.image_size) }) })); }
    catch (reason) { setError(reason.message); }
  };
  const lastEpoch = [...(job?.logs || [])].reverse().find((row) => row.validation);
  const losses = job?.losses || lastEpoch?.losses || {};
  const metrics = job?.validation || lastEpoch?.validation || {};
  const previewUrl = job?.job_id && (job.epoch || lastEpoch) ? `/api/deblur/jobs/${job.job_id}/preview?t=${job.epoch || 0}` : null;
  return <main className="deblur"><header className="hero"><div><p className="eyebrow">Paired motion restoration</p><h1>Deblurring</h1><p>Train the Ghost-DeblurGAN baseline or the explicitly non-official AGG paper reproduction, then test restoration against segmentation behavior.</p></div><div className="runtime"><strong>{health?.cuda_available ? health.cuda_device : "CPU"}</strong><span>PyTorch {health?.torch || "…"}</span></div></header>
    <nav className="subtabs"><button className={section === "train" ? "active" : ""} onClick={() => setSection("train")}>Training</button><button className={section === "evaluate" ? "active" : ""} onClick={() => setSection("evaluate")}>Hallucination & segmentation evaluation</button><button className={section === "export" ? "active" : ""} onClick={() => setSection("export")}>Inference export</button></nav>
    {error && <div className="error">{error}</div>}
    {section === "train" && <div className="columns"><section className="card form"><h2>Run configuration</h2>
      <div className="grid"><Field label="Architecture"><select value={form.architecture} onChange={(event) => set("architecture", event.target.value)}><option>GhostDeblurGAN</option><option>AGGDeblurGANReproduction</option></select></Field><Field label="Run name"><input value={form.run_name} onChange={(event) => set("run_name", event.target.value)} /></Field></div>
      {form.architecture === "AGGDeblurGANReproduction" && <p className="warning">Paper-guided reproduction only. It is not an official AGG-DeblurGAN implementation; ambiguous block details are documented in the model config.</p>}
      <Field label="Paired dataset path"><input value={form.dataset_root} onChange={(event) => set("dataset_root", event.target.value)} placeholder="…/deblur_dataset" /></Field>
      <div className="actions"><button onClick={validate}>Validate dataset</button>{validation && <span className={validation.valid ? "ok" : "bad"}>{validation.valid ? "Valid" : "Invalid"} · {validation.pair_count} pairs</span>}</div>
      {validation && <small>{validation.split_counts.train} train · {validation.split_counts.val} val · {validation.split_counts.test} test {validation.errors.length ? `· ${validation.errors.join("; ")}` : ""}</small>}
      <Field label="Output parent"><input value={form.output_dir} onChange={(event) => set("output_dir", event.target.value)} /></Field>
      <div className="grid three">{[["image_size", "Image size"], ["epochs", "Epochs"], ["batch_size", "Batch size"], ["learning_rate", "Learning rate"], ["workers", "Workers"], ["seed", "Seed"]].map(([name, label]) => <Field key={name} label={label}><input type="number" step={name === "learning_rate" ? "0.00001" : "1"} value={form[name]} onChange={(event) => set(name, event.target.value)} /></Field>)}</div>
      <div className="grid"><Field label="Device"><select value={form.device} onChange={(event) => set("device", event.target.value)}><option value="cpu">CPU</option><option value="cuda" disabled={!health?.cuda_available}>CUDA</option></select></Field><Field label="AMP"><select value={String(form.amp)} onChange={(event) => set("amp", event.target.value === "true")}><option value="true">Enabled on CUDA</option><option value="false">Disabled</option></select></Field></div>
      <details><summary>Losses and schedule</summary><div className="grid three">{[["perceptual_weight", "Perceptual"], ["l1_weight", "L1"], ["adversarial_weight", "Adversarial"], ["final_learning_rate", "Final LR"], ["linear_decay_start_epoch", "Decay begins"], ["weight_decay", "Weight decay"], ["generator_updates", "G updates"], ["discriminator_updates", "D updates"]].map(([name, label]) => <Field key={name} label={label}><input type="number" step="any" value={form[name]} onChange={(event) => set(name, event.target.value)} /></Field>)}</div><Field label="Local VGG19 state_dict (blank uses test-safe fallback)"><input value={form.vgg19_weights_path} onChange={(event) => set("vgg19_weights_path", event.target.value)} /></Field></details>
      <div className="grid"><Field label="Automatic recovery"><select value={String(form.auto_resume)} onChange={(event) => set("auto_resume", event.target.value === "true")}><option value="true">Resume this run automatically</option><option value="false">Always start from scratch</option></select></Field><Field label="Checkpoint every training steps"><input type="number" min="1" step="1" value={form.checkpoint_interval_steps} onChange={(event) => set("checkpoint_interval_steps", event.target.value)} /></Field></div>
      <Field label="Resume full checkpoint"><input value={form.resume_checkpoint} onChange={(event) => set("resume_checkpoint", event.target.value)} placeholder="…/last_checkpoint.pt" /></Field>
      <small>Recovery checkpoints are also saved after every epoch and when training is cancelled or raises an error. An explicit checkpoint path takes precedence.</small>
      <div className="actions"><button className="primary" onClick={start} disabled={["queued", "running", "cancelling"].includes(job?.state)}>Start training</button><button onClick={cancel} disabled={!job || !["queued", "running"].includes(job.state)}>Cancel</button></div>
    </section><section className="card status"><h2>Status <span className={`pill ${job?.state || "idle"}`}>{job?.state || "idle"}</span></h2>
      <div className="progress"><span style={{ width: `${Math.min(100, 100 * number(job?.epoch || 0) / number(form.epochs || 1))}%` }} /></div><p>Epoch {job?.epoch || 0}/{form.epochs} · batch {job?.batch || 0}/{job?.batches || 0}</p>
      <h3>Current losses</h3><div className="metrics">{Object.entries(losses).map(([name, value]) => <div key={name}><span>{name}</span><strong>{Number(value).toFixed(5)}</strong></div>)}</div>
      <h3>Validation metrics</h3><div className="metrics">{Object.entries(metrics).filter(([, value]) => value != null).map(([name, value]) => <div key={name}><span>{name}</span><strong>{Number(value).toFixed(4)}</strong></div>)}</div>
      {previewUrl && <img className="preview" src={previewUrl} alt="Blurred, restored, and sharp validation preview" />}
      {(job?.checkpoints || job?.result) && <div className="paths"><small>Best: {job?.checkpoints?.best_generator || job?.result?.best_generator || "not selected yet"}</small><small>Last: {job?.checkpoints?.last_checkpoint || job?.result?.last_checkpoint}</small></div>}
      {job?.recovery_checkpoint && <div className="paths"><small>Recovery checkpoint: {job.recovery_checkpoint}</small></div>}
      <h3>Live logs</h3><pre>{(job?.logs || []).slice(-80).map((row) => `${row.level}: ${row.message}`).join("\n")}</pre>
    </section></div>}
    {section === "evaluate" && <section className="card evaluation"><h2>GAN hallucination and segmentation disagreement</h2><p>Runs the same segmentation model on the original blurred crop and its restoration. A restored prediction is never automatically trusted.</p>
      <div className="grid">{[["checkpoint", "Generator checkpoint"], ["dataset_dir", "Paired dataset or real crop folder"], ["output_dir", "Output parent"], ["segmentation_model", "YOLO segmentation model (optional)"], ["split", "Dataset split"], ["image_size", "Image size"], ["class_risk_order", "Class risk order, low to high"], ["accepting_classes", "Accepting classes"]].map(([name, label]) => <Field key={name} label={label}><input value={evaluation[name]} onChange={(event) => setEvaluation({ ...evaluation, [name]: event.target.value })} /></Field>)}</div>
      <div className="actions"><button className="primary" onClick={startEvaluation} disabled={["queued", "running"].includes(evaluationJob?.state)}>Run evaluation</button></div><p>Status: <span className={`pill ${evaluationJob?.state || "idle"}`}>{evaluationJob?.state || "idle"}</span></p>
      {evaluationJob?.result && <div className="result"><strong>{evaluationJob.result.flagged_count} of {evaluationJob.result.sample_count} samples flagged</strong><span>{evaluationJob.result.output_dir}</span><p>Inspect disagreements.csv/json and before/after overlays before accepting a deblurred result.</p></div>}
    </section>}
    {section === "export" && <section className="card evaluation"><h2>Generator-only export</h2><p>Exports no discriminator. PyTorch and TorchScript are always attempted; ONNX uses fixed NCHW input at opset 17 and is numerically compared when ONNX Runtime is installed.</p><div className="grid">{Object.entries(exportForm).map(([name, value]) => <Field key={name} label={name.replaceAll("_", " ")}><input value={value} onChange={(event) => setExportForm({ ...exportForm, [name]: event.target.value })} /></Field>)}</div><button className="primary" onClick={runExport}>Export</button>{exportResult && <pre>{JSON.stringify(exportResult, null, 2)}</pre>}</section>}
  </main>;
}

function App() {
  const [tab, setTab] = useState("deblur");
  return <><header className="top"><strong>Skripsie Model Trainer</strong><nav><button className={tab === "models" ? "active" : ""} onClick={() => setTab("models")}>Model Training</button><button className={tab === "deblur" ? "active" : ""} onClick={() => setTab("deblur")}>Deblurring</button></nav></header>{tab === "models" ? <LegacyTraining /> : <Deblurring />}</>;
}

createRoot(document.getElementById("root")).render(<App />);
