from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .dataset import validate_paired_dataset
from .evaluation import DisagreementConfig, evaluate_deblurring
from .inference import export_generator
from .trainer import DeblurTrainer, DeblurTrainingConfig

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = PACKAGE_ROOT / "frontend" / "dist"


class TrainingRequest(BaseModel):
    config: dict[str, Any]


class EvaluationRequest(BaseModel):
    checkpoint: str
    dataset_dir: str
    output_dir: str
    segmentation_model: str | None = None
    split: str = "test"
    image_size: int = 256
    device: str = "cpu"
    class_risk_order: list[str] = Field(default_factory=lambda: ["Class 1", "Class 2", "Class 3"])
    accepting_classes: list[str] = Field(default_factory=lambda: ["Class 1"])


class ExportRequest(BaseModel):
    checkpoint: str
    output_dir: str
    image_size: int = 256
    device: str = "cpu"


def create_app() -> FastAPI:
    app = FastAPI(title="Skripsie Model Trainer", version="2.0.0")
    jobs: dict[str, dict[str, Any]] = {}
    cancel_events: dict[str, threading.Event] = {}
    lock = threading.Lock()
    training_lock = threading.Lock()
    if (FRONTEND_DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    def set_job(job_id: str, values: dict[str, Any]) -> None:
        with lock:
            if job_id in jobs:
                jobs[job_id].update(values)

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}

    @app.post("/api/deblur/validate")
    def validate(payload: dict[str, str]) -> dict[str, Any]:
        try: return validate_paired_dataset(payload["dataset_root"])
        except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/deblur/training/jobs")
    def start_training(payload: TrainingRequest) -> dict[str, str]:
        try: config = DeblurTrainingConfig.from_dict(payload.config); config.validate()
        except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc
        job_id, event = str(uuid.uuid4()), threading.Event()
        with lock:
            jobs[job_id] = {"job_id": job_id, "kind": "training", "state": "queued", "logs": [], "epoch": 0, "batch": 0, "result": None, "error": None, "run_dir": str(Path(config.output_dir).expanduser().resolve() / config.run_name)}
            cancel_events[job_id] = event

        def log(row: dict[str, Any]) -> None:
            with lock:
                job = jobs[job_id]; job["logs"] = [*job["logs"][-499:], row]
                for key in ("epoch", "batch", "batches", "losses", "validation", "checkpoints"):
                    if key in row: job[key] = row[key]

        def run() -> None:
            try:
                with training_lock:
                    set_job(job_id, {"state": "running"})
                    result = DeblurTrainer(config, log, event.is_set).fit()
                set_job(job_id, {"state": result["state"], "result": result})
            except Exception as exc:
                recovery = Path(config.output_dir).expanduser().resolve() / config.run_name / "last_checkpoint.pt"
                set_job(job_id, {
                    "state": "failed", "error": str(exc),
                    "recovery_checkpoint": str(recovery) if recovery.is_file() else None,
                })

        threading.Thread(target=run, name=f"deblur-train-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id}

    @app.post("/api/deblur/evaluation/jobs")
    def start_evaluation(payload: EvaluationRequest) -> dict[str, str]:
        job_id, event = str(uuid.uuid4()), threading.Event()
        with lock:
            jobs[job_id] = {"job_id": job_id, "kind": "evaluation", "state": "queued", "logs": [], "result": None, "error": None}
            cancel_events[job_id] = event

        def run() -> None:
            try:
                set_job(job_id, {"state": "running"})
                if event.is_set():
                    set_job(job_id, {"state": "cancelled"}); return
                result = evaluate_deblurring(
                    payload.checkpoint, payload.dataset_dir, payload.output_dir, segmentation_model=payload.segmentation_model,
                    split=payload.split, image_size=payload.image_size, device=payload.device,
                    config=DisagreementConfig(class_risk_order=tuple(payload.class_risk_order), accepting_classes=tuple(payload.accepting_classes)),
                )
                set_job(job_id, {"state": "complete", "result": result})
            except Exception as exc:
                set_job(job_id, {"state": "failed", "error": str(exc)})

        threading.Thread(target=run, name=f"deblur-eval-{job_id[:8]}", daemon=True).start()
        return {"job_id": job_id}

    @app.get("/api/deblur/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        with lock:
            if job_id not in jobs: raise HTTPException(status_code=404, detail="Unknown deblurring job.")
            return dict(jobs[job_id])

    @app.post("/api/deblur/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict[str, Any]:
        with lock:
            if job_id not in jobs: raise HTTPException(status_code=404, detail="Unknown deblurring job.")
            cancel_events[job_id].set(); jobs[job_id]["state"] = "cancelling"
            return dict(jobs[job_id])

    @app.get("/api/deblur/jobs/{job_id}/preview")
    def job_preview(job_id: str) -> FileResponse:
        with lock:
            if job_id not in jobs: raise HTTPException(status_code=404, detail="Unknown deblurring job.")
            run_dir = jobs[job_id].get("run_dir")
        previews = sorted((Path(run_dir) / "previews").glob("epoch-*.png")) if run_dir else []
        if not previews: raise HTTPException(status_code=404, detail="No validation preview has been written yet.")
        return FileResponse(previews[-1], media_type="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/deblur/export")
    def export(payload: ExportRequest) -> dict[str, Any]:
        try: return export_generator(payload.checkpoint, payload.output_dir, payload.image_size, payload.device)
        except Exception as exc: raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        path = FRONTEND_DIST / "index.html"
        if not path.is_file():
            return "<h1>Skripsie Model Trainer</h1><p>Build model_training/frontend with npm run build.</p>"
        return path.read_text(encoding="utf-8")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def fallback(path: str) -> str:
        index_path = FRONTEND_DIST / "index.html"
        if not index_path.is_file(): raise HTTPException(status_code=404, detail="Frontend is not built.")
        return index_path.read_text(encoding="utf-8")

    return app
