"""Freeze, verify, schedule and aggregate the final detection-only experiment."""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .colour import file_sha256, verify_paired_datasets
from .detection_data import prepare_detector_pairs
from .detection_metrics import EVALUATION_POLICY
from .experiment_tracking import dataset_identity, dataset_split_integrity
from .runtime import library_versions
from .train import YoloTrainingOptions, _run_training_process


def experiment_versions():
    from importlib.metadata import version
    return {**library_versions(), **{name: version(name) for name in ("faster-coco-eval", "PyYAML")}}

SEEDS = [0, 1, 2, 3, 4]
MODEL_IDS = ("yolo26n", "dfine_n")
POLICY = {"max_epochs": 300, "min_epochs": 100, "patience": 40, "validation_frequency": 1,
          "selection_metric": "box COCO mAP50-95", "selection_split": "val", "min_delta": 0.}
SUMMARY_FIELDS = ("completed_epochs", "best_validation_mAP50-95", "test_mAP50-95", "test_mAP50",
                  "precision", "recall", "F1", "selected_confidence", "training_seconds", "physical_batch",
                  "gradient_accumulation", "effective_batch")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def implementation_hash(upstream: Path):
    here = Path(__file__).parent
    files = sorted(here.glob("*.py"))
    files += sorted((here.parents[1] / "scripts").glob("controlled_detector_*.py"))
    files += sorted((upstream / "src").rglob("*.py")) + sorted((upstream / "configs").rglob("*.yml"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode() + path.read_bytes())
    return digest.hexdigest()


def baseline_recipes(horizontal_flip=0.):
    yolo = asdict(YoloTrainingOptions())
    yolo.pop("min_epochs_before_early_stop")
    extra = yolo.pop("extra_args")
    yolo.update(extra)
    yolo.update(mosaic=0., degrees=0., translate=0., scale=0., shear=0., perspective=0.,
                fliplr=horizontal_flip, flipud=0., hsv_h=0., hsv_s=0., hsv_v=0., bgr=0.,
                mixup=0., cutmix=0., copy_paste=0., max_det=100, conf=0.001, rect=False,
                batch=16, workers=0, epochs=300, imgsz=640, amp=True, task="detect", resume=False)
    return {
        "yolo26n": {"architecture": "Ultralytics YOLO26n", "model_size": "Nano", "training": yolo,
                     "recipe_source": "existing Nano AdamW transfer recipe; conservative conveyor geometry",
                     "nominal_accumulation": 1, "nominal_effective_batch": 16},
        "dfine_n": {"architecture": "official Peterande D-FINE-N", "model_size": "Nano",
                     "training": {"batch": 8, "workers": 0, "epochs": 300, "imgsz": 640, "amp": True,
                                  "optimizer": "AdamW", "head_lr": 0.0008, "backbone_lr": 0.0004,
                                  "weight_decay": 0.0001, "clip_max_norm": 0.1,
                                  "lr_scheduler": {"type": "MultiStepLR", "milestones": [500], "gamma": 0.1},
                                  "warmup_steps": 500, "ema": {"decay": 0.9999, "warmups": 1000},
                                  "horizontal_flip": horizontal_flip, "stage_transition_epoch": 280, "checkpoint_frequency": 300},
                     "recipe_source": "official Nano custom COCO fine-tuning optimiser and scheduler; final 20-epoch stage retained",
                     "nominal_accumulation": 1, "nominal_effective_batch": 8},
    }


def validate_tuning(ledger):
    if ledger.get("maximum_trials_per_architecture") != 3 or ledger.get("test_split_used") is not False:
        raise ValueError("Tuning must have an equal three-trial ceiling and use no test data")
    trials = ledger.get("trials", {})
    if set(trials) != set(MODEL_IDS) or len(trials["yolo26n"]) != len(trials["dfine_n"]):
        raise ValueError("Both architectures must receive the same completed tuning opportunity")
    if len(trials["yolo26n"]) > 3:
        raise ValueError("Tuning exceeds the frozen three-trial budget")
    for architecture in MODEL_IDS:
        if trials[architecture] and sum(t.get("selected") is True for t in trials[architecture]) != 1:
            raise ValueError("Exactly one trial must be selected for each tuned architecture")
        for trial in trials[architecture]:
            if trial.get("selection_split") != "val" or "configuration" not in trial or "selected" not in trial:
                raise ValueError("Every selected/discarded tuning trial must record its configuration and validation-only selection")
            if trial.get("test_metrics") is not None:
                raise ValueError("Tuning ledger contains held-out test results")
            if type(trial["selected"]) is not bool or not isinstance(trial.get("validation_mAP50-95"), (int, float)) or not 0 <= trial["validation_mAP50-95"] <= 1:
                raise ValueError("Every tuning trial must record its validation AP and a boolean selected/discarded decision")
    if trials["yolo26n"]:
        protocol = ledger.get("trial_protocol", {})
        if protocol.get("colour") != "rgb" or protocol.get("seed") != 0 or not 1 <= protocol.get("max_epochs", 0) <= 100:
            raise ValueError("Tuning needs one common RGB/seed-0 protocol with a 1-100 epoch ceiling")


def apply_selected_trials(recipes, ledger):
    allowed = {"yolo26n": {"optimizer", "lr0", "lrf", "weight_decay", "warmup_epochs", "warmup_momentum", "warmup_bias_lr", "cos_lr"},
               "dfine_n": {"head_lr", "backbone_lr", "weight_decay", "clip_max_norm", "warmup_steps", "lr_scheduler", "ema"}}
    for architecture in MODEL_IDS:
        for trial in ledger["trials"][architecture]:
            if not isinstance(trial["configuration"], dict) or set(trial["configuration"]) - allowed[architecture]:
                raise ValueError("Tuning configurations may override only documented architecture-specific optimisation parameters")
            if trial["selected"]:
                recipes[architecture]["training"].update(copy.deepcopy(trial["configuration"]))
                recipes[architecture]["recipe_source"] = "Selected by equal-budget validation-only preliminary trials; see tuning ledger"


def verify_experiment(manifest: dict, *, check_files=True):
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest.get("manifest_sha256") != digest_json(unsigned):
        raise ValueError("Frozen experiment manifest was modified")
    if manifest["policy"] != POLICY or manifest["evaluation"] != EVALUATION_POLICY:
        raise ValueError("Experiment duration, validation selection or evaluator differs from the controlled protocol")
    seeds = manifest["seeds"]
    if seeds != SEEDS and not manifest.get("fewer_seeds_reason"):
        raise ValueError("Fewer paired seeds require a recorded computational reason")
    expected = {(m, c, s) for m in MODEL_IDS for c in ("rgb", "hsv") for s in seeds}
    actual = [(job["model"], job["colour"], job["seed"]) for job in manifest["matrix"]]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Matrix is missing a paired detector/colour/seed run or contains duplicates")
    if set(manifest["recipes"]) != set(MODEL_IDS):
        raise ValueError("Final detection requires YOLO26n and official D-FINE-N only")
    validate_tuning(manifest["tuning"])
    for job in manifest["matrix"]:
        recipe = manifest["recipes"][job["model"]]
        if job["recipe_sha256"] != digest_json(recipe) or recipe["model_size"] != "Nano":
            raise ValueError("RGB/HSV architecture-specific recipes differ")
        training = recipe["training"]
        if training["epochs"] != 300 or training["imgsz"] != 640:
            raise ValueError("Final detector image size or duration differs")
        if job["model"] == "yolo26n":
            if any(training[k] != 0 for k in ("hsv_h", "hsv_s", "hsv_v", "bgr", "mosaic", "mixup", "cutmix")):
                raise ValueError("Controlled YOLO colour/mosaic/mixed-image augmentation must be disabled")
        if job["checkpoint"] != recipe["checkpoint"]:
            raise ValueError("A paired seed does not start from its original pretrained checkpoint")
    if check_files:
        if manifest["implementation_sha256"] != implementation_hash(Path(manifest["dfine_root"])):
            raise ValueError("Training implementation changed after freezing; prepare a separate new experiment")
        if manifest["library_versions"] != experiment_versions():
            raise ValueError("Framework versions changed after freezing")
        pair = manifest["paired_dataset"]
        verify_paired_datasets(Path(pair["rgb"]), Path(pair["hsv"]))
        for colour in ("rgb", "hsv"):
            identity = dataset_identity(Path(pair[colour]))
            for split in ("train", "val", "test"):
                for key in ("identity_sha256", "annotation_sha256"):
                    if identity["splits"][split][key] != pair["dataset_identity"]["splits"][split][key]:
                        raise ValueError("Dataset hashes/splits changed after freezing")
        # Recheck exact RGB/HSV source pixels, dimensions and source membership.
        checked = prepare_detector_pairs(manifest["source_dataset"], Path(manifest["dataset_cache"]))
        if checked["pair_fingerprint"] != pair["pair_fingerprint"] or checked["source_pixel_sha256"] != pair["source_pixel_sha256"]:
            raise ValueError("Source pixels or split configuration changed after freezing")
        if checked["split_integrity"] != pair["split_integrity"]:
            raise ValueError("Physical fruit/track/sequence grouping metadata changed after freezing")
        for key in ("HSV_FULL_pixel_sha256", "paired_config_sha256", "paired_sample_audit_sha256"):
            if checked[key] != pair[key]:
                raise ValueError("Frozen paired representations or source-pixel audit changed")
        for recipe in manifest["recipes"].values():
            if file_sha256(Path(recipe["checkpoint"])) != recipe["checkpoint_sha256"]:
                raise ValueError("Original pretrained checkpoint changed after freezing")
    return {"status": "PASS", "runs": len(actual), "duration": POLICY, "evaluation": EVALUATION_POLICY,
            "pair_status": "PASS", "split_integrity": manifest["paired_dataset"]["split_integrity"]}


def prepare_experiment(source, dfine_root, output_root, *, seeds=None, fewer_seeds_reason=None,
                       horizontal_flip=0., tuning_ledger=None, yolo_checkpoint=None):
    seeds = list(SEEDS if seeds is None else seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if seeds != SEEDS and not fewer_seeds_reason:
        raise ValueError("Use seeds 0-4, or explicitly record why fewer seeds are necessary")
    if horizontal_flip not in {0., 0.5}:
        raise ValueError("Horizontal flip is off, or 0.5 when physically valid")
    upstream = Path(dfine_root).resolve()
    if not (upstream / "configs/dfine/custom/dfine_hgnetv2_n_custom.yml").is_file():
        raise ValueError("Select the official Peterande D-FINE checkout, not D-FINE-seg")
    project = Path(__file__).resolve().parents[2]
    yolo_checkpoint = Path(yolo_checkpoint) if yolo_checkpoint else next(
        (p for p in (project / "yolo26n.pt", project.parent / "yolo26n.pt") if p.is_file()), project / "yolo26n.pt")
    checkpoint = upstream / "pretrained/dfine_n_coco.pth"
    if not yolo_checkpoint.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Cache the normal YOLO26n and official dfine_n_coco.pth checkpoints before freezing")
    if yolo_checkpoint.name != "yolo26n.pt":
        raise ValueError("The final experiment must start from the normal YOLO26n pretrained checkpoint")
    output_root = Path(output_root).resolve()
    cache = output_root / "_datasets"
    pair = prepare_detector_pairs(source, cache)
    directory = output_root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    directory.mkdir(parents=True, exist_ok=False)
    recipes = baseline_recipes(horizontal_flip)
    for model, path in (("yolo26n", yolo_checkpoint.resolve()), ("dfine_n", checkpoint)):
        recipes[model].update(checkpoint=str(path), checkpoint_sha256=file_sha256(path))
    tuning = copy.deepcopy(tuning_ledger) if tuning_ledger else {
        "maximum_trials_per_architecture": 3, "test_split_used": False,
        "trials": {"yolo26n": [], "dfine_n": []},
        "selection": "Frozen published/existing Nano transfer baselines; zero additional tuning trials for either architecture",
    }
    validate_tuning(tuning)
    apply_selected_trials(recipes, tuning)
    tuning["selected_recipes"] = copy.deepcopy(recipes)
    matrix = [{"run_id": f"{model}-{colour}-seed{seed}", "model": model, "colour": colour, "seed": seed,
               "recipe_sha256": digest_json(recipes[model]), "checkpoint": recipes[model]["checkpoint"]}
              for seed in seeds for model in MODEL_IDS for colour in ("rgb", "hsv")]
    manifest = {"schema_version": "controlled-detectors-v1", "created_at": datetime.now(timezone.utc).isoformat(),
                "output_dir": str(directory), "source_dataset": str(Path(source).resolve()), "dataset_cache": str(cache),
                "paired_dataset": pair, "dfine_root": str(upstream), "recipes": recipes, "seeds": seeds,
                "fewer_seeds_reason": fewer_seeds_reason, "policy": POLICY, "evaluation": EVALUATION_POLICY,
                "matrix": matrix, "tuning": tuning, "library_versions": experiment_versions(),
                "hardware_policy": "same NVIDIA RTX 4060 Laptop GPU; CUDA device 0", "implementation_sha256": implementation_hash(upstream)}
    manifest["manifest_sha256"] = digest_json(manifest)
    verify_experiment(manifest, check_files=False)
    path = directory / "experiment.json"
    write_json(path, manifest)
    write_json(directory / "verification.json", verify_experiment(manifest, check_files=False))
    with (directory / "experiment_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        expanded = []
        for job in matrix:
            recipe = recipes[job["model"]]
            t = recipe["training"]
            expanded.append({**job, "architecture": recipe["architecture"], "size": "Nano", "max_epochs": 300,
                             "min_epochs": 100, "patience": 40, "image_size": 640, "batch": t["batch"],
                             "nominal_accumulation": recipe["nominal_accumulation"], "nominal_effective_batch": recipe["nominal_effective_batch"],
                             "optimizer": t["optimizer"], "learning_rates": t.get("lr0", f"head={t.get('head_lr')};backbone={t.get('backbone_lr')}"),
                             "workers": t["workers"], "amp": t["amp"], "colour_augmentation": "off",
                             "horizontal_flip": t.get("fliplr", t.get("horizontal_flip")),
                             "dataset": pair["rgb"], "split_hashes": digest_json(pair["dataset_identity"]["splits"])})
        writer = csv.DictWriter(handle, fieldnames=list(expanded[0]))
        writer.writeheader()
        writer.writerows(expanded)
    return path


def matrix_text(manifest):
    lines = ["Frozen detector experiment: max=300, min=100, patience=40, COCO box AP, 640 px",
             "Run | physical batch | nominal accumulation | nominal effective batch | optimizer | learning rate | workers"]
    for job in manifest["matrix"]:
        recipe = manifest["recipes"][job["model"]]
        t = recipe["training"]
        lr = t.get("lr0", f"head={t.get('head_lr')}, backbone={t.get('backbone_lr')}")
        lines.append(f"{job['run_id']} | {t['batch']} | {recipe['nominal_accumulation']} | {recipe['nominal_effective_batch']} | {t['optimizer']} | {lr} | {t['workers']}")
    lines.append("Split verification: " + json.dumps(manifest["paired_dataset"]["split_integrity"]))
    lines.append("Duplicate annotations removed consistently for both detectors: " + json.dumps(manifest["paired_dataset"]["duplicate_annotations_removed"]))
    return "\n".join(lines)


def execute_experiment(manifest_path, output_callback=print):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = verify_experiment(manifest)
    import torch
    if not torch.cuda.is_available() or "RTX 4060 Laptop" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The frozen controlled experiment requires the NVIDIA RTX 4060 Laptop GPU")
    output_callback(matrix_text(manifest) + "\n")
    write_json(Path(manifest["output_dir"]) / "verification.json", verification)
    worker = Path(__file__).resolve().parents[2] / "scripts/controlled_detector_worker.py"
    for job in manifest["matrix"]:
        directory = Path(manifest["output_dir"]) / job["run_id"]
        metadata = directory / "run_metadata.json"
        if directory.exists():
            if metadata.is_file() and json.loads(metadata.read_text()).get("test_evaluation", {}).get("status") == "complete":
                output_callback(f"Already completed, preserving {job['run_id']}\n")
                continue
            raise FileExistsError(f"Preserving an incomplete prior run: {directory}; prepare a separate experiment to retry")
        directory.mkdir()
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1", PYTHONHASHSEED=str(job["seed"]), CUBLAS_WORKSPACE_CONFIG=":4096:8")
        _run_training_process([sys.executable, str(worker), "--manifest", str(manifest_path), "--run-id", job["run_id"]],
                              cwd=Path(__file__).resolve().parents[2], environment=environment,
                              output_callback=output_callback, output_log=directory / "trainer_console.log")
        aggregate_results(manifest_path)
    return aggregate_results(manifest_path)


def aggregate_results(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    verify_experiment(manifest, check_files=False)
    directory = Path(manifest["output_dir"])
    rows = []
    for job in manifest["matrix"]:
        path = directory / job["run_id"] / "run_metadata.json"
        if not path.is_file():
            continue
        m = json.loads(path.read_text(encoding="utf-8"))
        if m.get("test_evaluation", {}).get("status") != "complete":
            continue
        if m.get("manifest_sha256") != manifest["manifest_sha256"] or m.get("recipe_sha256") != job["recipe_sha256"]:
            raise ValueError("Cannot aggregate a result from a different frozen experiment")
        test, optimisation = m["final_test_metrics"], m["optimisation"]
        rows.append({"model": job["model"], "colour": job["colour"], "seed": job["seed"],
                     "completed_epochs": m["completed_epochs"], "best_validation_mAP50-95": m["best_metric"],
                     "test_mAP50-95": test["mAP50-95"], "test_mAP50": test["mAP50"],
                     **{key: test[key] for key in ("precision", "recall", "F1", "selected_confidence")},
                     "training_seconds": m["training_seconds"], "physical_batch": optimisation["physical_batch_size"],
                     "gradient_accumulation": m.get("observed_accumulation_steps", [optimisation["gradient_accumulation"]])[0]
                         if len(m.get("observed_accumulation_steps", [1])) == 1 else "variable: " + str(m["observed_accumulation_steps"]),
                     "effective_batch": optimisation["effective_batch_size"] if len(m.get("observed_accumulation_steps", [1])) == 1 else "variable"})
    summaries = []
    for model in MODEL_IDS:
        for colour in ("rgb", "hsv"):
            group = [r for r in rows if r["model"] == model and r["colour"] == colour]
            summary = {"model": model, "colour": colour, "completed_seeds": len(group), "expected_seeds": len(manifest["seeds"]),
                       "complete": len(group) == len(manifest["seeds"]), "statistics": {}}
            for field in SUMMARY_FIELDS:
                values = [r[field] for r in group if isinstance(r[field], (int, float))]
                summary["statistics"][field] = {"mean": statistics.mean(values) if values else None,
                                                 "std": statistics.stdev(values) if len(values) > 1 else None,
                                                 "std_definition": "sample standard deviation (ddof=1)"}
            summaries.append(summary)
    write_json(directory / "comparison.json", {"manifest_sha256": manifest["manifest_sha256"], "per_seed": rows,
                                                "summary": summaries, "complete": len(rows) == len(manifest["matrix"])})
    for name, records, fields in (("comparison_per_seed.csv", rows, ["model", "colour", "seed", *SUMMARY_FIELDS]),
                                   ("comparison_summary.csv", [{"model": s["model"], "colour": s["colour"], "completed_seeds": s["completed_seeds"],
                                      **{f"{f}_{k}": s["statistics"][f][k] for f in SUMMARY_FIELDS for k in ("mean", "std")}} for s in summaries],
                                    ["model", "colour", "completed_seeds", *[f"{f}_{k}" for f in SUMMARY_FIELDS for k in ("mean", "std")]])):
        with (directory / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
    return directory / "comparison.json"
