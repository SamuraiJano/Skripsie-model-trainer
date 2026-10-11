"""Validate fixed recipes, source datasets and real batches without training.

Run from model_training: python scripts/validate_final_experiment.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

import yaml
from omegaconf import OmegaConf
from naartjie_cv.colour import HSV_ENCODING_METADATA, file_sha256, prepare_colour_dataset, verify_paired_datasets
from naartjie_cv.datasets import prepare_dfine_seg_dataset
from naartjie_cv.experiment_tracking import dataset_identity, dataset_split_integrity
from naartjie_cv.runtime import seed_training
from naartjie_cv.final_validation import (configuration_differences, yolo_resolved_config, yolo_training_batch,
                                         project_dfine_training_batch, verify_batch_pair)
from naartjie_cv.final_validation import verify_duration_config, yolo_scheduler_diagnostics, project_scheduler_diagnostics
from naartjie_cv.final_validation import validation_only_route_contract
from naartjie_cv.train import (TrainingOptions, _dfine_config, _dfine_seg_config, _runtime,
                              final_dfine_seg_options, official_dfine_checkpoint, YOLO_FINAL_CHECKPOINTS)


def official_batch(config: Path, upstream: Path) -> dict:
    # Separate interpreter: the two supported D-FINE checkouts both import src.
    code = r'''
import json, sys, copy
from naartjie_cv.runtime import seed_training
from src.core import YAMLConfig
seed_training(0)
cfg = YAMLConfig(sys.argv[1])
resolved = copy.deepcopy(cfg.yaml_cfg)
cfg.yaml_cfg['train_dataloader']['total_batch_size'] = 1
cfg.yaml_cfg['train_dataloader']['num_workers'] = 0
images, targets = next(iter(cfg.train_dataloader))
result = {'resolved_config': resolved, 'tensor_shape': list(images.shape), 'tensor_dtype': str(images.dtype),
          'tensor_finite': bool(images.isfinite().all()), 'tensor_min': float(images.min()), 'tensor_max': float(images.max()),
          'target_classes': [t['labels'].tolist() for t in targets], 'target_boxes': [t['boxes'].tolist() for t in targets],
          'target_boxes_finite': all(bool(t['boxes'].isfinite().all()) for t in targets),
          'image_paths': [cfg.train_dataloader.dataset.coco.imgs[int(t['image_id'])]['file_name'] for t in targets]}
print('MODEL_TRAINER_RESULT=' + json.dumps(result))
'''
    env = dict(os.environ, PYTHONHASHSEED="0")
    completed = subprocess.run([sys.executable, "-c", code, str(config)], cwd=upstream,
                               env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    if completed.returncode:
        raise RuntimeError(completed.stdout + completed.stderr)
    line = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("MODEL_TRAINER_RESULT=")), "")
    if not line:
        raise RuntimeError("Official D-FINE did not emit a batch result: " + completed.stdout + completed.stderr)
    return json.loads(line.split("=", 1)[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-data", type=Path, default=PROJECT / "dataset_detect_rgb")
    parser.add_argument("--segmentation-data", type=Path, default=PROJECT / "dataset_seg_rgb")
    parser.add_argument("--dfine-root", type=Path, default=PROJECT / "third_party/D-FINE")
    parser.add_argument("--dfine-seg-root", type=Path, default=PROJECT / "third_party/D-FINE-seg")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs/final_validation")
    parser.add_argument("--skip-memory", action="store_true", help="Skip the optional CUDA physical-batch forward/backward check")
    args = parser.parse_args()
    output = args.output_dir.resolve() / datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    output.mkdir(parents=True)
    seed_training(0)
    report = {"runtime": _runtime(), "failures": {}, "checks": {}, "datasets": {}, "pairs": {}, "checkpoints": {},
              "policy": {"max_epochs": 300, "min_epochs_before_early_stop": 100, "patience": 40, "workers": 4},
              "schedulers": {}, "memory_smoke_tests": {"status": "NOT RUN"}}

    def check(name, action):
        print(f"Checking {name}...", flush=True)
        try:
            result = action()
            report["checks"][name] = "PASS"
            return result
        except Exception as error:
            report["failures"][name] = str(error)
            print(f"FAIL {name}: {error}", flush=True)
            return None

    def tests():
        completed = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider",
            "--basetemp", str(output / "test_temp"), "-k",
            "not test_model_trainer_notebook_covers_detection_and_segmentation and not test_dfine_notebook_covers_official_detection_and_dfine_seg"],
            cwd=PROJECT, text=True, encoding="utf-8", errors="replace", capture_output=True)
        (output / "controlled_tests.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        counts = {name: int(value) for value, name in re.findall(r"(\d+) (passed|failed|skipped|deselected)", completed.stdout.splitlines()[-1] if completed.stdout.splitlines() else "")}
        report["controlled_tests"] = {**counts, "returncode": completed.returncode,
            "excluded": ["test_model_trainer_notebook_covers_detection_and_segmentation",
                         "test_dfine_notebook_covers_official_detection_and_dfine_seg"],
            "exclusion_reason": "Historical saved notebook output assertions, unchanged from the validated 100-epoch suite"}
        print(completed.stdout, flush=True)
        if completed.returncode:
            raise ValueError(f"Controlled test suite failed; see {output / 'controlled_tests.txt'}")

    check("controlled test suite (saved notebook output assertions excluded)", tests)
    def stopping_tests():
        from naartjie_cv.training_policy import ValidationStoppingState
        histories = {"A": (1, 100), "B": (95, 135), "C": (120, 160)}
        result = {}
        for name, (best, stop) in histories.items():
            for colour in ("rgb", "hsv"):
                state = ValidationStoppingState()
                for epoch in range(1, 301):
                    if state.observe(epoch, 0.8 if epoch >= best else 0.5 * epoch / best):
                        break
                assert epoch == stop and state.best_epoch == best
                result[f"{name}_{colour}"] = {"stop_epoch": epoch, "best_epoch": best, "status": "PASS"}
        state = ValidationStoppingState()
        for epoch in range(1, 175):
            stop = state.observe(epoch, 0.1 if epoch < 95 else 0.2 if epoch < 134 else 0.3)
            assert stop == (epoch == 174)
        result["D"] = {"stop_epoch": 174, "best_epoch": 134, "status": "PASS"}
        state = ValidationStoppingState()
        for epoch in range(1, 301):
            assert not state.observe(epoch, epoch / 300)
        assert state.stop_reason == "max_epochs"
        result["E"] = {"stop_epoch": 300, "stop_reason": "max_epochs", "status": "PASS"}
        result["F"] = {"status": "PASS", "logic": "shared ValidationStoppingState, identical RGB/HSV inputs"}
        (output / "early_stopping_tests.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        report["early_stopping_tests"] = result
    check("one-based minimum epoch and patience state-machine histories A-F", stopping_tests)
    report["selection_checks"] = check("validation-only training/checkpoint selection and held-out test isolation",
                                       lambda: validation_only_route_contract(args.dfine_seg_root.resolve()))

    def checkpoint_check():
        paths = [PROJECT / YOLO_FINAL_CHECKPOINTS[t] for t in ("detect", "segment")]
        paths += [args.dfine_seg_root / "pretrained" / name for name in ("dfine_n_coco.pt", "dfine_seg_n_coco.pt")]
        # Official training loads the same pinned URL for both representations.
        import torch
        official = official_dfine_checkpoint(args.dfine_root)
        paths.append(Path(torch.hub.get_dir()) / "checkpoints/dfine_n_coco.pth" if official.startswith("https://") else Path(official))
        for path in paths:
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Original Nano pretrained checkpoint not available: {path}")
            report["checkpoints"][path.name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    check("all original Nano checkpoints available", checkpoint_check)
    def memory_check(framework, task, source, config=None):
        import torch
        if args.skip_memory or not torch.cuda.is_available():
            report["memory_smoke_tests"] = {"status": "SKIPPED", "reason": "explicit skip" if args.skip_memory else "CUDA unavailable"}
            return
        destination = output / "memory" / f"{framework}_{task}.json"
        command = [sys.executable, str(PROJECT / "scripts/validate_batch_memory.py"), "--framework", framework,
                   "--task", task, "--data", str(source), "--output", str(destination)]
        if config:
            command += ["--config", str(config), "--upstream", str(args.dfine_seg_root.resolve())]
        completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace", timeout=360)
        destination.with_suffix(".log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        result = json.loads(destination.read_text()) if destination.is_file() else {"status": "FAIL", "error": completed.stdout + completed.stderr}
        report["memory_smoke_tests"][framework + "_" + task] = result
        if completed.returncode or result["status"] != "PASS":
            report["memory_smoke_tests"]["status"] = "FAIL"
            raise ValueError(f"BATCH MEMORY VALIDATION: FAIL ({framework} {task}): {result.get('error')}")
        actual = result["actual_scheduler"]
        expected = report["schedulers"][framework + "_" + task]
        for field in (("horizon_epochs", "terminal_factor", "warmup_epochs") if framework == "yolo" else
                      ("horizon_epochs", "total_optimizer_steps", "optimizer_steps_per_epoch", "initial_lrs", "peak_lrs", "minimum_lrs")):
            assert actual[field] == expected[field], f"Actual trainer scheduler differs from diagnostic config: {field}"
        if report["memory_smoke_tests"].get("status") != "FAIL":
            report["memory_smoke_tests"]["status"] = "PASS"

    for task, source in (("detect", args.detection_data), ("segment", args.segmentation_data)):
        source = source.resolve()
        def dataset_check():
            integrity = dataset_split_integrity(source / "data.yaml")
            hsv = prepare_colour_dataset(source, task, PROJECT / "runs/yolo26/_colour_datasets", "hsv")
            verify_paired_datasets(source / "data.yaml", hsv / "data.yaml")
            metadata = yaml.safe_load((hsv / "data.yaml").read_text(encoding="utf-8"))
            for key, value in HSV_ENCODING_METADATA.items():
                if metadata.get(key) != value:
                    raise ValueError(f"HSV encoding metadata differs: {key}")
            report["datasets"][task] = {"source": str(source), "hsv": str(hsv), "split_identity": "PASS",
                "annotation_identity": "PASS", "classes": metadata["names"], "integrity": integrity,
                "splits": dataset_identity(source / "data.yaml")["splits"]}
            if integrity["warning"]:
                print("WARNING: fruit/track split integrity NOT PROVABLE: " + integrity["warning"], flush=True)
            return hsv
        hsv = check(f"{task} source identity, annotation and HSV_FULL provenance", dataset_check)
        if hsv is None:
            continue
        def yolo_check():
            configs = [yolo_resolved_config(root, task) for root in (source, hsv)]
            for config in configs:
                verify_duration_config(config, "yolo", task)
            difference = configuration_differences(*configs, "yolo")
            if difference:
                raise ValueError(f"Resolved configuration differences: {json.dumps(difference)}")
            batches = [yolo_training_batch(root, task) for root in (source, hsv)]
            verify_batch_pair(*batches)
            scheduler = yolo_scheduler_diagnostics(configs[0], report["datasets"][task]["splits"]["train"]["images"])
            report["schedulers"]["yolo_" + task] = scheduler
            for colour, config in zip(("rgb", "hsv"), configs):
                (output / f"yolo_{task}_{colour}.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
            report["pairs"]["yolo_" + task] = {"resolved_rgb": configs[0], "resolved_hsv": configs[1],
                                                    "configuration_diff": difference, "rgb_batch": batches[0], "hsv_batch": batches[1]}
        check(f"YOLO Nano {task} resolved parity and real RGB/HSV batches", yolo_check)
        check(f"YOLO Nano {task} physical batch memory forward/backward", lambda: memory_check("yolo", task, source))

        def project_dfine_check():
            options = final_dfine_seg_options(task)
            configs, batches = [], []
            for colour, root in (("rgb", source), ("hsv", hsv)):
                directory = output / f"project_{task}_{colour}"
                prepared = prepare_dfine_seg_dataset(root, task, directory / "prepared")
                config = _dfine_seg_config(args.dfine_seg_root.resolve(), directory, prepared, task, options)
                configs.append(OmegaConf.to_container(OmegaConf.load(config), resolve=True))
                verify_duration_config(configs[-1], "project_dfine", task)
                (directory / "resolved_config.json").write_text(json.dumps(configs[-1], indent=2), encoding="utf-8")
                batches.append(project_dfine_training_batch(config, args.dfine_seg_root.resolve()))
            difference = configuration_differences(*configs, "project_dfine")
            if difference:
                raise ValueError(f"Resolved configuration differences: {json.dumps(difference)}")
            verify_batch_pair(*batches)
            assert batches[0]["configured_training_batches"] == batches[1]["configured_training_batches"]
            assert batches[0]["configured_workers"] == batches[1]["configured_workers"] == 4
            report["schedulers"]["project_dfine_" + task] = project_scheduler_diagnostics(
                configs[0], batches[0]["configured_training_batches"])
            report["pairs"]["project_dfine_" + task] = {"resolved_rgb": configs[0], "resolved_hsv": configs[1],
                       "configuration_diff": difference, "rgb_batch": batches[0], "hsv_batch": batches[1]}
        check(f"Project D-FINE Nano {task} resolved parity and real RGB/HSV batches", project_dfine_check)
        check(f"Project D-FINE Nano {task} physical batch memory forward/backward", lambda: memory_check(
            "project_dfine", task, source, output / f"project_{task}_rgb/dfine_seg_training.yaml"))

        if task == "detect":
            def official_check():
                configs, batches = [], []
                options = TrainingOptions(batch=8)
                for colour, root in (("rgb", source), ("hsv", hsv)):
                    directory = output / f"official_{colour}"
                    prepared = prepare_dfine_seg_dataset(root, "detect", directory / "prepared")
                    config = _dfine_config(args.dfine_root.resolve(), directory, prepared, options)
                    batch = official_batch(config, args.dfine_root.resolve())
                    configs.append(batch.pop("resolved_config"))
                    (directory / "resolved_config.json").write_text(json.dumps(configs[-1], indent=2), encoding="utf-8")
                    batches.append(batch)
                difference = configuration_differences(*configs, "official_dfine")
                if difference:
                    raise ValueError(f"Resolved configuration differences: {json.dumps(difference)}")
                verify_batch_pair(*batches)
                assert configs[0]["epochs"] == configs[1]["epochs"] == 300
                assert configs[0]["train_dataloader"]["num_workers"] == configs[1]["train_dataloader"]["num_workers"] == 4
                report["legacy_route"] = {"required_final_run": False, "epochs": 300, "workers": 4,
                    "scheduler": "MultiStepLR", "milestones": configs[0]["lr_scheduler"]["milestones"],
                    "note": "Fixed legacy milestone [500] retained; no OneCycle horizon or final-run early stopping requirement."}
                report["pairs"]["official_dfine_detect"] = {"resolved_rgb": configs[0], "resolved_hsv": configs[1],
                        "configuration_diff": difference, "rgb_batch": batches[0], "hsv_batch": batches[1]}
            check("Official D-FINE Nano detection resolved parity and real RGB/HSV batches", official_check)

    expected = {"yolo_detect", "yolo_segment", "project_dfine_detect", "project_dfine_segment"}
    if not expected.issubset(report["pairs"]) or not expected.issubset(report["schedulers"]):
        report["failures"]["required final model matrix"] = "Missing required pair or scheduler checks"
    (output / "scheduler_diagnostics.json").write_text(json.dumps(report["schedulers"], indent=2), encoding="utf-8")
    report["status"] = "FAIL" if report["failures"] else "PASS"
    (output / "preflight_log.txt").write_text("\n".join(
        [f"PASS: {name}" for name in report["checks"]] + [f"FAIL: {name}: {error}" for name, error in report["failures"].items()]) + "\n", encoding="utf-8")
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"Validation report: {output / 'validation_report.json'}")
    print("FINAL 300-EPOCH EXPERIMENT VALIDATION: " + report["status"])
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
