"""Bounded real-model validation smoke; no training, optimizer updates or test images.

CPU/batch-1 checks confirm adapters. Use CUDA/physical batches on an idle GPU
to check training forward/backward memory without modifying the frozen recipe.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", choices=["yolo26n", "dfine_n"], required=True)
    parser.add_argument("--colour", choices=["rgb", "hsv"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    import torch
    from naartjie_cv.detector_experiment import verify_experiment, write_json
    from naartjie_cv.detector_worker import protocol_for, official_config
    from naartjie_cv.detection_data import detection_ground_truth
    from naartjie_cv.runtime import seed_training
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_experiment(manifest)
    job = next(j for j in manifest["matrix"] if j["model"] == args.model and j["colour"] == args.colour)
    seed_training(job["seed"])
    if args.batch < 1:
        raise ValueError("Preflight batch must be positive")
    directory = Path(tempfile.mkdtemp(prefix="detector-preflight-", dir=Path(manifest["output_dir"])))
    protocol = protocol_for(manifest, job, directory)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if args.model == "dfine_n":
        sys.path.insert(0, manifest["dfine_root"])
        from naartjie_cv.official_detection import install_official_geometry, predictions_from_output
        install_official_geometry()
        path = official_config(manifest, job, directory, protocol)
        from src.core import YAMLConfig
        from src.solver.det_solver import DetSolver
        cfg = YAMLConfig(str(path), device=args.device, use_amp=args.device.startswith("cuda"),
                         tuning=manifest["recipes"]["dfine_n"]["checkpoint"])
        cfg.yaml_cfg["val_dataloader"]["total_batch_size"] = args.batch
        cfg.yaml_cfg["train_dataloader"]["total_batch_size"] = args.batch
        solver = DetSolver(cfg)
        solver.eval()
        if args.backward:
            from collections import Counter
            loader = cfg.train_dataloader
            counts = Counter(a["image_id"] for a in loader.dataset.coco.anns.values())
            indices = sorted(range(len(loader.dataset)), key=lambda i: (-counts[loader.dataset.ids[i]], i))[:args.batch]
            images, targets = loader.collate_fn([loader.dataset[i] for i in indices])
        else:
            images, targets = next(iter(cfg.val_dataloader))
        images = images.to(solver.device)
        model = solver.model
        if args.backward:
            model.train()
            device_targets = [{k: v.to(solver.device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
            with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                output = model(images, targets=device_targets)
            losses = solver.criterion(output, device_targets)
            loss = sum(losses.values())
            assert torch.isfinite(loss)
            loss.backward()
        else:
            model.eval()
            with torch.no_grad():
                output = model(images)
                predictions = solver.postprocessor(output, torch.full((len(targets), 2), 640, device=solver.device))
                assert all(torch.isfinite(p["boxes"]).all() for p in predictions)
        result = {"loader_shape": list(images.shape), "model_parameters": sum(p.numel() for p in model.parameters()),
                  "loss": float(loss.detach()) if args.backward else None, "resolved_config": cfg.yaml_cfg,
                  "batch_object_counts": [len(t["boxes"]) for t in targets]}
    else:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel
        from naartjie_cv.yolo_detection import ControlledDetectionTrainer, CommonDetectionValidator, convert_batch_representation
        model = YOLO(job["checkpoint"])
        # Use the same dataset class-head transfer as training (also for forward checks).
        original = model.model
        model.model = DetectionModel(original.yaml, nc=len(detection_ground_truth(protocol["rgb_yaml"], "val")["categories"]), ch=3, verbose=False)
        model.model.load(original)
        model.model.to(args.device)
        loader = ControlledDetectionTrainer.__new__(ControlledDetectionTrainer)
        training = manifest["recipes"]["yolo26n"]["training"]
        loader.args = get_cfg(overrides={k: v for k, v in training.items() if v is not None})
        model.model.args = loader.args
        loader.data = {"names": {c["id"]: c["name"] for c in detection_ground_truth(protocol["rgb_yaml"], "val")["categories"]},
                       "nc": len(detection_ground_truth(protocol["rgb_yaml"], "val")["categories"]), "channels": 3}
        loader.model = model.model
        dataset = loader.build_dataset(manifest["paired_dataset"]["dataset_identity"]["splits"]["train" if args.backward else "val"]["path"],
                                       "train" if args.backward else "val", batch=args.batch)
        indices = sorted(range(len(dataset)), key=lambda i: (-len(dataset.labels[i]["cls"]), i))[:args.batch] if args.backward else list(range(min(args.batch, len(dataset))))
        batch = dataset.collate_fn([dataset[i] for i in indices])
        batch = convert_batch_representation(batch, args.colour)
        batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch["img"] = batch["img"].float() / 255
        if args.backward:
            model.model.train()
            with torch.autocast(device_type="cuda", enabled=args.device.startswith("cuda")):
                loss, _ = model.model(batch)
                loss = loss.sum()
            assert torch.isfinite(loss)
            loss.backward()
        else:
            model.model.eval()
            with torch.no_grad():
                output = model.model(batch["img"])
                assert torch.isfinite(output[0]).all()
        result = {"loader_shape": list(batch["img"].shape), "model_parameters": sum(p.numel() for p in model.model.parameters()),
                  "loss": float(loss.detach()) if args.backward else None,
                  "batch_object_counts": [len(dataset.labels[i]["cls"]) for i in indices]}
    result.update(status="PASS", model=args.model, colour=args.colour, device=args.device, requested_batch=args.batch,
                  backward=args.backward, optimizer_updates=0, test_inference_performed=False,
                  test_usage="pairing/hash audit only; no inference or selection",
                  batch_selection="densest training images" if args.backward else "first validation batch",
                  amp=args.device.startswith("cuda"), manifest_sha256=manifest["manifest_sha256"],
                  peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None)
    write_json(directory / "preflight.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "resolved_config"}, indent=2))
    print("Preflight report:", directory / "preflight.json")


if __name__ == "__main__":
    main()
