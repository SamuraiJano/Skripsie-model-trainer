"""Select saved detector checkpoints and export/verify deployment artifacts.

Every export goes into a new bundle. Source checkpoints and training files are
read only. Run one model per process to isolate the upstream D-FINE imports.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
LOCAL_TIME = timezone(timedelta(hours=2))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory():
    records = []
    for family, directory in (("yolo", "yolo26"), ("dfine", "dfine-seg")):
        for run in (PROJECT / "runs" / directory).iterdir():
            stamp = re.search(r"(\d{8}-\d{6})$", run.name)
            if not run.is_dir() or not stamp:
                continue
            metadata = read_json(run / "run_metadata.json") if (run / "run_metadata.json").exists() else {}
            task = metadata.get("task", "segment" if "segment" in run.name else "detect")
            colour = metadata.get("experiment", {}).get("colour_space")
            source = metadata.get("dataset", {}).get("source_root") or metadata.get("dataset", {}).get("root", "")
            colour_evidence = "experiment.colour_space"
            if colour is None and source:
                colour = "hsv" if "_hsv" in source.lower() else "rgb" if "_rgb" in source.lower() else None
                colour_evidence = "dataset source path (historical representation; not necessarily HSV_FULL)"
            checkpoint = run / ("weights/best.pt" if family == "yolo" else "model/model.pt")
            metric = metadata.get("selection", {}).get("best_validation_metric")
            metric_evidence = "selection.best_validation_metric"
            if metric is None and (run / "final_metrics.json").exists():
                final = read_json(run / "final_metrics.json")
                metric = final.get("metrics/mAP50-95(B)")
                metric_evidence = "final_metrics.json best-checkpoint validation"
            if metric is None and (run / "model/metrics.csv").exists():
                with (run / "model/metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                    for row in csv.DictReader(handle):
                        if list(row.values())[0] == "val" and "mAP_50_95" in row:
                            metric = float(row["mAP_50_95"])
                            metric_evidence = "model/metrics.csv val row (rounded)"
            records.append({"run_id": run.name, "run_dir": str(run), "stamp": stamp.group(1),
                            "family": family, "task": task, "colour": colour,
                            "colour_evidence": colour_evidence, "validation_map50_95": metric,
                            "metric_evidence": metric_evidence, "stop_reason": metadata.get("stop_reason"),
                            "checkpoint": str(checkpoint), "checkpoint_exists": checkpoint.is_file(),
                            "has_metadata": bool(metadata)})
    return sorted(records, key=lambda item: item["stamp"], reverse=True)


def prepare(output, mode, limit):
    output.mkdir(parents=True, exist_ok=True)
    records = inventory()
    candidates = [r for r in records if r["task"] == "detect"][:limit]
    selected = {}
    for family in ("yolo", "dfine"):
        for colour in ("rgb", "hsv"):
            choices = [r for r in candidates if r["family"] == family and r["colour"] == colour
                       and r["checkpoint_exists"] and r["validation_map50_95"] is not None]
            if mode == "latest-four":
                choices = [r for r in choices if r["stop_reason"] in {"max_epochs", "early_stopping"}]
                choice = max(choices, key=lambda r: r["stamp"])
            else:
                choice = max(choices, key=lambda r: r["validation_map50_95"])
            key = f"{family}_{colour}"
            selected[key] = dict(choice, checkpoint_sha256=sha256(choice["checkpoint"]))
            model_dir = output / key
            model_dir.mkdir(exist_ok=True)
            suffix = "best.pt" if family == "yolo" else "model.pt"
            staged = model_dir / suffix
            if not staged.exists():
                shutil.copy2(choice["checkpoint"], staged)
            if sha256(staged) != selected[key]["checkpoint_sha256"]:
                raise RuntimeError(f"Staged checkpoint mismatch for {key}")
            shutil.copy2(Path(choice["run_dir"]) / "run_metadata.json", model_dir / "source_run_metadata.json")
            if family == "dfine":
                shutil.copy2(Path(choice["run_dir"]) / "dfine_seg_training.yaml", model_dir / "source_training.yaml")
    manifest = {"created_at": datetime.now(LOCAL_TIME).isoformat(), "mode": mode,
                "requested_run_limit": limit, "available_detection_run_count": len(candidates),
                "runs_with_metadata": sum(r["has_metadata"] for r in records),
                "selection_metric": "validation mAP50-95; never test scores",
                "candidates": candidates, "selected": selected}
    write_json(output / "selection.json", manifest)
    print(json.dumps(selected, indent=2), flush=True)


def image_tensor(path, size=640):
    """Read existing model-space images: RGB files or H,S,V in PNG RGB slots."""
    import cv2
    import numpy as np
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read {path}")
    height, width = image.shape[:2]
    ratio = min(size / height, size / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = size - new_w, size - new_h
    padded = cv2.copyMakeBorder(resized, dh // 2, dh - dh // 2, dw // 2, dw - dw // 2,
                               cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255


def samples(metadata, count=3, staged_directory=None):
    import numpy as np
    split = metadata.get("dataset", {}).get("splits", {}).get("val", {})
    image_dir = Path(split.get("path") or Path(metadata["dataset"]["root"]) / "images/val")
    if staged_directory is not None and staged_directory.is_dir():
        image_dir = staged_directory
    elif not image_dir.is_dir():
        image_dir = Path(metadata["dataset"].get("source_root", metadata["dataset"]["root"])) / "images/val"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    chosen = [images[index] for index in np.linspace(0, len(images) - 1, count, dtype=int)]
    return [(str(p), image_tensor(p)) for p in chosen]


def comparison(reference, actual, *, atol=0.003, rtol=0.01):
    import numpy as np
    if reference.shape != actual.shape:
        return {"pass": False, "reference_shape": list(reference.shape), "actual_shape": list(actual.shape)}
    diff = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
    return {"pass": bool(np.all(np.isfinite(actual)) and np.allclose(reference, actual, atol=atol, rtol=rtol)),
            "shape": list(reference.shape), "max_absolute_difference": float(diff.max()),
            "mean_absolute_difference": float(diff.mean()), "atol": atol, "rtol": rtol}


def output_comparison(reference, actual, family):
    """YOLO's unused low-confidence TopK rows can differ in tie ordering."""
    import numpy as np
    if family != "yolo":
        return comparison(reference, actual)
    from scipy.optimize import linear_sum_assignment
    if reference.shape != actual.shape or not np.all(np.isfinite(actual)):
        return {"pass": False, "reason": "shape mismatch or non-finite output"}
    a, b = reference.reshape(-1, 6), actual.reshape(-1, 6)
    thresholds = {}
    for threshold in (0.01, 0.25, 0.5):
        first, second = a[a[:, 4] >= threshold], b[b[:, 4] >= threshold]
        detail = {"reference_detections": len(first), "exported_detections": len(second)}
        if first.shape != second.shape:
            detail["pass"] = False
        elif not len(first):
            detail["pass"] = True
        else:
            cost = np.abs(first[:, None, :4] - second[None, :, :4]).sum(axis=2)
            cost += (first[:, None, 5] != second[None, :, 5]) * 1e6
            rows, columns = linear_sum_assignment(cost)
            coord_diff = float(np.abs(first[rows, :4] - second[columns, :4]).max())
            score_diff = float(np.abs(first[rows, 4] - second[columns, 4]).max())
            detail.update(max_coordinate_difference_pixels=coord_diff, max_score_difference=score_diff,
                          **{"pass": bool(coord_diff <= 0.1 and score_diff <= 0.003
                                          and np.array_equal(first[rows, 5], second[columns, 5]))})
        thresholds[str(threshold)] = detail
    return {"pass": all(v["pass"] for v in thresholds.values()),
            "method": "Match detections by class and box at confidence 0.01, 0.25 and 0.5; unused TopK ties ignored",
            "coordinate_tolerance_pixels": 0.1, "score_tolerance": 0.003, "thresholds": thresholds}


def dfine_model(model_dir, metadata):
    import torch
    from omegaconf import OmegaConf
    upstream = Path(metadata["upstream"])
    sys.path.insert(0, str(upstream))
    from src.d_fine.dfine import build_model
    config = OmegaConf.load(model_dir / "source_training.yaml")
    model = build_model(config.model_name, len(config.train.label_to_name), False, "cpu",
                        img_size=list(config.train.img_size), in_channels=int(config.train.in_channels), task="detect")
    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    class RawDetector(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector

        def forward(self, images):
            result = self.detector(images)
            return result["pred_logits"], result["pred_boxes"]

    return RawDetector(model).eval()


def export_onnx(model_dir, record, metadata):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    destination = model_dir / "model.onnx"
    if record["family"] == "yolo":
        from ultralytics import YOLO
        from ultralytics.engine.exporter import Exporter
        loaded = YOLO(str(model_dir / "best.pt"))
        if destination.is_file():
            model = loaded.model.fuse().eval()
            model.model[-1].export = True
            model.model[-1].format = "onnx"
            model.model[-1].dynamic = False
        else:
            exporter = Exporter(overrides={"format": "onnx", "imgsz": 640, "batch": 1, "device": "cpu",
                                           "dynamic": False, "simplify": True, "opset": 19})
            result = exporter(model=loaded.model)
            shutil.copy2(result, destination)
            model = exporter.model
        names = ["output0"]
    else:
        import onnxsim
        from onnxscript.function_libs.torch_lib.ops import core as onnx_core
        sys.path.insert(0, str(PROJECT))
        from scripts.dfine_seg_model_tools import _lower_export_splits
        model = dfine_model(model_dir, metadata)
        names = ["logits", "boxes"]
        if not destination.is_file():
            torch.onnx.export(model, torch.zeros(1, 3, 640, 640), opset_version=19,
                              input_names=["images"], output_names=names, dynamo=True, optimize=False,
                              custom_translation_table={torch.ops.aten.mul.Scalar: onnx_core.aten_mul}).save(destination)
            graph = _lower_export_splits(onnx.load(destination))
            graph, valid = onnxsim.simplify(graph)
            if not valid:
                raise RuntimeError("ONNX graph simplification validation failed")
            onnx.save(graph, destination)
    onnx.checker.check_model(str(destination))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(destination), sess_options=opts, providers=["CPUExecutionProvider"])
    results = []
    for label, tensor in samples(metadata, staged_directory=model_dir.parent / "verification_images" / record["colour"]):
        with torch.inference_mode():
            reference = model(torch.from_numpy(tensor))
        if isinstance(reference, torch.Tensor):
            reference = (reference,)
        actual = session.run(None, {session.get_inputs()[0].name: tensor})
        checks = {name: output_comparison(ref.detach().cpu().numpy(), val, record["family"])
                  for name, ref, val in zip(names, reference, actual)}
        results.append({"image": label, "outputs": checks})
    passed = all(item["pass"] for row in results for item in row["outputs"].values())
    report = {"status": "validated" if passed else "failed_parity", "format": "onnx", "precision": "fp32",
              "file": str(destination), "sha256": sha256(destination), "samples": results,
              "outputs": [{"name": out.name, "shape": out.shape} for out in session.get_outputs()]}
    write_json(model_dir / "onnx_verification.json", report)
    if not passed:
        raise RuntimeError("ONNX predictions differ from source checkpoint; see verification JSON")


def export_tensorrt(model_dir, record, metadata):
    import numpy as np
    import onnxruntime as ort
    import tensorrt as trt
    import torch
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse((model_dir / "model.onnx").read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
    if hasattr(trt.BuilderFlag, "TF32"):
        config.clear_flag(trt.BuilderFlag.TF32)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(blob)
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    input_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
    outputs = [name for name in names if name != input_name]
    buffers = {name: torch.empty(tuple(context.get_tensor_shape(name)), device="cuda",
                               dtype=torch.from_numpy(np.empty((), dtype=trt.nptype(engine.get_tensor_dtype(name)))).dtype)
               for name in names}
    for name, tensor in buffers.items():
        context.set_tensor_address(name, tensor.data_ptr())
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    results = []
    for label, tensor in samples(metadata, staged_directory=model_dir.parent / "verification_images" / record["colour"]):
        buffers[input_name].copy_(torch.from_numpy(tensor))
        if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        torch.cuda.synchronize()
        expected = dict(zip((out.name for out in session.get_outputs()), session.run(None, {input_name: tensor})))
        checks = {name: output_comparison(expected[name], buffers[name].cpu().numpy(), record["family"]) for name in outputs}
        results.append({"image": label, "outputs": checks})
    passed = all(check["pass"] for row in results for check in row["outputs"].values())
    destination = model_dir / "nvidia" / "model_fp32.engine"
    destination.parent.mkdir(exist_ok=True)
    if passed:
        destination.write_bytes(bytes(blob))
    report = {"status": "validated" if passed else "failed_parity", "format": "tensorrt", "precision": "fp32",
              "file": str(destination) if passed else None, "tensorrt_version": trt.__version__,
              "gpu": torch.cuda.get_device_name(0), "platform": sys.platform,
              "portability": "Built for this NVIDIA GPU and TensorRT runtime; rebuild ONNX on other target systems.",
              "samples": results}
    if passed:
        report["sha256"] = sha256(destination)
    write_json(model_dir / "tensorrt_verification.json", report)
    if not passed:
        raise RuntimeError("TensorRT parity verification failed")


def export_ncnn(model_dir, record, metadata):
    if record["family"] == "dfine":
        raise RuntimeError("D-FINE Nano NCNN conversion failed native runtime verification. Use the verified ONNX model on Raspberry Pi 4.")
    import ncnn
    import numpy as np
    import onnxruntime as ort
    import torch
    destination = model_dir / "raspberry_pi4" / "ncnn_fp32"
    destination.parent.mkdir(exist_ok=True)
    from ultralytics.utils.export.ncnn import torch2ncnn
    if record["family"] == "yolo":
        from ultralytics import YOLO
        loaded = YOLO(str(model_dir / "best.pt"))
        model = loaded.model.fuse().eval()
        head = model.model[-1]
        if not head.end2end:
            raise RuntimeError("This export requires the YOLO26 one-to-one head")
        head.export = True
        head.format = "onnx"
        head.dynamic = False
        # The stock NCNN exporter disables end2end and exports the other head.
        # Preserve the trained one-to-one predictions, moving only unsupported
        # TopK selection to the consuming application. Boxes remain pixel xyxy.
        head.postprocess = lambda predictions: predictions
        dummy = torch.zeros(1, 3, 640, 640)
        with torch.inference_mode():
            model(dummy)
            model(dummy)
        torch2ncnn(model, dummy, destination, device=torch.device("cpu"))
    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.opt.use_fp16_storage = False
    net.opt.use_fp16_arithmetic = False
    if net.load_param(str(destination / "model.ncnn.param")) != 0:
        raise RuntimeError("NCNN could not load exported parameters")
    if net.load_model(str(destination / "model.ncnn.bin")) != 0:
        raise RuntimeError("NCNN could not load exported weights")
    inputs = net.input_names()
    outputs = ["out0"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(model_dir / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
    results = []
    for label, tensor in samples(metadata, staged_directory=model_dir.parent / "verification_images" / record["colour"]):
        print(f"NCNN verifying {Path(label).name}", flush=True)
        reference = session.run(None, {session.get_inputs()[0].name: tensor})
        extractor = net.create_extractor()
        if extractor.input(inputs[0], ncnn.Mat(tensor[0]).clone()) != 0:
            raise RuntimeError("NCNN input failed")
        checks = {}
        for index, name in enumerate(outputs):
            code, value = extractor.extract(name)
            if code != 0:
                raise RuntimeError(f"NCNN output {name} failed with code {code}")
            array = np.array(value)
            ref = reference[index][0]
            if record["family"] == "yolo":
                array = yolo_topk(array)
            checks[name] = output_comparison(ref, array, record["family"])
        results.append({"image": label, "outputs": checks})
    passed = all(item["pass"] for row in results for item in row["outputs"].values())
    report = {"status": "validated" if passed else "failed_parity", "format": "ncnn", "precision": "fp32",
              "directory": str(destination), "runtime_version": ncnn.__version__,
              "inputs": list(inputs), "outputs": list(outputs), "samples": results,
              "calibration_required": False, "verified_host": "Windows x86-64 CPU; Raspberry Pi hardware not attached"}
    if record["family"] == "yolo":
        report["contract"] = "Preserved one-to-one head; raw [8400,5] pixel xyxy and probability; host TopK, no NMS"
    write_json(model_dir / "ncnn_verification.json", report)
    if not passed:
        raise RuntimeError("NCNN numerical parity verification failed")


def yolo_topk(raw, maximum=300):
    import numpy as np
    scores = raw[:, 4:]
    indices = np.argsort(scores.max(axis=1))[::-1][:maximum]
    selected = scores[indices]
    flat_indices = np.argsort(selected.reshape(-1))[::-1][:maximum]
    anchors = indices[flat_indices // scores.shape[1]]
    classes = flat_indices % scores.shape[1]
    return np.column_stack((raw[anchors, :4], scores[anchors, classes], classes)).astype(np.float32)


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--mode", choices=("latest-four", "best"), default="latest-four")
    parser.add_argument("--limit", type=int, default=45)
    parser.add_argument("--model", choices=("yolo_rgb", "yolo_hsv", "dfine_rgb", "dfine_hsv"))
    parser.add_argument("--format", choices=("onnx", "tensorrt", "ncnn"))
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output, args.mode, args.limit)
        return
    import torch
    torch.set_num_threads(4)
    manifest = read_json(output / "selection.json")
    record = manifest["selected"][args.model]
    model_dir = output / args.model
    metadata = read_json(model_dir / "source_run_metadata.json")
    try:
        {"onnx": export_onnx, "tensorrt": export_tensorrt, "ncnn": export_ncnn}[args.format](model_dir, record, metadata)
        if sha256(record["checkpoint"]) != record["checkpoint_sha256"]:
            raise RuntimeError("Source checkpoint changed during export")
        (model_dir / f"{args.format}_error.json").unlink(missing_ok=True)
        print(f"VERIFIED {args.model} {args.format}", flush=True)
    except Exception:
        write_json(model_dir / f"{args.format}_error.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
