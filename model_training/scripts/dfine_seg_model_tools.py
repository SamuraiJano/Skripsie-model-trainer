"""Evaluate D-FINE-seg checkpoints and export them from the training notebook."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {path}")


def _load_run(run_dir: Path, checkpoint_name: str):
    from omegaconf import OmegaConf

    run_dir = run_dir.expanduser().resolve()
    metadata_path = run_dir / "run_metadata.json"
    config_path = run_dir / "dfine_seg_training.yaml"
    checkpoint = run_dir / "model" / checkpoint_name
    for path in (metadata_path, config_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(f"Required D-FINE-seg run file is missing: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("trainer") != "ArgoHA/D-FINE-seg" or metadata.get("task") != "segment":
        raise ValueError(f"This is not a D-FINE-seg instance-segmentation run: {run_dir}")
    upstream = Path(metadata["upstream"]).resolve()
    if not (upstream / "src" / "dl" / "train.py").is_file():
        raise FileNotFoundError(f"D-FINE-seg checkout is missing: {upstream}")
    config = OmegaConf.load(config_path)
    if config.task != "segment":
        raise ValueError(f"Expected task=segment in {config_path}")
    return run_dir, metadata, upstream, config, checkpoint


def _enter_upstream(upstream: Path) -> None:
    os.chdir(upstream)
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))


def _checkpoint_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lower_export_splits(model):
    """Replace PyTorch's sequence-valued split/getitem nodes with ONNX Split."""
    import numpy as np
    from onnx import helper, numpy_helper

    constants = {
        node.output[0]: numpy_helper.to_array(next(attr.t for attr in node.attribute if attr.name == "value"))
        for node in model.graph.node
        if node.op_type == "Constant" and any(attr.name == "value" for attr in node.attribute)
    }
    used_indices = {}
    for node in model.graph.node:
        if node.op_type == "aten_getitem":
            used_indices.setdefault(node.input[0], []).append(int(constants[node.input[1]]))

    split_outputs = {}
    rewritten = []
    for node in model.graph.node:
        if node.op_type in ("aten_split", "aten_split_with_sizes"):
            key = node.output[0]
            indices = used_indices.get(key)
            if not indices:
                raise RuntimeError(f"Unsupported unconsumed sequence output in ONNX export: {node.name}")
            if node.op_type == "aten_split":
                sizes = [int(constants[node.input[1]])] * (max(indices) + 1)
            else:
                sizes = list(map(int, constants[node.input[1]].tolist()))
            if max(indices) >= len(sizes):
                raise RuntimeError(f"Invalid split indices in ONNX export: {node.name}")
            names = [f"{key}_item_{index}" for index in range(len(sizes))]
            sizes_name = f"{key}_sizes"
            model.graph.initializer.append(numpy_helper.from_array(np.asarray(sizes, dtype=np.int64), sizes_name))
            axis = next(int(attr.i) for attr in node.attribute if attr.name == "dim")
            rewritten.append(helper.make_node("Split", [node.input[0], sizes_name], names, axis=axis, name=f"{node.name}_onnx"))
            split_outputs[key] = names
        elif node.op_type == "aten_getitem":
            index = int(constants[node.input[1]])
            rewritten.append(helper.make_node("Identity", [split_outputs[node.input[0]][index]], list(node.output), name=f"{node.name}_onnx"))
        else:
            rewritten.append(node)
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)
    remaining = {node.op_type for node in model.graph.node if node.op_type.startswith("aten_")}
    if remaining:
        raise RuntimeError(f"Unsupported PyTorch operations remain in ONNX export: {sorted(remaining)}")
    return model


def _compare_outputs(expected: dict, actual: dict, label: str, *, strict: bool = True) -> dict:
    import numpy as np

    comparison = {}
    if set(expected) != set(actual):
        raise RuntimeError(f"{label} output names differ: {sorted(expected)} versus {sorted(actual)}")
    for name, reference in expected.items():
        candidate = actual[name]
        if reference.shape != candidate.shape:
            raise RuntimeError(f"{label} {name} shape differs: {reference.shape} versus {candidate.shape}")
        maximum = float(np.max(np.abs(reference.astype(np.float64) - candidate.astype(np.float64))))
        mean = float(np.mean(np.abs(reference.astype(np.float64) - candidate.astype(np.float64))))
        matches = bool(np.allclose(reference, candidate, rtol=0.01, atol=0.003))
        comparison[name] = {"max_absolute_difference": maximum, "mean_absolute_difference": mean, "matches": matches}
    if strict and not all(item["matches"] for item in comparison.values()):
        raise RuntimeError(f"{label} changed the segmentation model's predictions: {comparison}")
    return comparison


def _decompose_instance_normalization(model):
    """Avoid a custom LiteRT op for ONNX InstanceNormalization on NCL tensors."""
    import numpy as np
    from onnx import helper, numpy_helper

    axes = "seg_instnorm_reduction_axes"
    channel_axes = "seg_instnorm_channel_axes"
    epsilon = "seg_instnorm_epsilon"
    model.graph.initializer.extend([
        numpy_helper.from_array(np.asarray([2], dtype=np.int64), axes),
        numpy_helper.from_array(np.asarray([0, 2], dtype=np.int64), channel_axes),
        numpy_helper.from_array(np.asarray(1e-5, dtype=np.float32), epsilon),
    ])
    rewritten = []
    for node in model.graph.node:
        if node.op_type != "InstanceNormalization":
            rewritten.append(node)
            continue
        node_epsilon = next((attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-5)
        if abs(node_epsilon - 1e-5) > 1e-8:
            raise RuntimeError(f"Unsupported InstanceNormalization epsilon in {node.name}: {node_epsilon}")
        prefix = node.name

        def add(operation, inputs, suffix, **attributes):
            name = f"{prefix}_{suffix}"
            rewritten.append(helper.make_node(operation, inputs, [name], name=name, **attributes))
            return name

        data, scale, bias = node.input
        mean = add("ReduceMean", [data, axes], "mean", keepdims=1)
        centered = add("Sub", [data, mean], "centered")
        squared = add("Mul", [centered, centered], "squared")
        variance = add("ReduceMean", [squared, axes], "variance", keepdims=1)
        denominator = add("Sqrt", [add("Add", [variance, epsilon], "variance_epsilon")], "denominator")
        normalized = add("Div", [centered, denominator], "normalized")
        scaled = add("Mul", [normalized, add("Unsqueeze", [scale, channel_axes], "scale")], "scaled")
        rewritten.append(helper.make_node("Add", [scaled, add("Unsqueeze", [bias, channel_axes], "bias")], list(node.output), name=f"{prefix}_output"))
    model.graph.ClearField("node")
    model.graph.node.extend(rewritten)
    return model


def evaluate_test(run_dir: Path, dataset_root: Path, checkpoint_name: str, device: str, output_dir: Path | None = None) -> Path:
    import torch

    from naartjie_cv.datasets import prepare_dfine_seg_dataset

    run_dir, metadata, upstream, config, checkpoint = _load_run(run_dir, checkpoint_name)
    dataset_root = dataset_root.expanduser().resolve()
    if not (dataset_root / "images" / "test").is_dir():
        raise FileNotFoundError(f"Dataset test images are missing: {dataset_root / 'images' / 'test'}")

    output_dir = (output_dir or run_dir / "test_evaluation" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    prepared = prepare_dfine_seg_dataset(dataset_root, "segment", output_dir / "prepared_dataset")
    test_info = prepared["splits"].get("test")
    if not test_info or not test_info["images"]:
        raise ValueError(f"No labelled test images were found in {dataset_root / 'images' / 'test'}")
    if {int(key): name for key, name in prepared["classes"].items()} != {
        int(key): name for key, name in metadata["dataset"]["classes"].items()
    }:
        raise ValueError("The current test dataset class names differ from this checkpoint's training classes")

    config.train.data_path = prepared["root"]
    config.train.path_to_save = str(output_dir / "evaluation")
    config.train.debug_img_path = str(output_dir / "debug_images")
    config.train.eval_preds_path = str(output_dir / "preview_images")
    config.train.device = device
    config.train.num_workers = 0
    config.train.pretrained_model_path = None
    config.train.use_wandb = False
    config.train.use_ema = False
    config.train.use_scheduler = False
    config.train.use_muon = False
    config.train.aux_optimizer = "adamw"
    config.train.to_visualize_eval = True
    if device == "cpu":
        config.train.amp_enabled = False

    _enter_upstream(upstream)
    from scripts.dfine_seg_runner import _install_consistent_letterbox
    from src.dl import dataset as dataset_module
    from src.dl.train import Trainer

    _install_consistent_letterbox(dataset_module)
    print(f"Testing {run_dir.name} on {test_info['images']} images from {dataset_root / 'images' / 'test'}")
    print(f"Checkpoint: {checkpoint}")
    trainer = Trainer(config)
    assert trainer.test_loader is not None
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    trainer.model.load_state_dict(state)
    metrics = trainer.evaluate(
        val_loader=trainer.test_loader,
        conf_thresh=trainer.conf_thresh,
        iou_thresh=trainer.iou_thresh,
        path_to_save=Path(config.train.path_to_save),
        extended=True,
        mode="test",
    )
    report = output_dir / "test_metrics.json"
    _write_report(report, {
        "run": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
        "dataset": str(dataset_root),
        "test_images": test_info["images"],
        "test_annotations": test_info["annotations"],
        "evaluation_letterbox_scaleup": True,
        "metrics": metrics,
        "preview_images": str(output_dir / "preview_images"),
        "plots": str(output_dir / "evaluation" / "plots" / "test"),
    })
    return report


def _onnx_model(config, checkpoint: Path, upstream: Path, export_dir: Path, raw: bool, overwrite: bool) -> Path:
    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxsim
    import torch
    from onnxscript.function_libs.torch_lib.ops import core as onnx_core

    _enter_upstream(upstream)
    from src.d_fine.configs import base_cfg
    from src.dl.export import DFINEPostProcessor, ExportWrapper, prepare_model

    export_dir.mkdir(parents=True, exist_ok=True)
    staged_checkpoint = export_dir / "model.pt"
    checkpoint_digest = _checkpoint_hash(checkpoint)
    checkpoint_changed = not staged_checkpoint.is_file() or _checkpoint_hash(staged_checkpoint) != checkpoint_digest
    if checkpoint_changed:
        shutil.copy2(checkpoint, staged_checkpoint)
    onnx_path = export_dir / ("model_raw.onnx" if raw else "model.onnx")
    source_hash_path = onnx_path.with_suffix(".sha256")
    export_signature = f"validated-v2:{checkpoint_digest}"
    if (
        onnx_path.is_file()
        and source_hash_path.is_file()
        and source_hash_path.read_text(encoding="ascii").strip() == export_signature
        and not overwrite
    ):
        print(f"Reusing ONNX: {onnx_path}")
        return onnx_path

    config.train.path_to_save = str(export_dir)
    config.train.device = "cpu"
    config.export.from_pretrained = False
    raw_model = prepare_model(config, config.train.device)
    if raw:
        class RawSegmentationOutputs(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, image):
                output = self.model(image)
                return output["pred_logits"], output["pred_boxes"], output["pred_masks"]

        model = RawSegmentationOutputs(raw_model).eval()
        names = ["logits", "boxes", "masks"]
    else:
        postprocessor = DFINEPostProcessor(
            num_classes=len(config.train.label_to_name),
            num_top_queries=base_cfg["DFINETransformer"]["num_queries"],
            use_focal_loss=base_cfg["matcher"]["use_focal_loss"],
        )
        model = ExportWrapper(raw_model, postprocessor, config.train.img_size).eval()
        names = ["labels", "boxes", "scores", "masks"]
    shape = (1, int(config.train.in_channels), *map(int, config.train.img_size))
    sample = torch.zeros(shape, dtype=torch.float32)
    source_path = onnx_path.with_name(f"{onnx_path.stem}_unoptimized.onnx")
    # The upstream optimizer changes D-FINE-seg predictions on this PyTorch version.
    # Export without it, then lower the remaining sequence-valued split operations.
    torch.onnx.export(
        model, sample, opset_version=19, input_names=["input"], output_names=names,
        dynamo=True, optimize=False,
        custom_translation_table={torch.ops.aten.mul.Scalar: onnx_core.aten_mul},
    ).save(source_path)
    try:
        graph = _lower_export_splits(onnx.load(str(source_path)))
    finally:
        source_path.unlink(missing_ok=True)
    graph, valid = onnxsim.simplify(graph)
    if not valid:
        raise RuntimeError("ONNX simplification did not validate the segmentation graph")
    onnx.save(graph, str(onnx_path))
    onnx.checker.check_model(str(onnx_path))
    test_image = np.random.default_rng(4).random(shape, dtype=np.float32)
    with torch.inference_mode():
        expected = dict(zip(names, (value.detach().numpy() for value in model(torch.from_numpy(test_image)))))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = dict(zip((item.name for item in session.get_outputs()), session.run(None, {"input": test_image})))
    parity = _compare_outputs(expected, actual, "PyTorch-to-ONNX export")
    _write_report(onnx_path.with_suffix(".parity.json"), parity)
    source_hash_path.write_text(export_signature + "\n", encoding="ascii")
    print(f"Validated ONNX: {onnx_path}")
    return onnx_path


def export_tflite(run_dir: Path, checkpoint_name: str, overwrite: bool, allow_unvalidated: bool = False) -> Path:
    run_dir, _, upstream, config, checkpoint = _load_run(run_dir, checkpoint_name)
    if importlib.util.find_spec("onnx2tf") is None or importlib.util.find_spec("ai_edge_litert") is None:
        raise RuntimeError(
            "TFLite export needs onnx2tf and ai-edge-litert in this Python environment. "
            "Install them, restart the notebook kernel, and retry."
        )
    import numpy as np
    import onnx
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    export_dir = run_dir / "exports" / "segmentation"
    onnx_path = _onnx_model(config, checkpoint, upstream, export_dir, raw=True, overwrite=overwrite)
    tflite_onnx_path = export_dir / "model_raw_tflite.onnx"
    if overwrite or not tflite_onnx_path.is_file() or tflite_onnx_path.stat().st_mtime_ns < onnx_path.stat().st_mtime_ns:
        graph = _decompose_instance_normalization(onnx.load(str(onnx_path)))
        onnx.save(graph, str(tflite_onnx_path))
        onnx.checker.check_model(str(tflite_onnx_path))
        shape = (1, int(config.train.in_channels), *map(int, config.train.img_size))
        sample = np.random.default_rng(4).random(shape, dtype=np.float32)
        original = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        converted = ort.InferenceSession(str(tflite_onnx_path), providers=["CPUExecutionProvider"])
        expected = dict(zip((item.name for item in original.get_outputs()), original.run(None, {"input": sample})))
        actual = dict(zip((item.name for item in converted.get_outputs()), converted.run(None, {"input": sample})))
        _compare_outputs(expected, actual, "InstanceNormalization decomposition")
    litert_dir = export_dir / "litert"
    tflite_files = sorted(litert_dir.glob(f"{tflite_onnx_path.stem}_*.tflite"))
    if not tflite_files or overwrite or max(path.stat().st_mtime_ns for path in tflite_files) < tflite_onnx_path.stat().st_mtime_ns:
        command = [
            sys.executable, "-m", "onnx2tf.onnx2tf",
            "-i", str(tflite_onnx_path), "-o", str(litert_dir),
            "-tb", "flatbuffer_direct",
            "-ois", f"input:1,{int(config.train.in_channels)},{int(config.train.img_size[0])},{int(config.train.img_size[1])}",
            "-k", "input", "-coion", "-v", "error",
        ]
        print("Converting raw segmentation ONNX to TFLite...")
        subprocess.run(command, cwd=upstream, check=True)
        tflite_files = sorted(litert_dir.glob(f"{tflite_onnx_path.stem}_*.tflite"))
    if not tflite_files:
        raise FileNotFoundError(f"TFLite conversion produced no .tflite file in {litert_dir}")

    tflite_path = next((path for path in tflite_files if "float32" in path.name.lower()), tflite_files[0])
    interpreter = Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one TFLite image input, found {len(inputs)}")
    image_input = inputs[0]
    interpreter.set_tensor(image_input["index"], np.zeros(tuple(image_input["shape"]), dtype=image_input["dtype"]))
    interpreter.invoke()
    outputs = interpreter.get_output_details()
    if len(outputs) != 3:
        raise RuntimeError(f"Expected logits, boxes, and masks in TFLite, found {len(outputs)} outputs")
    report = export_dir / "tflite_export.json"
    shape = tuple(map(int, image_input["shape"]))
    sample = np.random.default_rng(4).random(shape, dtype=np.float32)
    onnx_session = ort.InferenceSession(str(tflite_onnx_path), providers=["CPUExecutionProvider"])
    expected = dict(zip((item.name for item in onnx_session.get_outputs()), onnx_session.run(None, {"input": sample})))
    interpreter.set_tensor(image_input["index"], sample)
    interpreter.invoke()
    actual = {item["name"]: interpreter.get_tensor(item["index"]) for item in outputs}
    payload = {
        "status": "validated",
        "run": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
        "onnx": str(tflite_onnx_path),
        "tflite": str(tflite_path),
        "outputs": [{"name": item["name"], "shape": list(map(int, item["shape"]))} for item in outputs],
        "postprocessing": "Raw logits, normalized boxes, and mask logits; apply confidence filtering, box scaling, and mask resizing in the consuming application.",
    }
    try:
        payload["parity"] = _compare_outputs(expected, actual, "ONNX-to-TFLite conversion")
    except RuntimeError as error:
        payload["status"] = "failed_parity"
        payload["error"] = str(error)
        _write_report(report, payload)
        if not allow_unvalidated:
            raise
        print(f"WARNING: TFLite predictions differ from ONNX; saved unvalidated artifact at {tflite_path}")
        return report
    _write_report(report, payload)
    return report


def export_tensorrt(run_dir: Path, checkpoint_name: str, overwrite: bool, workspace_gib: int, allow_unvalidated: bool = False) -> Path:
    if importlib.util.find_spec("tensorrt") is None:
        raise RuntimeError(
            "TensorRT is not installed in this Python environment. Install the CUDA-matched "
            "TensorRT package (for this project: tensorrt-cu13), restart the kernel, and retry."
        )
    import tensorrt as trt
    import numpy as np
    import onnxruntime as ort
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT export requires an NVIDIA GPU with CUDA available")
    run_dir, _, upstream, config, checkpoint = _load_run(run_dir, checkpoint_name)
    export_dir = run_dir / "exports" / "segmentation"
    onnx_path = _onnx_model(config, checkpoint, upstream, export_dir, raw=True, overwrite=overwrite)
    engine_path = export_dir / "model_raw_fp32.engine"
    if not engine_path.is_file() or overwrite or engine_path.stat().st_mtime_ns < onnx_path.stat().st_mtime_ns:
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        major = int(str(trt.__version__).split(".")[0])
        flags = 0 if major >= 11 else 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(logger)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(index)) for index in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse the segmentation ONNX model:\n{errors}")
        settings = builder.create_builder_config()
        settings.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gib * (1 << 30))
        engine = builder.build_serialized_network(network, settings)
        if engine is None:
            raise RuntimeError("TensorRT could not build the segmentation engine; inspect the builder output")
        engine_path.write_bytes(bytes(engine))
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    if not {"logits", "boxes", "masks"}.issubset(names):
        raise RuntimeError(f"TensorRT engine is missing segmentation outputs: {names}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("TensorRT could not create an execution context")
    shape = (1, int(config.train.in_channels), *map(int, config.train.img_size))
    sample = np.random.default_rng(4).random(shape, dtype=np.float32)
    buffers = {}
    for name in names:
        tensor_shape = tuple(context.get_tensor_shape(name))
        if min(tensor_shape) < 1 or trt.nptype(engine.get_tensor_dtype(name)) != np.float32:
            raise RuntimeError(f"Unsupported TensorRT tensor shape or dtype: {name} {tensor_shape}")
        if name == "input":
            buffer = torch.from_numpy(sample).cuda().contiguous()
        else:
            buffer = torch.empty(tensor_shape, dtype=torch.float32, device="cuda")
        buffers[name] = buffer
        context.set_tensor_address(name, buffer.data_ptr())
    stream = torch.cuda.Stream()
    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT segmentation inference failed")
    stream.synchronize()
    actual = {name: buffer.cpu().numpy() for name, buffer in buffers.items() if name != "input"}
    onnx_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    expected = dict(zip((item.name for item in onnx_session.get_outputs()), onnx_session.run(None, {"input": sample})))
    parity = _compare_outputs(expected, actual, "ONNX-to-TensorRT conversion", strict=False)
    validated = all(item["matches"] for item in parity.values())
    report = export_dir / "tensorrt_export.json"
    _write_report(report, {
        "status": "validated" if validated else "failed_parity",
        "run": run_dir.name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _checkpoint_hash(checkpoint),
        "onnx": str(onnx_path),
        "engine": str(engine_path),
        "precision": "fp32",
        "tensorrt_version": trt.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "tensors": names,
        "parity": parity,
        "postprocessing": "Raw logits, normalized boxes, and mask logits; apply confidence filtering, box scaling, and mask resizing in the consuming application.",
    })
    if not validated and not allow_unvalidated:
        raise RuntimeError(f"TensorRT predictions differ from the PyTorch-validated ONNX model. See {report}")
    if not validated:
        print(f"WARNING: TensorRT predictions differ from ONNX; saved unvalidated artifact at {engine_path}")
    return report


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    test = subparsers.add_parser("test", help="Evaluate a checkpoint on dataset_seg_rgb/images/test")
    test.add_argument("--run-dir", type=Path, required=True)
    test.add_argument("--dataset", type=Path, required=True)
    test.add_argument("--checkpoint", choices=("model.pt", "last.pt"), default="model.pt")
    test.add_argument("--device", default="cuda")
    test.add_argument("--output-dir", type=Path)
    export = subparsers.add_parser("export", help="Export a D-FINE-seg instance-segmentation checkpoint")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--checkpoint", choices=("model.pt", "last.pt"), default="model.pt")
    export.add_argument("--format", choices=("tflite", "tensorrt"), required=True)
    export.add_argument("--overwrite", action="store_true")
    export.add_argument("--allow-unvalidated", action="store_true", help="Save the artifact despite failed numerical parity")
    export.add_argument("--workspace-gib", type=int, default=2)
    args = parser.parse_args()
    if args.action == "test":
        report = evaluate_test(args.run_dir, args.dataset, args.checkpoint, args.device, args.output_dir)
    elif args.format == "tflite":
        report = export_tflite(args.run_dir, args.checkpoint, args.overwrite, args.allow_unvalidated)
    else:
        report = export_tensorrt(args.run_dir, args.checkpoint, args.overwrite, args.workspace_gib, args.allow_unvalidated)
    print("MODEL_TRAINER_RESULT=" + str(report))


if __name__ == "__main__":
    main()
