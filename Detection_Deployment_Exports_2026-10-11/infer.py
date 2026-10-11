"""Portable RGB/HSV detector inference for the exported deployment bundle.

Input is an ordinary camera/image file. HSV models perform HSV_FULL conversion
before letterboxing. Use --prepared-hsv only for stored H,S,V PNG datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent


def preprocess(path, colour, prepared_hsv=False):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    height, width = image.shape[:2]
    if colour == "hsv" and not prepared_hsv:
        channels = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    else:
        channels = image[:, :, ::-1]
    ratio = min(640 / height, 640 / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    channels = cv2.resize(channels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = 640 - new_w, 640 - new_h
    left, top = dw // 2, dh // 2
    channels = cv2.copyMakeBorder(channels, top, dh - top, left, dw - left,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(channels.transpose(2, 0, 1)[None], dtype=np.float32) / 255
    return tensor, {"width": width, "height": height, "ratio": ratio, "left": left, "top": top}


def yolo_topk(raw, maximum=300):
    scores = raw[:, 4:]
    indices = np.argsort(scores.max(axis=1))[::-1][:maximum]
    selected = scores[indices]
    flat = np.argsort(selected.reshape(-1))[::-1][:maximum]
    anchors, classes = indices[flat // scores.shape[1]], flat % scores.shape[1]
    return np.column_stack((raw[anchors, :4], scores[anchors, classes], classes)).astype(np.float32)


def dfine_detections(logits, boxes, maximum=300):
    scores = 1 / (1 + np.exp(-np.clip(logits[0], -50, 50)))
    flat = np.argsort(scores.reshape(-1))[::-1][:maximum]
    queries, classes = flat // scores.shape[1], flat % scores.shape[1]
    selected = boxes[0, queries]
    centers, sizes = selected[:, :2] * 640, selected[:, 2:] * 640
    xyxy = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1)
    return np.column_stack((xyxy, scores[queries, classes], classes)).astype(np.float32)


class Detector:
    def __init__(self, model, backend="auto", threads=4):
        self.family, self.colour = model.split("_")
        self.model_dir = ROOT / model
        if backend == "auto":
            backend = "ncnn" if self.family == "yolo" else "onnx"
        self.backend = backend
        if backend == "ncnn":
            if self.family != "yolo":
                raise ValueError("Use ONNX for D-FINE on Pi. Its NCNN conversion failed runtime verification.")
            import ncnn
            self.ncnn = ncnn
            self.net = ncnn.Net()
            self.net.opt.use_vulkan_compute = False
            self.net.opt.num_threads = threads
            self.net.opt.use_fp16_storage = False
            self.net.opt.use_fp16_arithmetic = False
            folder = self.model_dir / "raspberry_pi4/ncnn_fp32"
            if self.net.load_param(str(folder / "model.ncnn.param")) != 0:
                raise RuntimeError("NCNN parameter loading failed")
            if self.net.load_model(str(folder / "model.ncnn.bin")) != 0:
                raise RuntimeError("NCNN weight loading failed")
        elif backend == "onnx":
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            self.session = ort.InferenceSession(str(self.model_dir / "model.onnx"), sess_options=opts,
                                                providers=["CPUExecutionProvider"])
        else:
            import tensorrt as trt
            import torch
            self.torch = torch
            self.logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self.logger, "")
            self.runtime = trt.Runtime(self.logger)
            engine_path = self.model_dir / "nvidia/model_fp32.engine"
            self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
            if self.engine is None:
                raise RuntimeError("Engine is incompatible with this GPU/runtime; rebuild from model.onnx")
            self.context = self.engine.create_execution_context()
            names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_name = next(n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
            self.outputs = [n for n in names if n != self.input_name]
            self.buffers = {n: torch.empty(tuple(self.context.get_tensor_shape(n)), device="cuda",
                                          dtype=torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(n)))).dtype)
                            for n in names}
            for name, value in self.buffers.items():
                self.context.set_tensor_address(name, value.data_ptr())
            self.stream = torch.cuda.Stream()

    def raw(self, tensor):
        if self.backend == "ncnn":
            extractor = self.net.create_extractor()
            if extractor.input("in0", self.ncnn.Mat(tensor[0]).clone()) != 0:
                raise RuntimeError("NCNN input failed")
            status, output = extractor.extract("out0")
            if status != 0:
                raise RuntimeError("NCNN inference failed")
            return {"output0": yolo_topk(np.array(output))[None]}
        if self.backend == "onnx":
            values = self.session.run(None, {self.session.get_inputs()[0].name: tensor})
            return dict(zip((item.name for item in self.session.get_outputs()), values))
        self.buffers[self.input_name].copy_(self.torch.from_numpy(tensor))
        self.stream.wait_stream(self.torch.cuda.current_stream())
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT inference failed")
        self.stream.synchronize()
        return {name: self.buffers[name].cpu().numpy() for name in self.outputs}

    def predict(self, path, confidence=0.25, prepared_hsv=False):
        tensor, geometry = preprocess(path, self.colour, prepared_hsv)
        outputs = self.raw(tensor)
        detections = (outputs["output0"][0] if self.family == "yolo"
                      else dfine_detections(outputs["logits"], outputs["boxes"]))
        selected = detections[detections[:, 4] >= confidence].copy()
        selected[:, [0, 2]] = (selected[:, [0, 2]] - geometry["left"]) / geometry["ratio"]
        selected[:, [1, 3]] = (selected[:, [1, 3]] - geometry["top"]) / geometry["ratio"]
        selected[:, [0, 2]] = np.clip(selected[:, [0, 2]], 0, geometry["width"])
        selected[:, [1, 3]] = np.clip(selected[:, [1, 3]], 0, geometry["height"])
        return [{"class_id": int(row[5]), "class_name": "naartjie", "confidence": float(row[4]),
                 "box_xyxy": row[:4].tolist()} for row in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", choices=("yolo_rgb", "yolo_hsv", "dfine_rgb", "dfine_hsv"), required=True)
    parser.add_argument("--backend", choices=("auto", "ncnn", "onnx", "tensorrt"), default="auto")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prepared-hsv", action="store_true")
    args = parser.parse_args()
    predictions = Detector(args.model, args.backend, args.threads).predict(args.image, args.confidence, args.prepared_hsv)
    print(json.dumps({"model": args.model, "detections": predictions}, indent=2))


if __name__ == "__main__":
    main()
