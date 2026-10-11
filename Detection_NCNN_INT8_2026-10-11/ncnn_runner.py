"""Portable YOLO26 NCNN inference and device benchmark. Input: ordinary camera RGB image."""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np

# NCNN owns the inference threads; avoid OpenCV oversubscription on a four-core Pi.
cv2.setNumThreads(1)

ROOT = Path(__file__).resolve().parent


def preprocess(path, colour):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {path}")
    height, width = image.shape[:2]
    channels = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL) if colour == "hsv" else image[:, :, ::-1]
    ratio = min(640 / height, 640 / width)
    new_w, new_h = round(width * ratio), round(height * ratio)
    channels = cv2.resize(channels, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw, dh = 640 - new_w, 640 - new_h
    left, top = dw // 2, dh // 2
    channels = cv2.copyMakeBorder(channels, top, dh - top, left, dw - left,
                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))
    tensor = np.ascontiguousarray(channels.transpose(2, 0, 1), dtype=np.float32) / 255
    return tensor, dict(width=width, height=height, ratio=ratio, left=left, top=top)


def topk(raw):
    if raw.shape != (8400, 5) or not np.isfinite(raw).all():
        raise ValueError(f"Invalid NCNN output: {raw.shape}")
    indices = np.argsort(raw[:, 4])[::-1][:300]
    return raw[indices].copy()


class Detector:
    def __init__(self, colour, variant="recommended", threads=None):
        import ncnn
        self.colour, self.ncnn = colour, ncnn
        if variant == "recommended":
            selection = json.loads((ROOT / "selection.json").read_text())[colour]
            variant = selection["variant"]
            device_file = ROOT / "device_selection.json"
            if device_file.exists():
                device_choice = json.loads(device_file.read_text()).get(colour)
                if device_choice and device_choice["variant"] in selection["pi_benchmark_candidates"]:
                    variant = device_choice["variant"]
                    if threads is None:
                        threads = device_choice["threads"]
        self.variant = variant
        self.folder = ROOT / colour / variant
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = False
        self.net.opt.num_threads = 4 if threads is None else threads
        self.net.opt.use_fp16_packed = False
        self.net.opt.use_fp16_storage = False
        self.net.opt.use_fp16_arithmetic = False
        self.net.opt.use_int8_inference = True
        if self.net.load_param(str(self.folder / "model.param")) != 0:
            raise RuntimeError("NCNN parameter loading failed")
        if self.net.load_model(str(self.folder / "model.bin")) != 0:
            raise RuntimeError("NCNN weight loading failed")

    def raw(self, tensor):
        extractor = self.net.create_extractor()
        if extractor.input("in0", self.ncnn.Mat(tensor).clone()) != 0:
            raise RuntimeError("NCNN input failed")
        status, output = extractor.extract("out0")
        if status != 0:
            raise RuntimeError("NCNN inference failed")
        return topk(np.array(output))

    def predict(self, path, confidence=0.25):
        tensor, geometry = preprocess(path, self.colour)
        detections = self.raw(tensor)
        selected = detections[detections[:, 4] >= confidence].copy()
        selected[:, [0, 2]] = (selected[:, [0, 2]] - geometry["left"]) / geometry["ratio"]
        selected[:, [1, 3]] = (selected[:, [1, 3]] - geometry["top"]) / geometry["ratio"]
        selected[:, [0, 2]] = np.clip(selected[:, [0, 2]], 0, geometry["width"])
        selected[:, [1, 3]] = np.clip(selected[:, [1, 3]], 0, geometry["height"])
        return [dict(class_id=0, class_name="naartjie", confidence=float(row[4]), box_xyxy=row[:4].tolist())
                for row in selected if row[2] > row[0] and row[3] > row[1]]


def benchmark(colour, variants, threads=4, repeats=100, warmup=15, images=None):
    import ncnn
    paths = images or sorted((ROOT / "benchmark_images").glob("*.jpg"))
    if not paths:
        raise ValueError("Benchmark requires at least one real input image")
    tensors = [preprocess(path, colour)[0] for path in paths]
    results = []
    for variant in variants:
        detector = Detector(colour, variant, threads)
        for index in range(warmup):
            detector.raw(tensors[index % len(tensors)])
        raw_ms, total_ms = [], []
        for index in range(repeats):
            tensor = tensors[index % len(tensors)]
            started = time.perf_counter()
            detector.raw(tensor)
            raw_ms.append((time.perf_counter() - started) * 1000)
        for index in range(repeats):
            started = time.perf_counter()
            detector.predict(paths[index % len(paths)])
            total_ms.append((time.perf_counter() - started) * 1000)
        results.append(dict(variant=variant, inference_topk_median_ms=float(np.median(raw_ms)),
                            inference_topk_p95_ms=float(np.percentile(raw_ms, 95)),
                            image_to_detections_median_ms=float(np.median(total_ms)),
                            image_to_detections_p95_ms=float(np.percentile(total_ms, 95))))
        del detector
    return dict(colour=colour, platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                ncnn=ncnn.__version__, threads=threads, repeats=repeats, warmup=warmup, results=results,
                note="Image-to-detections includes local image read/decode, preprocessing and TopK. No camera/display time.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, nargs="?")
    parser.add_argument("--colour", choices=("rgb", "hsv"), required=True)
    parser.add_argument("--variant", default="recommended")
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--threads", type=int, help="Inference threads (default 4, or the saved device choice)")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--tune", action="store_true", help="Benchmark 1,2,4 threads and save the fastest accuracy-qualified INT8 choice on this device")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.benchmark:
        selection = json.loads((ROOT / "selection.json").read_text())[args.colour]
        variants = ["fp32", *selection["pi_benchmark_candidates"]]
        variants = list(dict.fromkeys(variants))
        if args.tune:
            runs = [benchmark(args.colour, variants, threads, args.repeats, args.warmup) for threads in (1, 2, 4)]
            eligible = [dict(row, threads=run["threads"]) for run in runs for row in run["results"]
                        if row["variant"] in selection["pi_benchmark_candidates"]]
            fastest = min(eligible, key=lambda row: row["image_to_detections_median_ms"])
            device_file = ROOT / "device_selection.json"
            choices = json.loads(device_file.read_text()) if device_file.exists() else {}
            choices[args.colour] = fastest
            device_file.write_text(json.dumps(choices, indent=2) + "\n", encoding="utf-8")
            result = dict(runs=runs, selected=fastest)
        else:
            result = benchmark(args.colour, variants, args.threads or 4, args.repeats, args.warmup)
    else:
        if args.image is None:
            parser.error("Supply an image, or --benchmark")
        detector = Detector(args.colour, args.variant, args.threads)
        confidence = args.confidence
        if confidence is None:
            report = ROOT / "reports" / f"{args.colour}_{detector.variant}_val.json"
            confidence = json.loads(report.read_text())["confidence_selection"]["confidence"] if report.exists() else .25
        result = dict(colour=args.colour, variant=detector.variant, confidence=confidence,
                      detections=detector.predict(args.image, confidence))
    message = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(message + "\n", encoding="utf-8")
    print(message)


if __name__ == "__main__":
    main()
