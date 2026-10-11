"""Calibrate and assess verified YOLO NCNN exports without changing source models."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parents[2]
BUNDLE = WORKSPACE / "Detection_NCNN_INT8_2026-10-11"
SOURCE = WORKSPACE / "Detection_Deployment_Exports_2026-10-11"
DATA = WORKSPACE / "model_training/dataset_detect_rgb"
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(WORKSPACE / "model_training/src"))
from ncnn_runner import Detector, preprocess, benchmark
from naartjie_cv.detection_metrics import coco_ap, select_confidence, evaluate_test, EVALUATION_POLICY


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare():
    """Read only source data; stage inside the explicitly bounded new bundle."""
    assert BUNDLE.resolve().parent == WORKSPACE.resolve()
    manifest = dict(calibration_split="train", selection_split="val", final_assessment_split="test",
                    calibration_label_requirement=False, source=str(DATA), splits={})
    for split in ("train", "val", "test"):
        target = BUNDLE / "_work" / "images" / split
        target.mkdir(parents=True, exist_ok=True)
        gt = dict(info={}, categories=[dict(id=0, name="naartjie")], images=[], annotations=[])
        records, duplicates = [], 0
        for index, path in enumerate(sorted((DATA / "images" / split).glob("*"))):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            raw = path.read_bytes()
            copied = target / path.name
            copied.write_bytes(raw)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode {path}")
            height, width = image.shape[:2]
            image_id = index + 1
            gt["images"].append(dict(id=image_id, file_name=path.name, width=width, height=height))
            label = DATA / "labels" / split / (path.stem + ".txt")
            label_bytes = label.read_bytes()
            rows = np.asarray([list(map(float, line.split())) for line in label_bytes.decode().splitlines()
                               if line.strip()], dtype=np.float32).reshape(-1, 5)
            if len(rows):
                _, first = np.unique(rows, axis=0, return_index=True)
                duplicates += len(rows) - len(first)
                rows = rows[np.sort(first)]
            for category, cx, cy, bw, bh in rows:
                assert category == 0 and min(bw, bh) > 0
                bbox = [float((cx - bw / 2) * width), float((cy - bh / 2) * height),
                        float(bw * width), float(bh * height)]
                gt["annotations"].append(dict(id=len(gt["annotations"]) + 1, image_id=image_id,
                                             category_id=0, bbox=bbox, area=bbox[2] * bbox[3], iscrowd=0))
            records.append(dict(image=path.name, image_sha256=hashlib.sha256(raw).hexdigest(),
                                label_sha256=hashlib.sha256(label_bytes).hexdigest()))
        save(BUNDLE / "_work" / f"{split}_ground_truth.json", gt)
        manifest["splits"][split] = dict(count=len(records), annotation_count=len(gt["annotations"]),
                                          duplicate_label_rows_removed=duplicates, images=records)
        print(f"Staged {split}: {len(records)} images, {len(gt['annotations'])} boxes", flush=True)
    save(BUNDLE / "calibration_manifest.json", manifest)
    for colour in ("rgb", "hsv"):
        folder = BUNDLE / colour / "fp32"
        folder.mkdir(parents=True, exist_ok=True)
        for ext in ("param", "bin"):
            shutil.copyfile(SOURCE / f"yolo_{colour}/raspberry_pi4/ncnn_fp32/model.ncnn.{ext}", folder / f"model.{ext}")
    benchmark_images = BUNDLE / "benchmark_images"
    benchmark_images.mkdir(exist_ok=True)
    files = sorted((BUNDLE / "_work/images/val").glob("*"))
    for index in (0, len(files) // 2, len(files) - 1):
        shutil.copyfile(files[index], benchmark_images / files[index].name)


def tensors(colour, limit):
    folder = BUNDLE / "_work" / "calibration" / colour
    folder.mkdir(parents=True, exist_ok=True)
    sources = sorted((BUNDLE / "_work/images/train").glob("*"))
    if limit:
        sources = [sources[i] for i in np.linspace(0, len(sources) - 1, limit, dtype=int)]
    paths = []
    for index, source in enumerate(sources):
        target = folder / f"{index:04d}.npy"
        tensor, _ = preprocess(source, colour)
        np.save(target, tensor)
        paths.append(target.relative_to(BUNDLE).as_posix())
    listing = BUNDLE / "_work" / f"{colour}_{len(paths)}.list"
    listing.write_text("\n".join(paths) + "\n", encoding="utf-8")
    print(f"Prepared {len(paths)} {colour} calibration tensors (CHW float32 /255)", flush=True)
    return listing


def command(arguments, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        # EQ is a very costly per-channel search; use an explicit bound.
        completed = subprocess.run([str(a) for a in arguments], cwd=BUNDLE, stdout=stream, stderr=subprocess.STDOUT,
                                   timeout=600 if "method=eq" in arguments else None)
    if completed.returncode:
        raise RuntimeError(f"Command failed ({completed.returncode}): {log}\n{log.read_text()[-3000:]}")
    if "this layer need to be quantized, but no scale param" in log.read_text():
        raise RuntimeError(f"NCNN converter reported a missing scale despite returning success: {log}")
    print(f"Completed {log.stem}: {time.perf_counter() - started:.1f}s", flush=True)


def calibrate(colour, methods, limit=0):
    listing = tensors(colour, limit)
    source = BUNDLE / colour / "fp32"
    for method in methods:
        table = BUNDLE / "tables" / f"{colour}_{method}{'_smoke' if limit else ''}.table"
        table.parent.mkdir(exist_ok=True)
        log = BUNDLE / "logs" / f"{table.stem}.log"
        command([SOURCE / "tools/ncnn-tools/ncnn2table.exe", source / "model.param", source / "model.bin",
                 listing.relative_to(BUNDLE), table, "shape=[640,640,3]", "thread=8", f"method={method}", "type=1"], log)
        convert(colour, method + ("_smoke" if limit else ""), table)


def convert(colour, variant, table, exclude=(), preserve_input_range=False):
    if exclude or preserve_input_range:
        destination = BUNDLE / "tables" / f"{colour}_{variant}.table"
        lines = table.read_text().splitlines()
        # This NCNN release skips a layer only when its activation row is absent.
        # Removing just the weight row aborts the convolution pass, but the main
        # program still returns success. Remove BOTH rows and audit every layer.
        removed = set(exclude) | {name + "_param_0" for name in exclude}
        lines = [line for line in lines if line.split()[0] not in removed]
        if preserve_input_range:
            # Preprocessed RGB/HSV inputs have a known, fixed [0,1] domain.
            # Do not let a histogram dominated by padding clip them below 1.
            lines = ["conv_0 127.000000" if line.split()[0] == "conv_0" else line for line in lines]
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        table = destination
    source = BUNDLE / colour / "fp32"
    output = BUNDLE / colour / variant
    output.mkdir(parents=True, exist_ok=True)
    command([SOURCE / "tools/ncnn-tools/ncnn2int8.exe", source / "model.param", source / "model.bin",
             output / "model.param", output / "model.bin", table], BUNDLE / "logs" / f"{colour}_{variant}_convert.log")
    layers = output.joinpath("model.param").read_text().splitlines()[2:]
    conv = [line for line in layers if line.split()[0] in ("Convolution", "ConvolutionDepthWise", "InnerProduct")]
    quantized = [line.split()[1] for line in conv if any(t.startswith("8=") and t != "8=0" for t in line.split())]
    if not quantized:
        raise ValueError("Converter did not quantize any convolution weights")
    floating = [line.split()[1] for line in conv if line.split()[1] not in quantized]
    if set(floating) != set(exclude):
        raise ValueError(f"Precision audit failed: requested {list(exclude)}, actual float layers {floating}")
    save(output / "quantization.json", dict(table=table.relative_to(BUNDLE).as_posix(),
          quantized_convolution_layers=quantized, float_convolution_layers=floating,
          total_convolution_layers=len(conv), weight_bytes=(output / "model.bin").stat().st_size,
          preserve_input_range=preserve_input_range, requested_float_layers=list(exclude),
          floating_weight_storage="float16 (converter default); runtime arithmetic float32",
          param_sha256=sha256(output / "model.param"), weights_sha256=sha256(output / "model.bin")))


def evaluate(colour, variant, split="val", limit=0):
    detector = Detector(colour, variant, 4)
    gt = json.loads((BUNDLE / "_work" / f"{split}_ground_truth.json").read_text())
    if limit:
        chosen = [gt["images"][i] for i in np.linspace(0, len(gt["images"]) - 1, limit, dtype=int)]
        ids = {image["id"] for image in chosen}
        gt["images"] = chosen
        gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in ids]
    predictions, times = [], []
    for index, image in enumerate(gt["images"]):
        path = BUNDLE / "_work/images" / split / image["file_name"]
        started = time.perf_counter()
        output = detector.predict(path, .001)
        times.append((time.perf_counter() - started) * 1000)
        for detection in output:
            x1, y1, x2, y2 = detection["box_xyxy"]
            predictions.append(dict(image_id=image["id"], category_id=0, score=detection["confidence"],
                                    bbox=[x1, y1, x2 - x1, y2 - y1]))
        if (index + 1) % 100 == 0:
            print(f"{colour}/{variant}/{split}: {index+1}/{len(gt['images'])}", flush=True)
    metrics = coco_ap(gt, predictions)
    report = dict(colour=colour, variant=variant, split=split, images=len(gt["images"]),
                  annotations=len(gt["annotations"]), metrics=metrics, evaluation_policy=EVALUATION_POLICY,
                  median_image_to_detections_ms=float(np.median(times)),
                  timing_note="Diagnostic only; calibration/evaluation may run concurrently. Use the separate warmed benchmark.",
                  model_bytes=sum((detector.folder / f"model.{ext}").stat().st_size for ext in ("param", "bin")))
    if split == "val":
        report["confidence_selection"] = select_confidence(gt, predictions)
    elif not limit:
        selected = json.loads((BUNDLE / "reports" / f"{colour}_{variant}_val.json").read_text())["confidence_selection"]
        report["test_operating_metrics"] = evaluate_test(gt, predictions, selected)
    name = f"{colour}_{variant}_{split}" + ("_smoke" if limit else "")
    save(BUNDLE / "reports" / f"{name}.json", report)
    save(BUNDLE / "_work/predictions" / f"{name}.json", predictions)
    print(f"RESULT {name}: AP50-95={metrics['mAP50-95']:.6f}, AP50={metrics['mAP50']:.6f}, "
          f"median={report['median_image_to_detections_ms']:.1f}ms", flush=True)
    return report


def cleanup_tensors(colour):
    folder = (BUNDLE / "_work/calibration" / colour).resolve()
    assert folder.parent == (BUNDLE / "_work/calibration").resolve()
    for path in folder.glob("*.npy"):
        path.unlink()


def minmax_table(colour):
    """Use measured activation extrema instead of clipping the KL histogram."""
    rows = (BUNDLE / "tables" / f"{colour}_kl.table").read_text().splitlines()
    text = (BUNDLE / "logs" / f"{colour}_kl.log").read_text()
    maxima = {name: float(maximum) for name, maximum in
              re.findall(r"^(\S+)\s+: max =\s+([0-9.eE+-]+)", text, re.MULTILINE)}
    assert len(maxima) == 102
    rows = [f"{line.split()[0]} {127 / maxima[line.split()[0]]:.8f}"
            if line.split()[0] in maxima else line for line in rows]
    target = BUNDLE / "tables" / f"{colour}_minmax.table"
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
    convert(colour, "minmax", target)


def screen(colour, bases, limit=64, refine=False):
    """Screen explicit precision groups on a fixed evenly spaced validation subset."""
    output = [f"conv_{i}" for i in (78, 81, 84, 87, 90, 93)]
    regression = [f"conv_{i}" for i in range(76, 85)] + [f"conv_{i}" for i in (87, 90, 93)]
    head = [f"conv_{i}" for i in range(76, 94)] + [f"convdw_{i}" for i in range(213, 219)]
    stem2 = [f"conv_{i}" for i in range(2)]
    stem6 = [f"conv_{i}" for i in range(6)]
    stem11 = [f"conv_{i}" for i in range(11)]
    groups = dict(output=output, regression=regression, head=head,
                  stem2_output=stem2 + output, stem6_output=stem6 + output,
                  stem6_head=stem6 + head, stem11_head=stem11 + head)
    if refine:
        groups = dict(stem1_head=["conv_0"] + head, stem2_head=stem2 + head,
                      stem2_regression=stem2 + regression, stem6_regression=stem6 + regression)
    evaluate(colour, "fp32", limit=limit)
    for base in bases:
        table = BUNDLE / "tables" / f"{colour}_{base}.table"
        evaluate(colour, base, limit=limit)
        for suffix, excluded in groups.items():
            variant = f"{base}_{suffix}"
            convert(colour, variant, table, excluded)
            evaluate(colour, variant, limit=limit)


def freeze():
    """Freeze validation-only choices before test data is evaluated."""
    destination = BUNDLE / "selection.json"
    if destination.exists():
        raise ValueError("Selection is already frozen; do not reselect using test results")
    result = {}
    for colour in ("rgb", "hsv"):
        reports = [json.loads(path.read_text()) for path in (BUNDLE / "reports").glob(f"{colour}_*_val.json")]
        assert all(report["images"] == 504 for report in reports)
        reference = next(report for report in reports if report["variant"] == "fp32")
        eligible = [report for report in reports if report["variant"] != "fp32"
                    and all(reference["metrics"][metric] - report["metrics"][metric] <= .002
                            for metric in ("mAP50-95", "mAP50"))]
        if not eligible:
            raise ValueError(f"No {colour} INT8 variant met the conservative accuracy limit")
        # Identical precision/layout has the same convolution workload. Keep the
        # calibration choice with best AP, instead of benchmarking duplicate graphs.
        by_precision = {}
        for report in eligible:
            audit = json.loads((BUNDLE / colour / report["variant"] / "quantization.json").read_text())
            key = tuple(audit["float_convolution_layers"])
            if key not in by_precision or report["metrics"]["mAP50-95"] > by_precision[key]["metrics"]["mAP50-95"]:
                by_precision[key] = report
        eligible = list(by_precision.values())
        best = max(eligible, key=lambda report: report["metrics"]["mAP50-95"])
        result[colour] = dict(variant=best["variant"], confidence=best["confidence_selection"]["confidence"],
                             pi_benchmark_candidates=[r["variant"] for r in sorted(eligible, key=lambda r: r["variant"])],
                             validation_map50_95=best["metrics"]["mAP50-95"],
                             fp32_validation_map50_95=reference["metrics"]["mAP50-95"],
                             validation_drop_percentage_points=100 * (reference["metrics"]["mAP50-95"] - best["metrics"]["mAP50-95"]),
                             validation_report_sha256=sha256(BUNDLE / "reports" / f"{colour}_{best['variant']}_val.json"))
    result["policy"] = dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                            calibration="all 843 training images", selection="all 504 validation images; no test scores",
                            accuracy_limit_percentage_points=.2,
                            accuracy_limit_metrics=["mAP50-95", "mAP50"],
                            default="maximum validation mAP50-95 among eligible INT8 candidates",
                            duplicate_layout_rule="For identical float/INT8 layer placement, keep the highest validation AP calibration",
                            pi_tuning="fastest median image-to-detections time among accuracy-qualified INT8 candidates, with 1/2/4 threads",
                            pi_measured=False, format="NCNN mixed-precision INT8", ncnn_version="1.0.20260526")
    save(destination, result)
    print(json.dumps(result, indent=2), flush=True)


def test_frozen(colour):
    selected = json.loads((BUNDLE / "selection.json").read_text())[colour]
    for variant in ["fp32", *selected["pi_benchmark_candidates"]]:
        evaluate(colour, variant, split="test")


def benchmark_frozen():
    selection = json.loads((BUNDLE / "selection.json").read_text())
    for colour in ("rgb", "hsv"):
        variants = ["fp32", *selection[colour]["pi_benchmark_candidates"]]
        for variant in variants:
            assert (BUNDLE / "reports" / f"{colour}_{variant}_test.json").is_file()
        result = benchmark(colour, variants, 4, repeats=60, warmup=15)
        save(BUNDLE / "reports" / f"{colour}_workstation_benchmark.json", result)
        print(json.dumps(result, indent=2), flush=True)


def package():
    selection = json.loads((BUNDLE / "selection.json").read_text())
    rows, speed_rows = [], []
    include = [BUNDLE / name for name in ("README.md", "requirements.txt", "ncnn_runner.py", "selection.json",
               "source_models.json", "calibration_manifest.json", "runner_verification.json")]
    for colour in ("rgb", "hsv"):
        benchmark_report = BUNDLE / "reports" / f"{colour}_workstation_benchmark.json"
        measured = json.loads(benchmark_report.read_text())
        assert measured["repeats"] >= 60 and measured["warmup"] >= 15
        speeds = {r["variant"]: r for r in measured["results"]}
        reference_val = json.loads((BUNDLE / "reports" / f"{colour}_fp32_val.json").read_text())
        reference_test = json.loads((BUNDLE / "reports" / f"{colour}_fp32_test.json").read_text())
        for variant in ["fp32", *selection[colour]["pi_benchmark_candidates"]]:
            validation = json.loads((BUNDLE / "reports" / f"{colour}_{variant}_val.json").read_text())
            test = json.loads((BUNDLE / "reports" / f"{colour}_{variant}_test.json").read_text())
            assert validation["images"] == test["images"] == 504
            folder = BUNDLE / colour / variant
            audit_path = folder / "quantization.json"
            if audit_path.exists():
                audit = json.loads(audit_path.read_text())
                assert set(audit["float_convolution_layers"]) == set(audit["requested_float_layers"])
                assert len(audit["float_convolution_layers"]) + len(audit["quantized_convolution_layers"]) == 102
                assert sha256(folder / "model.bin") == audit["weights_sha256"]
                assert sha256(folder / "model.param") == audit["param_sha256"]
                quant_count = len(audit["quantized_convolution_layers"])
                include.extend([audit_path, BUNDLE / audit["table"]])
            else:
                quant_count = 0
                for ext in ("param", "bin"):
                    assert sha256(folder / f"model.{ext}") == sha256(SOURCE / f"yolo_{colour}/raspberry_pi4/ncnn_fp32/model.ncnn.{ext}")
            default = " (default)" if selection[colour]["variant"] == variant else ""
            v_ap = validation["metrics"]["mAP50-95"]
            t_ap = test["metrics"]["mAP50-95"]
            size_mb = validation["model_bytes"] / 1e6
            rows.append(f"| {colour.upper()} | {variant}{default} | {quant_count}/102 | {v_ap*100:.4f}% | "
                        f"{100*(reference_val['metrics']['mAP50-95']-v_ap):.4f} pp | {t_ap*100:.4f}% | "
                        f"{100*(reference_test['metrics']['mAP50-95']-t_ap):.4f} pp | {size_mb:.3f} MB |")
            speed = speeds[variant]
            speed_rows.append(f"| {colour.upper()} | {variant} | {speed['inference_topk_median_ms']:.2f} | "
                              f"{speed['inference_topk_p95_ms']:.2f} | {speed['image_to_detections_median_ms']:.2f} | "
                              f"{speed['image_to_detections_p95_ms']:.2f} |")
            include.extend(folder / f"model.{ext}" for ext in ("param", "bin"))
        include.append(benchmark_report)
    source = json.loads((BUNDLE / "source_models.json").read_text())
    for colour in ("rgb", "hsv"):
        assert sha256(SOURCE / f"yolo_{colour}" / "best.pt") == source[f"yolo_{colour}"]["checkpoint_sha256"]
    results = """# Measured NCNN INT8 results

All percentages below are COCO mAP50–95 under the common evaluator (max 100 detections per image), not the saved trainer metrics. Calibration: 843 training images. Validation: 504 images. Test: 504 images. Defaults and all speed alternatives were frozen using validation before test assessment. Positive loss means lower AP than FP32; `pp` means percentage points.

| Colour | Variant | INT8 convolutions | Validation AP | Validation loss | Test AP | Test loss | Param + weights |
|---|---|---:|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

## Workstation speed (not Raspberry Pi speed)

These measurements ran sequentially after calibration and all accuracy evaluations finished, with four NCNN CPU threads, 15 warmups and 60 timed repetitions per path on three real images. OpenCV uses one thread. FP16 arithmetic/storage and Vulkan are disabled. Image-to-detections includes local read/decode and exact preprocessing; inference includes CPU TopK. Median and p95 are milliseconds. Full machine/package details are in `source_models.json` and each benchmark report.

| Colour | Variant | Inference median | Inference p95 | Image-to-detections median | Image-to-detections p95 |
|---|---|---:|---:|---:|---:|
""" + "\n".join(speed_rows) + """

The defaults prioritize the highest measured validation accuracy among tested INT8 variants within 0.2 pp of FP32 for both AP50–95 and AP50. Raspberry Pi latency has not been measured. Run the supplied `--benchmark --tune` command on the Pi to select the fastest of the accuracy-qualified alternatives and the best of 1/2/4 threads. The models retain floating-point stems and detection layers to avoid the much larger losses observed with every convolution quantized.

Fully quantized KL, ACIQ and full-range variants, and smaller protected layer groups, were assessed before selection. Exact comparison reports are included. The 64-image screenings are summarized separately and were followed by full validation for finalists. EQ did not complete and is excluded from the results.
"""
    (BUNDLE / "RESULTS.md").write_text(results, encoding="utf-8")
    include.append(BUNDLE / "RESULTS.md")
    include.extend(path for path in (BUNDLE / "reports").glob("*.json") if "_smoke" not in path.name)
    screenings = []
    for path in sorted((BUNDLE / "reports").glob("*_val_smoke.json")):
        record = json.loads(path.read_text())
        if record["images"] != 64:
            continue
        audit_path = BUNDLE / record["colour"] / record["variant"] / "quantization.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            assert set(audit["float_convolution_layers"]) == set(audit["requested_float_layers"])
            record["quantization_audit"] = audit
        screenings.append(record)
    save(BUNDLE / "screening_summary.json", screenings)
    include.append(BUNDLE / "screening_summary.json")
    include.extend((BUNDLE / "benchmark_images").glob("*.jpg"))
    include = sorted(set(include))
    assert all(path.is_file() and path.resolve().is_relative_to(BUNDLE.resolve()) for path in include)
    manifest = dict(status="PASS", entries=[dict(path=path.relative_to(BUNDLE).as_posix(),
                     bytes=path.stat().st_size, sha256=sha256(path)) for path in include],
                    selection_sha256=sha256(BUNDLE / "selection.json"),
                    pi_speed_measured=False, native_runtime_verified="all seven delivered model variants, 504 validation and 504 test images each")
    save(BUNDLE / "bundle_verification.json", manifest)
    include.append(BUNDLE / "bundle_verification.json")
    archive = BUNDLE / "Raspberry_Pi4_NCNN_INT8_2026-10-11.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as stream:
        for path in include:
            stream.write(path, path.relative_to(BUNDLE).as_posix())
    with zipfile.ZipFile(archive) as stream:
        bad = stream.testzip()
        assert bad is None, bad
        assert len(stream.namelist()) == len(include)
    save(BUNDLE / "archive_verification.json", dict(status="PASS", archive=archive.name, bytes=archive.stat().st_size,
         sha256=sha256(archive), files=len(include), crc_check="PASS"))
    print(f"Packaged {len(include)} verified files: {archive} ({archive.stat().st_size} bytes)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "calibrate", "evaluate", "sweep", "screen", "refine", "convert", "minmax", "benchmark", "benchmark-frozen", "cleanup", "freeze", "test-frozen", "package"))
    parser.add_argument("--colour", choices=("rgb", "hsv"), default="rgb")
    parser.add_argument("--methods", nargs="+", default=["kl", "aciq"])
    parser.add_argument("--variant", default="fp32")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--table", type=Path)
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--preserve-input-range", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "calibrate":
        calibrate(args.colour, args.methods, args.limit)
    elif args.action == "evaluate":
        evaluate(args.colour, args.variant, args.split, args.limit)
    elif args.action == "sweep":
        for variant in args.methods:
            evaluate(args.colour, variant, args.split, args.limit)
    elif args.action == "convert":
        convert(args.colour, args.variant, args.table.resolve(), args.exclude, args.preserve_input_range)
    elif args.action == "cleanup":
        cleanup_tensors(args.colour)
    elif args.action == "minmax":
        minmax_table(args.colour)
    elif args.action == "screen":
        screen(args.colour, args.methods, args.limit or 64)
    elif args.action == "refine":
        screen(args.colour, args.methods, args.limit or 64, refine=True)
    elif args.action == "freeze":
        freeze()
    elif args.action == "test-frozen":
        test_frozen(args.colour)
    elif args.action == "package":
        package()
    elif args.action == "benchmark-frozen":
        benchmark_frozen()
    else:
        variants = args.methods
        result = benchmark(args.colour, variants, 4, repeats=60, warmup=15)
        save(BUNDLE / "reports" / f"{args.colour}_workstation_benchmark.json", result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
