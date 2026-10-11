# YOLO26 RGB and HSV: calibrated NCNN INT8

This bundle targets the Raspberry Pi 4 CPU. It contains the RGB and HSV YOLO26 Nano models selected from the latest completed runs, calibrated with all 843 training images. NVIDIA exports are unchanged. D-FINE is omitted because its NCNN conversion failed runtime verification; its earlier ONNX export remains the Pi deployment option.

The recommended variants use INT8 convolutions with selected sensitive layers running in floating point. These are **mixed precision**, not models with every operation in INT8. NCNN's converter stores the remaining convolution weights in float16; the supplied runner uses float32 arithmetic for those layers. Exact INT8/float layer lists and file hashes are in each model's `quantization.json`.

## Raspberry Pi setup

Use 64-bit Raspberry Pi OS and Python 3.10 or newer with a matching NCNN build. Extract the ZIP and run from its directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python ncnn_runner.py photo.jpg --colour rgb
python ncnn_runner.py photo.jpg --colour hsv
```

Both commands take an ordinary camera image. The HSV runner converts it using OpenCV **HSV_FULL (H, S, V each 0–255)** before resizing; do not supply an already encoded HSV image. The runner preserves aspect ratio, letterboxes to 640×640 with channel values 114, converts to CHW float32 and divides by 255 exactly once.

Dependencies are pinned to the evaluated versions. If pip cannot find an NCNN wheel for your Pi's Python/OS combination, use the [official NCNN Python build instructions](https://github.com/Tencent/ncnn/tree/20260526/python) and the `20260526` tag with INT8 enabled. The `.param`/`.bin` files are portable; the runtime must be built for ARM, and the workstation's Windows executables are not needed on the Pi.

The trained YOLO26 one-to-one head is preserved. NCNN input is `in0`; output `out0` is `[8400,5]` containing pixel xyxy boxes and one-class probability. The runner selects the 300 highest scores on the CPU and maps boxes back to the original image. No additional NMS is needed for this head. Class 0 is `naartjie`.

## Choose the speed setting on your Pi

The workstation measurements are not Pi performance measurements. Run these commands sequentially on the Pi with its normal cooling and power supply:

```bash
python ncnn_runner.py --colour rgb --benchmark --tune --repeats 100 --output rgb_pi_benchmark.json
python ncnn_runner.py --colour hsv --benchmark --tune --repeats 100 --output hsv_pi_benchmark.json
```

This warms up each model, benchmarks 1, 2 and 4 inference threads, and chooses the fastest accuracy-qualified INT8 variant using median image-to-detections time. It also measures FP32 as a reference. It saves `device_selection.json`, which subsequent inference uses automatically. The timing includes image read/decode, colour conversion, letterboxing, inference and TopK; it excludes camera capture and display. The FP32 model is only a reference and is not selected by this INT8 tuning command.

## Calibration and accuracy assessment

Calibration uses the training split only, without augmentations. Images are processed by the same function used for inference and supplied to `ncnn2table` as float32 CHW `.npy` tensors. Calibration does not need labels. Validation and test accuracy assessment does need labels.

KL and ACIQ were calibrated separately for RGB and HSV. A full-range variant uses the maximum absolute activation measured on all training images; another preserves the known input range [0,1]. Sensitive layer groups were screened on 64 evenly spaced validation images. Final candidates are evaluated on all 504 validation images. Thresholds maximize validation micro-F1. The selected models and thresholds are frozen before their final 504-image test assessment.

The common evaluator uses COCO IoU thresholds 0.50–0.95, a score floor of 0.001 and a maximum of 100 detections per image. Exact duplicate ground-truth label rows are removed consistently. Its AP numbers differ from the trainers' saved metrics because the evaluation policies differ; quantization loss is measured against the FP32 deployment model under this same evaluator.

The conservative selection target is at most 0.2 percentage points of validation mAP50–95 loss. `selection.json` records the measured choice, eligible Pi benchmark alternatives and the selection policy. `calibration_manifest.json` records dataset counts and source hashes. `reports/` contains metrics and workstation benchmarks. These checks establish a measured tradeoff among the tested candidates; they do not prove a global optimum or guarantee performance on new camera conditions.

The installed EQ algorithm was attempted and stopped: its per-channel search had not progressed past the first channel after several minutes. It was not used to select or certify a model. A native conversion issue was also caught by auditing each convolution: this NCNN version requires removing both activation and weight scale rows to leave a layer in floating point. The final outputs must pass an exact layer audit, rather than relying on the converter's exit code.

## Reproduce on the training workstation

The source workspace contains `model_training/scripts/quantize_ncnn_deployment.py` and the May 26, 2026 NCNN command-line tools in the earlier deployment bundle. It stages source images inside `_work/`, generates calibration tensors and tables, converts models, evaluates predictions and checks layer precision. Intermediate images, prediction caches and tensor files are not required on the Pi and are excluded from the ZIP.

References: [NCNN quantization and mixed precision](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference), [NCNN 20260526 conversion source](https://github.com/Tencent/ncnn/blob/20260526/tools/quantize/ncnn2int8.cpp), [NCNN Python package](https://pypi.org/project/ncnn/1.0.20260526/).

Measured results are written to `RESULTS.md` after validation and testing finish.
