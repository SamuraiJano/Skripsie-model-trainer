# Detection deployment exports — 11 October 2026

Best checkpoints from the **four latest completed** Nano detector runs. No models were retrained and no original checkpoints were modified.

| Model | Source run | Saved validation mAP50–95 | NVIDIA | Raspberry Pi 4 |
|---|---|---:|---|---|
| YOLO26n RGB | `yolo26n-detect-20261006-221958` | 0.98359 | `yolo_rgb/nvidia/model_fp32.engine` | `yolo_rgb/raspberry_pi4/ncnn_fp32/` |
| YOLO26n HSV | `yolo26n-detect-20261007-101115` | 0.9778584754167007 | `yolo_hsv/nvidia/model_fp32.engine` | `yolo_hsv/raspberry_pi4/ncnn_fp32/` |
| D-FINE Nano RGB | `dfine-seg-n-detect-20261007-122047` | 0.9824894666671753 | `dfine_rgb/nvidia/model_fp32.engine` | `dfine_rgb/raspberry_pi4/model.onnx` |
| D-FINE Nano HSV | `dfine-seg-n-detect-20261008-170850` | 0.9791693687438965 | `dfine_hsv/nvidia/model_fp32.engine` | `dfine_hsv/raspberry_pi4/model.onnx` |

D-FINE is the ArgoHA/D-FINE-seg implementation in detection mode. `selection.json` records the selection rule, source paths and SHA-256 hashes. Each model directory also contains its original best checkpoint, source metadata and a portable FP32 `model.onnx`.

## Raspberry Pi 4

Use 64-bit Raspberry Pi OS and a Python environment with `numpy`, OpenCV, `ncnn` and `onnxruntime` installed. Only inference dependencies are needed; PyTorch is not needed for NCNN or ONNX inference. Choose package versions with ARM64 wheels for the Pi's Python version. Copy or extract the Raspberry Pi ZIP with its folder structure intact.

```bash
python3 -m pip install numpy opencv-python-headless ncnn onnxruntime
python3 infer.py camera_photo.jpg --model yolo_rgb
python3 infer.py camera_photo.jpg --model yolo_hsv
python3 infer.py camera_photo.jpg --model dfine_rgb
python3 infer.py camera_photo.jpg --model dfine_hsv
```

The default backend is NCNN for YOLO and ONNX Runtime CPU for D-FINE. All models accept one 640 × 640 image; the runner handles resizing and padding. ONNX is also included for both YOLO models if you want to compare the same runtime on the Pi:

```bash
python3 infer.py camera_photo.jpg --model yolo_rgb --backend onnx
```

The NCNN models were verified using the CPU runtime on this Windows workstation. No Raspberry Pi hardware was attached, so Pi latency, memory consumption and installation have not been measured.

## NVIDIA workstation

Engines were built and executed on **NVIDIA GeForce RTX 4060 Laptop GPU, Windows, TensorRT 11.2.1.2**, with batch 1, fixed 640 × 640 input, FP32 and 2 GiB build workspace. YOLO HSV and D-FINE RGB required rebuilds with TF32 disabled to pass numerical verification. All four delivered engines passed their respective verification checks.

Use the existing workstation environment:

```powershell
..\.venv\Scripts\python.exe infer.py camera_photo.jpg --model yolo_rgb --backend tensorrt
..\.venv\Scripts\python.exe infer.py camera_photo.jpg --model yolo_hsv --backend tensorrt
..\.venv\Scripts\python.exe infer.py camera_photo.jpg --model dfine_rgb --backend tensorrt
..\.venv\Scripts\python.exe infer.py camera_photo.jpg --model dfine_hsv --backend tensorrt
```

These are plain TensorRT engines, without an Ultralytics metadata header. Use the supplied runner or the TensorRT API. Engines are tied to the build GPU/runtime/platform; rebuild the included ONNX on another NVIDIA GPU, Linux system or Jetson. The engine can be rebuilt using TensorRT's `trtexec`, for example:

```bash
trtexec --onnx=yolo_rgb/model.onnx --saveEngine=yolo_rgb/nvidia/model_fp32.engine --noTF32
```

## Exact preprocessing and outputs

- Class 0 is `naartjie`.
- RGB input tensor channel order: R,G,B. HSV input tensor order: H,S,V.
- HSV conversion is **OpenCV `COLOR_BGR2HSV_FULL`**, with all three channels using uint8 0–255. Ordinary HSV with hue 0–179 is incompatible.
- Convert to HSV **before** resizing and padding. Preserve aspect ratio, resize with bilinear interpolation, center-pad with channel values 114, convert to NCHW float32 and divide by 255 once.
- `infer.py` expects an ordinary RGB/camera image even when an HSV model is selected. It performs the conversion. For dataset PNGs storing H,S,V in RGB file slots, use `--prepared-hsv` to avoid converting twice.
- YOLO ONNX/TensorRT outputs: `[1,300,6]` pixel `x1,y1,x2,y2,confidence,class_id`.
- YOLO NCNN output: `[8400,5]` pixel `x1,y1,x2,y2,probability`, from the **same trained one-to-one head**. The runner performs TopK selection on the CPU. No NMS is required. The stock Ultralytics NCNN exporter switches to the other head; these exports preserve the original one-to-one predictions instead.
- D-FINE outputs: `logits [1,300,1]` and `boxes [1,300,4]`, with normalized center-x, center-y, width, height. The runner applies sigmoid, selects predictions, converts boxes and removes letterbox padding.

## NCNN calibration

**The delivered NCNN models use FP32 and need no calibration. No INT8 or FP16 model is represented as delivered.**

INT8 post-training quantization requires representative unlabeled images from the intended camera and scene: fruit sizes and counts, lighting, backgrounds, occlusion and negative/empty scenes. A few hundred varied images are a sensible starting point; representativeness matters more than a fixed count. Use training or validation/development data, and keep the held-out test split separate for final evaluation.

The existing RGB validation split and its verified, matching HSV_FULL view are available locally. Use separate calibration inputs for RGB and HSV, with the exact preprocessing above. For NCNN, preprocessed float32 **CHW `.npy` inputs without a batch dimension** avoid accidental stretching, channel reversal or double normalization. Labels are not required to build the calibration table, but labels are required to measure accuracy after quantization. Measure exported-model mAP before selecting an INT8 model for deployment.

See [Tencent NCNN INT8 calibration instructions](https://github.com/Tencent/ncnn/wiki/quantized-int8-inference) and [Ultralytics NCNN export documentation](https://docs.ultralytics.com/integrations/ncnn).

## Verification and D-FINE NCNN limitation

Every delivered ONNX export was checked against its PyTorch checkpoint on three real validation images. Every delivered engine/NCNN model was then compared with its verified ONNX counterpart. D-FINE verification compares raw tensor outputs. YOLO verification matches boxes, classes and confidence at 0.01, 0.25 and 0.5 because unused low-confidence TopK rows can have different tie ordering. This is export parity verification, not a full new dataset accuracy evaluation.

The D-FINE NCNN conversion contained unsupported `torch.topk` and `torch.gather` operations. An attempted custom CPU implementation encountered a native runtime crash. **No D-FINE NCNN model is included or certified usable.** Both D-FINE models are delivered as verified ONNX for Raspberry Pi CPU inference. Failed conversion diagnostics remain in the corresponding export log; the unusable model files were removed.

Per-model verification JSON files and `bundle_verification.json` record the results. `inference_runner_verification.json` checks the portable runner across the delivered backends and verifies live RGB-to-HSV_FULL preprocessing against the stored paired HSV samples.
