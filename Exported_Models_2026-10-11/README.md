# All exported detection models

All NCNN, ONNX and NVIDIA TensorRT exports are collected here. This folder contains copies; original exports and training checkpoints remain in their original locations.

| Model | Recommended Raspberry Pi NCNN INT8 | Portable ONNX | NVIDIA TensorRT FP32 |
|---|---|---|---|
| YOLO RGB | `NCNN_INT8/rgb/minmax_stem11_head/model.param` + `model.bin` | `yolo_rgb/model.onnx` | `yolo_rgb/nvidia/model_fp32.engine` |
| YOLO HSV | `NCNN_INT8/hsv/minmax_stem11_head/model.param` + `model.bin` | `yolo_hsv/model.onnx` | `yolo_hsv/nvidia/model_fp32.engine` |
| D-FINE RGB | Use ONNX; NCNN conversion failed runtime verification | `dfine_rgb/model.onnx` | `dfine_rgb/nvidia/model_fp32.engine` |
| D-FINE HSV | Use ONNX; NCNN conversion failed runtime verification | `dfine_hsv/model.onnx` | `dfine_hsv/nvidia/model_fp32.engine` |

Keep each NCNN `.param`/`.bin` pair together. The recommended NCNN models use INT8 mixed precision to preserve accuracy. Their validated speed alternatives and FP32 references are included under `NCNN_INT8/`. The original FP32 NCNN pairs also remain under the YOLO folders' `raspberry_pi4/ncnn_fp32/` paths so the original runner remains usable.

## NCNN INT8 on the Pi

Use an ordinary camera image for either colour model; HSV conversion is handled by the runner. From this folder:

```bash
python3 -m pip install -r NCNN_INT8/requirements.txt
python3 NCNN_INT8/ncnn_runner.py photo.jpg --colour rgb
python3 NCNN_INT8/ncnn_runner.py photo.jpg --colour hsv
python3 NCNN_INT8/ncnn_runner.py --colour rgb --benchmark --tune --output rgb_pi_benchmark.json
python3 NCNN_INT8/ncnn_runner.py --colour hsv --benchmark --tune --output hsv_pi_benchmark.json
```

See `NCNN_INT8/README.md` for setup and `NCNN_INT8/RESULTS.md` for measured accuracy and workstation speed. Pi hardware speed has not been measured.

## ONNX and NVIDIA engines

The supplied `infer.py` uses the four model folders here. Examples from this folder:

```bash
python3 infer.py photo.jpg --model dfine_rgb --backend onnx
python3 infer.py photo.jpg --model yolo_rgb --backend onnx
```

```powershell
..\.venv\Scripts\python.exe infer.py photo.jpg --model yolo_rgb --backend tensorrt
..\.venv\Scripts\python.exe infer.py photo.jpg --model dfine_hsv --backend tensorrt
```

ONNX inference requires NumPy, OpenCV and ONNX Runtime. The engines were built for Windows, NVIDIA RTX 4060 Laptop GPU and TensorRT 11.2.1.2. For a different GPU, platform or TensorRT version, rebuild an engine from the included ONNX. They are plain TensorRT engines without an Ultralytics metadata header.

All models use 640x640 inputs and class 0 `naartjie`. HSV models require HSV_FULL conversion before letterboxing, with all channels ranging from 0 to 255; both runners implement this correctly. `source_selection.json` records the four source runs. `collection_manifest.json` records source locations, sizes and SHA-256 hashes; every copied model file was checked against its verified export.
