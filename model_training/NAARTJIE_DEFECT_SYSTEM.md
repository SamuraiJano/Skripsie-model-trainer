# Naartjie Defect Detection System

This project trains and compares two YOLO instance segmentation models for naartjie defect detection:

- RGB YOLO segmentation model
- HSV YOLO segmentation model trained on HSV-converted image files

The classes are fixed:

```text
0: safe
1: minor_defect
2: major_defect
```

The HSV model is not using HSV augmentation as a substitute for HSV data. The HSV dataset is created by converting each RGB image with OpenCV `cv2.cvtColor(image, cv2.COLOR_BGR2HSV)`. Labels and split membership are copied unchanged from the RGB dataset.

During C++ HSV inference, the frame is converted to HSV and then channel-flipped to mirror Ultralytics' normal image-loading tensor order. This keeps the exported ONNX model consistent with how the HSV image files were seen during training.

## Project Layout

```text
project-root/
  background/
    CMakeLists.txt
    src/
    models/
    output_frames/
    logs/
  foreground/
  model_training/
    create_hsv_dataset.py
    train_rgb.py
    train_hsv.py
    export_models.py
    configs/
      naartjie_rgb.yaml
      naartjie_hsv.yaml
    dataset_rgb/
    dataset_hsv/
  README.md
```

## Dataset Contract

Place the RGB YOLO segmentation dataset here:

```text
model_training/dataset_rgb/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

Each image must have a matching YOLO segmentation `.txt` label file with the same stem under the matching `labels/{split}` directory.

## Create the HSV Dataset

From the project root:

```bash
cd model_training
python create_hsv_dataset.py
```

This writes:

```text
model_training/dataset_hsv/
  images/train|val|test
  labels/train|val|test
```

The images are HSV-converted. The labels are copied byte-for-byte, which guarantees the RGB and HSV datasets use the same labels and train/val/test split.

## Train RGB and HSV Models

Install the Python training dependencies in an environment with Ultralytics, OpenCV 5.0+, PyYAML, PyTorch, and ONNX:

```bash
conda env create -f environment.yml
conda activate naartjie-yolo26
python -m pip install -e .
```

Train the RGB model:

```bash
cd model_training
python train_rgb.py --model yolo26n-seg.pt --epochs 100 --imgsz 640
```

Train the HSV model:

```bash
python train_hsv.py --model yolo26n-seg.pt --epochs 100 --imgsz 640
```

The HSV training script disables Ultralytics HSV color augmentation with `hsv_h=0`, `hsv_s=0`, and `hsv_v=0`, because the image files are already HSV data.

Training outputs are saved under:

```text
model_training/runs/segment/naartjie_rgb/
model_training/runs/segment/naartjie_hsv/
```

## Export to ONNX

After both training runs have produced `best.pt`:

```bash
cd model_training
python export_models.py --imgsz 640
```

The exported deployment models are copied to:

```text
background/models/rgb_yolo_seg.onnx
background/models/hsv_yolo_seg.onnx
```

These ONNX files are generated artifacts and are not included as fake placeholders.

## Build and Run the C++ Background Service

Requirements:

- OpenCV 5.0+ development package
- ONNX Runtime C++ package
- CMake 3.20+
- A C++17 compiler
- 64-bit Raspberry Pi OS or another 64-bit Linux image for Pi deployment
- Active cooling for sustained inference runs

Configure and build:

```bash
cd background
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5
cmake --build build --config Release
```

Run:

```bash
./build/naartjie_backend --port 8080 --camera 0 --imgsz 640 --frame-skip 2 --threads 4
```

On Windows with a multi-config generator, the executable may be:

```powershell
.\build\Release\naartjie_backend.exe --port 8080 --camera 0 --imgsz 640 --frame-skip 2 --threads 4
```

The backend defaults are already tuned for a Raspberry Pi 4 without an AI HAT while preserving 640x640 model input: active-model-only inference, every second frame, four ONNX Runtime CPU threads, and a requested 640x480 15 FPS camera stream. Add `--run-both-models` only when you need RGB and HSV inference on every inference frame.

Background service endpoints:

```text
GET  /health
GET  /cameras
POST /set-camera      {"camera":0}
POST /start-camera
POST /stop-camera
POST /start-inference
POST /stop-inference
POST /start           legacy alias: start camera and inference
POST /stop            legacy alias: stop camera and inference
POST /set-model       {"model":"rgb"} or {"model":"hsv"} for the active Pi 4 inference model
GET  /status
GET  /latest-result
GET  /camera-frame.jpg     raw camera frame
GET  /latest-frame.jpg     raw camera frame compatibility alias
GET  /rgb-frame.jpg        RGB annotated frame when the RGB model is loaded
GET  /hsv-frame.jpg        HSV annotated frame when the HSV model is loaded
```

The background service saves the latest frames to:

```text
background/output_frames/latest-camera.jpg
background/output_frames/latest-rgb.jpg
background/output_frames/latest-hsv.jpg
```

Timing metrics are appended to:

```text
background/logs/metrics.csv
```

CSV columns:

```text
timestamp,model,fps,preprocess_ms,hsv_ms,inference_ms,total_ms,detection_count,major_defect_detected
```

## Run the React Dashboard

```bash
cd foreground
npm install
npm run dev
```

The dashboard defaults to:

```text
VITE_API_BASE_URL=http://localhost:8080
```

Set a different background-service URL by creating `foreground/.env`:

```text
VITE_API_BASE_URL=http://localhost:8080
```

The dashboard includes a camera selector, RGB/HSV active-model selector, camera start/stop controls, inference start/stop controls, a large raw camera feed, side-by-side RGB and HSV annotated feeds, per-model timing values, and a major-defect alert. If a model is not loaded or paused in Pi 4 single-model mode, its feed panel warns the user and the raw camera feed continues running.

## Benchmark RGB vs HSV

1. Start the background service with `--run-both-models` for simultaneous comparison, or leave the Pi 4 default active-model-only mode and switch models from the dashboard.
2. Make sure both `background/models/rgb_yolo_seg.onnx` and `background/models/hsv_yolo_seg.onnx` are present.
3. Run for a fixed number of frames or a fixed time window.
4. Compare rows in `background/logs/metrics.csv` by `model`.

Useful comparison metrics:

- Average FPS
- Average preprocessing time
- Average HSV conversion time
- Average inference time
- Average total frame time
- Major-defect count

## Quick Validation

Run the Python contract tests:

```bash
cd model_training
python -m pytest tests
```

Check C++ CLI help after building:

```bash
cd background
./build/naartjie_backend --help
```

Check foreground production build:

```bash
cd foreground
npm install
npm run build
```
