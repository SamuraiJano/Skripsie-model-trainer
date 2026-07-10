# New Model Trainer

This workspace is organized into three main folders:

```text
background/       C++ background services and deployment inference code
foreground/       React dashboard
model_training/   Python training code, notebooks, configs, datasets, and tests
```

## Model Training

```powershell
cd model_training
conda env create -f environment.yml
conda activate naartjie-yolo26
python -m pip install -e .
python -m pytest tests
```

The naartjie YOLO26 workflow starts at:

```text
model_training/notebooks/yolo26_naartjie_training.ipynb
```

The naartjie RGB/HSV training scripts are also in `model_training/`.

## Background

The C++ background service is in `background/`. The NCNN image-folder inference app is in:

```text
background/ncnn_infer/
```

The background service executable builds as `naartjie_backend`.
It requires OpenCV 5.0+ and ONNX Runtime; pass `OpenCV_DIR` to CMake if your OpenCV 5 install is not on the default package path.
On this Windows workspace, the local OpenCV 5 and ONNX Runtime packages are under `third_party/`, and the backend can be rebuilt with:

```powershell
powershell -ExecutionPolicy Bypass -File background\build_windows.ps1
```

Its defaults are tuned for a Raspberry Pi 4 without an AI HAT:

- `--imgsz 640`
- `--frame-skip 2`
- `--threads 4`
- `--camera-width 640 --camera-height 480 --camera-fps 15`
- active model only; pass `--run-both-models` when you explicitly need RGB and HSV inference every cycle

Export ONNX models for the live dashboard at the same input size:

```powershell
cd model_training
python export_models.py --imgsz 640
```

After starting the backend, verify the active OpenCV runtime at:

```text
http://localhost:8080/health
```

For Pi 4 deployment speed, prefer the INT8 NCNN export path:

```powershell
cd model_training
python scripts/export_ncnn.py --config config/project.yaml --weights runs/segment/naartjie_rgb/weights/best.pt --imgsz 640 --int8
```

## Foreground

```powershell
cd foreground
npm install
npm run build
npm run dev
```

The dashboard is served at `http://127.0.0.1:5173/` by default and expects the background API at `http://localhost:8080` unless `VITE_API_BASE_URL` is set.
It lets you select a laptop camera in the browser, start/stop camera preview separately from inference, and view one raw camera feed plus separate RGB and HSV model feeds. Inference still requires the C++ background API at `localhost:8080`; missing model files are reported in the matching feed panel.
