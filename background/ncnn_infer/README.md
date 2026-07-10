# Naartjie NCNN Inference

This is the Raspberry Pi 4 C++ image-folder inference app for exported Ultralytics NCNN models. It is the preferred CPU deployment path when no AI HAT, Coral TPU, or other accelerator is available.

## Build

Install OpenCV 5.0+ and NCNN development packages first, then:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

If CMake cannot find OpenCV 5, pass its package directory:

```bash
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv/lib/cmake/opencv5 -DCMAKE_BUILD_TYPE=Release
```

If CMake cannot find NCNN, pass its package directory:

```bash
cmake -S . -B build -DNCNN_DIR=/path/to/ncnn/lib/cmake/ncnn -DCMAKE_BUILD_TYPE=Release
```

## Run

```bash
./build/ncnn_infer \
  --model /path/to/best_ncnn_model \
  --input /path/to/images \
  --output /path/to/results \
  --classes class_1,class_2,class_3 \
  --conf 0.25 \
  --imgsz 640 \
  --source-width 640 \
  --source-height 480 \
  --threads 4
```

The defaults already match those Pi 4 values. From the repository root, export the model with matching size and INT8 quantization:

```bash
cd model_training
python scripts/export_ncnn.py --config config/project.yaml --weights runs/segment/naartjie_rgb/weights/best.pt --imgsz 640 --int8
```

For an RGB+HSV six-channel export:

```bash
./build/ncnn_infer --model /path/to/rgb_hsv_ncnn_model --input images --output results --channels rgbhsv
```

If your export uses different output blob names, pass them explicitly:

```bash
--input-name in0 --det-output out0 --proto-output out1
```
