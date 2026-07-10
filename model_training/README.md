# YOLO26 Naartjie Defect Training Pipeline

This project trains YOLO26 instance segmentation models for three naartjie defect classes using synthetic images generated from transparent foreground defect PNGs and clean naartjie backgrounds.

The main workflow is in `notebooks/yolo26_naartjie_training.ipynb`. The reusable code lives in `src/naartjie_cv`.

All commands below assume you are in the `model_training` folder.

## 1. Install Anaconda or Miniconda

`conda` is required. It was not available on PATH when this project was created.

Install one of these:

- Miniconda: https://docs.conda.io/projects/miniconda/
- Anaconda: https://www.anaconda.com/download

Open a fresh Anaconda Prompt or PowerShell after installing.

## 2. Create the environment

```powershell
conda env create -f environment.yml
conda activate naartjie-yolo26
python -m pip install -e .
python -m ipykernel install --user --name naartjie-yolo26 --display-name "Python (naartjie-yolo26)"
```

## 3. Verify GPU setup

```powershell
conda activate naartjie-yolo26
python scripts/smoke_check.py
```

Expected result: CUDA is available and the NVIDIA GPU name is printed.

## 4. Prepare input data

Place transparent foreground PNGs in class folders:

```text
data/raw/foreground/
  class_1/*.png
  class_2/*.png
  class_3/*.png
```

Place clean naartjie background images here:

```text
data/raw/background/*.jpg
data/raw/background/*.png
```

Foreground PNG alpha channels are used as exact defect masks.

## 5. Run the notebook

```powershell
conda activate naartjie-yolo26
jupyter lab notebooks/yolo26_naartjie_training.ipynb
```

The notebook will:

1. Verify the conda/PyTorch/Ultralytics runtime.
2. Generate 10,000 balanced synthetic RGB images.
3. Export COCO instance segmentation JSON and YOLO segmentation labels.
4. Generate an RGB+HSV 6-channel TIFF copy of the dataset.
5. Run three classical baselines for comparison.
6. Train an RGB YOLO26 instance segmentation model.
7. Train an experimental RGB+HSV multispectral YOLO26 model.
8. Resume interrupted runs from `last.pt`.
9. Evaluate precision, recall, F1, mAP, detection accuracy, and speed.
10. Review errors and hard examples in FiftyOne.
11. Export the best practical model to NCNN.

## 6. Classical baseline experiments

The project includes three classical baselines before the deep-learning comparison:

- `hsv_thresholding`: HSV thresholding, morphology, and contours.
- `lab_kmeans`: L*a*b* colour space, K-means clustering, and contours.
- `hsv_texture_classifier`: HSV fruit mask plus colour/texture features classified with KNN or SVM.

This follows the experimental motivation from Moreira et al., where HSV colour-space models are used as lighter baselines against deep-learning fruit defect detection/classification models. For naartjie defects, the thresholds in `config/project.yaml` are starting points and should be calibrated once real defect examples are available.

Run all three baselines on the test split:

```powershell
python scripts/run_classical_baselines.py --config config/project.yaml --split test
```

Outputs are written to:

```text
runs/classical/
```

Each baseline saves JSON/CSV metrics and annotated preview images.

## 7. Command-line workflow

All commands assume the conda environment is active.

```powershell
python scripts/generate_synthetic.py --config config/project.yaml
python scripts/run_classical_baselines.py --config config/project.yaml --split test
python scripts/train_yolo.py --config config/project.yaml --variant rgb
python scripts/train_yolo.py --config config/project.yaml --variant rgb_hsv
python scripts/evaluate_model.py --config config/project.yaml --weights runs/segment/naartjie_rgb/weights/best.pt --variant rgb
python scripts/fiftyone_model_review.py --config config/project.yaml --split test --weights runs/segment/naartjie_rgb/weights/best.pt --wait
python scripts/export_ncnn.py --config config/project.yaml --weights runs/segment/naartjie_rgb/weights/best.pt --imgsz 640 --int8
```

## 8. FiftyOne model review

FiftyOne is included as an evaluation and model-improvement tool, based on the official `voxel51/fiftyone` project and its Ultralytics/COCO evaluation support:

- GitHub: https://github.com/voxel51/fiftyone
- Ultralytics integration: https://docs.voxel51.com/integrations/ultralytics.html
- Evaluation guide: https://docs.voxel51.com/user_guide/evaluation.html
- COCO integration: https://docs.voxel51.com/integrations/coco.html

Use it after training to inspect masks, false positives, false negatives, low-quality labels, and hard examples:

```powershell
python scripts/fiftyone_model_review.py `
  --config config/project.yaml `
  --split test `
  --weights runs/segment/naartjie_rgb/weights/best.pt `
  --wait
```

This will:

- Import the generated COCO segmentations into a persistent FiftyOne dataset.
- Apply the trained Ultralytics model to the split.
- Run COCO-style detection/instance-segmentation evaluation.
- Save evaluation summaries to `runs/fiftyone/`.
- Export hard-example images and a manifest to `data/curation/fiftyone_hard_examples/`.
- Launch the FiftyOne App on the port in `config/project.yaml`.

To only open the dataset without applying a model:

```powershell
python scripts/fiftyone_model_review.py --config config/project.yaml --split test --wait
```

Use the hard-example manifest to decide what should be corrected, regenerated, or added to the next training iteration.

## 9. Raspberry Pi 4 C++ inference

The C++ app is in `../background/ncnn_infer`. It expects an Ultralytics NCNN export directory containing `.param` and `.bin` files. This is the preferred Raspberry Pi 4 path when you are not using an AI HAT or other NPU.

Build on the Pi after installing OpenCV 5.0+ and NCNN development packages:

```bash
cd ../background/ncnn_infer
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
```

Run image-folder inference:

```bash
./build/ncnn_infer \
  --model /path/to/best_ncnn_model \
  --input /path/to/images \
  --output /path/to/results \
  --conf 0.25 \
  --imgsz 640 \
  --source-width 640 \
  --source-height 480 \
  --threads 4 \
  --classes class_1,class_2,class_3
```

For the RGB+HSV export, add `--channels rgbhsv`. Keep the RGB model as the deployment fallback unless the RGB+HSV NCNN export and C++ parity check pass.

The live ONNX dashboard backend is also tuned for the Pi 4 by default: it runs the active model only, uses `--imgsz 640`, skips every second frame, and requests a 640x480 15 FPS camera stream. Re-export ONNX weights with `python export_models.py --imgsz 640` before running the backend with those defaults.

## Notes

- YOLO training resumes from `last.pt` when `training.resume: true`.
- Training image size uses `imgsz=736`, while Pi 4 deployment exports default to `imgsz=640` and source inference capture defaults to 640x480 before model letterboxing.
- If the final defect names change, update `classes` in `config/project.yaml`.
