# Skripsie Model Trainer

Training-only workspace for the final-year computer-vision project. It consumes a portable `*-training-export` produced by Data Labeller and trains:

- YOLO26n object detection (`fruit_box`)
- YOLO26n-seg instance segmentation (`defect_mask`)
- D-FINE Nano object detection (`fruit_box`) using the active ArgoHA/D-FINE-seg implementation
- D-FINE-seg Nano defect segmentation using that same implementation

YOLO and D-FINE detection/segmentation support controlled RGB and HSV comparisons: 100 epochs, seed 0, fixed batches, deterministic settings and no colour augmentation. HSV selection prepares a lossless HSV_FULL view from the same RGB source images and labels; all channels use 0..255 and model tensors use /255. The official Peterande/D-FINE Nano detector remains available as a separate legacy workflow. Final defaults and the preflight command are documented in [model_training/FINAL_EXPERIMENT_DEFAULTS.md](model_training/FINAL_EXPERIMENT_DEFAULTS.md).

Camera access, prediction workbenches, tracking, ONNX/NCNN deployment, Raspberry Pi operation, and benchmarking belong to the sibling Inference Runner. Trackers such as ByteTrack and BoT-SORT are configured at runtime; they are not trained here.

See [model_training/README.md](model_training/README.md) for setup, validation, training, resume, and artifact details.

The latest detector comparison is recorded in [Detection_Comparison_Results_2026-10-09/Comparison/metrics_report.md](Detection_Comparison_Results_2026-10-09/Comparison/metrics_report.md). Deployment export instructions and verification reports are in [Detection_Deployment_Exports_2026-10-11](Detection_Deployment_Exports_2026-10-11/README.md), with the YOLO RGB/HSV NCNN INT8 accuracy and speed results in [Detection_NCNN_INT8_2026-10-11/RESULTS.md](Detection_NCNN_INT8_2026-10-11/RESULTS.md).

GitHub contains the code, configuration, reports, logs, and metric plots. Datasets, image annotations, calibration and example images, videos, checkpoints, ONNX/TensorRT/NCNN model artifacts, and export archives stay local. The deployment bundle instructions describe local files that are intentionally absent from the repository; use the export scripts with your local checkpoints and dataset to recreate them.
