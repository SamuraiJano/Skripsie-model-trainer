# Skripsie Model Trainer

Training-only workspace for the final-year computer-vision project. It consumes a portable `*-training-export` produced by Data Labeller and trains:

- YOLO26n object detection (`fruit_box`)
- YOLO26n-seg instance segmentation (`defect_mask`)
- D-FINE-N object detection (`fruit_box`) using the official Peterande/D-FINE checkout
- D-FINE-seg object detection and instance segmentation using the separate ArgoHA implementation
- Ghost-DeblurGAN paired-image restoration (default baseline)
- `AGGDeblurGANReproduction`, an optional non-official reproduction of the 2025 AGG paper
- Deblurring hallucination, blur-routing, and before/after segmentation-disagreement evaluation

Camera access, prediction workbenches, tracking, ONNX/NCNN deployment, Raspberry Pi operation, and benchmarking belong to the sibling Inference Runner. Trackers such as ByteTrack and BoT-SORT are configured at runtime; they are not trained here.

See [model_training/README.md](model_training/README.md) for setup, validation, training, resume, and artifact details.
