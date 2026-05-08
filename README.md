# Skripsie

Repository scaffold for computer vision model training.

## Project layout

```text
data/
  shared/
    raw/
    processed/
    splits/
      train/
      val/
models/
  experiments/
```

- Put your dataset in `data/shared/` so multiple models can train from the same source data.
- Use `data/shared/splits/train/` and `data/shared/splits/val/` for training and validation sets.
- Save per-model checkpoints, logs, and exports in separate subfolders under `models/experiments/` (for example `models/experiments/yolov8/` and `models/experiments/resnet50/`).
