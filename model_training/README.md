# Model training workflow

## Input contract

Select the root directory ending in `-training-export`. `scripts/model_trainer.py validate` discovers and validates `export_manifest.json`, both YOLO `data.yaml` files, D-FINE COCO annotations, and the synthetic asset manifest before any training starts.

Validation blocks missing files, unsafe/non-portable paths, empty train/validation splits, non-contiguous class IDs, class-mapping mismatches, malformed normalized YOLO labels, invalid COCO pixel `xywh` boxes, missing images, and invalid D-FINE `area`/`iscrowd`. Negative images and unapproved source records are reported as warnings. A disabled test split must not leave empty test artifacts.

## Environment

The confirmed workstation environment is Python 3.12.6, Ultralytics 8.4.95, PyTorch 2.6.0+cu124, CUDA 12.4, and an NVIDIA RTX 4060 Laptop GPU. Recreate it with:

```powershell
conda env create -f environment.yml
conda activate naartjie-yolo26
python -m pip install -e .
```

## Notebook UI

For a form-based workflow covering both YOLO object detection and YOLO instance segmentation, open the included dashboard:

```powershell
conda activate naartjie-yolo26
jupyter lab notebooks/model_trainer_ui.ipynb
```

The notebook accepts either a standalone YOLO dataset or a complete Data Labeller export. Select **Object detection** or **Instance segmentation**, then validate the dataset, smoke-test the real Ultralytics loader, configure training, and start the run. The training button stays disabled until the long-running action is explicitly confirmed. It also shows the equivalent CLI command for reproducibility.

When installing without Conda, include the notebook dependencies:

```powershell
python -m pip install -e ".[notebook]"
```

D-FINE uses its own official checkout and dependencies. Clone `https://github.com/Peterande/D-FINE`, create the environment described upstream, and pass that checkout with `--dfine-root`. The Trainer never vendors or silently substitutes another implementation.

## Validate and smoke-test real loaders

```powershell
python scripts/model_trainer.py validate D:\exports\fruit-training-export
python scripts/model_trainer.py smoke-loaders D:\exports\fruit-training-export --dfine-root D:\src\D-FINE
```

The smoke command loads actual samples through the installed Ultralytics `YOLODataset` for detection and segmentation, then through the official D-FINE `CocoDetection` with `remap_mscoco_category=False`. D-FINE therefore receives the exported zero-based contiguous `category_id` values directly.

## Train and resume

```powershell
python scripts/model_trainer.py yolo-detect D:\exports\fruit-training-export --epochs 100 --imgsz 640 --batch 16 --workers 8 --device 0 --seed 42 --output-dir runs
python scripts/model_trainer.py yolo-segment D:\exports\fruit-training-export --epochs 100 --imgsz 640 --batch 16 --workers 8 --device 0 --seed 42 --output-dir runs
python scripts/model_trainer.py dfine-detect D:\exports\fruit-training-export --dfine-root D:\src\D-FINE --epochs 100 --batch 16 --workers 8 --device 0 --seed 42 --output-dir runs
```

Pass `--resume <checkpoint>` for any trainer. AMP is enabled by default; use `--no-amp` only for diagnosis. YOLO starts from `yolo26n.pt` or `yolo26n-seg.pt`; D-FINE starts from the official Nano custom config.

Each run records its exact configuration, runtime versions, source export hash, split sizes, class mapping, checkpoint locations, logs, and metrics in `run_metadata.json`. Ultralytics writes `results.csv` and `best.pt`/`last.pt`; D-FINE writes its native log and `.pth` checkpoints in the run directory.

## Synthetic-data boundary

```powershell
python scripts/model_trainer.py synthetic D:\exports\fruit-training-export
```

This reports foreground/background counts and the manifest path. A future generator must consume that manifest and create a new versioned dataset; it must never mutate the labelled export or invent a new split inside a training run.

## Tests

```powershell
python -m pytest -q
```
