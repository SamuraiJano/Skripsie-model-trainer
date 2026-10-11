# Model training workflow

For the **final object-detection comparison**, use [CONTROLLED_DETECTION_EXPERIMENT.md](CONTROLLED_DETECTION_EXPERIMENT.md) and `notebooks/controlled_detector_experiment.ipynb`. This separate workflow compares YOLO26n with official Peterande D-FINE-N across RGB/HSV_FULL and seeds 0–4, with max 300/min 100/patience 40, frozen paired recipes, and a common COCO evaluator. Run `python scripts/controlled_detector_experiment.py prepare` to verify the data and display the 20-run matrix without starting training. The dashboards and detection settings described below remain exploratory; segmentation is unchanged.

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

The notebook accepts either a standalone YOLO dataset or a complete Data Labeller export. Select **Object detection** or **Instance segmentation**, then validate the dataset, smoke-test the real Ultralytics loader, configure training, and start the run. The training button stays disabled until the long-running action is explicitly confirmed. It also shows a wrapped, correctly quoted equivalent CLI command for reproducibility.

The responsive **All YOLO parameters** sections expose the relevant Ultralytics 8.4.95 train settings, including early-stopping patience, RAM/disk dataset caching, checkpoint periods, optimizer and learning-rate settings, augmentation, mask settings, validation thresholds, plots, and profiling. The JSON override box accepts additional version-specific Ultralytics settings while protecting dataset and run-directory values managed by this workflow.

The final YOLO dashboard and CLI defaults use Nano, a 300-epoch maximum, 100-epoch minimum, patience 40, four workers and seed 0, 640 px images, fixed physical batch 16 and nominal batch 16. AdamW uses learning rate 0.001 and weight decay 0.0005. Detection uses mosaic 0.5; segmentation uses mosaic/MixUp off. Hue/saturation/value jitter and additional Albumentations colour transforms are off for both tasks and both colour spaces. Geometry uses translation 0.1, scale 0.25, rotations up to 10 degrees and horizontal flip 0.5. Task changes adjust unchanged task defaults and preserve custom values. Restart the kernel and rerun the setup and dashboard cells to load updated defaults. See [FINAL_EXPERIMENT_DEFAULTS.md](FINAL_EXPERIMENT_DEFAULTS.md) for the complete audited configuration and dataset readiness.

When installing without Conda, include the notebook dependencies:

```powershell
python -m pip install -e ".[notebook]"
```

### D-FINE notebook and dataset formats

Open the dedicated dashboard with:

```powershell
git -c http.sslBackend=schannel clone --depth 1 https://github.com/Peterande/D-FINE.git third_party/D-FINE
git -c http.sslBackend=schannel clone --depth 1 https://github.com/ArgoHA/D-FINE-seg.git third_party/D-FINE-seg
python -m pip install -e ".[notebook,dfine,dfine-seg]"
jupyter lab notebooks/dfine_model_trainer_ui.ipynb
```

The dashboard automatically discovers both checkouts under `third_party/` and uses the active notebook Python. Its setup cell installs missing training dependencies. On Windows, the Trainer downloads a missing D-FINE-seg pretrained checkpoint with Schannel-backed `curl.exe`, preserving certificate verification when Python's CA bundle cannot validate Hugging Face.

The dashboard keeps the upstream distinction explicit:

- [Peterande/D-FINE](https://github.com/Peterande/D-FINE) is the official, detection-only implementation. It does **not** read a YOLO `data.yaml` itself. The Trainer accepts a standalone YOLO detection dataset or complete Data Labeller export, validates it, and stages COCO JSON with pixel `bbox: [x, y, width, height]` annotations under the run directory.
- [ArgoHA/D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg) is a separate implementation supporting detection and instance segmentation. YOLO detection rows are `class_id xc yc width height`. YOLO instance-segmentation rows are `class_id x1 y1 x2 y2 ... xN yN`, using normalized polygon coordinates and at least three points.

For **Official D-FINE detection**, the notebook exposes image size, epochs, batch size, workers, seed, device, AMP, and resume directly. Expand **Official D-FINE advanced settings** for base and backbone learning rates, weight decay, gradient clipping, warmup steps, EMA, and additional upstream YAML overrides as JSON. The equivalent CLI command reflects these choices. Image size updates the training and validation resize operations, collate size, and model evaluation size together.

For official D-FINE and D-FINE-seg, the notebook validates the appropriate YOLO `data.yaml`, preserves its train/validation split, and creates a non-destructive COCO staging view under the run directory. Detection annotations contain pixel boxes; instance-segmentation annotations contain both pixel boxes and COCO polygon `segmentation` arrays. Images are hard-linked when the filesystem supports it and copied otherwise. The source dataset is never changed.

Each upstream uses its own checkout. The tested Windows setup shares this project's `.venv`; a separate upstream environment can still be selected in the dashboard. The Trainer never silently treats D-FINE-seg as the official implementation.

The dashboard opens on **D-FINE-seg instance segmentation** with `dataset_seg_rgb` as its dataset root. Switching to a detection trainer selects `dataset_detect_rgb` only when the path still matches the segmentation default; a custom path is preserved. Hover over a setting for a short explanation. If you already have the notebook open, rerun its dashboard cell to load updated defaults.

The D-FINE-seg workflow uses Nano for detection and segmentation with RGB or HSV, matching COCO-pretrained checkpoints, 640 px images, AMP (BF16 when supported, otherwise FP16), a 300-epoch maximum, 100-epoch minimum, patience 40, zero data workers and seed 0. Detection uses physical batch 4, accumulation 4, mosaic 0.5 and AdamW with the existing Nano base/head LR 0.0008, backbone LR 0.0004 and weight decay 0.0001. Segmentation uses physical batch 2, accumulation 8, mosaic off and the project Nano Muon + Adan/OneCycleLR/EMA recipe. Both target effective batch 16, preserve aspect ratio and use moderate geometry with colour transforms off. MixUp is absent from this implementation. Task changes preserve custom choices.

**Instance segmentation options** exposes aspect-ratio padding, mask-only fine-tuning, mask evaluation batch size, flip and rotation probabilities, maximum rotation, and blur probability. The selected values appear in the equivalent CLI preview and generated training YAML. Mosaic probability remains in **Advanced D-FINE-seg settings**. Mask-only fine-tuning expects a suitable segmentation checkpoint.

Detection exposes the same geometry and augmentation controls without the mask-only fields. **Checkpoint metric** controls both best-checkpoint selection and early stopping, and its value is preserved in the generated configuration and equivalent CLI command. Compare tuning candidates using validation data with equal tuning effort for each model; evaluate the held-out test set after settings and checkpoint selection are final.

The selected final checkpoint uses validation box mAP50–95 for detection and validation mask mAP50–95 for segmentation. The project YOLO segmentation trainer changes only the selection fitness from Ultralytics' combined box/mask score to the mask score; reported metrics remain unchanged. Patience counts epochs without improvement, including a flat zero-score plateau. Test evaluation runs only after training and checkpoint selection.

The project runner and segmentation test tool enlarge small validation/test images when letterboxing, matching training and the upstream inverse box/mask mapping. Older runs could report zero mask scores because their validation images were padded without enlargement. Their `model.pt` may therefore still be the initial checkpoint; evaluate `last.pt` before deciding the training failed. Compare both saved checkpoints on the original validation split, without retraining or changing the checkpoints:

```powershell
python scripts/dfine_seg_diagnose.py --run-dir runs/dfine-seg/YOUR_RUN --device cuda
```

This saves a timestamped `validation_diagnosis/.../diagnosis.json` comparing the last checkpoint with original and corrected preprocessing, and the selected best checkpoint with corrected preprocessing. Future runs started through this project's runner use the corrected validation path for early stopping and checkpoint selection. Direct upstream CLI calls do not install the project's compatibility fix.

Dataset validation calls an image with an empty segmentation label file a **negative image**: it has no annotated defect polygon. This is valid when the image truly has no target defect. Review these images if you expected a defect annotation; the notebook shows the warning count and a short sample of filenames.

## Validate and smoke-test real loaders

```powershell
python scripts/model_trainer.py validate D:\exports\fruit-training-export
python scripts/model_trainer.py smoke-loaders D:\exports\fruit-training-export --dfine-root D:\src\D-FINE
```

The smoke command loads actual samples through the installed Ultralytics `YOLODataset` for detection and segmentation, then through the official D-FINE `CocoDetection` with `remap_mscoco_category=False`. D-FINE therefore receives the exported zero-based contiguous `category_id` values directly.

## Train and resume

The final D-FINE experiments use `dfine-seg-detect` and `dfine-seg-segment`. The `dfine-detect` example below is the separate official Nano workflow.

```powershell
python scripts/model_trainer.py yolo-detect dataset_detect_rgb --workers 4 --device 0 --output-dir runs
python scripts/model_trainer.py yolo-segment dataset_seg_rgb --workers 4 --device 0 --output-dir runs
python scripts/model_trainer.py dfine-detect dataset_detect_rgb --dfine-root third_party\D-FINE --python-executable ..\.venv\Scripts\python.exe --epochs 300 --batch 8 --workers 4 --device 0 --seed 0 --output-dir runs
python scripts/model_trainer.py dfine-seg-detect dataset_detect_rgb --dfine-seg-root third_party\D-FINE-seg --python-executable ..\.venv\Scripts\python.exe --workers 0
python scripts/model_trainer.py dfine-seg-segment dataset_seg_rgb --dfine-seg-root third_party\D-FINE-seg --python-executable ..\.venv\Scripts\python.exe --workers 0
```

Pass `--resume <checkpoint>` to YOLO or the official D-FINE trainer. Resumed official D-FINE jobs use a new run directory and automatically carry forward the source run's `best_stg1.pth`, which the upstream solver reloads at its stage transition. AMP is enabled by default; use `--no-amp` only for diagnosis. YOLO detection starts from `yolo26n.pt`, and segmentation starts from `yolo26n-seg.pt`; official D-FINE fine-tunes the COCO-pretrained `dfine_n_coco.pth` using the Nano custom config. D-FINE-seg supports N/S/M/L/X sizes and defaults to the matching Nano detection or segmentation checkpoint.

To load the final Nano defaults in an already-open notebook, restart the kernel and rerun the setup and dashboard cells. Leave resume empty for a fresh pretrained run; explicit resume restores the checkpoint's historical architecture and settings.

Each final run records its model/variant/task/colour representation, pretrained source, exact settings, runtime versions, split image identities and annotation hashes, class mapping, physical/effective batches, optimizer and actual parameter-group LRs, best validation epoch/score and checkpoint path in `run_metadata.json`. Per-epoch validation selection records are written to `validation_history.jsonl`. YOLO writes `results.csv` and `weights/best.pt`/`last.pt`; active D-FINE writes `model/model.pt` (best) and `model/last.pt`.

D-FINE-seg defaults to **Data workers 0** in the dashboard, CLI and Python API, loading images in the trainer process to avoid Windows worker crashes and expose image-loading errors directly. Restart the notebook kernel and rerun its setup/dashboard cells to load these defaults. Explicit nonzero counts remain available; on Windows the runner prefetches one batch per training worker and loads validation/test images in the trainer process to avoid overlapping worker pools. The actual loader settings are recorded in `run_metadata.json` under `data_loading`. A generic worker exit does not establish its cause. If the log reports `MemoryError`, `_ArrayMemoryError`, or `Unable to allocate`, system RAM allocation failed; close other memory-heavy applications before retrying. Failed subprocesses show recovery instructions, the underlying traceback and the `trainer_console.log` path, and metadata preserves the original training error.

The legacy eight-run recipes use a **300-epoch maximum, 100-epoch minimum and 40-epoch patience**, with four YOLO workers and zero D-FINE-seg workers by default. Monitoring starts at epoch 1; stopping becomes legal after completing epoch 100 and 40 epochs without strict validation improvement. Detection selects validation box mAP50-95; segmentation selects validation mask mAP50-95. Test evaluation follows best-checkpoint selection. YOLO retains its native linear 300-epoch decay to `lr0 * 0.01` at the completed horizon boundary and three warmup epochs. Project D-FINE retains OneCycleLR with 10% warmup and counts every accumulated optimizer update, including the trailing partial window. Separate Nano detection backbone/head peaks remain intact. Metadata distinguishes maximum horizon, early stopping, manual interruption and errors. Preflight optionally runs one physical batch forward/backward per final architecture without updating weights. Start each colour run fresh from the same original pretrained checkpoint. See [the final recipe and precise epoch/scheduler conventions](FINAL_EXPERIMENT_DEFAULTS.md).

## Controlled RGB versus HSV

Select **Colour space: RGB/HSV** in either dashboard, or pass `--colour-space rgb` / `--colour-space hsv` with the same source dataset. The selector changes only representation; all architecture/task settings and starting weights remain fixed. The default `auto` uses dataset metadata or a `_hsv` folder name, otherwise RGB.

```powershell
python scripts/model_trainer.py yolo-detect dataset_detect_rgb --colour-space hsv
python scripts/model_trainer.py yolo-segment dataset_seg_rgb --colour-space hsv
python scripts/model_trainer.py dfine-seg-detect dataset_detect_rgb --dfine-seg-root third_party/D-FINE-seg --colour-space hsv
python scripts/model_trainer.py dfine-seg-segment dataset_seg_rgb --dfine-seg-root third_party/D-FINE-seg --colour-space hsv
```

HSV preparation uses `cv2.COLOR_BGR2HSV_FULL` and saves lossless PNG channels as **H,S,V** in PIL's three-channel file slots. YOLO and project D-FINE decode with OpenCV as V,S,H and reverse the channel order before tensor construction; official D-FINE reads H,S,V directly with PIL. H/S/V each use uint8 0..255; every tensor channel is divided by 255 once. Prepared HSV views preserve the same splits and label bytes. Metadata and the cache fingerprint explicitly include HSV_FULL; older caches are regenerated.

Historical HSV requires a paired RGB source through a sibling `_rgb` dataset or `source_data_yaml`. Every historical image must match its source in a proven HSV_FULL or ordinary HSV encoding and H,S,V or V,S,H storage order. The controlled view is always regenerated from RGB using HSV_FULL. An order declaration alone is insufficient provenance. The incomplete historical `dataset_seg_hsv` folder remains untouched: select HSV with `dataset_seg_rgb`.

The final preflight runs controlled tests, compares all resolved RGB/HSV configurations, checks source/annotation/split provenance, verifies original Nano checkpoints, and loads real RGB/HSV batches for every supported path. Numerical tests use a known source colour with FULL Hue above 179. Reports are saved under `runs/final_validation/`. Missing physical fruit identity is reported as NOT PROVABLE; known image or fruit overlap blocks training without rewriting splits.

The [latest 300-epoch validation report](runs/final_validation/20261006-185131-437706/final_report.md) records 124 controlled tests passed, zero failed, the same two historical saved-notebook-output exclusions, all 15 preflight checks passed and all four intended physical batches passing one forward/backward memory check without optimizer updates. Previous validation evidence remains intact. No final training has been launched.

```powershell
python scripts/validate_final_experiment.py
```

Seed 0 is applied to Python, NumPy, PyTorch, CUDA and D-FINE's independent Albumentations generators (including workers). cuDNN benchmarking is off; deterministic algorithms are requested with warnings for operations without deterministic CUDA implementations. Reproducibility across different hardware/library versions is not guaranteed. AMP/EMA and effective optimizer details are recorded; inspect actual AMP because frameworks can disable it on unsupported devices. Keep overrides and hardware/software identical between each RGB/HSV pair, and avoid resume for fresh comparisons.

## Synthetic-data boundary

```powershell
python scripts/model_trainer.py synthetic D:\exports\fruit-training-export
```

This reports foreground/background counts and the manifest path. A future generator must consume that manifest and create a new versioned dataset; it must never mutate the labelled export or invent a new split inside a training run.

## Tests

```powershell
python -m pytest -q
```
