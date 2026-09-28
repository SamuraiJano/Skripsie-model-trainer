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

The notebook accepts either a standalone YOLO dataset or a complete Data Labeller export. Select **Object detection** or **Instance segmentation**, then validate the dataset, smoke-test the real Ultralytics loader, configure training, and start the run. The training button stays disabled until the long-running action is explicitly confirmed. It also shows a wrapped, correctly quoted equivalent CLI command for reproducibility.

The responsive **All YOLO parameters** sections expose the relevant Ultralytics 8.4.95 train settings, including early-stopping patience, RAM/disk dataset caching, checkpoint periods, optimizer and learning-rate settings, augmentation, mask settings, validation thresholds, plots, and profiling. The JSON override box accepts additional version-specific Ultralytics settings while protecting dataset and run-directory values managed by this workflow.

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

For this laptop's RTX 4060 (8 GB VRAM), the segmentation defaults favor accuracy while leaving memory headroom: Small model, 640 px images, physical batch 2, four-step gradient accumulation (effective batch 8), BF16 when supported, 120 maximum epochs, 20-epoch early stopping, two Windows data workers, and mask evaluation batch 16. Aspect ratio is preserved; mosaic and blur are off by default so small defect masks are not distorted. These are conservative starting values, not a guarantee against out-of-memory errors if other GPU processes are active; reduce physical batch size to 1 first if needed. The Small COCO segmentation checkpoint downloads on first use if missing.

**Instance segmentation options** exposes aspect-ratio padding, mask-only fine-tuning, mask evaluation batch size, flip and rotation probabilities, maximum rotation, and blur probability. The selected values appear in the equivalent CLI preview and generated training YAML. Mosaic probability remains in **Advanced D-FINE-seg settings**. Mask-only fine-tuning expects a suitable segmentation checkpoint.

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

```powershell
python scripts/model_trainer.py yolo-detect D:\exports\fruit-training-export --epochs 100 --imgsz 640 --batch 16 --workers 8 --device 0 --seed 42 --output-dir runs
python scripts/model_trainer.py yolo-segment D:\exports\fruit-training-export --epochs 100 --imgsz 640 --batch 16 --workers 8 --device 0 --seed 42 --output-dir runs
python scripts/model_trainer.py yolo-detect D:\exports\fruit-training-export --patience 30 --cache disk --save-period 10 --optimizer AdamW --cos-lr
python scripts/model_trainer.py dfine-detect dataset_detect_rgb --dfine-root third_party\D-FINE --python-executable ..\.venv\Scripts\python.exe --epochs 100 --batch 8 --workers 0 --device 0 --seed 42 --output-dir runs
python scripts/model_trainer.py dfine-seg-detect dataset_detect_rgb --dfine-seg-root third_party\D-FINE-seg --python-executable ..\.venv\Scripts\python.exe --model-size n --workers 0
python scripts/model_trainer.py dfine-seg-segment D:\exports\fruit-training-export --dfine-seg-root third_party\D-FINE-seg --python-executable ..\.venv\Scripts\python.exe --model-size n --workers 0 --early-stopping 10
```

Pass `--resume <checkpoint>` to YOLO or the official D-FINE trainer. Resumed official D-FINE jobs use a new run directory and automatically carry forward the source run's `best_stg1.pth`, which the upstream solver reloads at its stage transition. AMP is enabled by default; use `--no-amp` only for diagnosis. YOLO starts from `yolo26n.pt` or `yolo26n-seg.pt`; official D-FINE starts from the Nano custom config. D-FINE-seg supports N/S/M/L/X sizes and selects the matching detection or segmentation pretrained checkpoint.

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

## Deblurring dashboard and CLI

Deblurring lives exclusively in `deblur/`; existing YOLO RGB/HSV, D-FINE, checkpoint, and deployment workflows are unchanged. Build and start the React dashboard:

```powershell
python -m pip install -e ".[test]"
cd frontend
npm install
npm run build
cd ..
python -m deblur dashboard --host 127.0.0.1 --port 8876
```

Open `http://127.0.0.1:8876` and select **Deblurring**. The tab validates paired folders, configures either architecture, starts/cancels a backend-owned training job, polls logs/losses/metrics without owning the training lifetime, displays validation previews, and reports best/last checkpoints. Defaults are 256 px, 200 epochs, batch 4, Adam at `1e-4` with no weight decay, 1:1 G/D updates, linear decay from epoch 10 to `1e-6`, seed 42, and AMP on supported CUDA. Every value is configurable.

Training recovery is enabled by default. The trainer atomically refreshes `last_checkpoint.pt` every 100 completed batches, after every epoch, and when cancellation or an exception is detected. Starting the same output directory and run name again automatically resumes that checkpoint, including the optimizer, schedulers, AMP scaler, random states, current epoch, and next batch. Set a different interval in the dashboard or with `--checkpoint-interval-steps`; use `--no-auto-resume` when a deliberate clean restart is required. A manually supplied `--resume-checkpoint` always takes precedence.

Equivalent CLI usage:

```powershell
python -m deblur validate D:\data\deblur_dataset
python -m deblur train D:\data\deblur_dataset --architecture GhostDeblurGAN --device cuda
python -m deblur train D:\data\deblur_dataset --architecture AGGDeblurGANReproduction --device cuda
python -m deblur train D:\data\deblur_dataset --resume-checkpoint runs\deblur\ghost_deblurgan\last_checkpoint.pt
python -m deblur train D:\data\deblur_dataset --checkpoint-interval-steps 25
```

The loader requires exact filename pairs under `split/blur` and `split/sharp`, matching RGB dimensions, unique pair IDs, and manifest agreement. It rejects cross-split filenames and source/perceptual/capture/fruit/tree/session leakage when the Data Labeller manifest is present. Paired crop/flip parameters are shared; validation is deterministic.

### Models, losses, checkpoints, and VGG setup

`GhostDeblurGAN` is the default lightweight baseline: a clean-room, attributed adaptation of the public GhostNet + HIN + FPN residual design. The training-only discriminator has full- and half-scale PatchGAN branches following DeblurGAN-v2. Generator content loss defaults to `0.7 × perceptual + 0.3 × L1`; total generator loss adds `0.005 ×` relativistic least-squares adversarial loss.

Automatic VGG downloads are disabled. For research training, install compatible torchvision and pass a local official torchvision VGG19 `state_dict` with `--vgg19-weights-path PATH`. If absent, the dashboard uses a deterministic multiscale/gradient fallback intended for offline tests and labels this clearly in the log; it is not a trained VGG substitute.

`last_checkpoint.pt` atomically stores both networks, optimizers, schedulers, AMP scaler, epoch/global step, next-batch position, partial epoch loss totals, best metrics, model/dataset config, seed, history, data-loader state, and Python/NumPy/Torch random states. `best_generator.pt` contains only the generator and deployment metadata. Best selection uses content error plus a Laplacian-change penalty, not PSNR/SSIM alone.

### AGG paper reproduction disclaimer

`AGGDeblurGANReproduction` implements Attention-Ghost (primary and cheap depthwise features, concatenation, SE channel attention, projection) and odd-channel-safe GHIN (half instance standardization with learnable `alpha`, `gamma`, and `beta` modulation). It uses the paper-described GhostNet widths, lateral projections, nearest-neighbour top-down fusion, common-width heads, concatenation, and residual RGB reconstruction.

There is no verified official public AGG-DeblurGAN implementation. The paper does not specify every expansion ratio, padding choice, or exact per-block ordering, so those are documented approximations in `deblur/models.py` and typed `DeblurModelConfig`. Do not claim this reproduction is official. Attribution and retained reference licences are in `third_party/`.

### Evaluation, blur routing, inference, and export

The dashboard’s **Hallucination & segmentation evaluation** runs one existing YOLO segmentation model on the original blurred crop and restored crop. It exports original/restored/optional sharp images, before/after overlays, and `disagreements.csv`, `disagreements.json`, and `summary.json`. Class-risk order and accepting classes are configurable. Flags cover less-severe transitions, large mask decreases, sharp-reference-supported dark-feature disappearance, unsupported new edges, colour shift, confidence jumps with disagreement, and high-confidence accepting output after an original defect prediction.

```powershell
python -m deblur evaluate best_generator.pt D:\data\deblur_dataset D:\reports --segmentation-model best.pt --class-risk-order "Class 1" "Class 2" "Class 3"
python -m deblur infer best_generator.pt D:\crops D:\restored --device cpu
python -m deblur export best_generator.pt D:\exports\ghost --image-size 256
python -m deblur calibrate-blur D:\calibration\acceptable D:\calibration\unacceptable D:\calibration\report --target-recall 0.95
```

The routing scorer deterministically computes variance of Laplacian after fixed-size preprocessing, with configurable inner crop, optional fruit mask, and kernel size. Scores below the calibrated low threshold route to deblurring; intermediate scores select the sharpest burst frame; scores above the high threshold bypass deblurring. Calibration derives CSV/JSON distributions and thresholds from manually labelled local crops—no internet example threshold is hard-coded.

Export produces a generator-only PyTorch checkpoint, TorchScript, and fixed-size ONNX opset 17 when optional dependencies exist. ONNX Runtime numerical comparison is recorded; an excessive mismatch fails verification. One-image, folder, and paired-dataset inputs are supported by the inference/evaluation commands. The discriminator is never exported. Raspberry Pi, NCNN, and Inference Runner integration are intentionally outside this change, and no real-time Pi claim is made.

### Safety and domain-gap warning

Synthetic blur is an initial training method. Final conclusions require real conveyor-blur validation. A sharper-looking GAN output does not prove improved defect detection: GANs can erase dark defects, invent edges, shift colour, or inflate confidence. Always compare the deblurred result with the original and review the disagreement report before safety or acceptance decisions.

Run all checks with:

```powershell
python -m pytest -q
cd frontend
npm test
npm run build
```
