# Final controlled RGB/HSV experiment

Final detection now uses the separate [controlled YOLO26n / official D-FINE-N experiment](CONTROLLED_DETECTION_EXPERIMENT.md). Its frozen recipes and common evaluator supersede the exploratory detection defaults below. The segmentation configuration below remains unchanged.

Updated 2026-10-06. No full training has been launched. Start each colour run independently from the same original pretrained checkpoint, with resume empty. Restart open notebook kernels and rerun setup/dashboard cells to load these defaults.

## Models and resolved recipes

All final models are Nano. Both RGB and HSV are available for every row.

| Task / implementation | Checkpoint | Physical batch x accumulation | Optimizer | Configured LR | Weight decay | Scheduler / warmup |
|---|---|---|---|---|---|---|
| YOLO26n detection | yolo26n.pt | 16 x 1 = 16 | AdamW | 0.001 | 0.0005 | Linear to 0.01 x initial LR; 3 warmup epochs |
| YOLO26n-seg segmentation | yolo26n-seg.pt | 16 x 1 = 16 | AdamW | 0.001 | 0.0005 | Same YOLO recipe |
| Project D-FINE-N detection (ArgoHA) | dfine_n_coco.pt | 4 x 4 = 16 | AdamW | Head 0.0008; backbone 0.0004 | 0.0001 | OneCycleLR; 10% warmup |
| Project D-FINE-N segmentation (ArgoHA) | dfine_seg_n_coco.pt | 2 x 8 = 16 | Muon + Adan | Base 0.0008; configured backbone 0.0004 | 0.000125 | OneCycleLR; 10% warmup |
| Official D-FINE-N detection (legacy supported route) | dfine_n_coco.pth | 8 x 1 = 8 | AdamW | Head 0.0008; backbone 0.0004 | 0.0001 | MultiStepLR milestones [500], gamma 0.1; LinearWarmup 500 steps |

Legacy eight-run defaults: **maximum 300 epochs, minimum 100 completed epochs, patience 40, 640 x 640, seed 0, deterministic true, AMP enabled, EMA enabled, multiscale off, cuDNN benchmarking off**. YOLO defaults to four workers; D-FINE-seg defaults to zero workers in both colour spaces to avoid Windows worker crashes. Explicit worker overrides remain available. Python, NumPy, PyTorch and CUDA are seeded. Project D-FINE's independent Albumentations generators and loader workers are seeded too. No automatic optimizer or batch selection is used. YOLO's first-epoch OOM batch reduction is explicitly disabled for controlled runs. The legacy official D-FINE reference route retains a valid 300-epoch maximum and four workers; it is outside this eight-run stopping policy. The separate controlled detection experiment uses zero workers for YOLO and official D-FINE.

Albumentations is imported before runtime seeding: its first lazy import otherwise shifts Python RNG state for only the first YOLO loader. Separate-process regression tests and real RGB/HSV batches verify matching geometric labels after this correction.

YOLO uses AdamW betas 0.9/0.999, warmup momentum 0.8, warmup bias LR 0, nominal batch 16 and gradient clipping 10. Its linear scheduler reaches LR 0.00001 at the completed 300-epoch boundary. Native epochs run from index 0 to 299: the last epoch uses factor 0.0133, and the mathematical horizon boundary at 300 has factor 0.01. Native warmup starts at zero LR and interpolates toward the scheduled initial LR over three epochs. Non-decay bias/norm groups retain zero weight decay.

Project D-FINE detection preserves backbone/head LR groups in OneCycle: peaks 0.0008/0.0016, first rates 0.000032/0.000064, default div_factor 25 and final_div_factor 10000, cycle_momentum false. AdamW betas are 0.9/0.999. Norm/bias groups retain zero weight decay. Gradient clipping is 0.1.

Project D-FINE segmentation preserves the existing Nano Muon + Adan recipe. Four Adan groups peak at 0.008 (base x 2 x adan_lr_mult 5); the Muon matrix group peaks at 0.016 (base x 10 x 2). Initial rates are 0.00032/0.00064. Adan betas are 0.98/0.92/0.99; Muon momentum is 0.95. The Nano Muon route does not maintain a separately lower backbone peak. Gradient clipping is 0.1. Both project paths use EMA momentum 0.9998 and BF16 AMP when supported, otherwise FP16.

Both project OneCycle schedulers use `ceil(len(train_loader) / b_accum_steps) * 300`, because the native loop flushes a trailing partial accumulation window. Scheduler advancement is attached to actual optimizer updates, including that final partial window and excluding FP16 GradScaler skips. With the current datasets, detection resolves 211 batches / accumulation 4 to 53 updates per epoch and 15,900 total; segmentation resolves 356 batches / accumulation 8 to 45 updates per epoch and 13,500 total. The existing 10% warmup, cosine cycle, group peaks and division factors remain unchanged. PyTorch uses schedule positions 0..total_steps-1 for the actual updates; its final post-update step has the native tiny cosine rebound.

Validation monitoring begins at epoch 1 and runs every epoch. Early stopping uses strict `new_metric > best_metric` (min_delta 0), so ties do not reset patience. A first finite metric, including zero, establishes the baseline. The shared state records the one-based current epoch, best meaningful metric/epoch and elapsed epochs since improvement. It cannot stop before completing epoch 100; afterwards 40 epochs without improvement terminates the run. A best at epoch 95 stops at 135, and a best at 120 stops at 160, assuming earlier improvements allow the run to reach those epochs. Continuous improvement reaches the 300-epoch maximum. RGB and HSV may finish at different epochs.

Checkpoint selection remains validation-only: box mAP50-95 for detection, mask mAP50-95 for segmentation. YOLO retains its native latest-equal-best checkpoint tie rule; project D-FINE retains the first equal-best checkpoint (strict improvement), including the existing initial zero-score fallback. Stopping patience always follows meaningful strict improvements, independently of checkpoint tie replacement. Last checkpoints remain available. Test evaluation occurs only after loading the selected best checkpoint. Run metadata includes max/min/patience, metric, best metric/epoch, actual completed stop epoch, stop reason (`early_stopping`, `max_epochs`, `manual_interrupt`, `error`), stopping state, scheduler horizon and actual optimizer updates.

Official D-FINE detection retains its official Nano custom configuration (`dfine_hgnetv2_n_custom.yml`, HGNetv2 B0), AdamW betas 0.9/0.999, norm/bias zero-decay groups, clipping 0.1, EMA decay 0.9999 with 1000 warmup updates, and its existing scheduler. It prefers the original checkpoint cached at `third_party/D-FINE/pretrained/dfine_n_coco.pth`, otherwise the upstream Nano release URL.

## Fixed augmentations

All photometric and colour augmentation is off in both representations. YOLO hsv_h/hsv_s/hsv_v/bgr are zero and augmentations is an empty list. Project D-FINE brightness/contrast, gamma, grayscale, noise and blur probabilities are zero. Official D-FINE removes RandomPhotometricDistort. No new colour transforms are introduced.

Geometry is retained unchanged:

- YOLO: rotation +/-10 degrees, translation 0.1, scale 0.25, horizontal flip 0.5; vertical flip/shear/perspective zero. Detection mosaic 0.5, segmentation mosaic zero; close mosaic in the final 10 epochs. MixUp/CutMix/copy-paste zero.
- Project D-FINE: preserve aspect ratio; horizontal flip 0.5, rotation probability 0.05 up to +/-10 degrees; vertical/90-degree flips and coarse dropout zero. Detection mosaic 0.5, segmentation zero; close in the final 10 epochs. Existing mosaic affine scale [0.5,1.5], degrees 0, translation 0.2, shear 2. MixUp is absent.
- Official D-FINE: existing RandomZoomOut (fill 0), RandomIoUCrop (p=0.8), horizontal flip, bounding-box sanitation and Resize 640. Inherited stage policy is retained; multiscale collation is disabled.

## HSV_FULL source-to-model pipeline

Selecting HSV with an original RGB dataset generates a lossless view of exactly its existing train/val/test splits and copies annotation bytes. Conversion is `cv2.COLOR_BGR2HSV_FULL`; **H, S and V each use uint8 0..255**. PNG slots are **H,S,V**, saved with PIL. There is no HSV-to-RGB conversion and no division by 179.

| Framework | Disk and loader route | Final model input |
|---|---|---|
| YOLO detection / segmentation | PNG H,S,V -> OpenCV V,S,H -> Format channel reversal -> CHW float /255 | H,S,V, each 0..1 |
| Project D-FINE detection / segmentation | PNG H,S,V -> OpenCV V,S,H -> BGR2RGB channel swap -> geometry -> Normalize mean 0/std 1, /255 -> ToTensorV2 | H,S,V, each 0..1 |
| Official D-FINE detection | PNG H,S,V -> PIL H,S,V -> geometry/Resize -> ConvertPILImage float /255, CHW | H,S,V, each 0..1 |

RGB follows the same numerical scaling and geometry within each pair, ending as R,G,B /255. Hue interpolation uses the existing geometric interpolation; changing that would introduce an additional experimental choice.

Generated data.yaml and colour_preparation.json explicitly record colour_space HSV, hsv_encoding HSV_FULL, hue/saturation/value ranges [0,255], hsv_storage_order HSV, tensor_channel_order HSV and tensor_scaling divide_by_255. Cache version `hsv-full-v2` and its fingerprint include the encoding and source image/annotation hashes. Old encoding metadata or altered prepared pixels trigger regeneration. Original source images and historical results remain intact.

Historical HSV is accepted only with verifiable RGB provenance (paired sibling or source_data_yaml), matching class/split/annotation identity and exact pixel correspondence in a recognised encoding/order. Controlled output is always computed anew from RGB using HSV_FULL. An explicit storage order alone is insufficient. Historical detection uses ordinary HSV in V,S,H storage; its originals remain intact. The incomplete dataset_seg_hsv folder remains intact: select HSV with dataset_seg_rgb.

## Preflight and reproducibility records

Run from model_training with the project Python:

```powershell
..\.venv\Scripts\python.exe scripts/validate_final_experiment.py
```

This command runs the controlled suite (excluding the two known saved-notebook-output assertions), proves class/split/annotation/provenance parity, compares every resolved setting with an explicit allowlist of representation/dataset/output fields, checks original Nano checkpoint availability and hashes, and loads real RGB/HSV batches through all five supported pipelines. Strong synthetic pixel tests use an HSV_FULL hue above 179. Failures are reported by check name; critical failures produce `FINAL 300-EPOCH EXPERIMENT VALIDATION: FAIL`. A JSON report, resolved pair configs and test log are retained under runs/final_validation/<timestamp>/.

Training entry points check image split disjointness and any available fruit/track identity mappings before preparing a colour view. Optional data.yaml fruit_identity/track_identity mappings use split/relative-image-name keys without extensions. export_manifest.json image rows can supply split, file_name and fruit_id/track_id/naartjie_id. Known overlap blocks the run; incomplete identity coverage reports NOT PROVABLE. Splits are never rewritten.

The current detection source has 843 train / 504 val / 504 test images; segmentation has 711 / 199 / 90. RGB/HSV relative source identities, annotations (including boxes/polygons), classes and split membership must match. **Physical fruit/track split integrity is NOT PROVABLE** in these datasets because complete identity metadata is absent. Different frames of the same fruit may therefore cross splits; image disjointness alone does not prove fruit disjointness.

Every final run saves framework/library/CUDA/PyTorch/device information, source and prepared datasets, representation metadata, checkpoint/hash, exact resolved settings and actual optimizer groups in run_metadata.json. YOLO also writes args.yaml; project D-FINE retains dfine_seg_training.yaml; official D-FINE records fully inherited YAML. Scheduler, warmup, clipping, AMP/EMA, batches, determinism and every augmentation setting are retained. Validation selects box mAP50-95 for detection and mask mAP50-95 for segmentation; held-out test evaluation stays separate.

Deterministic algorithms use warn_only=True for CUDA operations without deterministic kernels. Pair runs must share hardware and library versions; bitwise reproducibility across all operations/hardware is not claimed. Preflight loads small paired batches without model updates and, when CUDA is available, runs one forward/backward at each final architecture's physical batch size. The memory diagnostic uses native AMP forward and unscaled backward; final training retains native GradScaler behaviour. No optimizer update or training epoch runs. Explicit overrides and resume remain available for legacy work but can restore historical settings.

Saved notebook outputs are preserved. Their two existing content assertions remain separate from controlled-experiment validation.

Historical 100-epoch validation: **94 controlled tests passed**, two saved-notebook-output assertions excluded and independently confirmed to retain their existing failures. All **nine preflight checks passed**, including real RGB/HSV batches and empty resolved-configuration diffs for all five paths. Each batch was finite float32 with shape [1,3,640,640] and values in [0,1]. Checkpoint hashes and runtime versions remain in [the historical validation report](runs/final_validation/20261006-145230-577154/validation_report.json); the test log is retained beside it. New 300-epoch validation always creates a separate timestamped directory with policy checks, resolved configs, scheduler samples, artificial stopping histories, test logs and optional physical-batch memory evidence.

Latest 300-epoch validation: **124 controlled tests passed, zero failed**, with the same two historical saved-notebook-output assertions excluded. All **15 preflight checks passed**, including histories A-F, validation-only checkpoint selection/test isolation, 300-epoch schedulers, real RGB/HSV batches and empty residual configuration diffs for all four final pairs plus the official reference pair. All four physical-batch memory checks passed on the RTX 4060 Laptop GPU: peak allocated memory was 2.510 GiB for YOLO detection, 2.820 GiB for YOLO segmentation, 1.109 GiB for project detection and 1.194 GiB for project segmentation. The smokes performed zero optimizer updates and zero training epochs; no final training was launched. See [the consolidated report and diagnostic tables](runs/final_validation/20261006-185131-437706/final_report.md) and [the raw validation report](runs/final_validation/20261006-185131-437706/validation_report.json). Earlier evidence remains intact.
