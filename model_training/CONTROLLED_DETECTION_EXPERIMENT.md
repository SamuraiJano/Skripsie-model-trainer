# Final controlled detection experiment

Use `notebooks/controlled_detector_experiment.ipynb` or `scripts/controlled_detector_experiment.py` for the final detector comparison. The older dashboards remain available for exploratory detection and segmentation. Their summaries are not the common final evaluator.

The four groups are **YOLO26n RGB, YOLO26n HSV_FULL, official Peterande D-FINE-N RGB, and official Peterande D-FINE-N HSV_FULL**, each with seeds **0, 1, 2, 3, 4**. Every seed starts fresh from `yolo26n.pt` or official `pretrained/dfine_n_coco.pth`. D-FINE-seg is excluded. Nano models have different parameter counts and pretraining recipes; this experiment compares their pretrained Nano implementations, rather than matching capacity exactly.

## Fixed controls

| Setting | Final protocol |
|---|---|
| Duration | Maximum 300, minimum 100 completed epochs, patience 40 |
| Selection | Validation every epoch; common COCO box mAP50–95; strict improvement; first checkpoint wins ties |
| Images | 640 × 640, aspect-preserving resize and padding |
| Splits | Same existing exported train/validation/test partitions; no new random split |
| Colour | Canonical decoded RGB or OpenCV `COLOR_RGB2HSV_FULL`; uint8 channels then float / 255 once |
| Augmentation | Colour jitter, channel swaps, flips, rotation, translation, scale jitter, perspective, Mosaic, MixUp and CutMix off |
| Precision/hardware | AMP for training; CUDA device 0, RTX 4060 Laptop; fixed within each colour pair |
| Workers | 0; training `drop_last=False` |
| AP | One `faster_coco_eval.COCOeval_faster` implementation, IoU 0.50:0.05:0.95, 101 recall points, score floor 0.001, at most 100 predictions per image |
| P/R/F1 | Micro averages with class-aware greedy matching at IoU 0.50; confidence chosen by maximum validation F1, highest threshold wins ties |
| Test | Open for inference after training and validation confidence selection; once per final run |

Patience counts consecutive validation epochs without improvement, including those before epoch 100; the minimum prevents stopping before epoch 100. A flat run can therefore stop at epoch 100. A later improvement resets patience. The global best validation checkpoint may occur before epoch 100. AP uses the complete low-floor ranking, independently of the frozen operating confidence.

RGB and HSV are logical paired dataset views over the same original RGB files, with identical sample names, labels, splits and source dimensions. Preparation creates each sample's exact uint8 HSV_FULL array in memory and saves its pixel hash and dimensions in `paired_samples.json`; it does not duplicate the full-resolution image store. Both trainers use OpenCV-decoded canonical RGB, apply geometry, then choose RGB or HSV_FULL. This avoids HSV interpolation/padding, lossy HSV compression and repeated conversion. The view YAMLs explicitly declare runtime conversion and require the controlled detector adapters. Letterbox implementations may differ across architectures but remain identical within each architecture's RGB/HSV pair.

The canonical RGB view hard-links the original compressed image bytes where supported and copies only when necessary. Labels are copied, with duplicate rows removed using the same float32 equality rule YOLO would apply automatically; both families therefore receive the same annotations. Original labels remain untouched, raw and resolved label hashes are retained, and removal counts are recorded per split. This dataset currently includes 70 duplicate training annotations in one image. Periodic official D-FINE checkpoint frequency is 300 rather than every epoch; best and latest checkpoints still update normally, avoiding unnecessary disk consumption.

Image identities, annotations, source pixels, checkpoint bytes, grouping metadata, implementation files and installed framework versions are hashed. Known fruit, track or sequence IDs crossing splits block preparation. If identity coverage is incomplete, the dashboard and manifest explicitly report **NOT PROVABLE**. Existing exported partitions are preserved; UUID filenames alone cannot prove that related fruit frames are isolated. This limitation prevents a claim of “100% fair” or proven leakage-free data until reliable grouping metadata is supplied.

## Architecture-specific baselines

| Setting | YOLO26n | Official D-FINE-N |
|---|---|---|
| Physical batch / nominal accumulation | 16 / 1 | 8 / 1 |
| Nominal effective batch | 16 | 8 |
| Optimiser | AdamW | AdamW with native parameter groups |
| Initial LR | 0.001 | head 0.0008; backbone 0.0004 |
| Weight decay | 0.0005, native scaling/group rules | 0.0001; native bias/norm exclusions |
| LR schedule | Native linear decay, final fraction 0.01 | Native MultiStepLR, milestone 500, gamma 0.1 |
| Warm-up | Native 3 epochs | Native 500 updates |
| Clipping | Native YOLO norm 10 | Native norm 0.1 |
| EMA/loss | Native Ultralytics | Native official EMA (0.9999, 1000 warm-up updates) and D-FINE losses |

The D-FINE milestone 500 is inherited from its published custom Nano recipe and lies beyond this experiment's maximum; it is not presented as a validation-optimised decay schedule. Its native final-stage checkpoint/EMA refresh is retained with the transition at epoch 280, preserving the final 20-epoch stage at the extended maximum. All native behaviour and loaded checkpoint state are frozen within the colour pair. See the [official custom Nano configuration](https://github.com/Peterande/D-FINE/blob/master/configs/dfine/custom/dfine_hgnetv2_n_custom.yml) and [Ultralytics training settings](https://docs.ultralytics.com/modes/train/).

These are sensible transfer baselines, not a claim that both models have been exhaustively optimised. Both receive **zero additional tuning trials** by default. An optional ledger supports up to three preliminary configurations per architecture, with equal completed trial counts and one common RGB/seed-0/at-most-100-epoch protocol. Only architecture-specific optimisation parameters can change. Record each configuration, its validation result, and whether selected or discarded; exactly one must be selected per architecture. The selected overrides are applied before freezing and used by both colours. No test results belong in the tuning ledger. An example schema is:

```json
{
  "maximum_trials_per_architecture": 3,
  "test_split_used": false,
  "trial_protocol": {"colour": "rgb", "seed": 0, "max_epochs": 60},
  "trials": {
    "yolo26n": [{"configuration": {"lr0": 0.001}, "selection_split": "val", "validation_mAP50-95": 0.5, "selected": true}],
    "dfine_n": [{"configuration": {"head_lr": 0.0008, "backbone_lr": 0.0004}, "selection_split": "val", "validation_mAP50-95": 0.5, "selected": true}]
  }
}
```

D-FINE's native loop has one optimizer update per physical batch. Adding accumulation would change its scheduler/EMA/update semantics, so no artificial batch equality is imposed. YOLO retains native accumulation; observed values and optimizer update counts are recorded rather than claiming fixed behaviour from the form. Smaller final batches are retained and documented. Automatic YOLO OOM batch reduction is disabled: an unstable recipe fails visibly and requires a separately frozen experiment. GPU memory stability depends on the dataset's object density; these fixed baselines must be checked before a full launch. The existing dataset contains crowded training images.

## Commands

From `model_training`, using the shared `.venv` Python or the active notebook environment:

```powershell
python scripts/controlled_detector_experiment.py prepare
python scripts/controlled_detector_experiment.py verify "runs/controlled_detection/<new-id>/experiment.json"
python scripts/controlled_detector_experiment.py run "runs/controlled_detection/<new-id>/experiment.json"
python scripts/controlled_detector_experiment.py aggregate "runs/controlled_detection/<new-id>/experiment.json"
```

`prepare` verifies/pairs data and writes a separate frozen configuration and a 20-row CSV matrix; it does not train. `run` rechecks the hashes and displays the entire matrix before scheduling sequential isolated subprocesses. Completed runs are preserved and skipped on a later invocation. Incomplete run directories are preserved and block automatic reuse. Changing the code, framework, source dataset or checkpoint after preparation requires a new experiment. Use `--seeds 0 --fewer-seeds-reason "..."` only for a documented computational constraint, and `--horizontal-flip 0.5` only when mirrored conveyor views are physically valid.

Before a full launch, the bounded preflight command checks a real model and loader without optimizer updates or test images:

```powershell
python scripts/controlled_detector_preflight.py "<manifest>" --model dfine_n --colour hsv --device cpu --batch 1
python scripts/controlled_detector_preflight.py "<manifest>" --model dfine_n --colour hsv --device cuda:0 --batch 8 --backward
python scripts/controlled_detector_preflight.py "<manifest>" --model yolo26n --colour hsv --device cuda:0 --batch 16 --backward
```

Run CUDA memory checks when the GPU is idle, for both RGB and HSV. CPU batch-1 forward checks establish adapter compatibility, but do not establish GPU training-batch stability. Preflight reports are separate from final results and clearly record the actual batch/device, backward flag and zero optimizer updates.

Each run saves resolved optimiser groups, LR/warm-up/scheduler/EMA/clipping, actual precision/worker/batch/accumulation/update counts, effective configuration, runtime, dataset hashes, completed epochs, stop reason, best epoch/checkpoint, validation history, frozen confidence selection and final test predictions/metrics in `run_metadata.json` and companion files. Test inference is marked started before opening the test images, so failures do not silently trigger a second evaluation.

The dashboard displays a bounded recent console tail; complete logs are written to disk. Seeding and deterministic settings are recorded, but unsupported CUDA operations such as D-FINE grid-sampling backward use PyTorch's warning-only policy. Repeated seeds support a reproducible protocol and measured variation; bit-for-bit GPU replay is not guaranteed.

`comparison_per_seed.csv`, `comparison_summary.csv` and `comparison.json` include completed epochs, best validation AP, both test APs, P/R/F1, selected confidence, training seconds and batch information. Summary standard deviation is the sample standard deviation (`ddof=1`); it is empty for fewer than two completed seeds. Incomplete groups are labelled explicitly, and only results matching the frozen manifest and recipe hashes are aggregated. Segmentation and earlier trained models/results remain untouched.

The interpretation is: does HSV_FULL improve this pretrained detector when fine-tuned under its frozen architecture-specific recipe? It does not establish universal superiority of HSV independently of RGB pretraining.
