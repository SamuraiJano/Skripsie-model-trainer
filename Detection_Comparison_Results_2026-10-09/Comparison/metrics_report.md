# Latest four detection runs

Saved metrics are reproduced at their stored precision. No training or inference was run.

D-FINE here is the ArgoHA/D-FINE-seg implementation in detection mode.

## Test comparison

All four saved test evaluations contain 504 images. Values below are fractions, not percentages.

| Run | Precision | Recall | mAP50 | mAP50-95 | Saved F1 | Mean IoU | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO RGB | 0.9907167809595654 | 0.9917369452253173 | 0.9948259346820414 | 0.9843405471233158 | Not saved | Not saved | Not saved | Not saved | Not saved |
| YOLO HSV | 0.9882054422617946 | 0.9885172482415466 | 0.9947207258398857 | 0.9787257699527949 | Not saved | Not saved | Not saved | Not saved | Not saved |
| D-FINE RGB | 0.9927 | 0.9972 | 0.9901 | 0.9829 | 0.9949 | 0.9744 | 23413.0 | 173.0 | 65.0 |
| D-FINE HSV | 0.9931 | 0.9957 | 0.9901 | 0.9799 | 0.9944 | 0.9699 | 23377.0 | 163.0 | 101.0 |

YOLO does not save scalar F1, IoU or TP/FP/FN in its test JSON; these are marked Not saved rather than invented.

## Training and selection

| Run | Metadata completed epochs | Training CSV rows | Metadata best epoch | Best training CSV epoch | Selected validation mAP50-95 | Native training duration | Optimiser updates |
|---|---:|---:|---:|---:|---:|---|---:|
| YOLO RGB | 240 | 239 | 199 | 199 | 0.98359 | 5132.96 seconds | 12657 |
| YOLO HSV | 148 | 147 | 148 | 107 | 0.9778584754167007 | 3460.54 seconds | 7784 |
| D-FINE RGB | 246 | Not saved | 206 | Not saved | 0.9824894666671753 | 8.48 hours (rounded) | 13038 |
| D-FINE HSV | 209 | Not saved | 169 | Not saved | 0.9791693687438965 | 12.88 hours (rounded) | 11077 |

## Source details

### YOLO RGB

Run: `yolo26n-detect-20261006-221958`

Source: [Copied run folder](<../YOLO_RGB>)

Stop reason: `early_stopping`. Test images: 504.

Metadata reports 240 completed epochs; results.csv contains 239 training rows. The validation history contains a final best-checkpoint re-evaluation as an additional entry. Both source counts are preserved.

final_metrics.json combines final best-checkpoint validation P/R/AP with losses, learning rates and time from the last training CSV row. These are different evaluation contexts.

#### final_metrics.json

[Copied file](<../YOLO_RGB/final_metrics.json>)

| Saved metric | Value |
|---|---:|
| `metrics/precision(B)` | 0.9890627541181011 |
| `metrics/recall(B)` | 0.9908740123373182 |
| `metrics/mAP50(B)` | 0.9942892390309134 |
| `metrics/mAP50-95(B)` | 0.9835775487001911 |
| `fitness` | 0.9835775487001911 |
| `epoch` | 239.0 |
| `time` | 5132.96 |
| `train/box_loss` | 0.20173 |
| `train/cls_loss` | 0.08776 |
| `train/dfl_loss` | 0.00068 |
| `val/box_loss` | 0.16471 |
| `val/cls_loss` | 0.07307 |
| `val/dfl_loss` | 0.00063 |
| `lr/pg0` | 0.0002146 |
| `lr/pg1` | 0.0002146 |
| `lr/pg2` | 0.0002146 |

#### test_metrics.json

[Copied file](<../YOLO_RGB/test_metrics.json>)

| Saved metric | Value |
|---|---:|
| `metrics/precision(B)` | 0.9907167809595654 |
| `metrics/recall(B)` | 0.9917369452253173 |
| `metrics/mAP50(B)` | 0.9948259346820414 |
| `metrics/mAP50-95(B)` | 0.9843405471233158 |
| `fitness` | 0.9843405471233158 |

#### results.csv

[Copied file](<../YOLO_RGB/results.csv>)

All 239 rows and every column are included in all_metrics.json and both archives.

Last training row:

| Saved metric | Value |
|---|---:|
| `epoch` | 239 |
| `time` | 5132.96 |
| `train/box_loss` | 0.20173 |
| `train/cls_loss` | 0.08776 |
| `train/dfl_loss` | 0.00068 |
| `metrics/precision(B)` | 0.98877 |
| `metrics/recall(B)` | 0.99129 |
| `metrics/mAP50(B)` | 0.99368 |
| `metrics/mAP50-95(B)` | 0.98205 |
| `val/box_loss` | 0.16471 |
| `val/cls_loss` | 0.07307 |
| `val/dfl_loss` | 0.00063 |
| `lr/pg0` | 0.0002146 |
| `lr/pg1` | 0.0002146 |
| `lr/pg2` | 0.0002146 |

Best training CSV row:

| Saved metric | Value |
|---|---:|
| `epoch` | 199 |
| `time` | 4246.49 |
| `train/box_loss` | 0.21436 |
| `train/cls_loss` | 0.0901 |
| `train/dfl_loss` | 0.00074 |
| `metrics/precision(B)` | 0.98851 |
| `metrics/recall(B)` | 0.99108 |
| `metrics/mAP50(B)` | 0.99429 |
| `metrics/mAP50-95(B)` | 0.98359 |
| `val/box_loss` | 0.16483 |
| `val/cls_loss` | 0.07356 |
| `val/dfl_loss` | 0.00064 |
| `lr/pg0` | 0.0003466 |
| `lr/pg1` | 0.0003466 |
| `lr/pg2` | 0.0003466 |

All 240 validation-history records are retained in all_metrics.json.

### YOLO HSV

Run: `yolo26n-detect-20261007-101115`

Source: [Copied run folder](<../YOLO_HSV>)

Stop reason: `early_stopping`. Test images: 504.

Metadata reports 148 completed epochs; results.csv contains 147 training rows. The validation history contains a final best-checkpoint re-evaluation as an additional entry. Both source counts are preserved.

Best training CSV epoch is 107; metadata records best epoch 148. The latter corresponds to the final checkpoint re-evaluation entry. No source metadata was corrected.

final_metrics.json combines final best-checkpoint validation P/R/AP with losses, learning rates and time from the last training CSV row. These are different evaluation contexts.

#### final_metrics.json

[Copied file](<../YOLO_HSV/final_metrics.json>)

| Saved metric | Value |
|---|---:|
| `metrics/precision(B)` | 0.9861964749935924 |
| `metrics/recall(B)` | 0.9887454344686996 |
| `metrics/mAP50(B)` | 0.9943747918021916 |
| `metrics/mAP50-95(B)` | 0.9778584754167007 |
| `fitness` | 0.9778584754167007 |
| `epoch` | 147.0 |
| `time` | 3460.54 |
| `train/box_loss` | 0.24644 |
| `train/cls_loss` | 0.10291 |
| `train/dfl_loss` | 0.00085 |
| `val/box_loss` | 0.19346 |
| `val/cls_loss` | 0.09822 |
| `val/dfl_loss` | 0.00078 |
| `lr/pg0` | 0.0005182 |
| `lr/pg1` | 0.0005182 |
| `lr/pg2` | 0.0005182 |

#### test_metrics.json

[Copied file](<../YOLO_HSV/test_metrics.json>)

| Saved metric | Value |
|---|---:|
| `metrics/precision(B)` | 0.9882054422617946 |
| `metrics/recall(B)` | 0.9885172482415466 |
| `metrics/mAP50(B)` | 0.9947207258398857 |
| `metrics/mAP50-95(B)` | 0.9787257699527949 |
| `fitness` | 0.9787257699527949 |

#### results.csv

[Copied file](<../YOLO_HSV/results.csv>)

All 147 rows and every column are included in all_metrics.json and both archives.

Last training row:

| Saved metric | Value |
|---|---:|
| `epoch` | 147 |
| `time` | 3460.54 |
| `train/box_loss` | 0.24644 |
| `train/cls_loss` | 0.10291 |
| `train/dfl_loss` | 0.00085 |
| `metrics/precision(B)` | 0.98523 |
| `metrics/recall(B)` | 0.97936 |
| `metrics/mAP50(B)` | 0.99379 |
| `metrics/mAP50-95(B)` | 0.97748 |
| `val/box_loss` | 0.19346 |
| `val/cls_loss` | 0.09822 |
| `val/dfl_loss` | 0.00078 |
| `lr/pg0` | 0.0005182 |
| `lr/pg1` | 0.0005182 |
| `lr/pg2` | 0.0005182 |

Best training CSV row:

| Saved metric | Value |
|---|---:|
| `epoch` | 107 |
| `time` | 2542.24 |
| `train/box_loss` | 0.25658 |
| `train/cls_loss` | 0.1098 |
| `train/dfl_loss` | 0.0009 |
| `metrics/precision(B)` | 0.98625 |
| `metrics/recall(B)` | 0.98866 |
| `metrics/mAP50(B)` | 0.99436 |
| `metrics/mAP50-95(B)` | 0.97785 |
| `val/box_loss` | 0.20407 |
| `val/cls_loss` | 0.08805 |
| `val/dfl_loss` | 0.00083 |
| `lr/pg0` | 0.0006502 |
| `lr/pg1` | 0.0006502 |
| `lr/pg2` | 0.0006502 |

All 148 validation-history records are retained in all_metrics.json.

### D-FINE RGB

Run: `dfine-seg-n-detect-20261007-122047`

Source: [Copied run folder](<../D_FINE_RGB>)

Stop reason: `early_stopping`. Test images: 504.

The native D-FINE P/R/F1 and counts are computed at the configured confidence 0.5 and matching IoU 0.5. Separate validation/test F1-optimal thresholds in extended_metrics.csv describe the plotted confidence sweeps; they are not the thresholds used for the reported P/R/F1. AP uses the native full ranking.

#### test_metrics.json

[Copied file](<../D_FINE_RGB/test_metrics.json>)

| Saved metric | Value |
|---|---:|
| `mAP_50` | 0.9901 |
| `f1` | 0.9949 |
| `precision` | 0.9927 |
| `recall` | 0.9972 |
| `iou` | 0.9744 |
| `mAP_50_95` | 0.9829 |
| `TPs` | 23413.0 |
| `FPs` | 173.0 |
| `FNs` | 65.0 |

#### model/extended_metrics.csv

[Copied file](<../D_FINE_RGB/model/extended_metrics.csv>)

| Split | precision_naartjie | recall_naartjie | iou_naartjie | f1_naartjie | optimal_thresh |
| --- | --- | --- | --- | --- | --- |
| val | 0.9914 | 0.9942 | 0.9697 | 0.9928 | 0.65 |
| test | 0.9927 | 0.9972 | 0.9744 | 0.9949 | 0.6 |

#### model/metrics.csv

[Copied file](<../D_FINE_RGB/model/metrics.csv>)

| Split | mAP_50 | f1 | precision | recall | iou | mAP_50_95 | TPs | FPs | FNs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | 0.9898 | 0.9928 | 0.9914 | 0.9942 | 0.9697 | 0.9825 | 23410 | 204 | 136 |
| test | 0.9901 | 0.9949 | 0.9927 | 0.9972 | 0.9744 | 0.9829 | 23413 | 173 | 65 |

All 246 validation-history records are retained in all_metrics.json.

### D-FINE HSV

Run: `dfine-seg-n-detect-20261008-170850`

Source: [Copied run folder](<../D_FINE_HSV>)

Stop reason: `early_stopping`. Test images: 504.

The native D-FINE P/R/F1 and counts are computed at the configured confidence 0.5 and matching IoU 0.5. Separate validation/test F1-optimal thresholds in extended_metrics.csv describe the plotted confidence sweeps; they are not the thresholds used for the reported P/R/F1. AP uses the native full ranking.

#### test_metrics.json

[Copied file](<../D_FINE_HSV/test_metrics.json>)

| Saved metric | Value |
|---|---:|
| `mAP_50` | 0.9901 |
| `f1` | 0.9944 |
| `precision` | 0.9931 |
| `recall` | 0.9957 |
| `iou` | 0.9699 |
| `mAP_50_95` | 0.9799 |
| `TPs` | 23377.0 |
| `FPs` | 163.0 |
| `FNs` | 101.0 |

#### model/extended_metrics.csv

[Copied file](<../D_FINE_HSV/model/extended_metrics.csv>)

| Split | precision_naartjie | recall_naartjie | iou_naartjie | f1_naartjie | optimal_thresh |
| --- | --- | --- | --- | --- | --- |
| val | 0.991 | 0.9952 | 0.9671 | 0.9931 | 0.55 |
| test | 0.9931 | 0.9957 | 0.9699 | 0.9944 | 0.65 |

#### model/metrics.csv

[Copied file](<../D_FINE_HSV/model/metrics.csv>)

| Split | mAP_50 | f1 | precision | recall | iou | mAP_50_95 | TPs | FPs | FNs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | 0.9898 | 0.9931 | 0.991 | 0.9952 | 0.9671 | 0.9792 | 23433 | 212 | 113 |
| test | 0.9901 | 0.9944 | 0.9931 | 0.9957 | 0.9699 | 0.9799 | 23377 | 163 | 101 |

All 209 validation-history records are retained in all_metrics.json.
