# Latest four detection runs

Training and testing outputs for YOLO26n RGB/HSV and the ArgoHA D-FINE Nano detector RGB/HSV. All four saved test evaluations contain 504 images.

| Run | Copied folder | Original run ID | Files |
|---|---|---|---:|
| YOLO RGB | [YOLO_RGB](YOLO_RGB/run_metadata.json) | `yolo26n-detect-20261006-221958` | 38 |
| YOLO HSV | [YOLO_HSV](YOLO_HSV/run_metadata.json) | `yolo26n-detect-20261007-101115` | 38 |
| D-FINE RGB | [D_FINE_RGB](D_FINE_RGB/run_metadata.json) | `dfine-seg-n-detect-20261007-122047` | 65 |
| D-FINE HSV | [D_FINE_HSV](D_FINE_HSV/run_metadata.json) | `dfine-seg-n-detect-20261008-170850` | 68 |

## Comparison and complete metric histories

- [Metric report](Comparison/metrics_report.md): validation/test scores, final training metrics, durations and epoch information.
- [All metrics JSON](Comparison/all_metrics.json): exact saved values, configurations, all validation histories, every YOLO training CSV row and D-FINE per-epoch displayed training losses.
- [D-FINE training losses](Comparison/dfine_training_losses.md): loss histories for all 246 RGB and 209 HSV epochs.
- [File index](FILE_INDEX.md): every copied run output and comparison report.
- [Copy verification](copy_verification.json): SHA-256 verification against each source file.

## Training and testing files

- YOLO folders: `results.csv`, `results.png`, validation histories, train/validation batch images, P/R/F1/PR curves, confusion matrices, `final_metrics.json`, `test_metrics.json`, test graphs under `test_eval`, `args.yaml`, referenced dataset YAML and both best/latest weights.
- D-FINE folders: validation histories, native train/console logs, `model/metrics.csv`, `model/extended_metrics.csv`, validation/test graphs under `model/plots`, prediction images under `eval_preds`, training/resolved YAML, `test_metrics.json` and best/latest weights.

Prepared dataset annotation JSON files are included where they were stored in the run. The 1,851 prepared source images remain in the original dataset cache and the earlier complete archive.

Original run files and their raw metadata are unchanged. Raw metadata preserves the original absolute paths; the readable comparison report points to this folder. The report documents YOLO metadata epoch-count and best-epoch discrepancies. D-FINE training loss values retain the rounding of the native console display.
