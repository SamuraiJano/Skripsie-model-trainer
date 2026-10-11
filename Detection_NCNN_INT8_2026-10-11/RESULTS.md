# Measured NCNN INT8 results

All percentages below are COCO mAP50–95 under the common evaluator (max 100 detections per image), not the saved trainer metrics. Calibration: 843 training images. Validation: 504 images. Test: 504 images. Defaults and all speed alternatives were frozen using validation before test assessment. Positive loss means lower AP than FP32; `pp` means percentage points.

| Colour | Variant | INT8 convolutions | Validation AP | Validation loss | Test AP | Test loss | Param + weights |
|---|---|---:|---:|---:|---:|---:|---:|
| RGB | fp32 | 0/102 | 96.9908% | 0.0000 pp | 97.0500% | 0.0000 pp | 9.627 MB |
| RGB | minmax_stem11_head (default) | 67/102 | 96.8867% | 0.1041 pp | 96.9430% | 0.1070 pp | 2.744 MB |
| RGB | minmax_stem6_head | 72/102 | 96.7931% | 0.1976 pp | 96.9381% | 0.1119 pp | 2.683 MB |
| HSV | fp32 | 0/102 | 96.6528% | 0.0000 pp | 96.8021% | 0.0000 pp | 9.627 MB |
| HSV | minmax_stem11_head (default) | 67/102 | 96.5542% | 0.0986 pp | 96.7089% | 0.0932 pp | 2.744 MB |
| HSV | minmax_stem6_head | 72/102 | 96.5536% | 0.0992 pp | 96.7133% | 0.0889 pp | 2.683 MB |
| HSV | minmax_stem6_regression | 84/102 | 96.5301% | 0.1228 pp | 96.6851% | 0.1171 pp | 2.641 MB |

## Workstation speed (not Raspberry Pi speed)

These measurements ran sequentially after calibration and all accuracy evaluations finished, with four NCNN CPU threads, 15 warmups and 60 timed repetitions per path on three real images. OpenCV uses one thread. FP16 arithmetic/storage and Vulkan are disabled. Image-to-detections includes local read/decode and exact preprocessing; inference includes CPU TopK. Median and p95 are milliseconds. Full machine/package details are in `source_models.json` and each benchmark report.

| Colour | Variant | Inference median | Inference p95 | Image-to-detections median | Image-to-detections p95 |
|---|---|---:|---:|---:|---:|
| RGB | fp32 | 106.75 | 112.63 | 157.08 | 168.02 |
| RGB | minmax_stem11_head | 97.92 | 101.41 | 152.97 | 158.97 |
| RGB | minmax_stem6_head | 95.55 | 98.32 | 147.74 | 155.03 |
| HSV | fp32 | 107.79 | 111.65 | 143.13 | 147.51 |
| HSV | minmax_stem11_head | 98.41 | 100.65 | 132.27 | 137.01 |
| HSV | minmax_stem6_head | 95.45 | 97.64 | 130.66 | 134.73 |
| HSV | minmax_stem6_regression | 96.72 | 99.32 | 131.61 | 137.39 |

The defaults prioritize the highest measured validation accuracy among tested INT8 variants within 0.2 pp of FP32 for both AP50–95 and AP50. Raspberry Pi latency has not been measured. Run the supplied `--benchmark --tune` command on the Pi to select the fastest of the accuracy-qualified alternatives and the best of 1/2/4 threads. The models retain floating-point stems and detection layers to avoid the much larger losses observed with every convolution quantized.

Fully quantized KL, ACIQ and full-range variants, and smaller protected layer groups, were assessed before selection. Exact comparison reports are included. The 64-image screenings are summarized separately and were followed by full validation for finalists. EQ did not complete and is excluded from the results.
