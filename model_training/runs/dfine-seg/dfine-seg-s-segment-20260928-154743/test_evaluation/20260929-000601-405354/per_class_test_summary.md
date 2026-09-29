# D-FINE-seg class-by-class test results

Checkpoint: `model/model.pt` (best checkpoint). Test split: 90 images, 216 labelled instances. Confidence threshold: 0.50; match IoU threshold: 0.50. Precision, recall, and F1 use the segmentation-mask matching used by the upstream evaluator. Box AP uses all ranked top-K box predictions. The upstream evaluator generates masks only for detections above the confidence threshold, so mask AP uses those retained predictions.

| Class | Test instances | TP | FP | FN | Precision | Recall | F1 | Box AP50 | Mask AP50 | Mask AP50:95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| skin_wrinkling | 27 | 4 | 21 | 23 | 16.0% | 14.8% | 15.4% | 10.3% | 4.4% | 2.1% |
| dark_marking | 72 | 41 | 24 | 31 | 63.1% | 56.9% | 59.9% | 59.2% | 47.6% | 20.4% |
| wind_scarring | 92 | 59 | 34 | 33 | 63.4% | 64.1% | 63.8% | 65.3% | 53.3% | 26.7% |
| light_brown_discolouration | 24 | 3 | 14 | 21 | 17.6% | 12.5% | 14.6% | 13.5% | 9.6% | 3.9% |
| open_skin | 1 | 0 | 0 | 1 | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% | 0.0% |
| **Overall / macro AP** | **216** | **107** | **93** | **109** | **53.5%** | **49.5%** | **51.4%** | **33.6%** | **23.0%** | **10.6%** |

The model performs best on `wind_scarring` and `dark_marking`. It misses most `skin_wrinkling` and `light_brown_discolouration` instances at the configured threshold. The `open_skin` result is too uncertain to generalize from: there is only one labelled test instance, and no prediction above 0.50 confidence. Its box AP50 is nonzero because AP also evaluates lower-confidence predictions.

See [full metrics](test_metrics.json), [confusion matrix](evaluation/plots/test/confusion_matrix.png), [F1 versus threshold](evaluation/plots/test/f1_score_vs_threshold.png), and [precision/recall versus threshold](evaluation/plots/test/precision_recall_vs_threshold.png). The threshold plots use box matching, while the table's precision/recall/F1 use mask matching.
