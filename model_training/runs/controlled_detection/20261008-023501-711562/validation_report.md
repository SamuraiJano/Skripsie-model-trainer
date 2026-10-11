# Controlled detector experiment validation

The frozen matrix contains 20 planned runs; no final training or held-out test inference has started.

Maximum 300 epochs, minimum 100, patience 40, validation each epoch using one COCO box AP evaluator. RGB/HSV recipes are frozen per architecture.

Dataset: 843 training, 504 validation and 504 test images; all source partitions preserved. The shared annotation view consistently removes 70 redundant training annotations; source labels remain unchanged.

Related-fruit/track/sequence isolation is NOT PROVABLE because complete grouping metadata is absent.

42 controlled checks pass. Full suite: 185 pass and one pre-existing assertion fails because the user's D-FINE notebook contains saved outputs. Outputs are preserved.

## Physical batch checks

AMP forward/backward on the densest training images; zero optimizer updates and no test inference.

| Model | Representation | Physical batch | Peak allocated GPU memory |
|---|---|---:|---:|
| dfine_n | rgb | 8 | 2.080 GiB |
| dfine_n | hsv | 8 | 2.071 GiB |
| yolo26n | rgb | 16 | 2.429 GiB |
| yolo26n | hsv | 16 | 2.429 GiB |

The three earlier physical checks use the same frozen recipes, checkpoints, dataset and framework as this matrix. The subsequent implementation change only bounds notebook console memory; each raw report retains its source manifest hash.

## Storage incident

The initial lossless cache exhausted the remaining disk space and was removed. The final preparation hard-links original RGB bytes and creates HSV_FULL in memory with a per-sample audit. No prior model was deleted or overwritten.

The earlier D-FINE-seg run stopped at epoch 32 during this incident. Its last checkpoint loads and all 674 tensors are finite. It contains weights only, so the exact previous optimizer state cannot be restored. The run and checkpoint remain available.

The comparison CSV/JSON files currently mark all groups incomplete; the runner fills them from completed final runs and reports sample standard deviation (ddof=1).
