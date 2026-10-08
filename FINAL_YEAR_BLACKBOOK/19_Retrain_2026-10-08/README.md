# Retraining run, 2026-10-08

All models were retrained on the current `om_bottle` labels (1,167 reviewed images; 72 good, 1,095 defective) and the
Stage 2 v3 detection split. Test scores come from `registry.model_bench cls-test` (held-out test set, 238 images, thresholds chosen on validation).
Every new model is a CANDIDATE: none was activated, so the validation / approval gates in the Models page are still open.

## Classifiers (Stage 1) - held-out test

| Checkpoint | Architecture | Test macro-F1 | Precision | Recall | Exact match | PASS/REJECT acc | Model ms |
|---|---|---|---|---|---|---|---|
| 20261008-081732 | efficientnet_b0 | **0.9385** | 0.908 | 0.977 | 0.902 | **0.963** | 8.4 |
| 20261008-083918 | mobilenet_v3_small | 0.8292 | 0.875 | 0.866 | 0.813 | 0.919 | 7.1 |
| 20261008-090538 | convnext_tiny | 0.7946 | 0.745 | 0.875 | 0.813 | 0.919 | 9.2 |
| 20261008-082510 | resnet18 | 0.7539 | 0.670 | 0.875 | 0.764 | 0.959 | 4.4 |
| 20261008-083257 | efficientnet_b1 | 0.6896 | 0.640 | 0.756 | 0.776 | 0.870 | 13.2 |

Per-epoch history and per-defect validation scores: `classifiers/<stamp>/metrics.json`; test scores: `test_metrics.json`.
Validation F1 is saturated (1.0) for most runs, so the test column is the one to quote.

## Detectors (Stage 2, v3 split, 80 epochs max, 640 px, batch 8)

| Candidate | Parameters | Epochs run | Train minutes | Best validation mAP50 |
|---|---|---|---|---|
| yolov8n_v3 | 3.0 M | 48 (early stop) | 22.5 | 0.979 |
| yolov8s_v3 | 11.1 M | 80 | 48.7 | 0.985 |

Curves, confusion matrices and `results.csv` are under `yolo/<candidate>/train`, test-split plots under `yolo/<candidate>/test_eval`.
The v3 test split (52 images, 6 scenes) has no missing-cap bottle, so missing-cap detection is not validated by these numbers.

## Known limits (read before quoting)
- Only 72 good training images from 8 scenes; missing_cap comes from 3 scenes and missing_label from 2, so the rare classes cannot be judged honestly with one split.
- Retraining on the same labels did not beat the earlier active model (efficientnet_b0 20260919-164511, test F1 0.935).
- Segmentation: 546 label polygons are seeded but none reviewed; no segmentation model exists yet.

## Contents
- `classifiers/`, `yolo/`, `logs/` - this run (weights are not copied; they stay in `projects/om_bottle/models/` and `models/stage2_yolo/`)
- `previous_results/` - metrics, test metrics, registry and provenance of the earlier models. Their weights are kept in `models_archive/previous/` (not tracked by git).
