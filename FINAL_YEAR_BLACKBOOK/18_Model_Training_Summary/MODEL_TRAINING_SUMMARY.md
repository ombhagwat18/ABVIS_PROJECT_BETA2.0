# Model training summary

Generated from `models/MODEL_REGISTRY.json` (rebuilt by `python model_bench.py registry`); nothing typed by hand.
Validation F1 is saturated (1.0) on most checkpoints, so models are compared on the **held-out test set**, which is never used to pick an epoch or threshold.

## Stage 1: multi-label classifiers (om_bottle, input 192x448)

| Arch | Stamp | Epochs run | Batch | LR | Device | Train/val/test imgs | Val F1 | Test macro-F1 | Test exact-match | Test PASS/REJECT acc | False PASS | False REJECT | GPU ms | Size MB | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet18 | 20260918-120030 | 1/1 | 16 | 0.0003 | cpu | 879/264/- | 1.000 | - | - | - | - | - | - | 42.72 | legacy split (no held-out test recorded) |
| efficientnet_b0 | 20260918-191743 | 25/25 | 32 | 0.0003 | cpu | 879/264/- | 1.000 | - | - | - | - | - | - | 15.57 | legacy split (no held-out test recorded) |
| efficientnet_b1 | 20260918-220502 | 25/25 | 32 | 0.0003 | cpu | 879/264/- | 1.000 | - | - | - | - | - | - | 25.24 | legacy split (no held-out test recorded) |
| mobilenet_v3_small | 20260919-000152 | 25/25 | 32 | 0.0003 | cpu | 879/264/- | 0.848 | - | - | - | - | - | - | 5.93 | legacy split (no held-out test recorded) |
| convnext_tiny | 20260919-101315 | 25/25 | 32 | 0.0003 | cpu | 879/264/- | 1.000 | - | - | - | - | - | - | 106.19 | legacy split (no held-out test recorded) |
| efficientnet_b0 | 20260919-164511 | 25/25 | 32 | 0.0003 | cuda | 685/220/238 | 1.000 | 0.935 | 0.891 | 0.958 | 0 | 10 | 32.8 | 15.57 | test-evaluated |
| efficientnet_b1 | 20260919-222031 | 25/25 | 32 | 0.0003 | cuda | 685/220/238 | 1.000 | 0.845 | 0.681 | 0.878 | 10 | 19 | 31.3 | 25.24 | test-evaluated |
| mobilenet_v3_small | 20260920-102102 | 25/25 | 32 | 0.0003 | cuda | 685/220/238 | 0.864 | 0.631 | 0.672 | 0.895 | 6 | 19 | 21.7 | 5.93 | test-evaluated |
| resnet18 | 20260920-110121 | 8/25 | 32 | 0.0003 | cuda | 685/220/238 | 0.933 | 0.773 | 0.840 | 0.958 | 0 | 10 | 12.6 | 42.72 | test-evaluated |
| convnext_tiny | 20261003-101425 | 25/25 | 16 | 0.0003 | cuda | 685/220/238 | 1.000 | 0.764 | 0.681 | 0.920 | 0 | 19 | 19.8 | 106.19 | test-evaluated |
| resnet18 | 20261004-100320 | 25/25 | 32 | 0.0003 | cuda | 685/220/238 | 1.000 | 0.776 | 0.853 | 0.920 | 0 | 19 | 12.1 | 42.72 | test-evaluated |

Rows marked *legacy split* predate the held-out test set and cannot be honestly scored on it.
`resnet18 20260920-110121` ran only 8 epochs (an interrupted run); `20261004-100320` is the full 25-epoch rerun made on 2026-10-04 (test macro-F1 0.776 vs 0.773 at 8 epochs: the extra epochs changed almost nothing; validation F1 was already 1.0, and resnet18 stays behind EfficientNet-B0 at 0.935).

## Stage 2: component detectors (bottle / cap / label, 594 images, 25/7/7 scenes)

| Model | Deployed | Val mAP50 | Val mAP50-95 | Test mAP50 | Test mAP50-95 | Test P | Test R | Params | Size MB | GPU ms/img |
|---|---|---|---|---|---|---|---|---|---|---|
| det_yolov8n_stage2 | yes | 0.976 | 0.730 | 0.968 | 0.660 | 0.936 | 0.968 | 3006233 | 6.23 | 37.1 |
| det_yolov8n_stage2 | no | 0.986 | 0.852 | 0.979 | 0.793 | 0.960 | 0.983 | 3006233 | 6.23 | 39.1 |
| det_yolov8s_stage2 | no | 0.980 | 0.744 | 0.973 | 0.676 | 0.934 | 0.983 | 11126745 | 22.5 | 36.7 |

## Segmentation (label outline)

**Not trained.** `stage2_dataset/seg_annotations.json` holds 594 machine-seeded polygons, 0 reviewed. Training on unreviewed polygons would teach the model the seeding algorithm's mistakes, so it waits for human review (Annotate tab).

## Not covered

- Nothing here was measured on the physical machine. Test sets are small (238 images / 7 scenes for detection), so differences of a few points are within noise.
- `missing_cap` has no examples in the classifier data and is disabled; missing parts are found by the detector + decision rules instead.
