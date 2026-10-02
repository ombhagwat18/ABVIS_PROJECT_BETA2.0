# Per-Model Failure Summary — Held-Out Test Set (238 images)

| Model | Total FP | Total FN | Images with ≥1 incorrect label | Worst class (by F1) |
|---|---|---|---|---|
| EfficientNet-B0 | 28 | 0 | 26 / 238 | tilt_cap (F1 0.815) |
| EfficientNet-B1 | 74 | 10 | 76 / 238 | tilt_cap (F1 0.453) |
| MobileNetV3-Small | 82 | 36 | 78 / 238 | tilt_cap (F1 0.182) |
| ResNet18 | 38 | 12 | 38 / 238 | missing_label (F1 0.000) |

See `06_AI_Model_Development/Failure_Analysis/FAILURE_ANALYSIS.md` for the
per-class root-cause discussion.
