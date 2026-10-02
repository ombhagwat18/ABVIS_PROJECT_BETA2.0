# Dataset Limitations — Tabulated

| Limitation | Value / Evidence | Consequence observed |
|---|---|---|
| `missing_cap` has zero positive samples | 0 / 1,143 reviewed images | Unlearnable; unscored in all four experiments |
| `missing_label` is a thin class | 30 images, 2 distinct scenes | 0 validation positives in the current split |
| `missing_label` threshold untested on validation | 0 validation positives | 0/12 test recall on B1, MobileNetV3-Small, ResNet18; 12/12 on B0 |
| No genuine multi-defect images | 0 / 1,143 images have >1 defect flag | Multi-label capability not experimentally validated |
| Held-out test set is small | 238 images / 23 scenes | Test statistics reflect 23 independent samples, not 238 |
| Whole dataset is small | 1,143 images / 112 scenes | Ceiling on statistical confidence for every reported metric |

Full narrative discussion in `09_Limitations/DATASET_LIMITATIONS.md`.
