# Dataset Class Distribution

Positive image counts out of 1,143 reviewed images:

| Defect class | Positive images |
|---|---|
| damaged_bottle | 319 |
| damaged_label | 229 |
| skewed_bottle | 166 |
| tilt_cap | 145 |
| water_level | 112 |
| skewed_label | 70 |
| missing_label | 30 |
| missing_cap | 0 |

Per-class scene counts, and per-class positive counts broken down by
train/validation/test partition, are **to be populated from dataset
audit** — not computed during this documentation pass, except for
`missing_label`, whose scene distribution is documented in
`09_Limitations/DATASET_LIMITATIONS.md` (30 images, 2 scenes, 0 in
validation).
