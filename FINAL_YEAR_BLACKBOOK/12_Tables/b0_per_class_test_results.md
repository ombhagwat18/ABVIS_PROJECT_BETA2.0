# EfficientNet-B0 — Per-Class Held-Out Test Results

Checkpoint `20260919-164511`. Thresholds derived from B0's own validation
split; no test label used for threshold selection.

| Class | Support | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| damaged_bottle | 64 | 1.000 | 1.000 | 1.000 | 64 | 0 | 0 | 174 |
| damaged_label | 36 | 0.947 | 1.000 | 0.973 | 36 | 2 | 0 | 200 |
| skewed_bottle | 31 | 0.721 | 1.000 | 0.838 | 31 | 12 | 0 | 195 |
| skewed_label | 31 | 1.000 | 1.000 | 1.000 | 31 | 0 | 0 | 207 |
| tilt_cap | 22 | 0.688 | 1.000 | 0.815 | 22 | 10 | 0 | 206 |
| water_level | 23 | 0.852 | 1.000 | 0.920 | 23 | 4 | 0 | 211 |
| missing_label | 12 | 1.000 | 1.000 | 1.000 | 12 | 0 | 0 | 226 |
| missing_cap | 0 | unscored | unscored | unscored | — | — | — | — |
| GOOD (all-zero condition) | 19 | 1.000 | 0.474 | 0.643 | 9 | 0 | 10 | 219 |

GOOD is the implicit all-zero-defect condition, not one of the eight
stored defect classes.
