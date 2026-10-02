# Held-Out Test Evaluation Results — Stage 1

## Test set

238 images, 23 distinct scenes. Identical across all four architecture
evaluations (confirmed byte-identical `test_paths` lists). Held out before
training or validation began; never used for training, epoch selection, or
threshold selection for any of the four models.

## Threshold methodology

For every model below, per-class decision thresholds were selected using
only that model's own 220-image validation split (maximum-F1 sweep). No
test label was read or used to select or adjust any threshold. This is
stated explicitly per instruction, and is true for all results in this
document.

---

## EfficientNet-B0 — checkpoint `20260919-164511`

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

GOOD is not one of the eight stored defect classes; it is the derived,
implicit condition where a model predicts every defect class absent. It is
reported separately from the defect classes for that reason.

- Test macro-F1 (7 scored classes): **0.935**
- Test macro-precision: **0.887**
- Test macro-recall: **1.000**
- Exact-match accuracy: **89.1% (212/238)**
- Total false-positive label instances: **28**
- Total false-negative label instances: **0**
- Images with at least one incorrect label: 26 / 238
- Test-set inference latency / throughput: **13.89 ms / 72.0 FPS** (live-scoring methodology: per-image crop, letterbox, forward pass, measured during the test evaluation run itself)
- Model size: **15.57 MB**

`missing_cap` had zero positive images anywhere in the dataset and was
therefore disabled (unreachable threshold) and reported as unscored, not as
a passing or failing score.

---

## EfficientNet-B1 — checkpoint `20260919-222031`

| Class | Support | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| damaged_bottle | 64 | 0.711 | 1.000 | 0.831 | 64 | 26 | 0 | 148 |
| damaged_label | 36 | 1.000 | 1.000 | 1.000 | 36 | 0 | 0 | 202 |
| skewed_bottle | 31 | 0.554 | 1.000 | 0.713 | 31 | 25 | 0 | 182 |
| skewed_label | 31 | 1.000 | 1.000 | 1.000 | 31 | 0 | 0 | 207 |
| tilt_cap | 22 | 0.387 | 0.545 | 0.453 | 12 | 19 | 10 | 197 |
| water_level | 23 | 0.852 | 1.000 | 0.920 | 23 | 4 | 0 | 211 |
| missing_label | 12 | 0.000 | 0.000 | 0.000 | 0 | 0 | 12 | 226 |
| missing_cap | 0 | unscored | unscored | unscored | — | — | — | — |
| GOOD (all-zero condition) | 19 | 0.000 | 0.000 | 0.000 | 0 | 10 | 19 | 209 |

- Test macro-F1: **0.845** · macro-precision **0.786** · macro-recall **0.935**
- Exact-match accuracy: **68.1% (162/238)**
- Total false positives: **74** · total false negatives: **10**
- Images with at least one incorrect label: 76 / 238
- Test-set inference latency / throughput: **17.71 ms / 56.5 FPS**
- Model size: **25.24 MB**

---

## MobileNetV3-Small — checkpoint `20260920-102102`

| Class | Support | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| damaged_bottle | 64 | 1.000 | 1.000 | 1.000 | 64 | 0 | 0 | 174 |
| damaged_label | 36 | 1.000 | 0.972 | 0.986 | 35 | 0 | 1 | 202 |
| skewed_bottle | 31 | 0.721 | 1.000 | 0.838 | 31 | 12 | 0 | 195 |
| skewed_label | 31 | 0.351 | 0.839 | 0.495 | 26 | 48 | 5 | 159 |
| tilt_cap | 22 | 0.182 | 0.182 | 0.182 | 4 | 18 | 18 | 198 |
| water_level | 23 | 0.852 | 1.000 | 0.920 | 23 | 4 | 0 | 211 |
| missing_label | 12 | 0.000 | 0.000 | 0.000 | 0 | 0 | 12 | 226 |
| missing_cap | 0 | unscored | unscored | unscored | — | — | — | — |
| GOOD (all-zero condition) | 19 | 0.000 | 0.000 | 0.000 | 0 | 6 | 19 | 213 |

- Test macro-F1: **0.632** · macro-precision **0.587** · macro-recall **0.713**
- Exact-match accuracy: **67.2% (160/238)**
- Total false positives: **82** · total false negatives: **36**
- Images with at least one incorrect label: 78 / 238
- Test-set inference latency / throughput: **14.04 ms / 71.2 FPS**
- Model size: **5.93 MB**

---

## ResNet18 — checkpoint `20260920-110121` (trained with early stopping, patience = 5)

| Class | Support | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| damaged_bottle | 64 | 1.000 | 1.000 | 1.000 | 64 | 0 | 0 | 174 |
| damaged_label | 36 | 1.000 | 1.000 | 1.000 | 36 | 0 | 0 | 202 |
| skewed_bottle | 31 | 0.721 | 1.000 | 0.838 | 31 | 12 | 0 | 195 |
| skewed_label | 31 | 0.721 | 1.000 | 0.838 | 31 | 12 | 0 | 195 |
| tilt_cap | 22 | 0.688 | 1.000 | 0.815 | 22 | 10 | 0 | 206 |
| water_level | 23 | 0.852 | 1.000 | 0.920 | 23 | 4 | 0 | 211 |
| missing_label | 12 | 0.000 | 0.000 | 0.000 | 0 | 0 | 12 | 226 |
| missing_cap | 0 | unscored | unscored | unscored | — | — | — | — |
| GOOD (all-zero condition) | 19 | 1.000 | 0.474 | 0.643 | 9 | 0 | 10 | 219 |

- Test macro-F1: **0.773** · macro-precision **0.712** · macro-recall **0.857**
- Exact-match accuracy: **84.0% (200/238)**
- Total false positives: **38** · total false negatives: **12**
- Images with at least one incorrect label: 38 / 238
- Test-set inference latency / throughput: **16.71 ms / 59.8 FPS**
- Model size: **42.72 MB**

---

## Important scientific caveat

A test macro-F1 of 93.5% (EfficientNet-B0) describes performance on this
specific 238-image, 23-scene held-out test set under this specific
experimental protocol. It is not a claim of 93.5% accuracy under real
production conditions, which would involve camera variability, lighting
drift, bottle types, and defect presentations not represented in the
current dataset. Similarly, "no false-negative label instances were
observed for EfficientNet-B0 on the evaluated 238-image held-out test set"
is a statement about the specific test set evaluated, not a guarantee about
future or unseen data.
