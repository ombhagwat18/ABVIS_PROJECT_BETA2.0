# Failure Analysis — Stage 1

This document analyses where each of the four evaluated architectures
produced incorrect predictions on the 238-image held-out test set, using
validation-derived thresholds. All figures are taken directly from
`06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md`.

## Failure summary by model

| Model | Total FP | Total FN | Images with ≥1 incorrect label | Worst class (by F1) |
|---|---|---|---|---|
| EfficientNet-B0 | 28 | 0 | 26 / 238 | tilt_cap (F1 0.815) |
| EfficientNet-B1 | 74 | 10 | 76 / 238 | tilt_cap (F1 0.453) |
| MobileNetV3-Small | 82 | 36 | 78 / 238 | tilt_cap (F1 0.182) |
| ResNet18 | 38 | 12 | 38 / 238 | missing_label (F1 0.000) |

## `missing_label` — the most consistent failure

`missing_label` achieved 0/12 recall on the held-out test set for
EfficientNet-B1, MobileNetV3-Small, and ResNet18 (every false negative for
this class, for these three models, is a missed defect). EfficientNet-B0
was the only model to recall all 12 test positives for this class.

Root cause, traced to the dataset rather than to any one architecture: the
`missing_label` class has only 30 images across 2 distinct scenes in the
whole dataset, and the deterministic scene-based split places 0 of those
scenes into the validation partition. Every model's decision threshold for
`missing_label` was therefore selected with no positive validation example
to check it against, using a default fallback threshold. For three of the
four architectures, that untested threshold turned out to be unreachable
against real test-time probabilities. This should not be generalised as
proof that EfficientNet-B0 has permanently solved this class — it reflects
one architecture's probability calibration lining up, by observation, with
an untested threshold on one held-out set of 12 images.

## `tilt_cap` — degrades sharply on non-B0 architectures

`tilt_cap` scores F1 0.815 on both EfficientNet-B0 and ResNet18 (identical
figures for this class on both models), but degrades to F1 0.453 on
EfficientNet-B1 (10 missed, 19 false alarms) and F1 0.182 on
MobileNetV3-Small (18 missed, 18 false alarms out of 22 true positives).
This class shows the largest architecture-dependent variance of any scored
class.

## `skewed_label` — smallest well-represented class, threshold-sensitive

With only 31 test positives (and validation support as low as 9 for some
models), `skewed_label` shows the widest precision swing across
architectures: 1.000 (EfficientNet-B0) down to 0.351 (MobileNetV3-Small).
This is consistent with a class that has too few examples for its decision
threshold to be estimated robustly.

## `GOOD` (implicit all-zero-defect condition)

Recall of the implicit GOOD condition — a truly defect-free bottle
predicted as defect-free on every class — is 0.474 for EfficientNet-B0 and
ResNet18, and 0.000 for EfficientNet-B1 and MobileNetV3-Small. In every
case the failure mode is a false alarm (a genuinely good bottle flagged
with at least one spurious defect), not a missed genuine defect — GOOD's
false-negative count (a good bottle incorrectly flagged) is non-zero for
all four models, while GOOD's false-positive count (a defective bottle
incorrectly called good) is 0 for both EfficientNet-B0 and ResNet18.

## Why false negatives matter more than false positives here

In a QC pass/fail context, a false positive (a good bottle flagged as
defective) causes an unnecessary rejection, which is a throughput/cost
problem. A false negative (a defective bottle passed as good) ships a
defect to the customer, which is a quality-escape problem — generally the
more serious failure mode for an inspection system. This is why zero
observed false-negative label instances for EfficientNet-B0 on the
evaluated test set is highlighted as a meaningful result, separately from
its macro-F1 score.

## `missing_cap`

Not a failure in the sense above — it is entirely unscored across all four
models because the dataset contains zero positive images for this class.
No conclusion, positive or negative, can be drawn about any model's ability
to detect a missing cap from the current dataset.
