# Dataset Documentation

## Dataset summary

| Quantity | Value |
|---|---|
| Total reviewed images | 1,143 |
| Total distinct scenes | 112 |
| Number of defect classes | 8 |
| Reviewed vs. total | All 1,143 images used in Stage 1 have `reviewed=1`; unreviewed (inbox) images are excluded from training and evaluation by design. |

A "scene" is a group of near-duplicate consecutive video frames judged to
be the same physical bottle pass (detected by `dataset.scene_map()`, not
assigned manually). The dataset behaves as ~112 independent samples, not
1,143 — this is the ceiling on how much the reported metrics can
statistically mean, and is the reason splitting is done by scene rather
than by image (see "Why scene-based splitting" below).

## Class distribution (positive image counts, out of 1,143 reviewed images)

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

"GOOD" is not a stored column — it is the implicit condition where every
defect column is 0 for a given image. Per-class scene counts and a full
breakdown by train/validation/test partition are marked below as **"To be
populated from dataset audit"** where not already computed for Stage 1.

| Item | Status |
|---|---|
| Per-class scene counts (all classes) | To be populated from dataset audit |
| Per-class positive counts broken down by train/val/test | To be populated from dataset audit (only `missing_label`'s val/test breakdown was computed for Stage 1 — see below) |

## Deterministic scene-based split (confirmed, used identically across all four Stage 1 architecture experiments)

| Split | Images | Scenes |
|---|---|---|
| Training | 685 | 70 |
| Validation | 220 | 19 |
| Held-out test | 238 | 23 |
| **Total** | **1,143** | **112** |

Split seed: 0 (deterministic — confirmed identical `val_paths`/`test_paths`
lists across all four architecture checkpoints trained in Stage 1). Test
fraction: 0.15 (of scenes, before the train/validation split is taken from
the remainder).

## Why scene-based splitting is used

The dataset consists of consecutive video frames of a small number of
physical bottle passes; frames within one pass are visually near-identical.
A random per-image split would place near-duplicate frames of the *same*
bottle pass into both the training and test partitions. A model could then
score well on the test partition simply by having memorised the training
partition's near-duplicate frames, rather than by generalising to a new
bottle. Splitting by detected scene, so that every frame of a given bottle
pass falls entirely into one partition, removes this leakage path and is
the reason the reported held-out test results are treated as a meaningful
estimate of generalisation rather than of memorisation.

## Held-out test set

238 images across 23 scenes, held out before training or validation ever
began, and not touched for model selection, epoch selection, or threshold
selection at any point in the Stage 1 benchmark. It was evaluated exactly
once per checkpoint, using thresholds derived only from that checkpoint's
own validation split.

## Known dataset limitations

Documented in full in `09_Limitations/DATASET_LIMITATIONS.md`; summarised
here:

- `missing_cap` has zero positive images in the dataset — unlearnable and
  unscored in every Stage 1 experiment.
- `missing_label` has only 30 images across 2 distinct scenes, and 0
  validation positives in the current deterministic split.
- 0 of the 1,143 reviewed images contain more than one defect flag — the
  dataset currently contains no genuine multi-defect examples, even though
  the classifier architecture supports multi-label prediction.
