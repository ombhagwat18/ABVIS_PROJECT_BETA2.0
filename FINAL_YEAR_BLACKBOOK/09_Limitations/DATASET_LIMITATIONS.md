# Dataset Limitations

These are documented as experimental limitations of the current dataset,
not as defects hidden from the reader. They directly bound how the Stage 1
results should be interpreted.

## 1. `missing_cap` — zero positive samples

The dataset contains **0 positive images** for `missing_cap` out of 1,143
reviewed images. Consequently this class is unlearnable under the current
dataset and was disabled (given an unreachable decision threshold) in every
one of the four Stage 1 training runs. It is reported as **unscored**, not
as a passing or failing result, in every evaluation in this documentation.
No conclusion about the system's ability to detect a missing cap can be
drawn until positive examples exist.

## 2. `missing_label` — thin class, no validation signal

`missing_label` has only **30 images across 2 distinct scenes** in the
entire dataset. Under the deterministic scene-based split, **0 of those
scenes land in the validation partition** — meaning every model's decision
threshold for this class was chosen with no positive validation example to
verify it against.

The consequence, observed directly: EfficientNet-B1, MobileNetV3-Small, and
ResNet18 each achieved **0/12 recall** for `missing_label` on the held-out
test set (every one of the 12 test positives was missed). EfficientNet-B0
detected all 12 test positives for this class.

**This should not be generalised as proof that EfficientNet-B0 has solved
the `missing_label` class.** With only 2 scenes total for this class, one
scene's worth of images was available for the held-out test set and none
for validation — the sample size is too small to support a durable claim
of reliability in either direction. More images and, specifically, more
distinct scenes of this class are required before this class can be
considered reliably evaluated. See `15_Future_Work/FUTURE_WORK.md`.

## 3. No genuine multi-defect images

**0 of the 1,143 reviewed images** carry more than one defect flag. The
classifier's architecture is multi-label (capable of predicting more than
one defect simultaneously), but this capability has not been experimentally
exercised or validated against any real multi-defect bottle, because no
such labelled example currently exists in the dataset. Any claim about the
system's behaviour on a bottle with two or more simultaneous defects would
be speculative, not evidenced.

## 4. Scene count and the held-out test set

The dataset's 1,143 images reduce to **112 distinct scenes** (near-duplicate
consecutive-frame groups of the same physical bottle pass). This is the
real, independent sample size behind every reported metric — treating 1,143
images as 1,143 independent observations would overstate the statistical
power of any result.

The held-out test set contains **238 images across 23 scenes**. Splitting
by scene, rather than by individual image, is essential here: because
frames within a scene are visually near-identical, a per-image random split
would place near-duplicate frames of the same bottle pass into both
training and test, letting a model score well on the test partition by
recognising a frame it had effectively already memorised in training.
Scene-based splitting keeps every frame of a given bottle pass entirely
within one partition, so a high test score reflects generalisation to an
unseen bottle pass, not recall of a training image's near-duplicate.

## 5. General sample-size caveat

With only 112 scenes total (70 training, 19 validation, 23 test), the
dataset is small by the standards of a production computer-vision system.
Metrics reported in this documentation (macro-F1, precision, recall,
exact-match accuracy) are exact for the images and thresholds evaluated,
but their generalisation to a larger, more varied population of bottles,
lighting conditions, and camera setups has not been established and would
require a larger and more diverse dataset to test.
