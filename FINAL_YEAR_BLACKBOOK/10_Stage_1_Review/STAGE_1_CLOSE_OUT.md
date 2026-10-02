# Stage 1 Close-Out Review

## Objective

Establish a defensible, evidence-based baseline classifier architecture
for the bottle defect detection system, by training and evaluating multiple
candidate backbone architectures under an identical protocol, and to
honestly document the limitations of the current dataset.

## Work completed

- Built and reconciled a multi-label labelled image dataset
  (`labels.csv`) of 1,143 reviewed images across 8 candidate defect
  classes.
- Established a deterministic, scene-based train/validation/test split
  (685/220/238 images; 70/19/23 scenes) that avoids leaking near-duplicate
  frames across partitions.
- Implemented and used a held-out test evaluation protocol
  (`train.evaluate_test()` / equivalent scoring against recorded
  `test_paths`) that scores a checkpoint against images it never trained
  or validated on, using thresholds selected only from that checkpoint's
  own validation split.
- Trained and evaluated four candidate backbone architectures under this
  identical protocol: EfficientNet-B0, EfficientNet-B1,
  MobileNetV3-Small, and ResNet18.
- Added an opt-in early-stopping capability to the training script
  (`--patience N`, default off, verified not to change the behaviour of
  the three earlier experiments) and used it for the ResNet18 experiment.
- Documented dataset limitations that bound how the results should be
  interpreted.

## Dataset

1,143 reviewed images, 112 distinct scenes, 8 candidate defect classes. See
`05_Dataset_and_Data_Preparation/DATASET_DOCUMENTATION.md`.

## Methodology

Multi-label image classification with per-class sigmoid outputs and
per-class thresholds selected on a validation split, never on the held-out
test set. See `06_AI_Model_Development/Classification_Methodology/AI_METHODOLOGY.md`.

## Models evaluated

EfficientNet-B0, EfficientNet-B1, MobileNetV3-Small, ResNet18.
ConvNeXt-Tiny was **not** evaluated in Stage 1.

## Results

See `08_Experimental_Results/RESULTS_SUMMARY.md` for the consolidated
table. Under the evaluated protocol, EfficientNet-B0 achieved the highest
test macro-F1 (0.935), the highest exact-match accuracy (89.1%), and the
only zero-false-negative result among the four models on the 238-image
held-out test set.

## Key findings

1. Validation macro-F1 of 1.000 (observed for both EfficientNet-B0 and
   EfficientNet-B1) did not guarantee held-out test performance —
   EfficientNet-B1's test macro-F1 dropped to 0.845, illustrating why a
   genuinely held-out test evaluation is necessary before trusting a
   validation number.
2. `missing_label`'s complete test-set recall failure on three of four
   models is traceable to a specific, identified dataset gap (2 scenes
   total, 0 in validation), not to a general weakness of those
   architectures.
3. Early stopping (patience = 5) on ResNet18 reached its best validation
   epoch at epoch 3 and correctly stopped by epoch 8, avoiding roughly 17
   unnecessary epochs of training time compared to a fixed 25-epoch
   schedule, while retaining the same best-epoch checkpoint-selection rule
   used by the other three experiments.
4. False negatives, not false positives, are the more consequential error
   type for a QC pass/fail decision (a missed defect ships to the
   customer); EfficientNet-B0 was the only architecture with zero observed
   false negatives on the held-out test set.

## Limitations

See `09_Limitations/DATASET_LIMITATIONS.md` in full:
`missing_cap` has zero positive samples; `missing_label` is thin (30
images, 2 scenes) with zero validation positives; the dataset contains no
genuine multi-defect images; the held-out test set represents only 23
scenes, and the whole dataset only 112.

## Current selected baseline

EfficientNet-B0, checkpoint `20260919-164511`, selected under the evaluated
experimental protocol on the basis of held-out test macro-F1, exact-match
accuracy, and zero observed false negatives.

## What was learned

Comparing more than one architecture under an identical protocol surfaced
a result that a single-model study would have missed: a model can reach a
perfect validation score and still generalise worse than a model with a
lower validation score, and the gap is not visible without a genuinely
held-out, never-touched test set. It also became clear that class-level
failures (such as `missing_label`'s collapse) are frequently a dataset
sampling problem, not purely a modelling problem — a finding that directly
shapes the dataset-improvement items in `15_Future_Work/FUTURE_WORK.md`.
Scoping the benchmark to four architectures, rather than an exhaustive
sweep, was judged sufficient to produce a defensible, evidence-based
baseline selection for a final-year project.

## Next stage

Stage 2 will focus on planning a universal annotation and object-detection
(YOLO) capability. No Stage 2 implementation work has begun as of this
document.
