# Model Architecture Benchmark — Stage 1

## Objective

Compare four candidate classifier backbone architectures under an
identical training protocol and an identical, deterministic dataset split,
to select a defensible baseline architecture for the bottle defect
classifier — rather than accepting the first architecture trained.

## Architectures evaluated

1. EfficientNet-B0 (checkpoint `20260919-164511`)
2. EfficientNet-B1 (checkpoint `20260919-222031`)
3. MobileNetV3-Small (checkpoint `20260920-102102`)
4. ResNet18 (checkpoint `20260920-110121`, trained with early stopping,
   patience = 5)

**ConvNeXt-Tiny was not evaluated in Stage 1.** It is not included in any
comparison table in this documentation and must not be cited as an
evaluated result.

## Protocol held constant across all four experiments

- Same dataset, same `labels.csv` state, same deterministic scene-based
  split (685/220/238 images, 70/19/23 scenes — confirmed byte-identical
  path lists across all four checkpoints).
- Same batch size (32), learning rate (3e-4), optimizer (AdamW), test
  fraction (0.15), split seed (0).
- Same held-out test set, touched only after training was complete, and
  only for final evaluation.
- Same threshold-selection rule: per-class thresholds chosen on each
  checkpoint's own validation split, never on the test set.

## Validation-stage results

| Model | Validation macro-F1 | Best epoch / epochs run | Training time |
|---|---|---|---|
| EfficientNet-B0 | 1.000 | 8 / 25 | 216.2 s |
| EfficientNet-B1 | 1.000 | 3 / 25 | 453.9 s |
| MobileNetV3-Small | 0.864 | 7 / 25 | 192.2 s |
| ResNet18 | 0.933 | 3 / 8 (early-stopped, patience 5) | 145.3 s |

A validation macro-F1 of 1.000 (B0, B1) reflects the small size of the
validation split (220 images / 19 scenes) and does not, by itself,
establish generalisation — this is precisely why a held-out test
evaluation was performed for every model before any conclusion was drawn.

## Held-out test results (summary; full detail in `Test_Evaluation/HELD_OUT_TEST_RESULTS.md`)

| Model | Test macro-F1 | Test macro-precision | Test macro-recall | Exact-match accuracy | Total FP | Total FN | Test latency / FPS | Model size |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 0.935 | 0.887 | 1.000 | 89.1% (212/238) | 28 | 0 | 13.89 ms / 72.0 | 15.57 MB |
| EfficientNet-B1 | 0.845 | 0.786 | 0.935 | 68.1% (162/238) | 74 | 10 | 17.71 ms / 56.5 | 25.24 MB |
| MobileNetV3-Small | 0.632 | 0.587 | 0.713 | 67.2% (160/238) | 82 | 36 | 14.04 ms / 71.2 | 5.93 MB |
| ResNet18 | 0.773 | 0.712 | 0.857 | 84.0% (200/238) | 38 | 12 | 16.71 ms / 59.8 | 42.72 MB |

All figures above were obtained using validation-derived thresholds; no
test label was used to select or adjust any threshold for any model.

## Observation

Under the evaluated experimental protocol, EfficientNet-B0 achieved the
highest test macro-F1 and the highest exact-match accuracy among the four
architectures, and was the only architecture for which no false-negative
label instance was observed on the 238-image held-out test set. EfficientNet-B1,
MobileNetV3-Small, and ResNet18 each showed a larger gap between their
validation and test performance than EfficientNet-B0 did, most visibly on
the `missing_label` class (see `Failure_Analysis/FAILURE_ANALYSIS.md`).

## Selection

EfficientNet-B0 (checkpoint `20260919-164511`) was selected as the current
baseline architecture for this classifier on the basis of the held-out test
results above. This is a selection under the evaluated experimental
protocol and dataset, not a claim of optimality across all possible
architectures or a claim that the model is free of failure modes — see
`Failure_Analysis/FAILURE_ANALYSIS.md` and `09_Limitations/DATASET_LIMITATIONS.md`.

## Not evaluated in Stage 1

ConvNeXt-Tiny was intentionally not trained or evaluated, as a scope
decision for a final-year project benchmark (four architectures were judged
sufficient to establish a defensible baseline; see
`10_Stage_1_Review/STAGE_1_CLOSE_OUT.md`, "What was learned"). It is listed
here only to record that it was considered and deliberately excluded, not
that it was run.
