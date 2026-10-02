# AI Methodology — Classification Approach

## Task formulation

The bottle defect classifier is formulated as **multi-label image
classification**: for each input image, the model outputs one independent
probability per defect class (via a sigmoid activation per class, not a
single softmax over mutually exclusive classes). A bottle can carry more
than one defect flag simultaneously (e.g. skewed *and* label-damaged) in
principle, even though, as noted in the dataset limitations, no such
genuinely multi-defect example currently exists in the dataset.

This is a whole-image classification task, not object detection or
segmentation: the model does not localise a defect within the image or
produce a bounding box or pixel mask. It answers "is defect X present in
this image", per class.

## Model architecture

- A convolutional backbone (ImageNet-pretrained), with the final
  classification layer replaced by a linear layer with one output per
  defect class.
- Loss function: `BCEWithLogitsLoss` (binary cross-entropy per class,
  suited to multi-label problems where classes are not mutually
  exclusive), with a per-class `pos_weight` derived from the training
  split's class counts to counteract class imbalance (thin classes such as
  `missing_label` are up-weighted relative to common classes such as
  `damaged_bottle`).
- Output: per-class sigmoid probability, compared against a per-class
  threshold (see "Threshold selection" below) to produce a binary
  present/absent decision per class.

## Why multi-label, not multi-class

The defect classes are not mutually exclusive by definition (a bottle
could, in principle, be both damaged and skewed). A multi-class (single
softmax) formulation would force the model to pick exactly one label per
image, which is not a faithful model of the underlying problem, even
though the current dataset happens to contain no images with more than one
label set (see `09_Limitations/DATASET_LIMITATIONS.md`).

## Data augmentation

Small rotation and scale jitter, and brightness/contrast jitter, are
applied during training. Horizontal flipping is deliberately **not**
applied, because flipping would mirror printed label text and invert the
physical direction of skew — corrupting the semantics of the
`skewed_label`, `skewed_bottle`, and `damaged_label` classes.

## Threshold selection methodology (Stage 1 protocol)

For each trained checkpoint:

1. A per-class decision threshold is selected on the **validation split
   only**, by sweeping candidate thresholds and choosing the one that
   maximises that class's F1 score on validation data.
2. A class with zero positive images anywhere in the dataset is given an
   unreachable threshold (above the maximum possible sigmoid output),
   disabling it outright rather than letting it fire on an unconstrained,
   never-validated logit.
3. A class with training images but zero validation positives is left
   live, using its default/fallback threshold, but is explicitly reported
   as "unscored" on validation rather than given a fabricated perfect or
   zero score.
4. The held-out test set is scored, separately and only once per
   checkpoint, using exactly the thresholds selected in step 1 — the test
   labels are never read or used to select or adjust any threshold. This
   is the protocol followed for every architecture evaluated in Stage 1.

## Why this matters for the reported numbers

Because thresholds are fixed before the test set is touched, the reported
test-set precision/recall/F1 in
`06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md` are a
genuine estimate of generalisation, not a best-case number obtained by
retroactively picking the threshold that flatters the test set.
