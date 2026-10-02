# Problem Statement and Objectives

## Problem statement

Manual visual inspection of bottles on a quality-control line for defects
such as damage, skew, tilted caps, incorrect fill level, and label defects
is labour-intensive and prone to inconsistency. An automated, camera-based
inspection system that classifies defects and issues a PASS/FAIL verdict
per bottle can support a QC line, provided the underlying classifier is
evaluated rigorously enough that its reported accuracy can be trusted on
images it has never seen.

## Objectives — Stage 1 (completed)

1. Build a labelled, multi-label image dataset of bottle images across
   eight candidate defect classes, organised by project, with a
   `labels.csv` file as the source of truth.
2. Establish a data split that avoids leaking near-duplicate frames of the
   same physical bottle between training, validation, and test — the
   dataset consists of consecutive video frames of a small number of
   physical bottle passes ("scenes"), so a naive random split would
   overstate accuracy.
3. Train a multi-label defect classifier and measure its performance
   honestly on a held-out test set that is never used for training,
   validation-time model selection, or threshold tuning.
4. Compare more than one candidate backbone architecture under an identical
   protocol (same split, same batch size, learning rate, and test fraction)
   to select a defensible baseline architecture, rather than accepting a
   single untested model.
5. Document, rather than hide, the limitations of the current dataset (thin
   classes, absent classes, no genuine multi-defect examples).

## Objectives — future stages (not yet started)

See `15_Future_Work/FUTURE_WORK.md`. These are not part of Stage 1 and are
listed here only to state the project's intended direction; none of them
have been implemented or evaluated.

## Success criteria for Stage 1

- A working, reproducible training and evaluation pipeline
  (`train.py`, `dataset.py`, `infer.py`).
- At least one classifier architecture evaluated on a genuinely held-out
  test set, with per-class precision/recall/F1, confusion counts, and
  inference speed reported.
- A comparison across multiple architectures under the same protocol,
  producing a documented, evidence-based selection of a baseline
  architecture.
- Explicit documentation of dataset limitations that affect the confidence
  of the reported results (see `09_Limitations/DATASET_LIMITATIONS.md`).

All of the above have been achieved and are documented in this workspace.
