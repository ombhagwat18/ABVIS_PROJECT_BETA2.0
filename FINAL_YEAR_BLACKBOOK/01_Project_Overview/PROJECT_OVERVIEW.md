# Project Overview

## Title

Bottle Defect Detection System — a machine-vision quality-control
inspection application, developed as the first application of a longer-term
industrial AI vision inspection platform.

## Degree program context

B.Tech, Robotics & Automation Engineering — final-year project.

## What has been built (completed work)

A desktop application (`gui.py`, CustomTkinter) that:

- labels bottle images against a configurable, per-project set of defect
  classes (multi-label: a bottle can have more than one defect flag set),
- manages a project-based dataset layout (`projects/<slug>/...`) with a
  `labels.csv` file as the single source of truth for labels,
- trains a convolutional neural network classifier (EfficientNet-B0,
  ImageNet-pretrained backbone, multi-label sigmoid head) to detect eight
  candidate defect classes on bottles,
- performs live, multi-camera inspection with a PASS/FAIL verdict derived
  from per-defect probability thresholds,
- provides analysis tooling (per-defect precision/recall/F1, confusion
  matrices, checkpoint comparison) and camera benchmarking tools.

This is implemented as a whole-image multi-label classifier, not an object
detector — no bounding boxes or per-region localisation exist in the current
system. This is stated explicitly here because the long-term platform
direction (see `15_Future_Work/FUTURE_WORK.md`) includes object detection
and segmentation, which are not yet implemented.

## Long-term platform direction (not yet implemented)

The project is intended to evolve, in later stages, into a universal
industrial AI vision inspection platform supporting classification, object
detection, segmentation, annotation tooling, model training, a model
registry, camera/image acquisition, ROI/calibration, a decision engine, PLC
communication, conveyor control, pneumatic rejection, production
monitoring, inspection history/traceability, and industrial diagnostics.

**None of the items in the paragraph above are implemented as of Stage 1.**
Only the bottle classification pipeline described in "What has been built"
above exists and has been evaluated. The long-term direction is documented
in `15_Future_Work/FUTURE_WORK.md` as future work, not as completed
functionality.

## Stage 1 scope (this documentation package)

Stage 1 covers the classifier architecture benchmark: training and
evaluating four candidate backbone architectures
(EfficientNet-B0, EfficientNet-B1, MobileNetV3-Small, ResNet18) on the same
deterministic dataset split, and selecting a baseline architecture for the
bottle defect classifier based on held-out test performance. Full detail is
in `06_AI_Model_Development/` and `10_Stage_1_Review/STAGE_1_CLOSE_OUT.md`.

## Current selected baseline

EfficientNet-B0 (checkpoint `20260919-164511`) is the architecture selected
as the current baseline after the Stage 1 benchmark, on the basis of its
held-out test macro-F1 (0.935) and the absence of any false-negative label
instances on the 238-image held-out test set. See
`06_AI_Model_Development/Architecture_Benchmark/MODEL_ARCHITECTURE_BENCHMARK.md`
and `06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md` for
the full evaluation this selection is based on.
