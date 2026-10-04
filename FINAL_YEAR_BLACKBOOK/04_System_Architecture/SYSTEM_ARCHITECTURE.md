# System Architecture (Current Implementation)

This describes the application as it currently exists in the codebase. It
does not describe or claim any of the future platform capabilities listed
in `15_Future_Work/FUTURE_WORK.md`.

## Component overview

| Module | Role |
|---|---|
| `gui.py` | Desktop application (CustomTkinter). Tabs: Label, Defects, Train, Analysis, Live, Camera, Data health, Settings. |
| `dataset.py` | Central module holding project paths, `labels.csv` read/write, the crop/resize pipeline, and the scene-based train/validation/test split logic. |
| `train.py` | Trains a multi-label classifier (EfficientNet-B0 by default; other backbones selectable — see Stage 1 benchmark) and evaluates it against validation and held-out test sets. |
| `infer.py` | Loads a trained checkpoint and runs inference — both a `Model` class (single checkpoint) and a `CameraSet` class (multi-camera live inspection with a combined PASS/FAIL rule). |
| `calibrate.py` | Measures the bottle region-of-interest (ROI) once per project from sample frames. |
| `charts.py` | Draws line/bar/confusion charts directly on a Tkinter canvas (no external plotting library dependency in the running application). |
| `bench.py` | Benchmarks camera resolution/FPS/latency/sharpness combinations. |
| `migrate.py` | One-shot migration tool from an older single-project dataset layout. |
| `app.py` / `index.html` | An earlier browser-based (FastAPI) version of the application. Confirmed dead code — not imported by, or reachable from, the current desktop application. |

## Data flow (classification pipeline, as evaluated in Stage 1)

1. Images are labelled into `projects/<slug>/images/{+ve, -ve/<Class>,
   _inbox}/` and reconciled into `labels.csv` (one row per image, one
   column per defect class, multi-label).
2. `dataset.py` groups images into "scenes" (near-duplicate consecutive
   video frames of the same physical bottle) and splits scenes — not
   individual images, and not at random — into training, validation, and a
   genuinely held-out test partition.
3. `train.py` crops each image to the calibrated ROI, resizes/letterboxes
   it to a fixed input size, and trains a convolutional backbone with a
   per-class sigmoid output and per-class threshold.
4. Evaluation against the validation set drives model-selection (best
   epoch) and per-class threshold selection. Evaluation against the
   held-out test set (`train.evaluate_test()` / equivalent scoring
   against the checkpoint's own recorded `test_paths`) is performed
   separately, and only after training is complete, using the
   validation-derived thresholds — never thresholds fit on the test set.
5. `infer.py` applies a trained checkpoint's own baked-in ROI, input size,
   and per-class thresholds to score new images (live camera frames or,
   for Stage 1's benchmark, static held-out test images), producing a
   PASS/FAIL verdict per bottle.

## Crop / resize pipeline

A fixed, non-square (tall) input resolution is used because the physical
ROI on the bottle is tall and narrow; the same crop/resize code path is
shared by training and inference specifically to avoid a train/serve
mismatch (a previously identified and fixed defect in the codebase — see
`docs/design/PLAN.md` in the main project for the historical detail). This is stated
here because it is relevant to why the reported test metrics are
considered representative of what the live inference path would see on
the same images.

## What Stage 1 exercised, and what it did not

Stage 1 (this documentation package) exercised steps 2–4 above across four
candidate backbone architectures, using the existing static labelled image
dataset. It did not exercise live camera capture, the multi-camera fusion
rule in `infer.CameraSet.combined()`, PLC/conveyor/pneumatic integration
(none of which exists yet), or object detection/segmentation (also not yet
implemented).
