# Progress

Against `Bottle_Defect_Detection_Web_Dashboard_Architecture.docx`.
Updated 2026-08-23. ✅ done · 🟡 partial · ❌ not started

| | ✅ | 🟡 | ❌ |
|---|---|---|---|
| Stage 1 — Universal AI Module | 37 | 4 | 18 |
| Stage 2 — Machine Integration | 2 | 1 | 12 |

**Where it stands.** Dataset management is finished and multi-project:
`projects/om_bottle/` holds 1145 images (72 good, 1073 defective) across 7
classes with 8 checkpoints, and a project is created, uploaded to, trained and
run without touching code. Training, analysis, multi-camera live inspection,
camera benchmarking and real-time performance monitoring all work. Nothing has
run against the real camera on the line yet, and Stage 2 has not been started.

The remaining ❌ rows are three groups: things gated on the YOLO decision
below, things that need the Delta PLC physically present, and cosmetics
(rename/delete/export a model).

## Stage 1 — Universal AI Module

| § | Feature | Status | Note |
|---|---|---|---|
| 2.1 | Upload images | ✅ | per class, files or folder |
| 2.1 | Delete / manage images | ✅ | |
| 2.1 | Preview images | ✅ | |
| 2.1 | Dataset statistics | ✅ | |
| 2.1 | Class distribution | ✅ | |
| 2.1 | Train / validation split | ✅ | split by scene, not random |
| 2.1 | Test split | ❌ | 112 scenes too few, revisit at ~300 |
| 2.1 | Dataset versions | 🟡 | labels.csv snapshot per model + git, no UI |
| - | Appearance settings (font/control size) | ✅ | Settings tab, white/blue theme |
| 2.2 | Model selection YOLOv8n/s/m | ❌ | currently EfficientNet-B0 classifier |
| 2.2 | Epochs | ✅ | |
| 2.2 | Batch size | ❌ | hardcoded |
| 2.2 | Image resolution | 🟡 | per project in config.json, not in UI |
| 2.2 | Learning rate | ❌ | hardcoded |
| 2.2 | Optimizer | ❌ | hardcoded |
| 2.2 | Augmentation settings | ❌ | hardcoded |
| 2.2 | best.pt | ✅ | best-F1 checkpoint |
| 2.2 | last.pt | ❌ | |
| 2.2 | Confusion matrix | ✅ | per defect (multi-label has no single N×N) |
| 2.2 | Loss / precision / recall graphs | ✅ | Analysis tab, from recorded per-epoch history |
| 2.2 | F1-score | ✅ | per defect + macro |
| 2.2 | Training logs | ✅ | streams live |
| 2.3 | Store multiple models | ✅ | |
| 2.3 | Activate model | ✅ | |
| 2.3 | Deactivate model | ❌ | |
| 2.3 | Rename model | ❌ | |
| 2.3 | Delete model | ❌ | |
| 2.3 | Export / download model | ❌ | |
| 2.3 | Compare models | ✅ | re-scores each on its own held-out set |
| 2.3 | Select model for live | ✅ | |
| 2.3 | Separate cap / label / bottle models | ❌ | one combined multi-label model |
| 2.4 | Training mode | ✅ | |
| 2.4 | Testing on video files | ✅ | |
| 2.4 | Testing on saved images / datasets | ✅ | re-test any checkpoint from Analysis |
| 2.4 | Live camera feed | ✅ | |
| 2.4 | Multi-camera live inspection | ✅ | N cameras at once; any camera rejecting rejects the bottle |
| 2.4 | Bounding boxes / segmentation | ❌ | classifier has no boxes |
| 2.4 | Confidence score | ✅ | |
| 2.4 | PASS / FAIL decision | ✅ | |
| 2.4 | PLC status display | ❌ | |
| 2.4 | Bottle count / reject count | ❌ | |
| 2.5 | Precision | ✅ | |
| 2.5 | Recall | ✅ | |
| 2.5 | F1-score | ✅ | |
| 2.5 | mAP@50, mAP@50–95 | ❌ | detection-only metric |
| 2.5 | Average inference time | ✅ | 9.8 ms measured |
| 2.5 | FPS | ✅ | |
| 2.5 | CPU / GPU / memory usage | ✅ | psutil + torch.cuda; GPU %/temp need pynvml |
| 2.5 | False positives / negatives | ✅ | "show me the mistakes" |
| 2.6 | Current FPS | ✅ | |
| 2.6 | Processing latency | ✅ | per camera, live |
| 2.6 | Frame drops | ✅ | frames arriving while inference was busy |
| 2.6 | Bottle throughput | ❌ | needs the sensor trigger |
| 2.6 | PLC communication delay | ❌ | needs the PLC |
| 2.6 | Camera connection status | ✅ | live, per camera |
| 2.6 | CPU / GPU utilisation | ✅ | in the Live monitoring strip |
| 2.6 | Temperature monitoring | 🟡 | GPU only, and only with pynvml |
| 2.7 | Camera benchmarking | ✅ | Camera tab: FPS, latency, jitter, sharpness, brightness, clipping, CSV export |
| 2.8 | Model benchmarking table | 🟡 | our models compared; no YOLOv8n/s/m row |

## Stage 2 — Dynamic Machine Integration

| § | Feature | Status | Note |
|---|---|---|---|
| 3.1 | Conveyor calibration | ❌ | |
| 3.1 | Camera calibration (distortion, intrinsics, checkerboard) | ❌ | `calibrate.py` measures crop ROI only — different job |
| 3.1 | Lighting calibration | ❌ | |
| 3.2 | Detection controller (sensor → trigger → decide → reject) | ❌ | |
| 3.3 | Controller dashboard (conveyor / camera / PLC / AI status) | ❌ | |
| 3.4 | Confidence threshold | ✅ | per defect, tuned and adjustable |
| 3.4 | NMS threshold | ❌ | detection-only |
| 3.4 | Min / max bottle size | ❌ | detection-only |
| 3.4 | ROI | ✅ | measured or hand-set, per project |
| 3.4 | Reference line, reject delay, reject distance | ❌ | |
| 3.5 | PLC integration (Delta DVP, Modbus) | ❌ | |
| 3.6 | Live production dashboard (counts, cycle time, trends) | ❌ | |
| 3.7 | Database / inspection log | 🟡 | `rejects.csv` exists in the unused web version only |
| 3.7 | Reports by hour / shift / day / week / month | ❌ | |
| 3.7 | Export CSV / PDF / Excel | ❌ | |

## Also built, not in the doc

| Feature | Status |
|---|---|
| Multi-project (create, switch, isolated data + models) | ✅ |
| +ve / -ve folder tree → labels.csv automatically | ✅ |
| Multi-label (one bottle, several defects) | ✅ |
| Near-duplicate / scene detection | ✅ |
| Crop preview | ✅ |
| Snapshot from live into the dataset | ✅ |

## Two open decisions

1. **Web or desktop?** The doc says web dashboard. Current app is desktop (`gui.py`); `app.py` + `index.html` are ~600 lines behind and unused. Pick one and delete the other.
2. **YOLO or classifier?** The doc says YOLOv8 with boxes and mAP. Current app classifies whole images. Switching means drawing a box on every defect in all 1145 images first — that annotation does not exist, and neither does the tool to make it. Worth it only if a frame can hold more than one bottle, or you need to know *where* on the bottle the defect is.

Everything marked ❌ in 2.2–2.5 that mentions YOLO, mAP, NMS or boxes depends on decision 2.

## Next, in order

| # | Do | Why |
|---|---|---|
| 1 | Settle the two decisions above | Resolves ~5 ❌ rows by decision, not by work |
| 2 | Grow the good set to 300+ distinct bottles | 6% pass rate is backwards; the model has learned a world where defects are normal |
| 3 | Run it against the real camera | Every number so far comes from files; the capture path has never seen a device |
| 4 | Fix the `tilt_cap` labels, retrain | 19 Water Level frames are also tilt_cap — cheapest accuracy left |
| 5 | Benchmark the EMEET, pick a configuration | Camera tab is built; the numbers still have to be measured on the real rig |

Not yet: Stage 2 needs the Delta PLC in hand. Model rename/delete/export is
cosmetic.

**Note on model comparison.** The 8 existing checkpoints predate
recording which images each was validated on, so they cannot be re-scored
honestly — today's split would hand each model its own training images and
report a perfect 1.000 for every defect. They show “needs retrain” in the
comparison table. Anything trained from now on records it and compares properly.
