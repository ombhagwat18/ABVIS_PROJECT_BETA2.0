# Current System

What actually exists in this repository today -- nothing more. For what is *planned*, see
[FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) and [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

**Status vocabulary** used throughout this document:

| Label | Meaning |
|---|---|
| IMPLEMENTED | Code exists, is integrated where relevant, and has been exercised |
| PARTIAL | Some of it exists; the gaps are listed |
| TESTED SOFTWARE ONLY | A software self-test passes (fake cameras / fake models). **Not** a hardware test |
| HARDWARE UNVERIFIED | Software exists but has never been run against the physical machine |
| NOT IMPLEMENTED | Does not exist |

> **Nothing in this repository has been tested on the physical machine.** Every "tested" below means a
> software self-test.

---

## 1. What the application is

A Python desktop application (Tk / CustomTkinter, OpenCV, PyTorch, Ultralytics for offline YOLO work) for
the Bottle Defect Detection System. First intended application: **250 ml Bisleri bottle inspection on a
conveyor**. Entry point: `python gui.py` (or `run.bat` on Windows).

Tested environment: Windows 11, Python 3.8.0, torch 2.4.1+cu124, ultralytics 8.1.0, NVIDIA RTX 3050 Laptop
GPU (4 GB VRAM), 16 GB RAM.

## 2. Current pipeline (what runs in the live tab)

```
Camera thread (OpenCV, DirectShow, driver-default settings)
  -> Frame (camera_id, monotonic timestamp, per-session sequence number)
  -> crop to calibrated ROI -> letterbox -> centre-crop        (dataset.model_input)
  -> Stage 1 multi-label classifier (EfficientNet-style checkpoint, ~15 Hz cap)
  -> per-defect threshold -> PASS / REJECT / FAULT (per frame)
  -> CameraSet.combined()  (FAULT > REJECT > PASS across cameras)
  -> verdict label in the GUI
```

**The YOLO detector is not in this pipeline.** It is trained and evaluated offline; wiring it into the
runtime is the next development task ([IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), Phase 1).

## 3. Component status

### Application and data

| Component | Status | Notes |
|---|---|---|
| Desktop GUI (`gui.py`, 9 tabs) | IMPLEMENTED, TESTED SOFTWARE ONLY | `python gui.py --selftest` builds every tab against the real dataset with no device I/O |
| Multi-project structure (`projects/<slug>/`) | IMPLEMENTED | `dataset.py` rebinds module-level paths; labels in `labels.csv`, config in `config.json` |
| Stage 1 dataset (`projects/om_bottle`) | IMPLEMENTED | 1,143 images, all reviewed, 8 defect columns, only 72 "good" images. Scene-based split. **No held-out test split** |
| Stage 1 classifier training (`train.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | 5 backbones, 9 checkpoints, active: `efficientnet_b0` `20260919-164511` |
| Annotation Studio (`annotation_studio.py`, `annotate.py`) | IMPLEMENTED | Boxes and polygons, review flag, "next unannotated", YOLO detection and segmentation export. Segmentation export has never been used on real data (0 polygons) |
| Camera benchmark tab (`bench.py`) | IMPLEMENTED | Measures achieved FPS / latency / sharpness per mode. **The result is not applied to the live camera** |

### Stage 2 detection (offline)

| Component | Status | Notes |
|---|---|---|
| Stage 2 dataset (`stage2_dataset/`) | IMPLEMENTED | 594 images, 39 scenes: **418 train (25 scenes) / 89 val (7) / 87 test (7)**. 1,994 boxes: bottle 739, cap 709, label 546. All reviewed; scene-leakage check passed |
| YOLO export + validation | IMPLEMENTED | `export_yolo.py`, `validate_yolo_export.py`: 19/19 checks pass |
| YOLOv8n detector (`yolo_stage2_train.py`) | IMPLEMENTED (trained and test-evaluated); **not in the runtime** | See below |
| YOLOv8s | NOT IMPLEMENTED | Requested as a second candidate; skipped by the memory gate; not trained |

**YOLOv8n results** (`models/stage2_yolo/MODEL_PROVENANCE.json`, `training_metadata.json`):

| | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Validation (89 images) | 0.976 | 0.730 | 0.998 | 0.965 |
| **Held-out test (87 images, 7 scenes)** | **0.968** | **0.660** | 0.936 | 0.968 |

Per-class test mAP50-95: bottle 0.816, cap 0.593, label 0.571.
Trained 53 epochs (best epoch 33, early stopping), 640 px, batch 8, seed 0, 6.2 MB model.
Benchmark on the development laptop, batch 1: ~29 ms end to end on GPU (~34 FPS), ~248 ms on CPU (~4 FPS).

**Limitation -- read this before quoting the numbers:** the test set is only 7 scenes / 87 images / 396 boxes,
so the test metrics are indicative, not a production accuracy guarantee. Cap and label are clearly weaker than
bottle. Whether the images match the production camera and lighting is not recorded in the repository.

### Inspection software (`infer.py`, `inspection_trace.py`)

| Component | Status | Notes |
|---|---|---|
| Camera abstraction (`Camera`, `CameraSet`) | IMPLEMENTED, HARDWARE UNVERIFIED | Discovery by index probe, DirectShow, multiple cameras, newest-frame-wins, drop counting. No exposure/gain/focus control, no reconnect, no trigger |
| PASS / REJECT / FAULT | IMPLEMENTED, TESTED SOFTWARE ONLY | PASS and REJECT only come from a fresh valid score. FAULT for: not started, thread dead, driver error, no model, failed inference, NaN scores, nothing scored yet, stale frame, stale score (>1.0 s). FAULT clears itself on the next good frame (no latching) |
| Frame / session metadata | IMPLEMENTED, TESTED SOFTWARE ONLY | `camera_id`, `session`, per-session `seq`, `time.monotonic()` stamps, `latest_frame()`, `inspection()`. A restarted camera starts a new session; a stale thread cannot write into it |
| Inspection record (`InspectionRecord`, `TraceStore`) | PARTIAL, TESTED SOFTWARE ONLY | In-memory only. See [TRACEABILITY_PLAN.md](TRACEABILITY_PLAN.md) |
| Decision logic | PARTIAL | Per-frame, per-defect thresholds only. No rules, no temporal voting, no per-bottle decision |
| Diagnostics | IMPLEMENTED | FPS, read/inference/latency ms, dropped %, CPU/RAM/GPU (psutil/pynvml optional) |

### Machine side

| Component | Status | Notes |
|---|---|---|
| Sensor trigger, bottle tracking, inspection window | NOT IMPLEMENTED | |
| Timing model (distances, speed, actuator response) | NOT IMPLEMENTED | No physical values are recorded anywhere |
| PLC simulator prototype (`plc file/delta_sim_test.py`) | IMPLEMENTED as a standalone script; **not integrated; HARDWARE UNVERIFIED** | Manual keypad sender over Modbus ASCII to a local simulator. Its address map is **unverified** |
| ISPSoft project (`plc file/final_year/`) | EXISTS; contents **UNKNOWN** | The `.isp` is a proprietary binary that cannot be read here. Set to the ISPSoft simulation driver |
| Mock PLC, real PLC communication, reject controller | NOT IMPLEMENTED | |

## 4. Known limitations (summary)

- No hardware validation of any kind; PLC addresses/I-O mapping not verified.
- YOLO detector not integrated into the runtime; live inference is the Stage 1 classifier.
- Stage 1 validation scores are saturated (several checkpoints report macro-F1 1.0) and there is no test
  split, so they are not evidence of production accuracy. `missing_cap` has **zero** positive examples and is
  disabled; `missing_label` has 30 and scored 0 in the active model's validation.
- Hand-tuned live thresholds in `projects/om_bottle/config.json` (some at 0.05-0.1) are unvalidated.
- `time.monotonic()` on Windows ticks every ~15.6 ms; frame `seq` is the strict order.
- No per-bottle logic: at ~15 Hz a single physical bottle yields many independent verdicts.
- `gui.py` is large (~2.2k lines); `app.py` + `index.html` are an unused earlier web version.

## 5. Repository contents that are *not* in Git

Images (Stage 1 and Stage 2), the generated YOLO export, model weights (`*.pt`), logs and PLC editor
autosaves. The Stage 2 provenance (scripts, annotations, split, manifests) **is** tracked, and the weights
are described by checksum in `models/stage2_yolo/MODEL_PROVENANCE.json`.
