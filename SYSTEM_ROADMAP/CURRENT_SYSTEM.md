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

optional, beside the classifier (Live tab selector: "Classifier" | "Classifier + YOLO" | "YOLO only"):
  frame (whole, uncropped) -> detect.YoloDetector -> DetectionResult (bottle/cap/label boxes)
       -> overlay + detector state + trace fields.   Never a PASS/REJECT.
```

**YOLO TRAINING = COMPLETE. YOLO RUNTIME = CURRENT DEVELOPMENT. PHYSICAL MACHINE = NOT TESTED.**
The detector is wired in as an opt-in, *observational* path ([IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md), Phase 1).
The default (`Classifier`) is exactly the previous behavior.

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
| YOLOv8n detector (`yolo_stage2_train.py`) | IMPLEMENTED (trained and test-evaluated) | See below |
| YOLO runtime (`detect.py` + `infer.Camera` hook + Live tab) | **CURRENT DEVELOPMENT**: IMPLEMENTED, TESTED SOFTWARE ONLY, HARDWARE UNVERIFIED | See "YOLO runtime" below |
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
bottle. The images are 1780x1000 crops of desktop screenshots of the Iriun Webcam viewer
(`stage2_dataset/prepare_stage2.py`), not direct camera captures, so behaviour on live frames is unverified.

### YOLO runtime (`detect.py`, `infer.py`, Live tab) -- current development

| Item | Status |
|---|---|
| Contract | `Detection(class_id, class_name, confidence, x1, y1, x2, y2)` and `DetectionResult(camera_id, frame_seq, frame_ts, detections, frame_wh, infer_ms, model_id, conf_threshold)`. **Coordinates: absolute pixels in the original (whole, uncropped) frame passed to the detector; origin top-left; (x1,y1) top-left, (x2,y2) bottom-right.** No ROI, no resize in the contract (`scale_box()` converts for drawing). The Stage 1 ROI crop is not used |
| Classes | exactly `0 bottle, 1 cap, 2 label`; a model whose class names differ is refused |
| Loading | `models/stage2_yolo/stage2_best.pt`, verified against the sha256 in `MODEL_PROVENANCE.json`. Missing or non-matching weights raise `DetectorError` (no download, no substitute). Ultralytics 8.1.0 |
| Failure handling | inference exception, non-finite/out-of-range confidence, bad class id, degenerate box, bad frame -> `DetectorError` -> detector state FAULT -> inspection FAULT. **Never PASS** |
| Empty result | valid: state `NO DETECTIONS`, not an error and **not a defect** |
| Staleness | detections older than `max_age` (1.0 s, monotonic), or a stale frame -> detector FAULT; stale boxes are never exposed as current |
| Verdict role | **observational.** Detections never produce PASS/REJECT, and a missing cap/label is *not* turned into REJECT (absence can be occlusion, angle, blur, lighting, a false negative or a detector failure). With a classifier attached, its PASS/REJECT stands unless the detector faults. "YOLO only" has no inspection rules, so it reports FAULT `detector-only test mode` by design |
| Confidence | `DEV_CONF = 0.25`, a **development threshold, not validated for production**. Configurable via `settings.json` (`detector_conf`), separate from the implementation |
| Multi-camera | one shared detector (calls serialised by a lock); each camera runs independently and keeps its own `camera_id`, session, frame sequence and timestamps. **No synchronisation between cameras** |
| Trace | `InspectionRecord` gained optional `detector_state`, `detector_model_id`, `detector_ms`, `detections` (backward compatible; older records still load) |

**Software tests (fakes, no hardware):** parsing, empty result, invalid confidence/coordinates/class id, class-name mapping,
coordinate clipping and scaling, missing and foreign weights, inference exception, thread-sharing, camera fault, restart,
stale detections, empty detections leaving a PASS untouched, detector fault forcing FAULT, and two cameras with one
detector. Mutation-checked (8 deliberate breakages, all caught). A sanity check with the **real checkpoint** on 15 Stage 2
validation frames gives same-class IoU>=0.5 precision 1.00 / recall 0.96 at conf 0.25 (a coordinate-system check, not an evaluation).

**Measured runtime (RTX 3050 Laptop GPU, development machine; measured, not the earlier offline figure):**

| Measurement | Result |
|---|---|
| `detect()` alone, 89 val frames (1780x1000) | mean 42.7 ms, p50 30.2, p95 128, max 250 ms (~23 FPS ceiling; earlier offline benchmark: ~29 ms) |
| Through the real `Camera` loop, fake 30 fps capture of val frames | ~10.5 frames/s captured and ~10.5 detected/s; detection age at read mean ~100 ms, p95 ~170-220 ms; GPU utilisation samples 14-47 %; process CPU ~43 %; 0 dropped |
| Live camera index 0 (640x480, driver defaults), detector on | camera delivered ~7.5-8.7 fps (camera/exposure-limited here); detector 23-31 ms when alone |
| Two live cameras, one shared detector | ~8.7 fps each; detector time 84 and 121 ms per call (includes waiting for the shared lock/GPU) |

Capture and inference run in the **same thread** per camera, so inference time lowers that camera's capture rate; the
`dropped` counter does not capture driver-buffer staleness. These numbers are for this machine only.

**Live camera test (manual, driver defaults, no setting changed):** index 0 (640x480) viewed a room with **no bottle** -
the detector drew low-confidence false `label` (0.27-0.36) and `bottle` (0.40) boxes on a TV, a door and a strip at conf 0.25;
index 1 is the Iriun virtual webcam ("Please start Iriun Webcam", black frame) - correctly no detections. Plumbing, FAULT
handling and timing were verified live; **detection accuracy on a real bottle was not**, because none was in view. In one run
a camera reopened right after a two-camera session delivered only 7 frames and stalled (the staleness logic reported FAULT
correctly and the process then hung at exit); I could not reproduce it in three plain Stop/Start cycles. Cause unknown.

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
| ISPSoft project (`plc file/final_year/`) | EXISTS; ladder is an encrypted binary, unreadable here | Contract (X0 sensor, X1/X2 start/stop, Y1 conveyor, Y0 reject, M2 trigger, M0 PASS, M1 REJECT, T0/T1) is **user-stated**, tested by the user in the simulator. C0/C1 role unknown |
| PLC communication layer (`plc/`) | IMPLEMENTED; FAKE-PLC tested; simulator reads/faults verified; simulator write handshake NOT yet run; HARDWARE UNVERIFIED | `PLCService` is the single owner of the link; only M0/M1 are writable, once per trigger, never retried; not integrated with the inspection pipeline or GUI |
| Mock PLC, serial transport for the real PLC, reject controller | NOT IMPLEMENTED | |

## 4. Known limitations (summary)

- No hardware validation of any kind; PLC addresses/I-O mapping not verified.
- YOLO runtime is opt-in and observational; it is unvalidated on live bottle frames (training images are Iriun-viewer screenshots, 16:9; development cameras default to 640x480, 4:3). The default live path is still the Stage 1 classifier.
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
