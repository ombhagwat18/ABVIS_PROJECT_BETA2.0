# Current System

What actually exists in this repository today -- nothing more. For what is *planned*, see
[FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) and [PROGRESS_PLAN.md](PROGRESS_PLAN.md).

**Status vocabulary** used throughout this document:

| Label | Meaning |
|---|---|
| IMPLEMENTED | Code exists, is integrated where relevant, and has been exercised |
| PARTIAL | Some of it exists; the gaps are listed |
| TESTED SOFTWARE ONLY | A software self-test passes (fake cameras / fake models). **Not** a hardware test |
| HARDWARE UNVERIFIED | Software exists but has never been run against the physical machine |
| NOT IMPLEMENTED | Does not exist |

> **The full inspect -> M0/M1 -> reject cycle has never run on the physical machine.** The only physical
> contact so far is a first read-only serial link to the real Delta PLC (2026-10-04, USER-STATED; see CLAUDE.md).
> Every other "tested" below means a software self-test with fakes.
>
> Last reconciled against the code: **2026-10-06** (see `docs/audit/GAP_MATRIX_2026-10-05.md`). Older statements
> that said there is no per-bottle logic, no trigger/timing model, or that the PLC layer is not integrated
> were stale and have been corrected.

---

## 1. What the application is

A Python desktop application (Tk / CustomTkinter, OpenCV, PyTorch, Ultralytics for offline YOLO work) for
the Bottle Defect Detection System. First intended application: **250 ml Bisleri bottle inspection on a
conveyor**. Entry point: `python gui.py` (or `run.bat` on Windows).

Tested environment: Windows 11, Python 3.8.0, torch 2.4.1+cu124, ultralytics 8.1.0, NVIDIA RTX 3050 Laptop
GPU (4 GB VRAM), 16 GB RAM.

## 2a. Production pipeline (Production page -> `machine_cycle.py`)

```
X0 photo-eye -> ladder SET M2 -> PLCService trigger -> Bottle inspection_id (FIFO, time-stamped)
  -> each line camera collects its frames in ITS window: trigger + offset_mm / speed   (tracking.py)
     (non-blocking; bounded by the next bottle's window; CAMERA_ASSOCIATION_FAULT if not separable)
  -> AI stages per camera (classification / detection / segmentation)  -> decision.decide()
     (recipe, per-camera roles, frame vote, camera fusion FAULT > REJECT > PASS)
  -> PASS: M0 at once; REJECT: M1 at trigger + travel - T0; late REJECT never fired (FAULT)
  -> one final result per bottle -> production.db (SQLite) + evidence image + daily CSV; coded alarms
```

One machine state (`machine_state.py`) drives the Production banner and the status-bar LINE lamp.

## 2b. Live-tab pipeline (development view)

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
The detector is wired in as an opt-in, *observational* path ([PROGRESS_PLAN.md](PROGRESS_PLAN.md)).
The default (`Classifier`) is exactly the previous behavior.

## 3. Component status

### Application and data

| Component | Status | Notes |
|---|---|---|
| Desktop GUI (`gui.py` + `hmi.py`, 15 pages) | IMPLEMENTED, TESTED SOFTWARE ONLY | Light industrial theme (dark optional), OPERATOR mode (Production / History / Database / Health) and ENGINEER mode (all pages); every page scrolls both ways; screens say GOOD / DEFECT: <name>. `python gui.py --selftest` builds every page, runs the line against a fake PLC, writes its production record to a temp folder |
| Multi-project structure (`projects/<slug>/`) | IMPLEMENTED | `dataset.py` rebinds module-level paths; labels in `labels.csv`, config in `config.json` |
| Stage 1 dataset (`projects/om_bottle`) | IMPLEMENTED | 1,145 images (2 new skewed images imported 2026-10-05), all reviewed, 8 defect columns. Scene-based train/val/test split. **`missing_cap` = 0 positives**: the 12 user missing-cap images were imported, trained on, shown to teach a background shortcut, and removed again (undo batch `20261005-172406-105`) |
| Stage 1 classifier training (`train.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | 12 checkpoints on disk (`20261005-171833` REJECTED: white-background shortcut, see `shortcut_check.json`) (6 with held-out test results), active: `efficientnet_b0` `20260919-164511`. **Training no longer activates the new checkpoint** (`activate=False` default); it becomes a CANDIDATE |
| Model registry (`model_registry.py`, Models page) | IMPLEMENTED, TESTED SOFTWARE ONLY | CANDIDATE -> VALIDATED (held-out test + real-camera note) -> APPROVED -> ACTIVE, ARCHIVED / REJECTED, `models/deployments.jsonl`, rollback, sha-checked detector activation, critical-class (`missing_cap`) warning |
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
| Decision logic (`decision.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | Per bottle: recipe-driven detection rules, classification thresholds, segmentation rules, frame vote (majority/any/all), camera fusion FAULT > REJECT > PASS, per-camera roles (`judge`, `station_x`). Thresholds are development defaults |
| Alarms (`alarms.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | 25 coded alarms with severity, operator message, action; condition vs event; acknowledge via RESET FAULT; persisted |
| Production record (`production_store.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | SQLite: runs (job, product, recipe hash, model ids, mode), one row per bottle, alarms; evidence images by policy; History page |
| Diagnostics | IMPLEMENTED | FPS, read/inference/latency ms, dropped %, CPU/RAM/GPU (psutil/pynvml optional) |

### Machine side

| Component | Status | Notes |
|---|---|---|
| Sensor trigger, per-bottle inspection (`machine_cycle.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY (FAKE ladder) | X0 -> M2 trigger = one inspection_id; frames strictly after the trigger; X0 without trigger -> NOT INSPECTED |
| Time-based tracking (`tracking.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | No encoder: `TimePositionSource` (measured speed + tolerance -> uncertainty); `EncoderPositionSource` placeholder refuses. Staggered camera stations, association fault. **No distance or speed measured on the machine yet** |
| Speed calibration / line layout dialogs | IMPLEMENTED, TESTED SOFTWARE ONLY (logic self-tested) | Never used on the belt |
| Machine state (`machine_state.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY | OFFLINE / NOT_READY / READY / INITIALIZING / RUNNING / INSPECTING / STOPPING / FAULT / E_STOP / COMMUNICATION_FAULT + start checklist |
| ISPSoft project (`plc file/final_year/`) | DECODED (not encrypted) | 09:06 save: 7 nets, T0 K150 / T1 K50 vs stated K15 / K5. See PLC_COMMUNICATION.md section 0 |
| PLC communication layer (`plc/`) | IMPLEMENTED, integrated (Machine + Production); FAKE + SIMULATOR tested; **real PLC: read-only link only** | `PLCService` single owner; only M0/M1 writable (+ opt-in operator test bits); serial transport exists; physical write handshake not yet run |
| Mock PLC (`plc.test_simulation.FakePLC` / `FakeLadder`) | IMPLEMENTED | Scan emulation of the decoded ladder; used by every machine self-test |

### Data finding: missing cap (2026-10-05)

All 22 user missing-cap images are already in Stage 2 (10 neck close-ups = train scene 38; 12 full-bottle = test
scene 22). The current detector misses all 12 full-bottle ones (tamper ring read as a cap at ~0.70). The v3 split
moves scene 22 into train (`stage2_dataset/split_v3.py`). The v3 YOLOv8n candidate (`models/stage2_yolo/candidate_yolov8n_v3.json`): val missing cap 5/6, good 54/54 (val) and 50/50 (test), and it no longer reads the bare neck as a cap. It is a CANDIDATE, unvalidated on the machine.

### Logs, recipe, shifts, ladder check (2026-10-05)

| Component | Status |
|---|---|
| Structured logs (`applog.py`, `logs/*.log`, Health page viewer) | IMPLEMENTED, TESTED SOFTWARE ONLY |
| Recipe editor (`hmi.RecipeDialog`) | IMPLEMENTED, TESTED SOFTWARE ONLY |
| Shift reports (History) | IMPLEMENTED, TESTED SOFTWARE ONLY |
| Camera auto-reconnect while running | IMPLEMENTED, TESTED SOFTWARE ONLY (fake camera) |
| Ladder requirement check (`plc/ladder_check.py`) | IMPLEMENTED, run on the real project file (VERIFIED-FILE) |
| Ladder simulation (`plc/ladder_sim.py`) + `selfcheck.py` + engineer "Simulation check" | IMPLEMENTED: the real .isp is run in a scan simulator against 14 scenarios; all self-tests in one command (24 pass) |
| Stable verdict (`verdict.py`) | IMPLEMENTED, TESTED SOFTWARE ONLY: one GOOD / DEFECT per bottle, latched; operator overlay without boxes / bars |
| Database page + export (`production_export.py`) | IMPLEMENTED: tables in plain words, CSV for Excel, printable report |
| Threshold calibration (`calibrate_thresholds.py`) | IMPLEMENTED and applied to `om_bottle` (2026-10-05): test false alarms 28 -> 22, 1 / 219 defective passed, 10 / 19 good test bottles still called defective |

## 4. Known limitations (summary)

- No end-to-end hardware validation; only a read-only real-PLC link. No distance, speed or actuator timing measured.
- The ladder's one-bottle handshake (M2 held until answered, M1 masking triggers through T0 + T1) limits throughput;
  a downstream camera adds its travel time to the minimum bottle gap. Short gaps are reported NOT INSPECTED.
- Camera faults during a run make bottles FAULT; recovery is Stop -> Start (no automatic reopen mid-run).
- YOLO runtime is opt-in and observational; it is unvalidated on live bottle frames (training images are Iriun-viewer screenshots, 16:9; development cameras default to 640x480, 4:3). The default live path is still the Stage 1 classifier.
- Stage 1 validation scores are saturated (several checkpoints report macro-F1 1.0); judge models on the held-out
  test results. `missing_cap` has **zero** positive examples and is disabled in every classifier; the detector
  path (recipe: missing cap box) is the only missing-cap check and is unvalidated on EMEET frames.
- Hand-tuned live thresholds in `projects/om_bottle/config.json` (some at 0.05-0.1) are unvalidated.
- `time.monotonic()` on Windows ticks every ~15.6 ms; frame `seq` is the strict order.
- The Live tab still shows per-frame verdicts (development view); the Production line decides once per bottle.
- `gui.py` is large (~5k lines; new screens live in `hmi.py`); `legacy/web_dashboard/` is an unused earlier web version.

## 5. Repository contents that are *not* in Git

Images (Stage 1 and Stage 2), the generated YOLO export, model weights (`*.pt`), logs and PLC editor
autosaves. The Stage 2 provenance (scripts, annotations, split, manifests) **is** tracked, and the weights
are described by checksum in `models/stage2_yolo/MODEL_PROVENANCE.json`.
