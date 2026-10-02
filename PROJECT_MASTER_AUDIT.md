# PROJECT MASTER AUDIT

Date: 2026-10-02. Root: `E:\MACHINE LEARNING PROJECT\Datasets\Bottle-train-OpenCV-main_vesion two`.
Method: static reading of code, docs, logs and data files, plus counting of dataset files. The app was not run, no model was trained, and no source, data, config or DB was changed. The only file created is this report (approved by the user). Numbers are quoted from the file named beside them. Anything unverified is marked UNKNOWN.

Status legend: GREEN = functional, YELLOW = partial, ORANGE = experimental, RED = missing/blocking, BLUE = needs testing.

---

## 1. Executive summary

- The project is a working **desktop dataset-labelling, classifier-training and frame-verdict tool** (Tk/customtkinter). It is not yet an inspection machine.
- **Exists:** multi-project dataset management, multi-label classifier training (5 backbones), per-defect thresholds, live multi-camera scoring on OpenCV, a box/polygon annotation studio, and a fully reviewed 594-image Stage 2 detection dataset with YOLO export.
- **Absent:** trigger, geometry/measurement, rule-based defect analysis, per-bottle decision engine, PLC link, sensor/tracking/reject timing, traceability (IDs, counts, history, reports, alarms), job/recipe layer. PROGRESS.md itself marks these "not started".
- **No YOLO detector has been trained** on the Stage 2 data. There are no detection metrics anywhere.
- Camera acquisition for live inspection uses driver defaults; there is no exposure/gain/focus/white-balance control, no sync, no reconnect, and EMEET NOVA 4K behaviour is untested.
- **PLC addresses are UNKNOWN.** The only ISPSoft file (`sell.isp`) has no ladder rungs or I/O.
- Time left: about 13 days to the 15 Oct machine-test target and 18 days to the 20 Oct submission. The critical path is long (see sections 17 and 18).

## 2. Project structure

### Entry points
| Entry | Purpose |
|---|---|
| `run.bat` | Installs torch if missing, migrates old layout, calibrates ROI, then runs `gui.py` |
| `gui.py` | Desktop app, `--selftest` flag |
| `train.py` | CLI training, `--demo` |
| `calibrate.py` | ROI measurement |
| `migrate.py` | One-shot layout migration into `projects/` |
| `stage2_dataset/review_app.py` | Standalone KEEP/REJECT/AMBIGUOUS reviewer |
| `stage2_dataset/annotation_workflow.py` | Opens Annotation Studio on the Stage 2 candidates |
| `app.py` + `index.html` | Old FastAPI web UI. **Dead**: nothing imports or launches it (CLAUDE.md agrees) |

### Source modules
| Module | Lines | Role | Active |
|---|---|---|---|
| `gui.py` | 2181 | App window and 8 tab classes (+ Annotate from studio) | Yes |
| `dataset.py` | 957 | Projects, labels.csv, cache, scene split, settings; global-state hub | Yes |
| `annotation_studio.py` | 868 | Box/polygon editor tab | Yes |
| `annotate.py` | 490 | annotations.json data layer, YOLO export | Yes |
| `train.py` | 519 | Classifier training and evaluation | Yes |
| `infer.py` | 416 | Model, Camera, CameraSet, verdict | Yes |
| `bench.py` | 225 | Camera and system benchmark | Yes |
| `charts.py` | 207 | Canvas charts | Yes |
| `calibrate.py` | 110 | ROI estimate by edge energy | Yes |
| `yolo_train.py` | 280 | Ultralytics synthetic smoke test | Experimental |
| `migrate.py` | 173 | One-shot migration | Rarely |
| `app.py` | 433 | FastAPI routes | Dead |
| `stage2_dataset/*.py` (6) | 201-753 | prepare, review, export, validate | Standalone |

Import graph: `gui` imports `annotation_studio`, `bench`, `charts`, `dataset`, `infer`; `dataset` is the hub; `annotate` is used by `annotation_studio` and `yolo_train`; `train` is loaded by the GUI.

### Config, data, docs
- `settings.json` (per-machine UI/bench prefs, gitignored), `requirements.txt` (torch, torchvision, opencv-python, customtkinter, pillow, optional psutil; **missing fastapi and ultralytics**).
- `projects/active.txt` = `bottle_detection`. `projects/om_bottle/config.json`: ROI `[826,0,841,1927]` on 2537x1927, per-class thresholds, `active_model: 20260918-120030`.
- Datasets: `All Datasets/` (1145 images), `stage2_dataset/`, `yolo_detection_dataset/`, `YOLO_EXPORT/`, `projects/`.
- Docs: CLAUDE.md, PLAN.md, PROGRESS.md (2026-08-23), README.md (title only), `docs/superpowers/specs/...multi-project-design.md`, `FINAL_YEAR_BLACKBOOK/` (about 40 files), `pipeline.drawio`, `docs/n8n/*`, a web-dashboard architecture .docx.
- **Stale docs:** CLAUDE.md says "not a git repo" and gui.py 1917 lines (git exists with one commit; file is 2181). PLAN.md says eight tabs (nine now). PROGRESS.md has no Annotate/Stage 2/YOLO content and counts 8 checkpoints (9 exist).
- **Dead/temp/leftover:** `app.py`, `index.html`, `YOLO_EXPORT/snap_056.txt`, `yolo_detection_dataset/images/` (606 images, no labels, unreferenced), `gui_run.log`, `selftest_out.log`, `tktest.log`, `__pycache__`, `docs/n8n`.
- **Tests:** none (no tests dir, no CI). Only per-module `selftest()`/`demo()`.

## 3. Current architecture

Single-process Tk app with a flat module layout. `dataset.py` rebinds about 15 module globals (`PROJECT`, `TASK`, `IMAGE_ROOT`, `MODELS`, `LABELS_CSV`, ...) in `use_project()` and is imported by everything. Long jobs go through `run_bg` (thread + queue + pump). A project is a folder: `project.json`, `labels.csv` or `annotations.json`, `config.json`, `models/<stamp>/`. Task type (classification / detection / segmentation) is recorded but only classification is wired into train/infer (`dataset.py:24-28`).

```
Camera(OpenCV DSHOW) -> fixed ROI crop -> letterbox -> EfficientNet multi-label sigmoid
  -> per-defect threshold OR -> per-frame PASS/REJECT -> GUI overlay
```
Nothing downstream of the GUI overlay exists.

## 4. Existing functionality (GUI)

Tabs: Label, Defects, Train, Analysis, Live, Camera, Data health, Annotate, Settings.

| Area | Status | Evidence |
|---|---|---|
| Navigation, shared refresh, background threads | WORKING | `gui.py:97-160, 240-255` |
| Project management | WORKING (no rename/delete) | `gui.py:183-238`, `dataset.py:61-136` |
| Dataset management (grid, multi-select, apply, delete, mark good) | WORKING | `gui.py:263-538` |
| Defect management | WORKING | `gui.py:540-700` |
| Data health (ROI, duplicate/scene scan) | WORKING | `gui.py:1425-1575` |
| Annotation | PARTIAL | boxes/polygons/export exist; disabled for classification projects; 1 image annotated in `bottle_detection`; no class add/rename in studio |
| Training | WORKING, classification only | `gui.py:715-745, 1025`; no task gate: a detection project can start a classifier |
| YOLO training in GUI | MISSING | no ultralytics reference in `gui.py` |
| Inference / Live | WORKING in code, BLUE (never run on real line) | `gui.py:1047-1420`; `PROGRESS.md:16,130` |
| Camera + benchmark | WORKING in code | `gui.py:1816-1975`, `bench.py` |
| Analysis (curves, confusion, compare) | WORKING | `gui.py:1578-1815` |
| Settings | WORKING | `gui.py:1976-2095` |
| Stage 2 review / annotation tools | ORANGE (standalone, not reachable from GUI) | `review_app.py:1-14` |
| Web UI | UNUSED | `app.py`, `index.html` |

Log evidence: `selftest_out.log` (31 Aug) "all 8 tabs built" predates the Annotate tab, so it is stale. `gui_run.log` has a DSHOW "can't be used to capture by index" warning. No recent run log exists.

## 5. Dataset status

### stage2_dataset (verified by counting, 2026-10-02)
| Item | Value |
|---|---|
| Files in `clean/` and `source/` | 606 each; 594 KEEP, 12 REJECT (`review_manifest.csv`) |
| Split | 418 train / 89 val / 87 test (matches `split_manifest.csv`, `split.json`, on-disk labels) |
| Scene grouping | 39 scenes: 25 train / 7 val / 7 test; no scene crosses splits |
| Boxes (`annotations.json`) | 1994 = 739 bottle / 709 cap / 546 label; 0 polygons |
| Per split (bottle/cap/label) | train 446/435/350 = 1231; val 137/131/99 = 367; test 156/143/97 = 396 |
| Status | all 594 `reviewed: true` |
| YOLO export | `yolo_export/labels/{train,val,test}` = 418/89/87 files; `data.yaml` nc=3 |
| Validation | `validate_log.txt` 19/19 checks pass |
| Quality notes | 4 boxes over 0.95 side (close-ups), 245 boxes touch an edge |

Previously reported figures (594, 418/89/87, 1994, 739/709/546) are **confirmed** as still current.

Gaps: `annotation_candidates.json/csv` still says `pending` for all 594 (stale vs `annotations.json`). No dataset versioning (`schema_version` only). Git has one commit and nearly all data is untracked. `yolo_detection_dataset/` (606 images, no labels) and `YOLO_EXPORT/snap_056.txt` are leftovers.

### Classification data
- `All Datasets/`: 1145 images, 9 folders: Damaged Bottle 319, Damaged Label 229, Good Bottle 72, Missing Label 30, **Missing Cap 0**, Skewed Bottle 167, Skewed Label 71, Tilt Cap 145, Water Level 112.
- `projects/om_bottle`: `labels.csv` 1143 rows, `scenes.json`, 9 checkpoints. About 112 distinct bottles, 72 good, 1073 defective (`PROGRESS.md:11-14`).

## 6. Annotation status

- `annotate.py`: normalized cxcywh boxes, flat-point polygons, class order = YOLO id; `validate_annotations`, derived status (`pending/annotated/reviewed`), `next_unannotated`, `progress_counts`, `export_yolo_detection`, `export_yolo_segmentation`. Extra keys (confidence, annotator) are tolerated.
- `annotation_studio.py`: box draw/resize, polygon (right-click finish, 3+ points), select/delete, mark reviewed, next unannotated, export buttons, `selftest`. No class management inside the studio.
- Candidate/review workflow: exists as separate scripts (`prepare_stage2.py`, `review_app.py`, `annotation_workflow.py`).

Feasibility of future modes (assessment only):
| Mode | Current capability | Effort / needs |
|---|---|---|
| Manual | Yes | none |
| AI-assisted | Schema ready (extra keys, `reviewed:false` -> "annotated") | Low: need a trained detector and a "propose boxes" call into the studio |
| Auto / batch | Not implemented | Medium: batch loop, confidence storage, threshold policy |
| Human review queue | Partial (`image_status`, `next_unannotated`, standalone review_app) | Low-medium: filter by confidence/status in the GUI |
No trained detector exists to drive any of the AI modes.

## 7. AI status

| Item | Status |
|---|---|
| Classifier training (`train.py`): BCE + pos_weight, early stopping | Implemented |
| Backbones: EfficientNet-B0, B1, MobileNetV3-Small, ResNet18, ConvNeXt-Tiny | Implemented, all trained at least once |
| Validation / test evaluation (`score_checkpoint`, `evaluate_test`) | Implemented |
| Thresholds per defect (config + GUI) | Implemented |
| Inference (`infer.py`) | Classifier only |
| Benchmarking | Camera/system only (`bench.py`); model benchmark exists only as Blackbook documents |
| Ultralytics / YOLO | Only `yolo_train.py` synthetic smoke test (1 epoch, 64 px, 8 images, yolov8n). Not a quality test |
| YOLO training on stage2, inference, export, mAP | **Not done / planned**; no `runs/` dir; `validate_log` says "Do NOT start training"; ultralytics 8.1.0 installed, `check_det_dataset` passes |
| Segmentation, anomaly detection | Not implemented |

### Results (never mix the two tables)

**Validation** (`projects/om_bottle/models/<stamp>/metrics.json`; split 685 train / 220 val / 238 test; cuda; 192x448):
| Model | Checkpoint | Val macro-F1 | Latency / FPS | Size MB |
|---|---|---|---|---|
| EfficientNet-B0 | 20260919-164511 | 1.0 | 9.16 ms / 109.2 | 15.57 |
| EfficientNet-B1 | 20260919-222031 | 1.0 | 13.48 ms / 74.2 | 25.24 |
| MobileNetV3-Small | 20260920-102102 | 0.864 | 4.95 ms / 201.8 | 5.93 |
| ResNet18 | 20260920-110121 | 0.9333 | 7.75 ms / 129.0 | 42.72 |

Older checkpoints (879/264 split, no test set, CPU): ResNet18 `20260918-120030` F1 1.0, 53.43 ms, 18.7 FPS; EffNet-B0 `20260918-191743` 1.0; EffNet-B1 `20260918-220502` 1.0; MobileNetV3 `20260919-000152` 0.8477; ConvNeXt-T `20260919-101315` 1.0, 106.19 MB, 111.39 ms, 9.0 FPS.

**Held-out test** (238 images, 23 scenes; source only `FINAL_YEAR_BLACKBOOK/06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md`, repeated in `12_Tables/model_comparison.md`; **not reproducible from `metrics.json`**, which holds only `n_test` and `test_paths`):
| Model | Test macro-F1 | Exact match | FP / FN | Latency / FPS |
|---|---|---|---|---|
| EfficientNet-B0 | 0.935 | 89.1% (212/238) | 28 / 0 | 13.89 ms / 72.0 |
| EfficientNet-B1 | 0.845 | 68.1% | 74 / 10 | 17.71 ms / 56.5 |
| MobileNetV3-Small | 0.632 | 67.2% | 82 / 36 | 14.04 ms / 71.2 |
| ResNet18 | 0.773 | 84.0% | 38 / 12 | 16.71 ms / 59.8 |

Notes: `missing_cap` has no positives and is unscored; ConvNeXt-T not test-evaluated; training times (B0 216.2 s, B1 453.9 s, MobileNetV3 192.2 s, ResNet18 145.3 s) are from the Blackbook only; latency columns differ between tables because they come from different runs. B0 `20260919-164511` is the Blackbook's chosen baseline, but `config.json` `active_model` is the oldest ResNet18. **No YOLO metrics exist.** Only `.pt` files inside the project tree were checked.

## 8. Camera status

All code is OpenCV (`infer.py`, `bench.py`, `gui.py`).

| Capability | Status | Evidence |
|---|---|---|
| Discovery | Implemented (probe indices 0..N-1) | `infer.py:49-74` |
| Selection | Implemented (tick boxes, video file) | `gui.py:1261-1274` |
| Live preview | Implemented | `gui.py:1320-1342` |
| Resolution / FPS | Only in benchmark; live uses driver defaults | `bench.py:127-129`, `infer.py:185-238` |
| Exposure, gain, focus, white balance | **Missing** in main project (exposure/MJPG/buffer only in outside `cam_capture.py`; no gain/focus/WB anywhere) | grep |
| Frame capture / saving | Implemented (JPEG q95 to inbox) | `dataset.py:495-508` |
| Multiple cameras | Implemented (thread per camera, shared model) | `infer.py:292-366` |
| Synchronised acquisition | **Missing** | no timestamp alignment |
| Health | Partial (fps, latency, dropped, error, CPU/RAM/GPU) | `infer.py:155`, `gui.py:1344-1388` |
| Reconnect / error handling | **Missing** (thread exits on failed read) | `infer.py:205-211` |
| Triggering | **Missing** (free-run about 15 Hz) | `infer.py:214` |
| EMEET NOVA 4K | BLUE: no evidence of any test; DSHOW index warning seen in `gui_run.log`; autofocus control not exposed |

Outside the project, `Datasets\cam_capture.py` (identical to `Projectx\cam_capture.py`) requests MJPG 3840x2160@30, manual exposure -6, buffer 1 and a DSHOW settings dialog. It is a separate tool.

## 9. Inspection pipeline status

| Stage | Status |
|---|---|
| Trigger | RED, absent |
| Camera acquisition | YELLOW (free-run, defaults) |
| Preprocessing | YELLOW (fixed ROI crop, letterbox, ImageNet norm; `dataset.py:529-606`) |
| AI detection | ORANGE: classification only, no boxes |
| Geometry analysis | RED, absent (`calibrate.py` only measures a crop ROI) |
| Defect analysis | YELLOW (learned classes only) |
| Decision engine | YELLOW (minimal) |
| PLC result | RED, absent |

Conclusion: separate components, not a pipeline. The chain ends at an on-screen verdict.

## 10. Defect analysis status

All defects are classifier outputs; none is measured.

| Defect | Status | Detail |
|---|---|---|
| Bottle presence | MISSING | `new_bottle` is a label with threshold 1.01 (can't fire) |
| Bottle damage | PARTIAL | `damaged_bottle`, thr 0.45; 319 training images |
| Bottle deformation / skew | PARTIAL | `skewed_bottle`, thr 0.95 |
| Cap presence | MISSING | no separate check |
| Cap missing | PARTIAL, effectively non-functional | `missing_cap` has 0 images; thr 1.01 (`PLAN.md:144`) |
| Cap tilt | PARTIAL | `tilt_cap`, thr 0.2 |
| Label presence / missing | PARTIAL | `missing_label`, thr 0.15; 30 images |
| Label skew | PARTIAL | `skewed_label`, thr 0.2 |
| Label damage | PARTIAL | `damaged_label`, thr 0.2 |
| Water level | PARTIAL (weak) | `water_level`, thr 0.1; trained on 31 frames, validated on 15 (`PLAN.md:135`); no fill-line measurement |
| Other configurable defects | PARTIAL | new defects can be added as dataset classes; no rule-based defect types |

Data caveat: 1145 images come from about 112 bottles; the model has never seen the real line.

## 11. Decision engine status

- Exists only as `infer.py:117-119`: reject if any defect probability >= its threshold (default 0.5). Multi-camera: reject if any camera rejects (`infer.py:360-366`).
- Thresholds are per project in `config.json`, tunable in the GUI (`gui.py:1168-1190, 981`).
- Missing: geometry/measurement input, rules, severity, temporal voting/debounce, per-bottle (rather than per-frame) result, defect reason list for the machine, fail-safe on model/camera loss. `combined()` returns PASS when no cameras are live (`infer.py:360-366`), which is unsafe for production.
- Status: YELLOW (a prototype, not a dedicated engine).

## 12. PLC status

Main project: **none**. No serial/Modbus/Ethernet code, no ladder, no handshake, timeout or reconnect. `PLAN.md:403` lists the reject-signal interface as an open question.

Outside the project (separate):
- `Datasets\plc_modbus_gui.py` (369 lines, Tk): manual Modbus coil write (fn 05) / read (fn 01) via pymodbus; RTU/ASCII serial or TCP; defaults ASCII, 7 data bits, even parity, 9600 baud, slave 1, TCP 192.168.1.5:502, 1 s timeout; polls every 0.5 s; no reconnect. `BIT_MAP = {M0:0, M1:1, M2:2}` is manual toggling, not wired to inspection.
- `Datasets\PLC_Modbus_GUI_Documentation.md`: claims M0-M2 map to coils 0-2. **Unverified** against the DVP-14SS2 manual or a real ladder. Its Ethernet assumption does not fit a DVP-14SS2 body.
- `Datasets\sell.isp` / `sell.ini`: project config only (`CPU=000E`, `COM_NUM=5`, `CON_INTERFACE=2`). No rungs, no M/X/Y/D devices, no comm parameters. Some fields look like an AH-series template, so the target CPU is unconfirmed. No `.dvp/.dvpx` files exist under `E:\MACHINE LEARNING PROJECT`.

| Item | Status |
|---|---|
| Protocol, interface | UNKNOWN (Modbus RTU/ASCII over RS-232/485 is plausible; not verified) |
| Registers / bits for trigger, result, handshake, fault | **UNKNOWN** |
| Connection code for inspection | MISSING |
| Handshake, timeout, reconnect, fault handling | MISSING |

## 13. Hardware integration status

Conveyor, camera enclosure, 2x EMEET NOVA 4K, lighting, photoelectric sensor, PLC, Festo DSNU cylinder, 5/2 solenoid: **no software support for any of these.**

- Photoelectric trigger, bottle tracking, conveyor timing, camera trigger, inspection timing, reject timing, pneumatic timing, pass/fail synchronisation: all absent (grep for sensor/tracking/conveyor/reject/gpio/serial found nothing in the main project; PROGRESS.md lines 65-66, 78-79, 90-103 say "not started").
- `app.py` `/api/reject-log` only appends to `rejects.csv`, is never called, and is dead code.
- Lighting: no control or configuration anywhere.

## 14. Production / traceability status

| Item | Status |
|---|---|
| Inspection ID | MISSING |
| Timestamp | Partial: capture filenames only (`cap_YYYYmmdd-HHMMSS_ms.jpg`) |
| Product/job | Partial: project name |
| Images per inspection | MISSING (dataset capture only) |
| Result, defect reason, confidence | In-memory per camera only |
| Model version | Partial: model stamp in project config; not stored per inspection |
| PLC result, reject status | MISSING |
| Production / pass / fail counts | MISSING (`PROGRESS.md:66`) |
| History, reports | MISSING (only a benchmark CSV export, `gui.py:1960`) |
| Alarms | MISSING |
| Logs | Development logs only |

## 15. Code quality

- **Oversized:** `gui.py` 2181 lines with 9 classes; `dataset.py` 957; `annotation_studio.py` 868; `validate_yolo_export.py` 753.
- **Global state:** `dataset.py` mutates about 15 module globals and runs `_boot()` at import. Threads read them while the main thread can switch project. Camera stop is guarded on switch; training is not (not verified).
- **Duplication:** colour palette repeated in `gui.py`, `infer.py`, `annotation_studio.py`, `review_app.py`; `app.py` duplicates label/train/live logic; repeated selftest scaffolding.
- **Hard-coding:** default input 192x448 (`dataset.py:44-45`) assumes a tall bottle; `calibrate.py` assumes edge energy on a flat dark backdrop (PAD 0.06, RATIO 0.15); `run.bat` hard-codes "OM Bottle"; `annotation_workflow.py` hard-codes counts and names; `+ve/-ve` folders and a `reviewed` column are reserved names. No "Bisleri" string in `.py`/`.html` (only `PLAN.md:3`). No absolute paths outside the repo.
- **Fragile dependencies:** fastapi and ultralytics used but unlisted; ultralytics weights cached in `~/.cache`.
- **Error handling:** 12 broad `except Exception` in `gui.py`, 5 in `dataset.py`; `project_title`/`project_task` silently fall back (`dataset.py:94-111`); none in `train.py` / `annotation_studio.py`. Task type is not enforced in Train/Live.
- **Tests:** none automated; selftests need real data.
- **Repo hygiene:** one commit, nearly everything untracked; `.gitignore` does not cover `All Datasets/`, `stage2_dataset/clean/`, `yolo_detection_dataset/`, `YOLO_EXPORT/`, so `git add .` would stage thousands of images. Code, datasets and thesis documents are mixed in one folder.
- **Config oddities:** thresholds of 1.01 disable two classes; `active_model` is not the chosen model.

## 16. Industrial architecture gap analysis

| Area | CURRENT | TARGET | GAP |
|---|---|---|---|
| Job / product | Project folder with ROI, thresholds, input size, task | Job = product + dataset + classes + cameras + lighting + ROI + model + rules + PLC + reject | No cameras, lighting, rules, PLC, reject config; no job layer above project |
| Classification | Multi-label classifier, 5 backbones | Selectable per job | Works; needs integration into decision engine |
| Detection | Dataset + export ready, no trained model | Detector locating bottle/cap/label | Train, validate, integrate |
| Segmentation | Annotation/export only | Optional | No training/inference |
| Anomaly detection | None | Optional | Entirely missing |
| Traditional vision (threshold, edge, blob, pattern, measurement) | Only ROI edge-energy | Geometry, tilt, fill level | Missing |
| OCR, barcode/QR | None | Optional | Missing |
| Cameras | Single/multi OpenCV, defaults | 2 synchronised 4K, controlled | Controls, sync, reconnect, trigger |
| Trigger | None | Sensor-driven | Missing |
| Decision engine | Per-frame OR of thresholds | Rules + AI + geometry -> PASS/FAIL + reasons, per bottle | Mostly missing |
| PLC | None | Handshake, trigger, result | Missing; addresses unknown |
| Reject logic | None | Tracking + timed actuator | Missing |
| Traceability | None | ID, images, result, counts, reports, alarms | Missing |
| Model versioning | Timestamped checkpoint folders | Registry with dataset linkage | Partial |
| Dataset versioning | None | Versioned snapshots | Missing |
| Auto annotation / active learning | Schema hooks only | Pre-label, review queue, loop | Missing |
| Multi product / multi job | Multi-project folders | Switchable jobs | Partial |
| Testing, deployment | Selftests; run.bat | Automated tests, stable runtime | Missing |

## 17. Critical blockers

1. No trigger/sensor/tracking/reject software and no PLC link.
2. PLC program and I/O map unknown; no verified ladder (`sell.isp` is empty of logic, possibly the wrong series).
3. Camera layer lacks exposure/focus/WB control, reconnect and sync; EMEET behaviour untested; live inspection untested on real line.
4. No real detector trained; classifier's `missing_cap` has no data; `water_level` very thin; the model has never seen real-line imagery.
5. Decision engine is per-frame, with no fail-safe (no cameras => PASS).
6. No traceability or counts.
7. No automated tests; task type not enforced in GUI.
8. Hardware (conveyor, enclosure, lighting, cylinder, valve) readiness is not documented anywhere in the repo.

## 18. Recommended development order

Target: 15 Oct machine testing (13 days), 20 Oct submission.

1. **Now (days 1-2):** confirm the physical I/O: obtain or recreate the actual ISPSoft ladder and the PLC wiring list (X/Y/M assignments), PLC model/COM port. Test the EMEET cameras in the enclosure; fix lighting and capture real-line images.
2. **Days 1-4 (parallel):** train a YOLOv8 detector on stage2 (bottle/cap/label), report validation vs held-out test mAP separately; capture and label real-line frames (including missing-cap samples).
3. **Days 3-6:** camera layer: lock resolution/FPS/exposure/focus, MJPG, reconnect, 2-camera capture.
4. **Days 5-8:** per-bottle decision engine: voting across frames/cameras, rules, defect reasons, fail-safe default FAIL, with a thresholds config.
5. **Days 6-9:** PLC link (pymodbus/pyserial) once addresses are verified: trigger-in, result-out, handshake, timeout, reconnect.
6. **Days 8-10:** trigger + reject timing (sensor to camera to reject delay), run dry on bench before conveyor.
7. **Days 10-12:** traceability (ID, timestamp, images, result, counts, CSV log), simple job config file.
8. **Day 13 (15 Oct):** full-machine test; fix.
9. **16-20 Oct:** retest, freeze, update docs and blackbook, final report.

Defer: segmentation, OCR/barcode, anomaly detection, active learning, auto-annotation, model registry.

## 19. Risks

- PLC/ladder unknown and possibly mismatched series: highest schedule risk.
- Domain gap: all training data predates the real enclosure, lighting and cameras; test numbers (B0 F1 0.935, 28 FP) may not transfer.
- Classifier test results exist only as documents; they should be regenerated reproducibly.
- Thin classes (missing cap = 0, water level = 31, missing label = 30).
- No automated tests; global state; threads can race on project switch.
- Camera driver behaviour (DSHOW index warning, autofocus drift on a conveyor).
- Time: 13 days for about 6 new subsystems.
- Documentation drift and untracked data: risk of loss; no backup or commit history.

## 20. Final readiness assessment

| Component | Status | Evidence | Missing work | Priority |
|---|---|---|---|---|
| Dataset tooling (classification) | GREEN | `gui.py`, `dataset.py` | versioning | Low |
| Stage 2 detection dataset | GREEN | counts verified, 19/19 validation | candidate file stale | Low |
| Annotation studio | YELLOW | boxes/polygons/export; class mgmt absent | AI-assist, review queue | Medium |
| Classifier training/eval | GREEN | 5 backbones, metrics | reproducible test metrics | Medium |
| Classifier on real line | BLUE | never run on line | on-line validation | High |
| YOLO detection | RED | smoke test only | train, evaluate, integrate | High |
| Camera acquisition | YELLOW | basic OpenCV | controls, sync, reconnect | High |
| EMEET NOVA 4K | BLUE | no evidence | hardware test | High |
| Trigger | RED | none | sensor input | High |
| Geometry | RED | none | measurement | Medium |
| Defect analysis | YELLOW | classifier classes only | rules, thin classes | High |
| Decision engine | YELLOW | `infer.py:117` | per-bottle, fail-safe | High |
| PLC | RED | none; addresses UNKNOWN | ladder, driver, handshake | High |
| Reject/tracking | RED | none | timing logic | High |
| Traceability | RED | none | log, counts, reports | Medium |
| Job architecture | YELLOW | project config | camera/rules/PLC config | Medium |
| Tests / code health | RED | no tests; globals | tests, gui split | Medium |
| Documentation | YELLOW | stale | refresh | Low |

Overall: **not ready for the physical machine.** Roughly the front end and dataset tooling are done; the machine-control half (trigger, decision, PLC, reject, traceability) has not started. Reaching the 15 Oct target requires starting the PLC/ladder, camera and decision-engine work immediately and treating AI model quality on real-line images as a parallel track.

Items not verified in this audit: any `yolo*.pt` outside the project tree, the actual training-thread guard on project switch, and the contents of stale docs beyond their headers.

---

AUDIT COMPLETE — NO FILES MODIFIED (only this report, `PROJECT_MASTER_AUDIT.md`, was created, as approved)
