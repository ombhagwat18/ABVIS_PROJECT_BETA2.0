# Gap matrix and implementation record: 2026-10-05

This audit set out to turn the existing application into a coherent, production-oriented inspection
platform, **without** creating a second application. It covers three steps:

- Phase 0: a read-only audit, reconciling `CLAUDE.md`, `CURRENT_SYSTEM.md` and the actual code.
- A gap matrix.
- What was implemented in this pass.

Evidence labels (`VERIFIED` / `USER-STATED` / `INFERRED`, and `FAKE` / `SIMULATOR` / `PHYSICAL`) keep their
meaning from `docs/roadmap/PLC_COMMUNICATION.md`.

> **Everything below marked done is SOFTWARE VERIFIED** (self-tests with fake PLC, fake cameras and fake
> models). **Nothing in this pass was run on the physical machine.**

## 1. Documentation drift found (Phase 0)

| Claim | Where | Reality in code (VERIFIED by reading and running) |
|---|---|---|
| "No per-bottle logic", "Decision logic PARTIAL: per-frame only" | CURRENT_SYSTEM.md section 3 | `decision.py` (frame vote, camera fusion, recipe) and `machine_cycle.py` (one decision per bottle) exist and are FAKE-tested |
| "Sensor trigger, bottle tracking, timing model: NOT IMPLEMENTED" | CURRENT_SYSTEM.md, Machine side | X0 -> M2 trigger, time-based FIFO, deadline REJECT dispatch, `timing_problem()` exist |
| "PLC layer not integrated with the inspection pipeline or GUI" | CURRENT_SYSTEM.md | Integrated: Machine and Production tabs, one `PLCService` |
| "Mock PLC, serial transport: NOT IMPLEMENTED" | CURRENT_SYSTEM.md | `FakePLC` / `FakeLadder` and `SerialTransport` exist; first read-only physical link on 2026-10-04 (CLAUDE.md, USER-STATED) |
| "Desktop GUI, 9 tabs", "gui.py ~2.2k lines" | CURRENT_SYSTEM.md | 11 tabs before this pass, 14 after; gui.py ~5k lines |
| "Auto annotation: DEFERRED" | FEATURE_STATUS.md | `autoannotate.py` + Annotate-tab proposals exist (FAKE-tested) |
| "No held-out test split" | CURRENT_SYSTEM.md | `split_train_val_test` + `test_metrics.json` exist for 6 classifier checkpoints |
| `missing_cap` has 0 positives | both | **Still true** (VERIFIED 2026-10-05: `labels.csv` 1,143 rows, `missing_cap` = 0). Images the user mentioned adding are not in `labels.csv` |
| `train.run()` promotes every run | CLAUDE.md | Was true; **changed in this pass** (see below) |

## 2. Gap matrix (requested capability -> state before -> state now)

| Capability | Before this pass | Now (all SOFTWARE VERIFIED unless noted) |
|---|---|---|
| One machine state | Each widget derived its own (LINE lamp, Production label, Machine label) | `machine_state.py`: 10 states from one set of facts; Production banner and status-bar LINE lamp show the same value |
| Start readiness checks | PLC connected + timing only | PLC link, PLC RUN, cameras, models, timing (incl. stations and calibration), E-stop, recipe; START disabled until READY |
| Startup / initialising | Line started as soon as models loaded | INITIALIZING until every line camera delivers a first frame (8 s), else CAMERA_DISCONNECTED and not started |
| Orderly STOP | Stop left inspecting / scheduled bottles in the FIFO | Bottles still inspecting or scheduled end as FAULT ("remove by hand"), never silently dropped; run closed in the store |
| Time-based tracking | travel = distance / speed in `machine_cycle` | `tracking.py`: `PositionSource` seam, `TimePositionSource` (speed plus tolerance gives an explicit uncertainty), `EncoderPositionSource` refuses to exist (no encoder) |
| Speed calibration wizard | None | Engineer dialog: stopwatch or typed runs, at least 3 runs, mean/min/max/spread, refused above tolerance, timestamped record in settings |
| Sequential (staggered) cameras | All cameras read at the trigger instant | `camera_stations` (offset mm downstream, wall side, role, per-camera rules); each camera's window is trigger + offset/speed; non-blocking collection; frames bounded by the next bottle's window; CAMERA_ASSOCIATION_FAULT for neighbours too close to separate |
| Camera roles | None | Per-camera decision overrides (`judge` = parts this camera judges, `station_x`) in `decision.decide(per_camera=...)` |
| Layout validation | T0 vs travel only | `line_problems()`: downstream camera without speed, camera past the reject station, decision ready after the REJECT dispatch time (+ margin) |
| Coded alarms | Free-text list | `alarms.py`: 25 codes (severity, operator message, action), condition and event alarms, acknowledge (RESET FAULT), dedup, persisted |
| Persistent traceability | Daily CSV | `production_store.py`: SQLite runs (job, product, recipe hash, model ids, settings), one row per bottle, alarms, evidence images by policy (async writer, disk check); CSV kept |
| Per-stage timings | Total inference ms | capture wait, classification, detection, decision, PLC ms, trigger->decision, trigger->command per bottle; p50/p95 on screen |
| History / reports | None | History page: day, filter, counts, yield, defect distribution, bottles/hour, alarms, evidence viewer, CSV export |
| System health | Partial (Camera tab) | Health page: computer, PLC, cameras, models, storage, alarms -> HEALTHY / WARNING / FAULT |
| Model registry / deployment | Checkpoint folders; training auto-activated | `model_registry.py`: CANDIDATE -> VALIDATED (held-out test + real-camera note) -> APPROVED -> ACTIVE, archive, `deployments.jsonl`, rollback; detector activation sha-checked; Models page. `train.run()` no longer activates by default |
| Critical-class visibility | Hidden in per-defect tables | Registry flags every classifier: `missing_cap` has **no test positives** |
| Operator / engineer UX | One dark training GUI, Production 4th in its group | Light industrial theme (dark optional), operator mode (Production / History / Health only), large START / STOP / RESET FAULT, engineer row hidden from operators |
| AI test mode (no PLC) | `machine_cycle.py --bench` (CLI) | TEST INSPECTION button: same models, decision engine and recipe; result, overlay and timing on screen; files in `captures/` |
| Camera-only test | Live tab / Camera tab | CAMERA TEST: chosen line cameras live with brightness / contrast / sharpness / clipping; capture test frame |
| Bandwidth awareness | Hint text | Delivered vs requested resolution shown; CAMERA_BANDWIDTH_PROBLEM alarm |
| Simulator vs real PLC | Small text | SIMULATOR / REAL PLC badge on Production and Machine |
| Self-test pollution | GUI self-test wrote fake bottles into the real project's production CSV | Self-test writes to a temporary folder |

## 3. Not done in this pass (honest list)

| Item | Why / what is needed |
|---|---|
| YOLO multi-candidate training, classifier retraining | Must not train before a dataset audit of the new images (and none of the user's new missing-cap images are in `labels.csv`). Existing tooling (`model_bench.py yolo / cls-train / cls-test`, `vision_data.py`) already runs it; candidates then appear in Models |
| Missing-cap hard negatives (bare neck, tamper ring, partial cap) | Needs real EMEET images collected on the machine, annotated (Annotate tab plus proposals), then a detector candidate trained and benchmarked on `det-defects` |
| Camera auto-reconnect while running | A camera fault makes the bottle FAULT and raises CAMERA_DISCONNECTED; recovery is Stop -> Start (no app restart). An automatic reopen during a run is not implemented |
| Per-camera ROI for the classifier | The classifier ROI is per checkpoint; per-camera ROI for a second view needs a second classifier or ROI-free detection |
| Recipe GUI editor | Recipe is still `config.json` `"inspection"` (validated at start) |
| Structured per-subsystem log files | Production events go to SQLite (bottles, alarms, runs) and the CSV; PLC events stay in `PLCService.events()`. Separate rotating app / camera / AI log files are not added |
| Shift reports, user login | Day-level history only; no users / roles beyond the engineer PIN (a convenience lock, not security) |
| Ladder changes (K150 -> K15, multi-bottle reject FIFO in the PLC) | The ladder belongs to the user; never edited here |

## 4. Physical blockers (unchanged by software)

1. **T0 / T1** in the saved ladder are K150 / K50 (15 s / 5 s) against the stated K15 / K5; `settings.json` says 0.75 s.
   `plc_t0_s` must equal what is in the PLC. Measure it on the real PLC (Machine tab, T0 value) before any REJECT test.
2. **One-bottle handshake**: M2 stays ON until answered (a REJECT is answered at its dispatch time), and M1 masks
   triggers through T0 + T1. A downstream camera lengthens the time to answer. Bottles closer than that are
   NOT INSPECTED (detected and alarmed, not hidden).
3. **USB**: two EMEET Nova 4K on one USB 2.0 hub give one 1080p stream. Use separate USB 3 root ports.
4. **No measured distances or speed**: run the speed calibration and enter the line layout.
5. **Models**: no classifier can detect `missing_cap` (0 positives); the detector path is the only missing-cap
   check, and it is unvalidated on EMEET frames.

## 5. Second pass (2026-10-05, afternoon)

| Item from section 3 | Now |
|---|---|
| Camera auto-reconnect while running | DONE: background reopen every 3 s, counted, logged; bottles in the gap FAULT (SOFTWARE VERIFIED, fake camera) |
| Recipe GUI editor | DONE: `hmi.RecipeDialog`, validated against detector classes, refused while running |
| Structured per-subsystem logs | DONE: `applog.py`, 7 channels, event-based, searchable on Health |
| Shift reports | DONE: History per shift (configurable, night shift across midnight) |
| Active-learning queues (hard negatives, missing components) | DONE: Annotate "Next: missing part" / "Next: doubtful part" |
| PLC ladder requirements | DONE (document + checker): `docs/hardware/PLC_LADDER_REQUIREMENTS.md`, `python -m plc.ladder_check`. The ladder itself is the user's; not edited |
| Missing-cap data + retraining | DONE, with a negative result for the classifier (shortcut, REJECTED, images removed, undoable) and a v3 detector candidate that needs machine validation. See README "Missing cap" |
| Per-camera classifier ROI, user login, PC heartbeat | NOT DONE: heartbeat needs a ladder rung first (R13); ROI per camera needs a second classifier; login deferred |

## 6. Third pass (2026-10-05 evening, status 2026-10-06)

| Request | Now |
|---|---|
| One direct answer per bottle, no boxes / bars, industry-style | DONE: `verdict.py` + operator overlay; GOOD / DEFECT: <name>; boxes and bars only with "Engineer details" |
| Good bottles shown as defective, better confidence | PARTLY: thresholds re-chosen from validation, steady verdict; the model itself is limited by 72 good images, so more real good bottles + retraining are still needed |
| Auto-annotation prediction of defects | DONE / documented: classifier defect pre-labels (Label tab) use the calibrated thresholds; Annotate tab queues; `docs/guides/AUTO_ANNOTATION.md` (with sources) |
| Simulation check for engineers only, simulator and real PLC | DONE: engineer-only, live PLC snapshot, switch to remove it later (`show_simulation_check`) |
| Panels full of text hide buttons -> scroll both ways | DONE: every page is a `ScrollHost` |
| Explain and add the database, exportable | DONE: Database page, CSV + report export, `docs/guides/DATABASE.md` |
| Restart the system | The app was restarted but stopped by the OS under memory pressure (another training job on the PC) |
| Docs for ChatGPT review | `docs/PROJECT_BRIEF_FOR_REVIEW.md` |
