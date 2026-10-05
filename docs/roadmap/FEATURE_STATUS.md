# Feature Status

Factual status only -- no rankings. **Maintain this file whenever something changes**; it is the source of
truth the README links to.

**Status values:** `COMPLETE` (works for its intended purpose today) - `PARTIAL` - `SOFTWARE ONLY` (works in
software; never run on hardware) - `HARDWARE REQUIRED` (needs the physical machine to build/verify) - `NEXT` -
`FUTURE` - `DEFERRED` (intentionally postponed until the machine is proven).

**Horizon values:** `NOW` (current scope, see [CURRENT_SCOPE.md](CURRENT_SCOPE.md)) - `NEXT` - `FUTURE`.

"SOFTWARE ONLY" means a software self-test or offline evaluation exists. It is never evidence of hardware
success. Nothing in this repository has been validated on the physical machine except a read-only PLC link (2026-10-04).

| Feature | Current status | Now/Next/Future | Notes |
|---|---|---|---|
| Project management | COMPLETE | NOW | Multi-project folders, switch/create in the GUI. Not a job/recipe system |
| Dataset management | PARTIAL | NOW | Stage 1 images/labels managed in the GUI; Stage 2 dataset built by scripts in `stage2_dataset/`. Label tab: keyboard toggle, one-image multi-label inspector, balance strip, trash + `label_log.csv` + undo, import of a pre-sorted folder (new product or current), AI pre-labels (suggestions only, never training data until accepted). Software self-tested only; the AI pre-label path was tested with fake suggestions, not run against a real checkpoint in the self-test. No users/roles: the log records the OS login |
| Classification | SOFTWARE ONLY | NOW | Stage 1 multi-label classifier; held-out test results for 7 checkpoints. **`missing_cap` cannot be learned from the current data**: the 12 user images are one white-background bottle, and the 2026-10-05 candidate trained on them flagged every white-background bottle (shortcut; REJECTED, images removed, undoable). Needs missing-cap bottles photographed in the classifier's own setup |
| Detection | PARTIAL | NOW | YOLOv8n components (bottle/cap/label), not defects; opt-in runtime and the line's detection task. **Every trained detector misses full-bottle missing caps (0/12: tamper ring read as cap, ~0.70)** because that scene was test-only. v3 split puts it in train (`stage2_dataset/split_v3.py`); v3 candidate trained 2026-10-05: val missing cap 5/6, good 54/54 val + 50/50 test, learns the full-bottle bare neck (12/12 on its own training frames, not evidence; v1 active gets 0/12). CANDIDATE until validated on the machine |
| Segmentation | PARTIAL (interface only) | NOW | `segment.py` runtime interface (`YoloSegmenter`, label area / fill measurements) and decision rules (`label_area_low`, `label_damaged`) exist and are FAKE-tested. **No segmentation model is trained**: selecting the Segmentation task refuses to start the line with that reason; an inspection that needs it is FAULT |
| Annotation Studio | COMPLETE | NOW | Boxes + polygons, review flag, YOLO detection/segmentation export |
| YOLO training | COMPLETE | NOW | YOLOv8n only: 53 epochs, best epoch 33; YOLOv8s not trained |
| YOLO runtime | PARTIAL (SOFTWARE ONLY) | NOW | **Current development.** `detect.py` + Camera hook + Live-tab selector; boxes/confidence/detector state shown; observational (never PASS/REJECT); detector fault or stale -> FAULT. Live-tested only without a bottle in view; dev confidence 0.25 is not validated |
| Camera manager | PARTIAL | NOW | Discovery with device names, requested mode + delivered resolution/FPS/frame age, delivered-vs-requested + CAMERA_BANDWIDTH_PROBLEM, camera_controls lock, rotate, CAMERA TEST with image quality. **Auto-reconnect while the line runs** (background reopen every 3 s, bottles in the gap FAULT, attempts counted and logged). FAKE-tested; not tried on the EMEETs. No hardware trigger |
| Multi-camera | PARTIAL (2x EMEET Nova 4K tested live) | NOW | Both EMEETs stream together at 640x480 (~30 fps each). **Only one streams at 1080p/720p**: both are on one USB 2.0 hub (VID_1A40 PID_0101), measured 2026-10-03; needs separate USB ports. Per-bottle fusion FAULT > REJECT > PASS in `decision.py`. Cameras are not hardware-synchronised |
| Camera synchronization | SOFTWARE ONLY (time-based) | NOW | No hardware sync. Staggered stations (`camera_stations`, `tracking.py`): each camera's frames are taken in its own window (trigger + offset / speed), bounded by the next bottle, CAMERA_ASSOCIATION_FAULT when neighbours cannot be separated. FAKE-tested; never run on the belt |
| Trigger sensor | SOFTWARE ONLY | NOW | X0 -> M2 trigger drives the line (`machine_cycle`); X0 edges without a trigger are counted (`BOTTLE_UNTRIGGERED`). Physical photo-eye untested |
| Bottle tracking | SOFTWARE ONLY | NOW | Sensor-triggered association: each M2 trigger = one `inspection_id`, frames used are the ones captured strictly after the trigger. No visual tracking |
| Inspection window | SOFTWARE ONLY | NOW | `inspect_frames` per camera within `inspect_window_s` after the trigger (+ `capture_delay_s`); frame vote in `decision.py`. Values not tuned on the machine |
| Temporal voting | SOFTWARE ONLY | NOW | Frame vote per camera (`decision.py` majority / any / all) over the frames of one bottle |
| Decision engine | SOFTWARE ONLY | NOW | `decision.py`: one decision layer for classification / detection (**per-project inspection recipe**, `config.json` `inspection`: anchor + required parts + zones; default = bottle/cap/label: missing/misplaced cap, missing label, no bottle -> FAULT) / segmentation; frame vote; camera fusion; every failure -> FAULT. Thresholds are development defaults. No fault latching |
| PASS / REJECT / FAULT | SOFTWARE ONLY | NOW | Fail-safe states with freshness checks; software-tested with fake captures |
| Machine state | SOFTWARE ONLY | NOW | `machine_state.py`: one state (OFFLINE ... COMMUNICATION_FAULT) + start checklist; START enabled only when READY |
| Traceability | SOFTWARE ONLY | NOW | Every bottle persisted with run (job, product, recipe hash, model ids, SIMULATOR/REAL), decision, defects, PLC outcome, timings, per-camera windows, evidence path. `inspection_trace.TraceStore` (in-memory) still unused by the line |
| PLC communication | SOFTWARE ONLY (simulator) | NOW | `plc/`: Modbus ASCII client, `PLCService` (single owner, M2 trigger events, at-most-once PASS/REJECT, REJECT ack = M2 cleared because the ladder holds M1, X0 bottle accounting). FAKE-PLC tests pass against a scan emulation of the 09:06 ladder; simulator reads/faults verified; **09:06 ladder not yet run on the simulator (simulator had no program)**. Physical PLC untested. See `PLC_COMMUNICATION.md` 0.1 / K |
| Ladder simulation check | SOFTWARE ONLY | NOW | `plc/ladder_sim.py` + GUI Simulation check: the user's real .isp run in a scan simulator vs 14 software-contract scenarios (current ladder: structure OK, T0/T1 presets FAIL, masking/one-bottle LIMIT, no E-stop/interlock/timeout WARN); `selfcheck.py` runs every self-test. Does not read OR branches; not a hardware test |
| Stable operator verdict | SOFTWARE ONLY | NOW | `verdict.py`: one GOOD / DEFECT: <name> / CHECKING / NO BOTTLE / FAULT per bottle, decided over several frames and latched until the bottle leaves; engineer view with boxes and score bars behind "Engineer details" |
| Scrolling pages | SOFTWARE ONLY | NOW | every page scrolls up/down and left/right (`ScrollHost`) |
| Database page and export | SOFTWARE ONLY | NOW | `hmi.DatabaseTab`, `production_export.py`; CSV and printable report; `docs/guides/DATABASE.md` |
| Threshold calibration | SOFTWARE ONLY | NOW | `calibrate_thresholds.py` (--apply / --restore); does not fix the good-bottle false alarms (72 good images) |
| Mock PLC | SOFTWARE ONLY | NOW | `plc.test_simulation.FakePLC` + `FakeLadder`: Modbus server + scan-by-scan emulation of the decoded ladder (nets 1-7, presets configurable), used by every PLC / machine-cycle / GUI self-test |
| Delta PLC | HARDWARE REQUIRED | NOW | First read-only serial link 2026-10-04 (USER-STATED). `plc.ladder_check` decodes the saved `final_year.isp` (unchanged since 2026-10-03 20:24, 7 networks) and checks 13 requirements: trigger / handshake / reject cycle / conveyor PRESENT; T0/T1 MISMATCH (15 s / 5 s); M10/M11, E-stop input, Y0 interlock, answer timeout, heartbeat ABSENT; triggers masked during a reject. See `docs/hardware/PLC_LADDER_REQUIREMENTS.md`. What is in the real PLC's memory is not yet compared |
| Conveyor | HARDWARE REQUIRED | NOW | Y1 read and shown (never written). Speed not measured: `conveyor_mm_s` = 0 until it is |
| Reject actuator | HARDWARE REQUIRED | NOW | Y0 read and tracked per REJECT bottle (never written; the PLC times it). Cylinder never fired on hardware |
| Timing engine | SOFTWARE ONLY | NOW | Time-based (no encoder): `tracking.TimePositionSource`, speed calibration wizard (>= 3 runs, spread limit, timestamped), line layout dialog, `line_problems()` blocks Start (T0 >= travel, unmeasured T0 > 5 s, downstream camera without speed, decision after REJECT dispatch). REJECT at trigger + travel - T0; late REJECT never fired. **No distance or speed measured on the machine yet** |
| Production counters | SOFTWARE ONLY | NOW | Production tab: Total / PASS / REJECT / FAULT / queue / not inspected / missed rejects; per-bottle row in `projects/<slug>/production/<date>.csv` |
| Evidence capture | SOFTWARE ONLY | NOW | `production_store.py`: first frame + overlay per bottle by `evidence_policy` (default REJECT_AND_FAULT), async writer, disk check; path in the bottle record |
| Job / recipe | SOFTWARE ONLY | NOW | Job + product per run; recipe editor dialog (engineer) for `config.json` `inspection` (anchor + parts + search/zone), validated against the detector classes, refused while the line runs; recipe hash stored per run |
| Model registry | SOFTWARE ONLY | NOW | `model_registry.py` + Models page: CANDIDATE / VALIDATED / APPROVED / ACTIVE / ARCHIVED / REJECTED, gated activation, `deployments.jsonl`, rollback, critical-class warning. Training no longer auto-activates |
| Dataset versioning | DEFERRED | FUTURE | Today: `labels.csv` copy per checkpoint; `schema_version` in Stage 2 JSON; hashes in `MODEL_PROVENANCE.json` |
| Traditional vision | DEFERRED | FUTURE | Only Sobel (ROI calibration) and Laplacian (sharpness) are used |
| Geometry measurement | FUTURE | FUTURE | No measurement in the code; shape defects are learned, not measured |
| Lighting configuration | HARDWARE REQUIRED | FUTURE | Not represented in software |
| Auto annotation | SOFTWARE ONLY | NOW | `autoannotate.py`: detector proposals kept out of `boxes` until accepted; least-sure-first ordering. FAKE-tested |
| Active learning | SOFTWARE ONLY | NOW | Annotate tab queues: least sure, missing part (bottle without cap/label = likely real defect captures), doubtful part (cap/label at 0.25-0.80 = hard-negative candidates). Reorders unreviewed images only |
| Anomaly detection | DEFERRED | FUTURE | |
| OCR | DEFERRED | FUTURE | |
| Barcode | DEFERRED | FUTURE | |
| SQLite | SOFTWARE ONLY | NOW | `projects/<slug>/production/production.db`: runs, bottles, alarms (stdlib sqlite3). FAKE-tested |
| Dashboard | SOFTWARE ONLY | NOW | Production page is the operator HMI (light theme, one machine state, both cameras, current bottle, counters, alarms, START / STOP / RESET FAULT, CAMERA TEST, TEST INSPECTION); Health page; operator / engineer modes |
| User login | DEFERRED | FUTURE | |
| Reports | SOFTWARE ONLY | NOW | History page: per day or per shift (A/B/C, configurable, night shift across midnight): counts, yield, defect distribution, bottles/hour, alarms, CSV export |
| Alarm management | SOFTWARE ONLY | NOW | `alarms.py`: 25 coded alarms (severity, message, action), condition / event, RESET FAULT acknowledges, persisted, shown on Production / History / Health |
| Maintenance | PARTIAL | NOW | Health page (computer / PLC / cameras / models / storage / alarms) + structured event logs (`applog.py`, 7 channels, searchable on Health) |
| Remote monitoring | DEFERRED | FUTURE | |
| Closed-loop retraining | DEFERRED | FUTURE | |
