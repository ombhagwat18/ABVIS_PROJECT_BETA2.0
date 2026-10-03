# Feature Status

Factual status only -- no rankings. **Maintain this file whenever something changes**; it is the source of
truth the README links to.

**Status values:** `COMPLETE` (works for its intended purpose today) - `PARTIAL` - `SOFTWARE ONLY` (works in
software; never run on hardware) - `HARDWARE REQUIRED` (needs the physical machine to build/verify) - `NEXT` -
`FUTURE` - `DEFERRED` (intentionally postponed until the machine is proven).

**Horizon values:** `NOW` (current scope, see [CURRENT_SCOPE.md](CURRENT_SCOPE.md)) - `NEXT` - `FUTURE`.

"SOFTWARE ONLY" means a software self-test or offline evaluation exists. It is never evidence of hardware
success. Nothing in this repository has been validated on the physical machine.

| Feature | Current status | Now/Next/Future | Notes |
|---|---|---|---|
| Project management | COMPLETE | NOW | Multi-project folders, switch/create in the GUI. Not a job/recipe system |
| Dataset management | PARTIAL | NOW | Stage 1 images/labels managed in the GUI; Stage 2 dataset built by scripts in `stage2_dataset/` |
| Classification | SOFTWARE ONLY | NOW | Stage 1 multi-label classifier is the live path. No test split; saturated validation scores; `missing_cap` has 0 positives |
| Detection | PARTIAL | NOW | YOLOv8n: components (bottle/cap/label) only, not defects. Available as an opt-in runtime path; unvalidated on live bottle frames |
| Segmentation | PARTIAL (interface only) | NOW | `segment.py` runtime interface (`YoloSegmenter`, label area / fill measurements) and decision rules (`label_area_low`, `label_damaged`) exist and are FAKE-tested. **No segmentation model is trained**: selecting the Segmentation task refuses to start the line with that reason; an inspection that needs it is FAULT |
| Annotation Studio | COMPLETE | NOW | Boxes + polygons, review flag, YOLO detection/segmentation export |
| YOLO training | COMPLETE | NOW | YOLOv8n only: 53 epochs, best epoch 33; YOLOv8s not trained |
| YOLO runtime | PARTIAL (SOFTWARE ONLY) | NOW | **Current development.** `detect.py` + Camera hook + Live-tab selector; boxes/confidence/detector state shown; observational (never PASS/REJECT); detector fault or stale -> FAULT. Live-tested only without a bottle in view; dev confidence 0.25 is not validated |
| Camera manager | PARTIAL | NOW | DirectShow index discovery **with device names** (`infer.camera_names`), requested capture mode (`line_capture_wh`, MJPG), delivered resolution/FPS/frame age shown; disconnect / no frames / frame timeout -> FAULT. Probing serialised (two concurrent DirectShow scans crashed the app, fixed). No exposure/gain/focus control, no auto-reconnect, no hardware trigger |
| Multi-camera | PARTIAL (2x EMEET Nova 4K tested live) | NOW | Both EMEETs stream together at 640x480 (~30 fps each). **Only one streams at 1080p/720p**: both are on one USB 2.0 hub (VID_1A40 PID_0101), measured 2026-10-03; needs separate USB ports. Per-bottle fusion FAULT > REJECT > PASS in `decision.py`. Cameras are not hardware-synchronised |
| Camera synchronization | FUTURE | NEXT | Cross-camera frame alignment does not exist |
| Trigger sensor | SOFTWARE ONLY | NOW | X0 -> M2 trigger drives the line (`machine_cycle`); X0 edges without a trigger are counted (`BOTTLE_UNTRIGGERED`). Physical photo-eye untested |
| Bottle tracking | SOFTWARE ONLY | NOW | Sensor-triggered association: each M2 trigger = one `inspection_id`, frames used are the ones captured strictly after the trigger. No visual tracking |
| Inspection window | SOFTWARE ONLY | NOW | `inspect_frames` per camera within `inspect_window_s` after the trigger (+ `capture_delay_s`); frame vote in `decision.py`. Values not tuned on the machine |
| Temporal voting | NEXT | NOW | Not started |
| Decision engine | SOFTWARE ONLY | NOW | `decision.py`: one decision layer for classification / detection (missing/misplaced cap, missing label, no bottle -> FAULT) / segmentation; frame vote; camera fusion; every failure -> FAULT. Thresholds are development defaults. No fault latching |
| PASS / REJECT / FAULT | SOFTWARE ONLY | NOW | Fail-safe states with freshness checks; software-tested with fake captures |
| Traceability | PARTIAL | NOW | In-memory `InspectionRecord` / `TraceStore`; now also optional detector fields (state, model id, ms, boxes). `job_id`, `evidence_path` placeholders. See [TRACEABILITY_PLAN.md](TRACEABILITY_PLAN.md) |
| PLC communication | SOFTWARE ONLY (simulator) | NOW | `plc/`: Modbus ASCII client, `PLCService` (single owner, M2 trigger events, at-most-once PASS/REJECT, REJECT ack = M2 cleared because the ladder holds M1, X0 bottle accounting). FAKE-PLC tests pass against a scan emulation of the 09:06 ladder; simulator reads/faults verified; **09:06 ladder not yet run on the simulator (simulator had no program)**. Physical PLC untested. See `PLC_COMMUNICATION.md` 0.1 / K |
| Mock PLC | SOFTWARE ONLY | NOW | `plc.test_simulation.FakePLC` + `FakeLadder`: Modbus server + scan-by-scan emulation of the decoded ladder (nets 1-7, presets configurable), used by every PLC / machine-cycle / GUI self-test |
| Delta PLC | HARDWARE REQUIRED | NOW | Ladder decoded from `final_year.isp` (09:06 save: 7 nets, T0 K150 / T1 K50 vs stated K15 / K5). Physical PLC not connected; serial link untested |
| Conveyor | HARDWARE REQUIRED | NOW | Y1 read and shown (never written). Speed not measured: `conveyor_mm_s` = 0 until it is |
| Reject actuator | HARDWARE REQUIRED | NOW | Y0 read and tracked per REJECT bottle (never written; the PLC times it). Cylinder never fired on hardware |
| Timing engine | SOFTWARE ONLY | NOW | Time-based FIFO in `machine_cycle.py`: travel = distance / speed (else T0), `scheduled_reject_time`, REJECT dispatched at `scheduled - T0` on a deadline loop, late REJECT not fired (FAULT + alarm), Y0 observed vs schedule. **No distances or speeds measured yet** |
| Production counters | SOFTWARE ONLY | NOW | Production tab: Total / PASS / REJECT / FAULT / queue / not inspected / missed rejects; per-bottle row in `projects/<slug>/production/<date>.csv` |
| Evidence capture | NEXT | NEXT | `evidence_path` field reserved; no images saved |
| Job / recipe | DEFERRED | FUTURE | Today: one project = dataset + ROI + thresholds + active model |
| Model registry | DEFERRED | FUTURE | Today: timestamped checkpoint folders with metrics; GUI "Use" switches the active model |
| Dataset versioning | DEFERRED | FUTURE | Today: `labels.csv` copy per checkpoint; `schema_version` in Stage 2 JSON; hashes in `MODEL_PROVENANCE.json` |
| Traditional vision | DEFERRED | FUTURE | Only Sobel (ROI calibration) and Laplacian (sharpness) are used |
| Geometry measurement | FUTURE | FUTURE | No measurement in the code; shape defects are learned, not measured |
| Lighting configuration | HARDWARE REQUIRED | FUTURE | Not represented in software |
| Auto annotation | DEFERRED | FUTURE | |
| Active learning | DEFERRED | FUTURE | |
| Anomaly detection | DEFERRED | FUTURE | |
| OCR | DEFERRED | FUTURE | |
| Barcode | DEFERRED | FUTURE | |
| SQLite | DEFERRED | FUTURE | Deliberately not built; see [TRACEABILITY_PLAN.md](TRACEABILITY_PLAN.md) |
| Dashboard | PARTIAL | NOW | Production tab in the desktop app is the operator view (PLC, cameras, AI task/model, last bottle, counters, FIFO/history, alarms). Not a web dashboard; no login/reports |
| User login | DEFERRED | FUTURE | |
| Reports | DEFERRED | FUTURE | |
| Alarm management | DEFERRED | FUTURE | FAULT states exist; no alarm handling |
| Maintenance | DEFERRED | FUTURE | |
| Remote monitoring | DEFERRED | FUTURE | |
| Closed-loop retraining | DEFERRED | FUTURE | |
