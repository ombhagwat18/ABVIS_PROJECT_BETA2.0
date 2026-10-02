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
| Detection | PARTIAL | NOW | YOLOv8n trained and test-evaluated offline; not loaded by the runtime |
| Segmentation | PARTIAL | DEFERRED | Annotation/export code exists; no polygons annotated, no model |
| Annotation Studio | COMPLETE | NOW | Boxes + polygons, review flag, YOLO detection/segmentation export |
| YOLO training | COMPLETE | NOW | YOLOv8n only: 53 epochs, best epoch 33; YOLOv8s not trained |
| YOLO runtime | NEXT | NOW | Phase 1: load `stage2_best.pt` behind the existing Camera/Inspection contract |
| Camera manager | PARTIAL | NOW | Index discovery, DirectShow, driver defaults. No exposure/gain/focus, reconnect or hardware trigger |
| Multi-camera | SOFTWARE ONLY | NOW | Threads + shared model + FAULT > REJECT > PASS fusion |
| Camera synchronization | FUTURE | NEXT | Cross-camera frame alignment does not exist |
| Trigger sensor | HARDWARE REQUIRED | NOW | Photoelectric sensor not represented in software |
| Bottle tracking | NEXT | NOW | Nothing knows that frames belong to one physical bottle. Sensor-triggered capture may replace full tracking |
| Inspection window | NEXT | NOW | Not started |
| Temporal voting | NEXT | NOW | Not started |
| Decision engine | PARTIAL | NOW | Per-frame thresholds only; no rules for detections, no fault latching |
| PASS / REJECT / FAULT | SOFTWARE ONLY | NOW | Fail-safe states with freshness checks; software-tested with fake captures |
| Traceability | PARTIAL | NOW | In-memory `InspectionRecord` / `TraceStore`; `job_id`, `evidence_path` placeholders. See [TRACEABILITY_PLAN.md](TRACEABILITY_PLAN.md) |
| PLC communication | NEXT | NOW | No integration code in the application |
| Mock PLC | NEXT | NOW | Not started; `plc file/delta_sim_test.py` is only a manual sender |
| Delta PLC | HARDWARE REQUIRED | NOW | ISPSoft project exists but cannot be read here; I/O mapping **unverified** |
| Conveyor | HARDWARE REQUIRED | NOW | Not represented in software |
| Reject actuator | HARDWARE REQUIRED | NOW | Not represented in software |
| Timing engine | NEXT | NOW | No distances, speeds or actuator timings recorded; needs measurements |
| Production counters | NEXT | NEXT | Needs persisted records first |
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
| Dashboard | DEFERRED | FUTURE | The existing Tk GUI is a development tool, not a production dashboard |
| User login | DEFERRED | FUTURE | |
| Reports | DEFERRED | FUTURE | |
| Alarm management | DEFERRED | FUTURE | FAULT states exist; no alarm handling |
| Maintenance | DEFERRED | FUTURE | |
| Remote monitoring | DEFERRED | FUTURE | |
| Closed-loop retraining | DEFERRED | FUTURE | |
