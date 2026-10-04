# Future Enhancements

Everything here is **deliberately deferred** until the physical inspection machine is proven (one bottle sensed,
inspected, decided, communicated to the PLC and correctly rejected -- see [CURRENT_SCOPE.md](CURRENT_SCOPE.md)).

These items are **not abandoned**. They are the path from "one working machine" to an industrial platform. They
are postponed because building them first would mean polishing software around a machine that does not yet
work, and because several of them (history, reports, recipes) are only meaningful once real inspections exist.

**None of this is implemented.** Where a small foundation exists it is named.

## Database and persistence

- **SQLite / production persistence** for inspection records, counters and history.
  Foundation: in-memory `InspectionRecord` / `TraceStore` with `to_dict()`/`from_dict()` ready for storage.
  See [TRACEABILITY_PLAN.md](TRACEABILITY_PLAN.md).

## User interface

- **Advanced industrial dashboard**: production counts, live reject statistics, alarm banner, operator workflow.
  (The current Tk GUI is a development and labelling tool.)

## Security

- Login, users, **roles and permissions**.
- Configuration and recipe protection; audit trail of changes (today, moving a threshold slider rewrites
  `config.json` immediately with no record).
- Model deployment approval.

## Vision

- **OCR** (e.g. batch/date codes) and **barcode / QR**.
- **Anomaly detection** (defects not seen in training).
- **Traditional vision tools**: thresholding, edge, blob analysis, contours, pattern matching, color analysis.
- **Geometry measurement** (fill level, cap/label position and skew as measured values, not only classified).

## AI workflow

- **Auto annotation** and **AI-assisted annotation**.
- **Active learning** and a **human review queue** in the main GUI (a standalone reviewer exists in
  `stage2_dataset/review_app.py`).
- A **retraining loop** from production data.
- Segmentation models (annotation and export code exist; no data or model).

## Platform

- **Multiple jobs / recipe management**: product, classes, cameras, lighting, ROI/calibration, model, rules,
  timing, PLC and reject configuration as one versioned object. (Today: one project.)
- **Model registry**, **dataset versioning**, **deployment approval and rollback**. (Today: timestamped
  checkpoints with metrics, GUI rollback, hashes recorded in `models/stage2_yolo/MODEL_PROVENANCE.json`.)
- **Remote monitoring**; cloud or SaaS features.

## Production

- Production history, **reports**, **alarms** management, maintenance information, **analytics**.

## Hardware-side items that follow the first working cycle

- Hardware camera trigger, cross-camera frame synchronization, lighting control, camera health with automatic
  reconnect, fault latching and reset workflow.

## Why this order

The same rule applies to every item: *prove the minimum machine first*. Each deferred feature is listed in
[FEATURE_STATUS.md](FEATURE_STATUS.md) with status `DEFERRED` or `FUTURE`, and Phase 9 of
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) is where they re-enter.
