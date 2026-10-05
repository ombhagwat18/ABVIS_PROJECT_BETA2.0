# Future Enhancements

*Status 2026-10-06.* Items that are still **not built**, deferred until the physical machine is proven (one bottle sensed,
inspected, decided, communicated to the PLC and correctly rejected -- see [CURRENT_SCOPE](CURRENT_SCOPE.md)). They are
postponed, not abandoned.

## Already built (no longer future)

SQLite production record, evidence pictures, History / Database pages and CSV / report export, shift reports, coded
alarms, event logs, model registry with gated activation and rollback, recipe editor, auto-annotation proposals with
active-learning queues, operator / engineer modes, engineer PIN. See [FEATURE_STATUS](FEATURE_STATUS.md).

## Still future

### Security
- Login, named users, **roles and permissions** (today: an engineer PIN, a convenience lock only).
- Audit trail of configuration changes (moving a threshold or editing a recipe rewrites a file with no change log
  beyond the deployment log and the saved previous thresholds).
- Recipe / configuration protection, change control.

### Vision
- **OCR** (batch / date codes) and **barcode / QR**.
- **Anomaly detection** (defects not seen in training; useful for missing / rare defects).
- **Traditional vision tools** (thresholding, edge, blob, pattern matching, colour) and **geometry measurement**
  (fill level, cap / label position and skew as measured values).
- Segmentation models (annotation and export code exist; no data, no model).

### AI workflow
- A **retraining loop** from production data (evidence pictures already collected).
- Foundation-model-assisted labelling.
- Dataset versioning beyond the hashes in the provenance files.

### Platform
- **Multi-job / product wizard** with one versioned object per product (classes, cameras, lighting, ROI, model, rules,
  timing, PLC settings).
- **Remote monitoring**, cloud or SaaS features.
- Maintenance schedules, analytics beyond the current day / shift report.

### Machine side (needs ladder AND software changes together)
- **PLC protocol v2**: several bottles in flight, no trigger masking during a reject
  ([PLC_LADDER_REQUIREMENTS](../hardware/PLC_LADDER_REQUIREMENTS.md) section 3.4).
- **PC <-> PLC heartbeat watchdog** and PLC-side answer timeout.
- **Encoder** (the `PositionSource` abstraction is ready; `EncoderPositionSource` refuses to exist until one is installed).
- Hardware camera trigger, cross-camera frame synchronisation, lighting control.

## Why this order
Prove the minimum machine first. Each item is listed in [FEATURE_STATUS](FEATURE_STATUS.md) with status `DEFERRED` or `FUTURE`.
