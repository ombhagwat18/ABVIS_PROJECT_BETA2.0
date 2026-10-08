# Traceability

*Status 2026-10-06.* The production record is **persistent and implemented** (software-tested; not yet run on the machine).

## 1. What is recorded today

| Where | What | Code |
|---|---|---|
| `projects/<project>/production/production.db` (SQLite) | **runs** (each START: job, product, models, recipe fingerprint, SIMULATOR / REAL PLC, line settings), **inspections** (one row per bottle: result, defects, confidence, command and PLC answer, timings, per-camera frame windows, evidence path, full record as JSON), **alarms** (raise / acknowledge / clear) | `production_store.py` |
| `projects/<project>/production/evidence/<date>/` | first frame + overlay per bottle, by the evidence policy (NONE / ALL / REJECT_ONLY / FAULT_ONLY / REJECT_AND_FAULT / SAMPLE:n) | `production_store.py` |
| `projects/<project>/production/<date>.csv` | daily CSV (kept) | `machine_cycle.py` |
| `logs/<channel>.log` | event logs: app, camera, ai, plc, machine, alarm, production | `applog.py` |
| `models/deployments.jsonl`, `models/model_status.json` | model activation history, who validated / approved what and why | `model_registry.py` |
| `models/stage2_yolo/MODEL_PROVENANCE.json` + candidate records | dataset -> annotations -> split -> export -> training -> checkpoint (sha256) -> test | `yolo_stage2_train.py`, `model_bench.py` |
| `projects/<project>/label_log.csv` | every label edit, reversible | `dataset.py` |

Every bottle can be traced to the run (job, product), the models and recipe version, the PLC mode, and the picture.
Read and export it on the **Database** page: [DATABASE](../guides/DATABASE.md).

## 2. Older in-memory trace (`inspection_trace.py`)

`InspectionRecord` / `TraceStore` (bounded, thread-safe, in memory) is the earlier frame-level record. The production
line does **not** use it; it uses `production_store`. It remains as a tested data contract.

## 3. Gaps (honest)

- No batch / lot number, operator name or shift owner in the record (job and product are typed on the Production page).
- No audit trail of configuration edits (thresholds, recipe, settings).
- No user login; the engineer PIN is not a security control.
- Evidence is the first frame used, not every frame; there is no image retention / clean-up policy besides the
  free-disk check (stops writing below 1 GB).
- The record has not been exercised against a physical line; field names may need to change after commissioning.
