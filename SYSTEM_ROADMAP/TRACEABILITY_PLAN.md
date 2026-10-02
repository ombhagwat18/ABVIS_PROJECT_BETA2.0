# Traceability Plan

Traceability today means **software-level inspection records**. It is a data contract and an in-memory store,
not a production traceability system. No database is planned until the machine cycle works.

## 1. Current: `InspectionRecord` and `TraceStore` (`inspection_trace.py`)

`InspectionRecord.from_inspection(camera.inspection())` builds a frozen record. `TraceStore` is a bounded
(default 1000), thread-safe, newest-first in-memory history. The trace layer **records** what the camera
concluded; it never decides PASS/REJECT/FAULT.

### Fields and whether they are populated today

| Field | Meaning | Today |
|---|---|---|
| `inspection_id` | `<run_id>-<camera_id>-<session>-<seq>` for PASS/REJECT (the scored frame); `...-F<n>` for FAULT. Unique across camera restarts and app launches | Populated |
| `run_id` | Random token per process launch (session/seq restart from 1 every launch) | Populated |
| `timestamp` | Wall-clock epoch seconds, for humans/reports only, never freshness | Populated |
| `state` | PASS / REJECT / FAULT, from `infer.py` | Populated |
| `decision` | What is to be done with the bottle | Populated, but **always equal to `state`** -- no decision policy exists yet |
| `camera_id` | Source id (e.g. `"0"`) | Populated |
| `session_id` | Camera session counter (increments on each start) | Populated |
| `frame_seq`, `frame_ts` | Newest captured frame: per-session sequence and monotonic stamp | Populated (`None` before the first frame) |
| `result_seq`, `result_ts` | The frame that was actually scored | Populated for PASS/REJECT; `None` for FAULT |
| `reasons` | Why FAULT (tuple) | Populated for FAULT |
| `hits` | Defects at/above threshold | Populated for REJECT |
| `model_id` | Checkpoint stamp that produced the score (for FAULT: the model loaded, or `None`) | Populated |
| `processing_ms` | Inference time of the scored frame (`perf_counter`); **not** capture-to-result latency | Populated for PASS/REJECT; `None` for FAULT |
| `project_id` | Active project slug | Populated |
| `job_id` | Job/recipe id | **Placeholder -- always `None`** (no job system) |
| `evidence_path` | Saved evidence image | **Placeholder -- always `None`** (no images are saved) |

Unavailable information is `None` / empty; values are never invented.

### Limits of the current implementation

- **In-memory only:** nothing survives a restart; evicted records are gone (`TraceStore.evicted` counts them).
- **No persistence, no database, no evidence images, no production history, no counters, no reports.**
- **Not wired into the application:** the GUI does not create records yet. A producer must record each new
  `result_seq` exactly once -- the store does not deduplicate.
- Timestamps use `time.monotonic()`, which ticks every ~15.6 ms on Windows.
- Tested with a fake capture and model, not with the real camera or a real checkpoint end to end.

## 2. Next (small, in-memory/file level)

- Create records from the live inspection path, once per scored result (and for FAULT transitions).
- Associate the detector/classifier model and version (including the YOLO checkpoint checksum) with each record.
- Evidence association: save the triggering frame for REJECT/FAULT and fill `evidence_path`.
- Simple production counters (total / pass / reject / fault) derived from records.
- Per-bottle inspection ids once an inspection window exists (Phase 2 of [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)).

## 3. Future (deferred until the machine is proven)

- Persistent storage and queryable history (SQLite is the likely choice -- **not built, deliberately**).
- Reports, dashboards, alarm history.
- Audit trail of configuration and model changes; traceability from a rejected bottle back to model, dataset
  and training run.

## Training-side traceability (already in place, as files)

`models/stage2_yolo/MODEL_PROVENANCE.json` records the chain *dataset -> annotations -> split -> YOLO export ->
training configuration -> checkpoint (sha256) -> validation -> test*, with hashes of `annotations.json`,
`split.json`, `scene_map.json` and the export tree. Together with `stage2_dataset/` scripts this lets a future
engineer answer which data produced which model and how it was evaluated.
