# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A bottle-inspection system for a QC conveyor (first target: 250 ml bottles), in two halves:

- **Vision / data tooling** — a CustomTkinter desktop app (`gui.py` ~5k lines + `hmi.py`) to label
  images, manage defect classes, train a multi-label classifier (Stage 1), annotate boxes and
  polygons, and run live multi-camera inspection with PASS / REJECT / FAULT verdicts. A
  YOLOv8n component detector (Stage 2) is trained and available as an opt-in runtime path.
- **Machine path** — `plc/` talks Modbus ASCII to a Delta DVP-SS2 PLC (the ISPSoft simulator, and since
  2026-10-04 a first read-only link to the real PLC), and `decision.py` + `machine_cycle.py` turn a PLC trigger
  into one decision per bottle and a timed PLC command. **The full inspect -> M0/M1 -> reject cycle has not been
  run on the physical machine**; every "tested" in this repo means a software self-test with fakes unless
  `docs/roadmap/PLC_COMMUNICATION.md` labels it PHYSICAL.
- **Real PLC link gotchas** (COM5, RS-232, ASCII 9600 7E1, station 1): Delta **COMMGR keeps the COM port open
  while its driver exists**, even with ISPSoft offline, so the app gets "Access is denied" -- delete the COMMGR
  driver/exit it first (one program per port). A running `gui.py` rewrites `settings.json` `plc_com` from its
  Machine-tab dropdown (it once reverted COM5 to a Bluetooth COM10), so stop the app before editing it. The
  saved ladder still has T0 K150 / T1 K50 (15 s / 5 s) -- `plc_t0_s` must equal what is actually in the PLC.
  Net 1 is `X1 -> SET Y1` (latched), so the conveyor runs until X2; software cannot write Y1 by design.

Git repo on `main`; `.gitignore` is whitelist-style (see below).

`legacy/web_dashboard/` (`app.py` FastAPI + `index.html`) are an earlier browser-based version of the app: dead
code (nothing imports them, `run.bat` never launches them, and `app.py`'s write endpoints
have no auth). The desktop app is the application. Don't extend them; ask before deleting.

## Running it

`run.bat` does the full setup and launch (picks the CUDA or CPU torch wheel, migrates an old
`All Datasets/` layout, runs `calibrate.py` if no ROI is set, then `python gui.py` — kept as
`python.exe` so tracebacks stay visible). Manually: `python calibrate.py` once (measures the crop
ROI -> `projects/<slug>/config.json`), then `python gui.py`; `python train.py --epochs 25` trains
from the terminal. `torch`/`torchvision` are installed separately (CUDA-vs-CPU wheel); `psutil` is
optional (without it the Camera tab's CPU/RAM readings are blank). Not in `requirements.txt` but
imported lazily where needed: `ultralytics==8.1.0` (every YOLO path), `pyserial` (the "Real PLC
(serial)" link), `comtypes` (camera names). Missing ones fail only that feature, not startup.

Dev environment: Windows 11, **Python 3.8** (keep `from __future__ import annotations`; no
3.9+ syntax at runtime), torch 2.4.1+cu124, RTX 3050 4 GB, 16 GB RAM. Shell is PowerShell /
Git Bash; there is no lint or format config.

`projects/active.txt` decides which project the app (and `gui.py --selftest`) opens. The
labelled Stage 1 data is `om_bottle`; `bottle_detection` is a near-empty detection project.


## Tests

There is no separate test suite — every module is its own self-check, run
directly:

```bash
python dataset.py            # demo() + project_demo(): CSV/crop/scene-split correctness
python train.py --demo
python infer.py              # tri-state PASS/REJECT/FAULT, freshness, frame metadata, restart safety
python inspection_trace.py   # InspectionRecord, ids, TraceStore, live Camera -> record
python detect.py             # detector contract/validation (fakes) + real-checkpoint sanity check
python calibrate.py --demo
python charts.py
python bench.py
python migrate.py --demo
python vision_data.py        # unified dataset prep: class mapping, cross-folder scene merge, leak fix, box checks
python decision.py           # per-bottle decision rules, frame vote, camera fusion
python segment.py            # segmentation runtime interface (fake model; no weights exist yet)
python autoannotate.py       # model box proposals: kept out of boxes/export until accepted
python machine_cycle.py      # full cycle: FAKE PLC emulating the decoded ladder + fake cameras/detector
                             #   (incl. staggered camera stations, association fault, SQLite record)
python tracking.py           # time-based position source (no encoder), camera stations, association, calibration
python machine_state.py      # the one machine state + start checklist
python alarms.py             # coded alarms: dedup, condition vs event, acknowledge
python production_store.py   # SQLite runs / bottles / alarms, evidence policy
python model_registry.py --selftest   # CANDIDATE -> VALIDATED -> APPROVED -> ACTIVE gates, rollback (temp dir)
python applog.py             # structured event logs: 7 channels, search
python stage2_dataset/split_v3.py --selftest   # v3 split (missing-cap scene in train), input untouched
python stage2_dataset/seg_pipeline.py --selftest
python gui.py --selftest     # builds every real tab (incl. Machine against a fake PLC), no device I/O

# plc/ is a package: these MUST be run with -m (running the file directly -> ImportError)
python -m plc.test_simulation          # protocol, addressing, write policy, FAULTs (fake PLC)
python -m plc.test_service             # PLCService + FakeLadder contract
python -m plc.commissioning --selftest
python -m plc.handshake_test --fake
python -m plc.ladder_check --selftest  # ISPSoft .isp decoder + 13 ladder requirement checks
python -m plc.ladder_check             # READ-ONLY report on plc file/final_year/final_year.isp vs settings.json
python -m plc.ladder_sim               # runs that ladder in a scan simulator vs the software contract (S1-S14)
python -m plc.ladder_sim --selftest
python selfcheck.py [--full]           # every module self-test in its own process (the GUI "Simulation check" button)
python verdict.py                      # stable one-verdict-per-bottle tracker
python production_export.py            # database rows in words, CSV, printable report
python calibrate_thresholds.py [--apply|--restore]   # thresholds from validation, checked on test
```

There is no single-test runner: each file runs all its checks; to run one, import the module
and call that check function. `python -m plc.handshake_test --real` and
`python -m plc.sim_comm_test` need the ISPSoft simulator running (Modbus TCP on
127.0.0.1:10002) and write M0/M1/M2 there.

Stage 2 data checks: `stage2_dataset/validate_yolo_export.py` runs 19 export
checks but **writes `yolo_export/EXPORT_REPORT.md`** when run; to avoid touching
the export, load it as a module and point `REPORT_PATH` somewhere else. All of
the above are software self-checks (fake captures, fake models); none is a
hardware test.

**Known stale self-test:** `stage2_dataset/review_app.py --selftest` asserts that
every row of `review_manifest.csv` is still `unreviewed`. The review is finished
(606 reviewed: 594 KEEP, 12 REJECT), so it now fails by design -- it is not a
regression, and it is intentionally left unmodified.

Run the self-check(s) for any module you touch before considering a change
done. `gui.py --selftest` (`selftest()` in `gui.py`) is the closest thing to an integration test — constructs the
real `App`, cycles every tab in `App.TABS`, exercises `LabelTab` selection
modes, `AnalysisTab.draw`/`draw_defect` with and without metrics history
(including a simulated pre-history checkpoint), pushes font scale to both
extremes and back, builds/tears down `LiveTab` video panes for two fake
cameras without opening real devices, and feeds `BenchTab` fake rows
(success and error cases) — all without touching real hardware.

## Architecture

### Everything funnels through `dataset.py`'s module globals

`dataset.py` holds ~13 module-level path constants (`D.ROOT`,
`D.IMAGE_ROOT`, `D.LABELS_CSV`, `D.MODELS`, `D.CONFIG_JSON`, etc.), all `Path()` by
default. `use_project(name)` rebinds every one of them to point at `projects/<name>/...`
and clears the config cache (inference reads thresholds from `config.json` on every
scored frame — up to ~15/sec per camera — so `load_config` only re-parses the file when
its mtime changes). Every other
module reads these as `D.NAME` **at call time** (attribute lookup on the
module), so switching the active project requires touching only
`dataset.py` — `gui.py`, `train.py`, `infer.py` and `calibrate.py` all
follow for free. Don't thread a project object through call signatures;
extend this pattern instead. `use_project` also persists the choice to
`projects/active.txt`, which is what the app reopens on next launch
(`_boot()`, runs at import time).

App-wide UI preferences (font scale, label-grid page size, camera probe
depth, bench seconds, monitor refresh Hz) live in a separate top-level
`settings.json`, not per-project — see `D.load_settings`/`save_settings`.

### `labels.csv` is the source of truth, not the folders

One row per image (`path, <defect columns...>, reviewed`), one column per
defect (multi-label — a bottle can be skewed *and* have a damaged label).
`GOOD` = every defect column is `0`; there is no `good` column, so "good"
can never contradict a defect flag. `reviewed=0` marks an image nobody has
looked at yet, distinct from "reviewed and good" — unreviewed images are
excluded from training so an unlabelled capture can't silently teach
"good bottle."

Folder position only **seeds** a label the first time an image is imported.
`sync_new_images` reconciles folders into the CSV on
every load but **never rewrites an existing row** — this is deliberate: a
folder can express one defect per image, the model needs several, and a
naive re-import would silently erase manually-added second labels. This
exact scenario (a hand-added `water_level` flag on a `Tilt Cap`-folder
image surviving a rescan) is asserted in `dataset.project_demo()`.

Exact folder → CSV mapping (`implied()`):
- `+ve/x.jpg` → no defect columns set, `reviewed=1` (good).
- `-ve/<Class>/x.jpg` → that column `=1`, `reviewed=1`.
- `-ve/x.jpg` (dropped straight in `-ve/`, no class subfolder) → no columns
  set, `reviewed=0` — goes to inbox, **never** straight to good.
- `_inbox/x.jpg` → `reviewed=0`.

```
projects/<slug>/
  project.json          display name as typed
  images/
    +ve/                 good — every column 0
    -ve/<Class>/          one sub-folder per class == one labels.csv column
    _inbox/               unlabelled: reviewed = 0
  labels.csv   config.json   models/<stamp>/   cache/   thumbs/   scenes.json
projects/active.txt      which project slug the app opens on
```

Adding a `-ve/<NewClass>/` folder adds a defect column with no code change.
Renaming a class must rename its `-ve/` folder too — `rename_defect`
(`dataset.py`) does both together, or a stale folder can resurrect the old
label on the next rescan. Deleting a class drops the column and moves its
images to `_inbox/` (`reviewed=0`) rather than "good" — a folder can never
resurrect a deleted column as a pass. See
`docs/design/2026-08-09-multi-project-design.md` for the full
consequences table.

Only `labels.csv`, `config.json` and `project.json` are git-tracked per
project (see `.gitignore`'s whitelist-style rules); images, caches,
thumbnails and model weights (~2.3 GB) are not.

**Label edits are logged and reversible (`dataset.py`).** `apply_labels`, `set_labels`
(exactly this set of defects) and `delete_images` append one row per changed image to
`projects/<slug>/label_log.csv` (`time,user,batch,action,path,before,after`; the `labels.csv`
schema is unchanged, since any extra column there would be read as a defect). `delete_images`
moves files to `projects/<slug>/trash/<batch>/` (outside `images/`, so a rescan never re-imports
them); `undo(batch)` / `last_undoable()` restore labels and trashed files. Model pre-labels live in
`cache/suggestions.json` (`load/save/drop_suggestions`) and are **never** written to `labels.csv`:
an image stays `reviewed=0`, i.e. out of training, until a person accepts or corrects it.

**Importing another product's dataset:** `plan_import(folder)` + `run_import(folder, mapping)`
copy a folder already sorted one sub-folder per class into a project (`good/ok/pass...` -> GOOD,
`raw/unsorted...` -> inbox, anything else -> a defect column named by `slug()`). GUI:
`ImportDialog` ("Import dataset...", into a new product or the current one). A new object also
needs its own ROI (`calibrate.py` / Data health -> Re-measure) and a retrain.

### Crop / resize pipeline, and why it's tall not square

Fixed camera, black background → the bottle ROI is measured once by
`calibrate.py` and stored in `config.json` as `roi` (`[x,y,w,h]` in
absolute pixels) + `roi_frame` (the `[W,H]` of the frame it was measured
on, so a differently-sized capture rescales the ROI instead of silently
cropping the wrong region — historical bug, see below). Measurement uses
edge energy (Sobel gradient magnitude), not a brightness threshold, because
the bottle is transparent and nearly as dark as the backdrop; the
column/row profile is thresholded **relative to its own floor/peak**
(`_span`) to survive a lighting gradient and a table
line crossing every frame. The final ROI is the union of the middle 95% of
boxes measured across ~60 sample images, so one bad
frame can't blow the box out to full-frame size.

Input is **192×448** (`D.INPUT_WH`, tall), not a square: the ROI is tall
and narrow, and letterboxing it into a square would waste over half the
frame on black padding — exactly where the meniscus line (`water_level`)
and label print (`damaged_label`) live. `dataset.py`'s `crop` / `letterbox`
/ `center_crop` / `model_input` / `cached_crop` are
the single code path used by *both* training and live inference —
`model_input(full_bgr, cfg, wh)` is the canonical entry point: crop →
letterbox to `cache_wh` → center-crop to `input_wh`, and `dataset.demo()`
asserts this is pixel-identical to what training's `cached_crop` +
center-crop produces. Don't add a second crop/resize path; route any new
consumer through `dataset.model_input`. The on-disk crop cache is **PNG**,
not JPEG — JPEG re-encoding was measured to shift `water_level`'s predicted
probability by ~0.15.

### Training

`train.py`: EfficientNet-B0 backbone (torchvision,
ImageNet-pretrained, `build()`), one linear multi-label
head, `BCEWithLogitsLoss` with per-column `pos_weight` (`clip(neg/pos, 1,
50)` from training-split counts) to handle class imbalance. Augmentation
(`_aug`) does small rotation/scale jitter and brightness/
contrast jitter but **no horizontal flip** — flipping would mirror label
text and invert skew direction, corrupting the `skewed_label`/
`skewed_bottle`/`damaged_label` classes.

Split is **by detected scene** (`dataset.scene_map`/`split_train_val`), not random and not fixed-size blocks — the dataset
is consecutive video frames of near-duplicate bottles, so a random split
leaks near-duplicates between train/val and reports fake accuracy.
`scenes.json` caches the grouping; delete it to force a rescan.
`split_train_val_test` additionally carves out a held-out **test** set (`test_paths` in
`metrics.json`, scored only by `train.evaluate_test()`, never used to pick an epoch or
threshold). Caveat: `scene_map` groups frames *within one class folder* only; `vision_data.py`
found 403 cross-folder same-scene pairs and builds a leak-free split, but existing
checkpoints used the per-folder one. Validation F1 is saturated (~1.0) on every checkpoint —
judge models on the test set (`FINAL_YEAR_BLACKBOOK/.../HELD_OUT_TEST_RESULTS.md`).

`train.run()` **no longer promotes itself** (since 2026-10-05): `activate=None` reads
`settings.json` `auto_activate_trained_model` (default false), so a new checkpoint is a CANDIDATE and
`config.json` is untouched. Activation goes through `model_registry.py` (Models page): CANDIDATE ->
VALIDATED (needs a held-out test result + a written real-camera validation) -> APPROVED -> ACTIVE; the
previous model is ARCHIVED, `models/deployments.jsonl` logs it, `rollback()` re-activates it. Classifier
activation writes `active_model` + that checkpoint's thresholds; detector activation sets
`detector_weights` after checking the candidate's sha256 (and writes a `MODEL_PROVENANCE.json` beside it,
which `detect.verify_checkpoint` reads). `python train.py --activate` keeps the old CLI behaviour.
`model_bench.py` trains/scores candidates (it snapshots and restores `active.txt` and `config.json`).

A defect column with zero training positives gets an unreachable threshold
(`1.01`, since sigmoid outputs are ∈ [0,1]) and is disabled outright rather
than allowed to fire on an unconstrained logit; a defect with training
images but zero *validation* positives is left live but reported as
"unscored" rather than given a fake 0.000/1.000 score (`evaluate()`).

Checkpoints under `projects/<slug>/models/<stamp>/` contain `model.pt`
(state dict + `arch`, `defects` order, `input_wh`, `cache_wh`, `roi`,
`roi_frame` all baked in), a copy of `labels.csv` as of that run, and
`metrics.json` (per-epoch history, per-defect precision/recall/F1,
`val_paths` — the exact validation set used, so a checkpoint can be
honestly re-scored later without ever leaking training images into
validation). `train.score_checkpoint()` refuses to score a checkpoint that
never recorded `val_paths`. Rolling back to an older model always feeds it
the crop/ROI it was trained on, regardless of the project's current
`config.json`, because `infer.Model` reads those fields from the
checkpoint itself, not from live config.

### Inference (`infer.py`) and the safety model

**The default runtime inference is the Stage 1 classifier.** The Stage 2 YOLOv8n
detector is trained and test-evaluated (see below) and has an opt-in, *observational*
runtime path: `detect.py` (`YoloDetector`, `Detection`, `DetectionResult`) plugged in
as `Camera.detector` and chosen in the Live tab ("Classifier" / "Classifier + YOLO" /
"YOLO only"). It finds bottle/cap/label **components, not defects**, and never produces
PASS/REJECT -- do not turn a missing box into a defect (absence can be occlusion,
angle, blur, lighting or a false negative). A detector failure or stale detection makes
the inspection FAULT. Box coordinates are absolute pixels in the original whole frame
(no ROI, no resize). Class names are read from the model; `YoloDetector(require=...)` refuses a
model lacking a class the project's inspection recipe needs. `DEV_CONF = 0.25` is a development threshold; configure it with
`settings.json` `detector_conf`. Weights are checked against the sha256 in
`MODEL_PROVENANCE.json` and are never downloaded.

`Model` loads a checkpoint and always crops using the ROI/input size **baked
into that checkpoint**, not the project's live config -- this is what makes
rollback safe. `verdict(probs, thresholds)` is the original boolean threshold
rule (kept; `train.py` uses it). `decide(probs, thresholds)` is the runtime
version: **PASS / REJECT / FAULT**, where an empty or non-finite (NaN) score is a
FAULT, never a PASS. `PASS`, `REJECT`, `FAULT` are defined once, in `infer.py`;
import them, don't redefine them.

Camera driver controls live in `settings.json` `camera_controls` (`{"<index>": {"focus": ..,
"exposure": .., "wb_temperature": .., "rotate": 90}}`): `open_capture` applies them (auto modes
off first) and records what the driver read back in `infer.applied_controls`; `rotate` is applied
in the grab thread before the frame gets its seq, so every consumer sees the same rotated frame.

`Camera` runs one background grab thread per source plus a separate scoring thread
(`_infer_loop`) that always scores the **newest** frame -- a frame that arrives mid-inference
is skipped and counted (`self.dropped`), never queued. Scoring used to run inline in the grab
loop, which stalled capture for the length of every inference; each result still carries the
seq/capture time of the frame it scored, so freshness and the FAULT rules are unchanged. Inference is capped at ~15 Hz. Video-file
sources are paced to their native FPS and loop on EOF. Every captured frame
gets a `time.monotonic()` stamp (`frame_ts`) and a per-session sequence number
(`frame_seq`, from 1; `session` counts `start()`s, so `(session, seq)` never
repeats). `latest_frame()` returns a `Frame(camera_id, ts, seq, image)`.
**Wall-clock time is never used for freshness.** On Windows `time.monotonic()`
ticks every ~15.6 ms, so adjacent frames can share a timestamp -- `seq` is the
strict order.

`Camera.inspection()` returns an `Inspection` (state, hits, reason, frame/result
seq + ts, model_id, infer_ms); `Camera.result()` is the same thing as the
original 3-tuple `(state, hits, reason)`. **PASS/REJECT only come from a fresh,
valid score.** Everything else is FAULT with a reason: not started, thread
dead, driver error, no model, failed inference (the thread survives and drops
its scores), nothing scored yet, or a frame/score older than
`MAX_RESULT_AGE_S` (1.0 s). A FAULT clears itself on the next good frame -- there
is no latching yet. Each camera thread is pinned to its own stop event so a
wedged driver that wakes after a restart cannot write into the new session.

`CameraSet` manages N cameras (keyed by `str(source)`, one shared `Model`).
`combined()` returns `(state, detail)` with precedence **FAULT > REJECT > PASS**:
PASS only if every *armed* camera has a fresh PASS; no armed camera at all is a
FAULT. (If cameras watch independent lines, use each camera's own result.)

`inspection_trace.py` turns an `Inspection` into an `InspectionRecord` and
holds a bounded, thread-safe in-memory `TraceStore`. It records what the
camera decided; it never decides. Nothing calls it yet: the line's persistent record is
`production_store.py` (below) plus the daily CSV `machine_cycle.py` writes. In-memory
only: no persistence, no evidence images, no database. `job_id` and
`evidence_path` are always `None`; `decision` currently equals `state`. Do not
name a module `trace.py` -- it shadows the standard library (it was renamed for
that reason).

### Stage 2 detection pipeline (training offline; runtime opt-in)

`stage2_dataset/` holds the 594-image detection dataset (classes bottle, cap,
label; 1,994 boxes; scene split 25/7/7 scenes = 418/89/87 images). Its
provenance -- scripts, `annotations.json`, `split.json`, `scene_map.json`,
manifests, audit reports -- IS tracked. The images (`clean/`, `source/`) and the
generated `yolo_export/` are NOT; rebuild the export with
`stage2_dataset/export_yolo.py`. `yolo_export/data.yaml` has an absolute path;
`stage2_dataset/data.yaml` is the portable copy (pass it to Ultralytics as an
absolute path). Don't edit the export in place: `training_metadata.json` records
its tree hash.

`yolo_stage2_train.py` trains candidates on train only, selects on val, then
evaluates test once. Result and the full chain back to the data are in
`models/stage2_yolo/MODEL_PROVENANCE.json`. **Windows gotcha:** Ultralytics
`val()` defaults to 8 dataloader workers (~500 MB each); on a 16 GB machine
that exhausted the paging file and hung runs. Always pass `workers=` to *both*
`train()` and every `val()`. `legacy/yolo_train_smoke_test.py` is an older synthetic smoke test,
not the real training script. Weights (`*.pt`) are never committed.

**Missing cap (found 2026-10-05; read before adding data or retraining).** All 22 user missing-cap images
(`All Datasets/Missing Cap/snap001-022`, white background, unlabelled bottles, Iriun-viewer screenshots) are
ALREADY in Stage 2: snap013-022 = neck close-ups = train scene 38 (img_635-644); snap001-012 = full bottle =
part of test scene 22 (img_328-341; scene 22 is the whole 35-frame white-background session). No detector
ever trained on a full-bottle bare neck, so v1 n / v1 s / v2 n all call the green tamper ring a "cap" (conf
~0.70) and score 0/12 on missing cap (`models/stage2_yolo/defects_*_test.json`). **v3**
(`stage2_dataset/split_v3.py`, `split_v3.json`, `yolo_export_v3/`; `model_bench.py yolo --data v3`,
`det-defects --split-version v3`) = v2 boxes + scene 22 moved test -> train (whole scene, audit rule):
test drops to 52 images / 6 scenes and has NO missing-cap bottle, so a v3 model's missing-cap ability can
only be validated on the real machine. The CLASSIFIER must not get these images: one bottle on a white
background vs a black-background dataset -> candidate `20261005-171833` learned "white = missing cap"
(25/25 capped white bottles flagged at 1.00, `shortcut_check.json`), so it is REJECTED in the registry and
the 12 copies were removed from `om_bottle` with `dataset.delete_images` (undo batch `20261005-172406-105`).
The classifier needs missing-cap bottles photographed in ITS setup (black enclosure, labelled Bisleri).

Segmentation (label outline) is in progress and has **no trained model**:
`stage2_dataset/seg_pipeline.py` (seed → annotate in Annotation Studio → export → validate →
train) keeps polygons in its own `seg_annotations.json`, because `annotations.json` is
hash-pinned in `MODEL_PROVENANCE.json` and must stay frozen. A seeded polygon is unreviewed
and is never exported. `segment.py` is the runtime interface; without weights it raises
`SegmenterError`, so any inspection that requires segmentation is FAULT.

### Machine path: PLC trigger → per-bottle decision → PLC command

```
X0 photo-eye -> ladder SET M2 -> PLCService Trigger
  -> machine_cycle.Inspector: frames captured AFTER the trigger, every line camera
  -> AI stage(s) -> decision.decide() -> one PASS / REJECT / FAULT per bottle
  -> software FIFO keyed by inspection_id, each bottle with its own deadline
  -> PASS: M0 now;  REJECT: M1 at (trigger + travel_time - T0)   -> PLC times Y0 itself
  -> daily production CSV
```

- **`plc/`** — `protocol.py` (Modbus ASCII frames, LRC), `address_map.py` (the single device
  map and write policy), `client.py` (`PLCClient` + `TcpTransport` for the simulator /
  `SerialTransport` for the real PLC), `service.py`. **`PLCService` is the one owner of the
  link**: one worker thread does every transaction; application code (GUI, machine cycle)
  talks to `PLCService`, never to `PLCClient`. Only M0 (PASS) / M1 (REJECT) are writable (exception: the
  opt-in commissioning `PLCService.operator_write` -- M10/M11 conveyor pulses and M2 "virtual bottle", off
  unless `settings.json` `plc_operator_controls` is true; see `docs/guides/BENCH_TEST_DECISION_ENGINE.md`);
  Y outputs are never written — the ladder owns conveyor, reject delay and pulse.
  `simulator_test_write` (X0/X1/X2/M0/M1/M2 stimuli) refuses any non-loopback transport.
- **Command rules you must not weaken:** one answer per trigger; the trigger is marked
  answered *before* the write; nothing is ever retried (an unknown outcome is reported as
  `WRITE_FAILED` / `NOT_ACKED` / `ACK_LOST`); preconditions (M2=1, M0=M1=0) are re-read right
  before the write; success = the PLC clears the command bit and M2 (M1 is held through the
  reject cycle, `HELD_UNTIL_DONE`). A trigger seen again after reconnect is a new trigger
  flagged `after_reconnect`.
- **`decision.py`** — pure functions, no I/O. Each AI stage turns one frame into findings;
  detection findings follow the project's **inspection recipe** (`config.json` `"inspection"`:
  `anchor` class + `parts` with `required` / `search` / `zone`; absent = `default_recipe()`, the
  bottle/cap/label rule). A new product needs data + a detector + a recipe, not code. `Inspector`
  and `--bench` load the recipe; `LiveTab.build_detector` validates it (`recipe_problems`).
  `decide()` votes the findings across the bottle's frames (`majority` default) and fuses cameras
  FAULT > REJECT > PASS. A missing detection box becomes a defect only through the vote,
  and "no bottle found" is FAULT. `RULES` thresholds are development defaults, overridable
  via `settings.json` `decision_rules`.
- **Safety latch (software layer only; the hardware E-stop must cut power by itself):** `MachineCycle.halt(reason)`
  latches; while halted no M0/M1 is ever sent, each trigger becomes a FAULT bottle "NOT ANSWERED", scheduled REJECTs
  are cancelled. Halts come from the Production tab STOP button, a hardware E-stop input (`estop_device`, e.g. `X3`;
  `estop_active_high` false = NC contact; an unreadable input counts as pressed), `fault_latch_after` consecutive
  FAULT bottles (default 3), or a crashed cycle. `reset()` is refused while the E-stop input still reads pressed.
- **`autoannotate.py`** — detector proposals live under `"proposals"` in `annotations.json`, never in `"boxes"`, so
  `annotate.export_yolo_*` cannot export them; accept moves them to `boxes` (`source: "auto"`), the image stays
  `reviewed: false`. Annotate tab: Propose boxes / Accept / Reject / Next: least sure / Next: missing part
  (bottle found, cap or label not: likely real missing-component captures) / Next: doubtful part (cap or label
  proposed at 0.25-0.80: hard-negative candidates such as a tamper ring). Queues only reorder unreviewed images.
- **PLC ladder requirements** — `plc/ladder_check.py` decodes the user's ISPSoft `.isp` (binary header of
  varying length, then raw deflate; header searched for) and checks 13 requirements (trigger, handshake,
  reject cycle, T0/T1 == settings, M10/M11 operator bits, E-stop input, Y0 interlock, trigger masking, answer
  timeout, heartbeat). `docs/hardware/PLC_LADDER_REQUIREMENTS.md` explains each and gives the rungs to add.
  As of 2026-10-05 the file (and every backup since 2026-10-03 20:24) is the 7-network ladder; R8-R13 are
  absent / mismatched. Read-only: never write the ladder. `plc/ladder_sim.py` interprets the decoded networks
  in a virtual-time scan simulator (SET/RST/OUT/TMR/CNT, rising-edge contacts) and runs scenarios S1-S14 against
  the software contract; it does NOT read series/parallel wiring (several contacts per network = AND, flagged,
  OR-dependent scenarios SKIP). `hmi.SimulationCheckDialog` (Production engineer row + Machine page) runs it and
  `selfcheck.run_all` (code check) side by side: ladder FAIL = ladder / settings wrong, self-test FAIL = code wrong.
- **`machine_cycle.py`** — `Inspector` + `MachineCycle`: one deadline-driven thread. Every
  bottle ends with exactly one final result. FAULT is physically rejected by default
  (`fault_action: "REJECT"`). A REJECT that would miss its deadline is not fired late (answered
  with M0, recorded as FAULT, alarm to remove by hand). Line settings (`line_cameras`,
  `inspect_frames`, `inspection_to_reject_mm`, `conveyor_mm_s`, `plc_t0_s`, …) live in
  `settings.json`; distance/speed of 0 means "not measured" and T0 is used as travel time.
  `timing_problem()` blocks Start line when T0 >= travel (every REJECT would be late) or T0 > 5 s
  unmeasured (the saved K150 ladder).
  Driven from the GUI's **Production** tab. Line cameras run capture-only; the Inspector runs
  the models per bottle on frames stamped strictly *after* the trigger (`>`, not `>=`: the
  coarse Windows clock otherwise lets in a frame of the previous bottle).
- **Time-based tracking, no encoder (`tracking.py`)** — `PositionSource` is the seam (`TimePositionSource`
  in use; `EncoderPositionSource` raises until an encoder exists). `settings.json` `camera_stations`
  (`{"<source>": {name, role, offset_mm, side, rules: {station_x, judge}}}`) places each line camera
  `offset_mm` downstream of the photo-eye; its frame window is `trigger + offset / conveyor_mm_s`. The
  Inspector collects **non-blocking** (`begin` / `poll` / `evaluate`; the loop keeps serving triggers and
  deadlines), bounds a camera's frames by the next bottle's window, and refuses a downstream camera's
  evidence as `CAMERA_ASSOCIATION_FAULT` when another bottle was sensed closer than window + 2 x uncertainty
  (`speed_tolerance_pct`). Per-camera `rules` reach `decision.decide(per_camera=...)` (`judge` = recipe
  parts that camera judges). `machine_cycle.line_problems(cfg, cameras)` = `timing_problem` + saved speed
  calibration + `tracking.station_problems`; any problem blocks Start. No distance/speed is hardcoded:
  0 = not measured. The speed calibration wizard (`hmi.SpeedCalibrationDialog`) saves `conveyor_mm_s` and a
  timestamped `speed_calibration` record.
- **One machine state (`machine_state.py`)** — `state(facts)` -> OFFLINE / NOT_READY / READY / INITIALIZING /
  RUNNING / INSPECTING / STOPPING / FAULT / E_STOP / COMMUNICATION_FAULT, `readiness(facts)` the start
  checklist. `ProductionTab.facts()` gathers cached facts; the banner and the LINE lamp both show
  `ProductionTab.mstate`. Never derive a machine state in a widget.
- **Alarms and record** — `alarms.AlarmManager` (one per App, `app.alarms`): coded alarms (CATALOG: severity,
  message, action), condition (`set_condition`) vs event (`raise_`), RESET FAULT = `acknowledge_all`.
  `MachineCycle._alarm(text, code, key)` raises them. `production_store.ProductionStore` (`app.store()`, per
  project, `projects/<slug>/production/production.db`): `runs` (job, product, recipe hash, model ids, mode,
  settings), one `inspections` row per finished bottle, `alarms`; evidence images by `evidence_policy` on a
  bounded background writer (never delays a PLC command). The full frame is dropped from memory after
  recording. `stop()` finishes bottles still INSPECTING / SCHEDULED as FAULT (never silently dropped).
- **Bottle accounting** — with `PLCService(watch_x0=True)` (the Production tab sets it) X0 is
  polled before M2; an X0 rise with no M2 rise since the previous X0 read is a
  `BOTTLE_UNTRIGGERED` event and becomes a FAULT "NOT INSPECTED" bottle. The REJECT ack is
  "M2 cleared" because the ladder holds M1 through T0 + T1 (`address_map.HELD_UNTIL_DONE`).
- **`FakeLadder`** (`plc/test_simulation.py`) is a scan-by-scan emulation of the decoded 09:06
  ladder. It commits each scan atomically and serves X from a scan-latched image; keep it that
  way, since mid-scan reads produced phantom triggers. Bottles are queued behind a clear beam
  (`gap_s`). Timing tests on it must allow ~15.6 ms Windows sleep jitter.
- **The ladder** (`plc file/final_year/final_year.isp`; not encrypted: a 0xAA-byte header,
  then a raw-deflate stream, `zlib.decompressobj(-15)`) is owned by the user, not this repo — never change it. Its decoded nets, simulator
  behaviour and known limits (one-bottle M0/M1/M2 handshake; triggers masked while M1 is held;
  the 09:06 save fixed the old one-scan Y0 flash but still has T0 K150 / T1 K50 against the stated
  K15 / K5 contract) are in `docs/roadmap/PLC_COMMUNICATION.md` §0.
  Read that before touching anything PLC-related, and keep its evidence labels (VERIFIED /
  USER-STATED / INFERRED; FAKE / SIMULATOR / PHYSICAL) honest.

**One verdict, in words (`verdict.py`).** Operators never see per-frame scores. `Camera._feed_tracker` feeds a
`VerdictTracker` once per scored frame (classifier hits + the recipe's missing-part findings from the detector,
presence = the recipe anchor box); the tracker ignores the first frames, needs `min_frames` valid frames, keeps a
defect only if it is in >= `vote` of them, then LATCHES GOOD / DEFECT until the bottle has been absent
`absent_frames` frames. No detector => presence unknown => rolling window, no latch. Invalid frames x `fault_frames`
=> FAULT, never GOOD. `Camera.overlay_frame` draws only banner + coloured border unless `Camera.details`
(Live "Engineer details") is on. Display words: PASS -> GOOD, REJECT -> DEFECT (`verdict.shown_result`), defect
names via `verdict.pretty`; the PLC / DB / code keep PASS / REJECT / FAULT. The line's decision is still
`decision.py`; the tracker only DISPLAYS.

**Thresholds** come from `calibrate_thresholds.py` (validation F1, middle of the perfect band, floor 0.30, recall
guard; test only reported). `--apply` keeps the old ones in `config.json` `thresholds_before_calibration`;
`--restore` undoes it. 2026-10-05 on 20260919-164511: test false alarms 28 -> 22, 1/219 defective passed, 10/19 good
test bottles still called defective: the model is limited by only 72 good training images, not by the thresholds.

**Pages scroll.** Every `NavShell` page is a `ScrollHost` (canvas + both scroll bars, min size per page in
`NavShell.MIN_SIZE`); `App._page_wheel` scrolls the page only when the nearest scrollable widget IS the page.
New pages: build into `app.tabs.tab(name)` as before.

**Database page** (`hmi.DatabaseTab`, `production_export.py`): see `docs/guides/DATABASE.md`.

**Logs (`applog.py`)** — `logs/<channel>.log` (app, camera, ai, plc, machine, alarm, production), rotating,
one line per EVENT (never per frame). Hooks: `App._on_alarm` (alarm), `PLCService` listener (plc),
`MachineCycle._alarm` / `_finish` / start / stop (machine, production, ai), `ProductionTab._camera_watch`
(camera state changes + reconnect attempts). `applog.search()` feeds the Health page viewer. Self-tests point
`applog.setup()` at their temp folder: never let a test write fake bottles into `logs/`.

### GUI (`gui.py`, ~5k lines, + `hmi.py`)

**Theme:** `theme.py` is the one palette: a **light industrial HMI** by default, the older dark one with
`settings.json` `"ui_theme": "dark"` (chosen once at import, applied at the next start, like the text size).
Grey surfaces, green/red/amber only for PASS/REJECT/FAULT, blue only for selection; camera images always
sit on the dark `VIDEO_BG`; text on a state-coloured button is `ACC_T`. `gui.py`, `hmi.py`,
`annotation_studio.py`, `charts.py` and the OpenCV overlays in `infer.py` import from it; do not add hex
colours elsewhere. `theme.apply_ctk()` also rewrites CustomTkinter's stock widget colours.

**Operator / engineer:** `App.ui_mode` (`settings.json` `ui_mode`, header button, optional `engineer_pin`).
OPERATOR shows only `App.OPERATOR_PAGES` (Production, History, Health) via `NavShell.show_only`; pages are
never destroyed. The Production page's ENGINEER row (`ProductionTab.eng`: task, cameras, timing,
calibration / layout dialogs, HALT latch, simulator feed) is hidden for the operator. New production screens
(`HistoryTab`, `HealthTab`, `ModelsTab`, the two line dialogs) live in `hmi.py`, same `(app, parent)` +
`refresh()` contract. The GUI self-test sets `app.production_dir` to a temp folder: never let a test write
into a real project's production record.

Single `App(ctk.CTk)` with a `NavShell` (left rail grouped DATA / MODEL / RUNTIME / SYSTEM, same
`add/tab/get/set` API as the `CTkTabview` it replaced) and a status bar of PLC / LINE / CAMERAS /
MODEL lamps (`App.update_lamps`, cached state only, ~2 Hz from `pump`). A label edit calls
`App.data_changed()` (re-read labels, mark Defects/Train/Data health stale, refreshed when shown)
instead of the full `reload()`. `App.open_project(name)` is the one way to switch project.
`reload()` refreshes only the visible tab and marks the others stale (`_stale`, refreshed in
`on_page` when first shown): refreshing all eleven cost ~20 s at start-up (3,000+ CustomTkinter
widgets). Per-frame / per-card widgets (Live and Production camera panes, Label cards) are plain
`tk` widgets with `ImageTk.PhotoImage`, and the Live frames are drawn on a worker thread: a CTk
widget costs ~4x as much to create/redraw. **Text size** (`settings.json` `font_scale`) is saved
at once and applied only at the next start (`App.restart`): rescaling a running window redraws
every widget (20+ s, looked frozen), and with a scrollable page on screen it also recursed
`CTkScrollbar.set` <-> `update_idletasks` (guarded in `theme._guard_scrollbar`). Don't call
`ctk.set_widget_scaling` on a built window. `App.TABS` (14): **Production**, History, Health, Label,
Defects, Train, Analysis, Models, Live, Machine, Camera, Data health, Annotate, Settings (`all_tabs()` must
list the tab objects in exactly this order; Settings stays last for the self-test). The rail groups them
(`App.GROUPS`: PRODUCTION / DATA / MODEL / ENGINEERING / SYSTEM). In-file class order differs (MachineTab,
ProductionTab, LabelTab, ... in `gui.py`; HistoryTab, HealthTab, ModelsTab in `hmi.py`; `AnnotationTab` in
`annotation_studio.py`) — don't assume file position implies UI position.

**Production start sequence** (`ProductionTab`): `start_line` saves the timing row, requires
`machine_state.state(facts) == READY`, loads models in the background, `_go` opens the line cameras
(capture only), `_check_warm` keeps the state INITIALIZING until every camera delivered a frame
(`WARMUP_S`, else CAMERA_DISCONNECTED and no start), `_launch` builds the `MachineCycle` with
`app.alarms` + `app.store()`. STOP = `stop_line` (orderly); the software HALT latch is the engineer
`halt_line`; RESET FAULT = `reset_halt` (clears the halt, acknowledges alarms). CAMERA TEST and TEST
INSPECTION (`machine_cycle.bench_inspect`, no PLC) refuse while the line runs. While running,
`_camera_watch` reopens a dead line camera in a background thread at most every `RECONNECT_S` (bottles in the
gap are FAULT, nothing else pauses). Engineer row also opens `hmi.RecipeDialog` (edits `config.json`
`"inspection"`, validated against `detect.CLASS_NAMES`, refused while the line runs). History has shift
reports (`settings.json` `shifts`, default A 06-14 / B 14-22 / C 22-06; `production_store.shift_span`,
`summary_range`).

`App` owns one `PLCService` (`self.plc`), built from `settings.json` `plc_*` keys by
`plc_link()`; it auto-connects only to the simulator, never to a serial port. `MachineTab`
shows PLC I/O and sends **operator-armed test** PASS/REJECT commands only. `ProductionTab`
runs the inspection line (`machine_cycle`) on the same `PLCService` and `CameraSet`: Start
line stops the Live tab's cameras, project switches stop the line first, and the selftest
drives it against `FakeLadder` (restoring `settings.json`, which Start line writes).
Two EMEET Nova 4K on one USB 2.0 hub give only one 1080p stream (measured); see
`docs/hardware/HARDWARE_INTEGRATION.md`. The app calls `gc.disable()` and runs GC only from `App.pump()` on
the Tk thread: automatic GC on a background thread finalised Tk objects there and froze the
PLC worker. Don't re-enable it, and don't block the Tk loop for long.

Tk is not thread-safe: widgets are touched only on the main thread.
Background work (training, thumbnail decode, camera capture) reports back
through `App.q`, a `queue.Queue` drained every 60ms by `App.pump()` via `self.after`. Use `App.run_bg(work, done)` or `App.post(fn)` for any new background
operation rather than touching widgets from a thread directly. Every tab
class takes `(app, parent)`, stores `self.app`, and exposes a `refresh()`
called uniformly by `App.reload()` — follow this pattern for new tabs.
Switching projects (`switch_project`/`new_project`)
always stops cameras first, since a running camera thread must not keep
scoring frames against a project it no longer belongs to. (`TrainTab.start` does *not*: a
camera left running through training keeps the old `Model` but picks up the new
thresholds that `train.run()` writes. `AnalysisTab.activate` does reload the model.)

Tab reference: read the `Tab` classes in `gui.py` (on-screen order is `App.TABS`). Non-obvious: the
"Camera" tab is `BenchTab` and "Data health" is `DataTab`; `BenchTab` picks the recommended
camera setting by **sharpness**, not raw FPS (a defect the camera can't resolve is invisible at
any frame rate) and refuses to run while LiveTab's cameras are open; `AnalysisTab.compare`
distinguishes checkpoints that predate the current val split (need retrain) from honestly
re-scored ones.

`bgr_to_ctk()` is the shared OpenCV-BGR-frame → `CTkImage`
helper used across Label/Train/Live/Data/Bench tabs. `train` and
`calibrate` are imported lazily inside the functions that need them
(`TrainTab.start`, `AnalysisTab.compare`, `DataTab.recalibrate`) to keep
GUI startup fast.

### Charts without matplotlib (`charts.py`)

Draws `line_chart`, `bar_chart`, and `confusion` (a single defect's 2×2
matrix — multi-label means there's no single combined N×N matrix)
directly on a Tk `Canvas`, deliberately avoiding matplotlib (a 40MB
dependency with its own event loop and theme conflicts) for what's just a
few chart types over a few hundred points. Exposes color constants
(`INK, DIM, LINE, BLUE, RED, GREEN, AMBER`, etc.) that `gui.py` uses
directly for consistent theming, e.g. recall bars colored green/amber/red
by threshold.

### Camera/system benchmarking (`bench.py`)

`benchmark_camera()` measures what a camera setting *actually delivers*
(OpenCV reports the requested mode, not the achieved one, especially under
auto-exposure) — real wall-clock FPS, latency, jitter, and frame quality
(`sharpness` = variance of Laplacian, `frame_quality` = brightness/
contrast/sharpness/%clipped). `system_stats()` samples CPU/RAM/GPU,
degrading gracefully (`None`, not `0.0`) when `psutil`/`pynvml` aren't
installed or there's no CUDA device — a dashboard silently claiming 0% GPU
on a GPU-less machine would be a lie.

### Migration (`migrate.py`)

One-shot move of the legacy single-project layout (`All Datasets/<Class>/`,
`data/`, `models/`) into `projects/<slug>/...`. Dry-run by default
(`python migrate.py` prints the plan only); `--run` actually **renames**
folders in place (not copies — ~1GB, instant on the same volume) and
rewrites `labels.csv` paths row-by-row via `remap()`, preserving every
hand-added label exactly rather than re-importing (which would lose them).
Already run for this repo's original dataset; only relevant again if
another old-layout dataset needs importing.

## Known bugs already fixed — don't reintroduce these

Documented in full in `docs/design/PLAN.md`; summarized here so new code doesn't
regress them:

1. **Random per-block split** could put both of a rare class's blocks into
   train, producing a fake `0.000` recall. Fixed by scene-based splitting
   with a per-folder quota (`dataset.split_train_val`).
2. **Fixed-size 25-frame blocks** didn't match real scene boundaries
   (actual runs are 18-31 frames), inflating several defects to a fake
   1.000/1.000. Fixed by detecting real scene runs (`dataset.scene_map`).
3. **Train/serve skew**: validation used a center-cropped cached crop while
   live inference letterboxed directly, and the cache was JPEG (shifting
   pixel values). Fixed by routing both paths through
   `dataset.model_input`/`cached_crop` and switching the cache to lossless
   PNG; `dataset.demo()` asserts pixel-identical output between the two
   paths.
4. **ROI didn't scale with frame size** — a differently-sized camera feed
   cropped the wrong region and the model read pure background confidently.
   Fixed by storing `roi_frame` alongside `roi` and rescaling in
   `dataset.crop`.
5. **A defect with zero training images could still fire** on an
   unconstrained logit. Fixed by giving such defects an unreachable
   (`1.01`) threshold in `train.py`.

## Key docs already in the repo

- `docs/PROJECT_BRIEF_FOR_REVIEW.md` - one self-contained brief (hardware, architecture, status, known problems, review
  questions) for a reviewer or an AI assistant; keep its numbers in step with FEATURE_STATUS when they change.

All documentation is indexed in `docs/README.md` (hardware, guides, roadmap, audit, design). Camera placement / line timing: `docs/hardware/CAMERA_PLACEMENT_AND_LINE_PLAN.md`; tab purposes: `docs/guides/TAB_GUIDE.md`.

- `docs/design/PLAN.md` — original design doc: data shape, the five bugs above in full
  detail, rationale for every non-obvious choice (input size, ROI
  measurement, scene-based split, deliberately skipped features like
  bounding boxes, auth, or Docker training — see its closing note: "add any
  of these when the constraint that rules them out stops being true").
- `docs/audit/SYSTEM_AUDIT_2026-10-03.md` — whole-system audit (findings ranked by severity, doc drift,
  recommended order of work). Written before `decision.py` / `machine_cycle.py` existed, so
  its "camera verdict is not connected to the PLC" finding is partly addressed in software.
- `docs/roadmap/PLC_COMMUNICATION.md` — the PLC contract, decoded ladder, simulator
  measurements and fault matrix. `docs/roadmap/VISION_DATASET.md` — dataset readiness
  for classification / detection / segmentation (`python vision_data.py build|export-cls|validate`).
- `docs/roadmap/` — the current, maintained description of the project:
  what exists (`CURRENT_SYSTEM.md`), what is in scope now (`CURRENT_SCOPE.md`),
  what is deliberately deferred (`FUTURE_ENHANCEMENTS.md`), the target
  architecture, the phased plan, per-feature status, hardware and traceability.
  Start there, and keep `FEATURE_STATUS.md` honest.
- `docs/design/2026-08-09-multi-project-design.md` — design
  rationale for the multi-project layout described above, including the
  full folder-rename/delete consequences table.
