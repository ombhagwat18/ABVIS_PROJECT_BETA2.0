# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A bottle-inspection system for a QC conveyor (first target: 250 ml bottles), in two halves:

- **Vision / data tooling** — a CustomTkinter desktop app (`gui.py`, ~3k lines) to label
  images, manage defect classes, train a multi-label classifier (Stage 1), annotate boxes and
  polygons, and run live multi-camera inspection with PASS / REJECT / FAULT verdicts. A
  YOLOv8n component detector (Stage 2) is trained and available as an opt-in runtime path.
- **Machine path** — `plc/` talks Modbus ASCII to a Delta DVP PLC (so far only the ISPSoft
  simulator), and `decision.py` + `machine_cycle.py` turn a PLC trigger into one decision per
  bottle and a timed PLC command. **Nothing has been run on the physical machine**; every
  "tested" in this repo means a software self-test with fakes.

Git repo on `main`; `.gitignore` is whitelist-style (see below).

`app.py` (FastAPI) + `index.html` are an earlier browser-based version of the app: dead
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
python machine_cycle.py      # full cycle: FAKE PLC emulating the decoded ladder + fake cameras/detector
python stage2_dataset/seg_pipeline.py --selftest
python gui.py --selftest     # builds every real tab (incl. Machine against a fake PLC), no device I/O

# plc/ is a package: these MUST be run with -m (running the file directly -> ImportError)
python -m plc.test_simulation          # protocol, addressing, write policy, FAULTs (fake PLC)
python -m plc.test_service             # PLCService + FakeLadder contract
python -m plc.commissioning --selftest
python -m plc.handshake_test --fake
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
`docs/superpowers/specs/2026-08-09-multi-project-design.md` for the full
consequences table.

Only `labels.csv`, `config.json` and `project.json` are git-tracked per
project (see `.gitignore`'s whitelist-style rules); images, caches,
thumbnails and model weights (~2.3 GB) are not.

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

`train.run()` ends by writing the new checkpoint's `thresholds` **and** `active_model` into the
project's `config.json`, i.e. every run promotes itself. `model_bench.py` exists to train/score
candidates without that side effect (it snapshots and restores `active.txt` and `config.json`).

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

`Camera` runs one background grab thread per source and always infers on the
**newest** frame -- a frame that arrives mid-inference is dropped and counted
(`self.dropped`), never queued. Inference is capped at ~15 Hz. Video-file
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
camera decided; it never decides. Nothing calls it yet (not the GUI, not
`machine_cycle.py`, which writes its own CSV log). In-memory
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
`train()` and every `val()`. `yolo_train.py` is an older synthetic smoke test,
not the real training script. Weights (`*.pt`) are never committed.

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
  talks to `PLCService`, never to `PLCClient`. Only M0 (PASS) / M1 (REJECT) are writable;
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
  K15 / K5 contract) are in `SYSTEM_ROADMAP/PLC_COMMUNICATION.md` §0.
  Read that before touching anything PLC-related, and keep its evidence labels (VERIFIED /
  USER-STATED / INFERRED; FAKE / SIMULATOR / PHYSICAL) honest.

### GUI (`gui.py`, ~3k lines)

Single `App(ctk.CTk)` with a `CTkTabview`. On-screen order (`App.TABS`): Label, Defects,
Train, Analysis, Live, **Machine**, **Production**, Camera, Data health, **Annotate**, Settings. That differs
from the in-file class order (MachineTab, ProductionTab, LabelTab, DefectsTab, TrainTab, LiveTab, DataTab,
AnalysisTab, BenchTab, SettingsTab; `AnnotationTab` lives in `annotation_studio.py`) — don't
assume file position implies UI position.

`App` owns one `PLCService` (`self.plc`), built from `settings.json` `plc_*` keys by
`plc_link()`; it auto-connects only to the simulator, never to a serial port. `MachineTab`
shows PLC I/O and sends **operator-armed test** PASS/REJECT commands only. `ProductionTab`
runs the inspection line (`machine_cycle`) on the same `PLCService` and `CameraSet`: Start
line stops the Live tab's cameras, project switches stop the line first, and the selftest
drives it against `FakeLadder` (restoring `settings.json`, which Start line writes).
Two EMEET Nova 4K on one USB 2.0 hub give only one 1080p stream (measured); see
`SYSTEM_ROADMAP/HARDWARE_INTEGRATION.md`. The app calls `gc.disable()` and runs GC only from `App.pump()` on
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

Documented in full in `PLAN.md`; summarized here so new code doesn't
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

- `PLAN.md` — original design doc: data shape, the five bugs above in full
  detail, rationale for every non-obvious choice (input size, ROI
  measurement, scene-based split, deliberately skipped features like
  bounding boxes, auth, or Docker training — see its closing note: "add any
  of these when the constraint that rules them out stops being true").
- `SYSTEM_AUDIT_2026-10-03.md` — whole-system audit (findings ranked by severity, doc drift,
  recommended order of work). Written before `decision.py` / `machine_cycle.py` existed, so
  its "camera verdict is not connected to the PLC" finding is partly addressed in software.
- `SYSTEM_ROADMAP/PLC_COMMUNICATION.md` — the PLC contract, decoded ladder, simulator
  measurements and fault matrix. `SYSTEM_ROADMAP/VISION_DATASET.md` — dataset readiness
  for classification / detection / segmentation (`python vision_data.py build|export-cls|validate`).
- `SYSTEM_ROADMAP/` — the current, maintained description of the project:
  what exists (`CURRENT_SYSTEM.md`), what is in scope now (`CURRENT_SCOPE.md`),
  what is deliberately deferred (`FUTURE_ENHANCEMENTS.md`), the target
  architecture, the phased plan, per-feature status, hardware and traceability.
  Start there, and keep `FEATURE_STATUS.md` honest.
- `docs/superpowers/specs/2026-08-09-multi-project-design.md` — design
  rationale for the multi-project layout described above, including the
  full folder-rename/delete consequences table.
