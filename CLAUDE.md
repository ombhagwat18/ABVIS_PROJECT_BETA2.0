# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A desktop app (`gui.py`, CustomTkinter, 1917 lines) that trains and runs a
multi-label defect classifier for bottles on a QC line: label images,
manage defect classes, train an EfficientNet-B0 classifier, and run live
multi-camera inspection with PASS/FAIL verdicts. Not a git repository (no
`.git`), though a `.gitignore` exists for if/when one is initialized.

`app.py` (FastAPI, 433 lines) + `index.html` (530 lines) are an earlier
browser-based version of the same app. Confirmed dead code: nothing in the
repo imports `app.py` or references `index.html`, and `run.bat` never
launches them. Don't extend them without checking with the user first —
see the "web vs. desktop" open decision in `PROGRESS.md`.

## Running it

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # or /cpu
pip install -r requirements.txt

python calibrate.py       # first run only: measures the crop ROI -> projects/<slug>/config.json
python gui.py              # the desktop dashboard
```

`requirements.txt` covers `opencv-python`, `customtkinter`, `pillow`, and
optional `psutil` (without it the CPU/RAM readings in the Camera tab's
performance panel are blank; everything else still works). `torch`/
`torchvision` are installed separately because of the CUDA-vs-CPU wheel
choice.

`run.bat` does all of the above automatically: checks Python is on PATH,
detects an NVIDIA GPU (`Get-CimInstance Win32_VideoController`) to pick the
CUDA or CPU wheel, runs `migrate.py "OM Bottle" --run` if an old `All
Datasets/` layout is found, runs `calibrate.py` if no ROI is configured
yet, then launches `python gui.py` (not `pythonw.exe`, so tracebacks stay
visible in the console).

`python train.py --epochs 25` trains from the terminal instead of the Train
button in the GUI.

## Tests

There is no separate test suite — every module is its own self-check, run
directly:

```bash
python dataset.py            # demo() + project_demo(): CSV/crop/scene-split correctness
python train.py --demo
python infer.py
python calibrate.py --demo
python charts.py
python bench.py
python migrate.py --demo
python gui.py --selftest     # builds every real tab against the real dataset, no device I/O
```

Run the self-check(s) for any module you touch before considering a change
done. `gui.py --selftest` (implemented at `gui.py:1836`, entry point at
`gui.py:1913`) is the closest thing to an integration test — constructs the
real `App`, cycles every tab in `App.TABS`, exercises `LabelTab` selection
modes, `AnalysisTab.draw`/`draw_defect` with and without metrics history
(including a simulated pre-history checkpoint), pushes font scale to both
extremes and back, builds/tears down `LiveTab` video panes for two fake
cameras without opening real devices, and feeds `BenchTab` fake rows
(success and error cases) — all without touching real hardware.

## Architecture

### Everything funnels through `dataset.py`'s module globals

`dataset.py` (899 lines) holds ~13 module-level path constants (`D.ROOT`,
`D.IMAGE_ROOT`, `D.LABELS_CSV`, `D.MODELS`, `D.CONFIG_JSON`, etc., defined
around `dataset.py:19-38`), all `Path()` by default. `use_project(name)`
(`dataset.py:53`) rebinds every one of them to point at `projects/<name>/...`
and clears the config cache (`dataset.py:121-142`, since inference reads
thresholds from `config.json` on every scored frame — up to ~15/sec per
camera — and only re-parses the file when its mtime changes). Every other
module reads these as `D.NAME` **at call time** (attribute lookup on the
module), so switching the active project requires touching only
`dataset.py` — `gui.py`, `train.py`, `infer.py` and `calibrate.py` all
follow for free. Don't thread a project object through call signatures;
extend this pattern instead. `use_project` also persists the choice to
`projects/active.txt`, which is what the app reopens on next launch
(`_boot()`, `dataset.py:106`, runs at import time).

App-wide UI preferences (font scale, label-grid page size, camera probe
depth, bench seconds, monitor refresh Hz) live in a separate top-level
`settings.json`, not per-project — see `D.load_settings`/`save_settings`
(`dataset.py:149-171`).

### `labels.csv` is the source of truth, not the folders

One row per image (`path, <defect columns...>, reviewed`), one column per
defect (multi-label — a bottle can be skewed *and* have a damaged label).
`GOOD` = every defect column is `0`; there is no `good` column, so "good"
can never contradict a defect flag. `reviewed=0` marks an image nobody has
looked at yet, distinct from "reviewed and good" — unreviewed images are
excluded from training so an unlabelled capture can't silently teach
"good bottle."

Folder position only **seeds** a label the first time an image is imported.
`sync_new_images` (`dataset.py:245`) reconciles folders into the CSV on
every load but **never rewrites an existing row** — this is deliberate: a
folder can express one defect per image, the model needs several, and a
naive re-import would silently erase manually-added second labels. This
exact scenario (a hand-added `water_level` flag on a `Tilt Cap`-folder
image surviving a rescan) is asserted in `dataset.project_demo()`.

Exact folder → CSV mapping (`implied()`, `dataset.py:206`):
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
(`_span`, `calibrate.py:22`) to survive a lighting gradient and a table
line crossing every frame. The final ROI is the union of the middle 95% of
boxes measured across ~60 sample images (`calibrate.py:54`), so one bad
frame can't blow the box out to full-frame size.

Input is **192×448** (`D.INPUT_WH`, tall), not a square: the ROI is tall
and narrow, and letterboxing it into a square would waste over half the
frame on black padding — exactly where the meniscus line (`water_level`)
and label print (`damaged_label`) live. `dataset.py`'s `crop` / `letterbox`
/ `center_crop` / `model_input` / `cached_crop` (`dataset.py:504-625`) are
the single code path used by *both* training and live inference —
`model_input(full_bgr, cfg, wh)` is the canonical entry point: crop →
letterbox to `cache_wh` → center-crop to `input_wh`, and `dataset.demo()`
asserts this is pixel-identical to what training's `cached_crop` +
center-crop produces. Don't add a second crop/resize path; route any new
consumer through `dataset.model_input`. The on-disk crop cache is **PNG**,
not JPEG — JPEG re-encoding was measured to shift `water_level`'s predicted
probability by ~0.15.

### Training

`train.py` (398 lines): EfficientNet-B0 backbone (torchvision,
ImageNet-pretrained, `build()` at `train.py:65`), one linear multi-label
head, `BCEWithLogitsLoss` with per-column `pos_weight` (`clip(neg/pos, 1,
50)` from training-split counts) to handle class imbalance. Augmentation
(`_aug`, `train.py:50`) does small rotation/scale jitter and brightness/
contrast jitter but **no horizontal flip** — flipping would mirror label
text and invert skew direction, corrupting the `skewed_label`/
`skewed_bottle`/`damaged_label` classes.

Split is **by detected scene** (`dataset.scene_map`/`split_train_val`,
`dataset.py:630-704`), not random and not fixed-size blocks — the dataset
is consecutive video frames of near-duplicate bottles, so a random split
leaks near-duplicates between train/val and reports fake accuracy.
`scenes.json` caches the grouping; delete it to force a rescan.

A defect column with zero training positives gets an unreachable threshold
(`1.01`, since sigmoid outputs are ∈ [0,1]) and is disabled outright rather
than allowed to fire on an unconstrained logit; a defect with training
images but zero *validation* positives is left live but reported as
"unscored" rather than given a fake 0.000/1.000 score (`evaluate()`,
`train.py:83`).

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

### Inference (`infer.py`, 416 lines)

`Model` (`infer.py:77`) loads a checkpoint and always crops using the ROI/
input size **baked into that checkpoint**, not the project's live config —
this is what makes rollback safe. `verdict(probs, thresholds)`
(`infer.py:117`) is simple threshold logic: any defect probability at or
above its configured threshold (default `0.5` if unconfigured) fails the
bottle.

`Camera` (`infer.py:122`) runs one background grab thread per source and
always infers on the **newest** frame — a frame that arrives mid-inference
is dropped and counted (`self.dropped`), never queued, so latency can't
build an unbounded backlog. Inference is capped at ~15 Hz regardless of
camera FPS. Video-file sources are paced to their native FPS and loop on
EOF.

`CameraSet` (`infer.py:292`) manages N simultaneous cameras for
multi-camera live inspection, keyed by `str(source)` so re-adding a source
replaces rather than duplicates it, and shares one `Model` instance across
all cameras. `combined()` (`infer.py:360`) is the fusion rule: **REJECT if
ANY camera rejects** — rationale being that multiple angles of the same
bottle share a verdict (a defect only one angle can see is still a
defect); if cameras are meant to watch independent lines instead, use each
camera's own verdict rather than `combined()`.

### GUI (`gui.py`, 1917 lines)

Single `App(ctk.CTk)` (`gui.py:53`) with a `CTkTabview`. **Note:** the
on-screen tab order (`App.TABS`, `gui.py:101`: Label, Defects, Train,
Analysis, Live, Camera, Data health, Settings) differs from the in-file
class definition order (LabelTab, DefectsTab, TrainTab, **LiveTab**,
**DataTab**, **AnalysisTab**, **BenchTab**, SettingsTab) — don't assume
file position implies UI position.

Tk is not thread-safe: widgets are touched only on the main thread.
Background work (training, thumbnail decode, camera capture) reports back
through `App.q`, a `queue.Queue` drained every 60ms by `App.pump()`
(`gui.py:120`) via `self.after`. Use `App.run_bg(work, done)`
(`gui.py:136`) or `App.post(fn)` (`gui.py:133`) for any new background
operation rather than touching widgets from a thread directly. Every tab
class takes `(app, parent)`, stores `self.app`, and exposes a `refresh()`
called uniformly by `App.reload()` — follow this pattern for new tabs.
Switching projects (`switch_project`/`new_project`, `gui.py:148-159`)
always stops cameras first, since a running camera thread must not keep
scoring frames against a project it no longer belongs to.

Tab reference:
- **LabelTab** (`gui.py:198`) — paginated thumbnail grid (default 60/page),
  filter by mode (all/inbox/good/defective/pos/neg) or filename search,
  multi-select with keyboard shortcuts (digits 1-9 toggle a defect, `g`
  marks good, `a` selects page, `Esc` clears), bulk apply/clear/delete, all
  routed through `D.apply_labels`/`D.delete_images` in the background.
- **DefectsTab** (`gui.py:442`) — add/rename/delete defect classes, upload
  images into a class's `-ve/` folder or into `+ve/`, shows per-class
  WITH/WITHOUT counts and a "thin — aim for 100+" warning.
- **TrainTab** (`gui.py:601`) — epoch count + "Train now" (spawns a thread
  calling `train.run(epochs, log=emit)`), a live log, per-defect metrics
  table (VAL+/THR/PREC/RECALL/F1/MISSED) with a false-positive/negative
  mistake gallery, and a model-version list with per-checkpoint "Use"
  (`activate()`, `gui.py:726`, which also derives `thresholds` from that
  model's metrics).
- **AnalysisTab** (`gui.py:1303`) — loss/macro-F1 curves, per-defect
  recall bars, a confusion matrix for a chosen defect, and a
  "re-test every model on today's labels" comparison table
  (`compare()`/`show_compare()`) that distinguishes checkpoints needing
  retrain (predate the current val split) from honestly re-scored ones —
  all drawn via `charts.py`.
- **LiveTab** (`gui.py:772`) — multi-camera live dashboard: camera
  discovery/picker, video grid with PASS/FAIL overlay per camera plus a
  combined verdict (`app.cams.combined()`), a real-time performance panel
  (FPS/latency/drop% via `Camera.stats()` + `bench.system_stats()`), a
  snapshot-into-a-label capture panel, and per-defect threshold sliders
  bound live to `config.json`.
- **DataTab** ("Data health", `gui.py:1150`) — ROI editing (manual entry +
  "Re-measure" via `calibrate.run()`, "Preview crop" side-by-side), a
  near-duplicate-frames scan (`dataset.scene_map`) explaining how many
  distinct bottles N images actually represent, and dataset health counts.
- **BenchTab** ("Camera" tab, `gui.py:1552`) — sweeps a camera through
  resolution/FPS combos via `bench.benchmark_camera`, shows
  requested-vs-actual FPS/latency/jitter/sharpness/inference-time, and
  highlights `bench.best_combo()`'s recommended setting (picked by
  **sharpness**, not raw FPS — a defect the camera can't resolve is
  invisible at any frame rate). Guards against running concurrently with
  LiveTab's cameras.
- **SettingsTab** (`gui.py:1712`) — font scale, camera probe depth, bench
  duration, monitor refresh rate, label-grid page size; persisted via
  `D.save_settings`. Shows read-only paths for the active project.

`bgr_to_ctk()` (`gui.py:46`) is the shared OpenCV-BGR-frame → `CTkImage`
helper used across Label/Train/Live/Data/Bench tabs. `train` and
`calibrate` are imported lazily inside the functions that need them
(`TrainTab.start`, `AnalysisTab.compare`, `DataTab.recalibrate`) to keep
GUI startup fast.

### Charts without matplotlib (`charts.py`, 207 lines)

Draws `line_chart`, `bar_chart`, and `confusion` (a single defect's 2×2
matrix — multi-label means there's no single combined N×N matrix)
directly on a Tk `Canvas`, deliberately avoiding matplotlib (a 40MB
dependency with its own event loop and theme conflicts) for what's just a
few chart types over a few hundred points. Exposes color constants
(`INK, DIM, LINE, BLUE, RED, GREEN, AMBER`, etc.) that `gui.py` uses
directly for consistent theming, e.g. recall bars colored green/amber/red
by threshold.

### Camera/system benchmarking (`bench.py`, 225 lines)

`benchmark_camera()` measures what a camera setting *actually delivers*
(OpenCV reports the requested mode, not the achieved one, especially under
auto-exposure) — real wall-clock FPS, latency, jitter, and frame quality
(`sharpness` = variance of Laplacian, `frame_quality` = brightness/
contrast/sharpness/%clipped). `system_stats()` samples CPU/RAM/GPU,
degrading gracefully (`None`, not `0.0`) when `psutil`/`pynvml` aren't
installed or there's no CUDA device — a dashboard silently claiming 0% GPU
on a GPU-less machine would be a lie.

### Migration (`migrate.py`, 173 lines)

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
- `PROGRESS.md` — current feature status against
  `Bottle_Defect_Detection_Web_Dashboard_Architecture.docx`, and the two
  open decisions that gate a chunk of remaining work:
  1. **Web vs. desktop** — the architecture doc specifies a web dashboard;
     the actual app is desktop (`gui.py`). `app.py`/`index.html` are the
     unused web version.
  2. **YOLO/detection vs. classifier** — the doc specifies YOLOv8 with
     bounding boxes and mAP; the current app is a whole-image classifier.
     Switching requires bounding-box annotation that doesn't exist yet and
     is only worth it if a frame could hold more than one bottle or defect
     *location* matters.
  Check this before assuming a doc-spec'd feature (YOLOv8, bounding boxes,
  mAP) should be added — it's blocked on a decision, not forgotten.
- `docs/superpowers/specs/2026-08-09-multi-project-design.md` — design
  rationale for the multi-project layout described above, including the
  full folder-rename/delete consequences table.
