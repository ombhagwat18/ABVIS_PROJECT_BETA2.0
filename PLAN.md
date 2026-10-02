> **HISTORICAL design document (Stage 1 classifier era).** Its rationale for the crop pipeline, scene-based split and the known-bugs list is still valid. Statements that bounding boxes / detection were "deliberately skipped" are **superseded**: a Stage 2 YOLO detection dataset and YOLOv8n model now exist (offline, not yet in the runtime). Current scope and status: [`SYSTEM_ROADMAP/`](SYSTEM_ROADMAP/README.md).

# Bottle Inspection — Plan

Live multi-label defect detection for Bisleri 250ml bottles, with a dashboard to
label images, add new defect types, and retrain.

---

## How to run

**Double-click `run.bat`.** It installs what is missing on first run (picking the
CUDA or CPU PyTorch build by looking for an NVIDIA card), measures the crop
region if it has never been measured, and opens the app.

By hand:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python calibrate.py       # measure the crop region -> projects/<slug>/config.json
python gui.py             # the desktop dashboard
```

`python train.py --epochs 25` trains from the terminal instead of the Train button.
Every module self-checks: `python dataset.py`, `python train.py --demo`,
`python infer.py`, `python calibrate.py --demo`, `python charts.py`,
`python bench.py`, `python migrate.py --demo`, `python gui.py --selftest`.

| File | Does |
|---|---|
| `dataset.py` | projects, `labels.csv` read/write, crop, cache, scene split, label/defect operations |
| `migrate.py` | one-shot move of the old single-project layout into `projects/` |
| `calibrate.py` | measures where the bottle sits, writes the ROI |
| `train.py` | fine-tune, per-defect metrics, thresholds, checkpoints |
| `infer.py` | camera thread, model, verdict, annotated frame |
| `gui.py` | the whole desktop UI (CustomTkinter), eight tabs |
| `charts.py` | line/bar/confusion drawing straight on a Tk canvas, no matplotlib |
| `bench.py` | camera benchmarking and CPU/GPU/RAM sampling |
| `run.bat` | setup-on-first-run launcher |

The UI is a desktop app. `app.py` and `index.html` are the earlier browser
version, now unused — they still work if you ever want the dashboard reachable
from another machine on the line, but nothing calls them.

Tk is not thread-safe, so `gui.py` touches widgets only on the main thread;
training, thumbnail decoding and camera capture all report back through a queue
drained by `after()`.

### Projects

Each inspection job is a folder under `projects/`, and nothing is shared between
them. The header dropdown switches; **+ New project** creates one.

```
projects/om_bottle/
  project.json          the name as typed
  images/
    +ve/                good — every column 0
    -ve/<Class>/        one sub-folder per class = one column in labels.csv
    _inbox/             unlabelled: reviewed = 0
  labels.csv  config.json  models/<stamp>/  cache/  thumbs/  scenes.json
projects/active.txt     which one the app opens on
```

**Folder position is the label**, so adding a sub-folder under `-ve/` adds a
class with no code change and no mapping table. But a folder can only say *one*
defect per image and the model is multi-label, so folders seed a row once and
`labels.csv` is the truth after that — a rescan adds new files and never
rewrites a row you edited. That rule is what stops a re-import undoing the
`tilt_cap` corrections described above. A file dropped straight into `-ve/` with
no class goes to the inbox, not to good.

`migrate.py` moves the old single-project layout in (rename, not copy).

### What's not in the repo

Gitignored: everything under `projects/<slug>/` except the small text files, plus
`sample/` — ~2.3 GB of images, video and weights. `labels.csv`, `config.json` and
`project.json` **are** tracked, and `labels.csv` keys on paths relative to
`images/`, so a fresh clone needs the images restored to the layout those paths
already name.

Then `python calibrate.py` to write `config.json` — the ROI is measured on your
camera and is not portable — and one train run to produce `models/`. Until both
exist, only the **Label** and **Defect types** tabs do anything useful.

`calibrate.py` finds the object by edge energy, which assumes something tall on a
dark backdrop. A project whose object is a different shape sets the ROI by hand
in the Data health tab, and `input_wh` in `config.json`; `cache_wh` follows it.

### First results

Trained on the RTX 4070 Ti SUPER, 3.4 s/epoch, 25 epochs. Inference 9.8 ms/frame
(~100 FPS of headroom).

```
defect             val+   thr   prec  recall     F1
damaged_bottle       72  0.15  1.000   1.000  1.000
damaged_label        58  0.70  1.000   1.000  1.000
missing_label        18  0.40  1.000   1.000  1.000
skewed_bottle        43  0.85  1.000   1.000  1.000
skewed_label         37  0.40  0.925   1.000  0.961
tilt_cap             29  0.90  1.000   0.793  0.885
water_level          31  0.30  0.886   1.000  0.939
                          macro-F1 = 0.969  over 7/7 scored defects
```

**Read these numbers with the sample size in mind.** 1145 images are ~112
distinct scenes, and validation is ~24 of them. The 1.000s mean "got every
validation scene right", and there are only three to six scenes behind most of
them. They are not wrong, they are thin. More distinct bottles is the only thing
that moves them.

**Every single validation mistake is one confusion**: 19 `Water Level` frames
flagged `tilt_cap`, and 4 the other way. Looking at the crops, the model is right
and the label is incomplete — those Water Level bottles have a visibly unseated
cap, a clear gap between the cap skirt and the neck ring. The single-label
folders could not record both defects. Tick `tilt_cap` on those frames from the
**Show me the mistakes** panel and retrain; `tilt_cap` precision should jump.

### Five bugs found by building it, worth knowing about

1. **The split gave two classes zero validation.** Flipping a coin per block put
   both of `missing_label`'s blocks in train 64% of the time, and it reported a
   meaningless `0.000` recall that read like failure. Now a per-folder quota with
   scene-sized groups, and a defect with no validation positives says so instead
   of scoring 0.
2. **Fixed 25-frame blocks did not match the real scenes.** The runs are 18–31
   frames, so scenes still straddled the split and six defects scored a fake
   1.000/1.000. The split now detects the runs and keeps each one whole — verified
   zero scenes span the split.
3. **Train/serve skew, twice.** Validation centre-cropped a cached crop while
   inference letterboxed straight to the input size; and the cache was JPEG q95,
   which moved pixels ~2 grey levels. Together they made live inference disagree
   with validation on 15 of 31 frames for `water_level` — a model that scores 0.94
   and fails on the line. Both paths now run through one function, the cache is
   lossless PNG, and a self-check asserts the two views are identical pixels.
4. **The ROI did not follow the frame size.** It is measured in absolute pixels
   on 2537×1927 images, so a half-resolution video cropped the right-hand edge of
   the picture and the model read pure background — confidently, and with a
   verdict that never changed. `config.json` now records `roi_frame`, the size the
   box was measured on, and the crop rescales to whatever frame it is given.
   This was open question 5, and it bites the moment a real camera differs.
5. **A defect with zero training images could still fire.** `missing_cap` has
   never seen a positive example, so its logit was never constrained downwards
   and it read 0.83 on unfamiliar input — handing an operator a reject reason
   that does not exist in the dataset. Any defect with no training images is now
   given an unreachable threshold and is disabled outright. No data, no claim.

### Built differently from the plan below, and why

1. **Images stay in `All Datasets/`** — not copied to `data/images/`. That avoids
   duplicating ~1 GB, and keying `labels.csv` on the relative path also removes
   the filename collision between folders (`snap_001.jpg` exists in several).
2. **A reserved `reviewed` column.** Without it, "reviewed and good" and "nobody
   has looked at this yet" are the same all-zero row, and the inbox is impossible.
   Unreviewed images are excluded from training — otherwise every unlabelled
   capture teaches the model that it is a good bottle.
3. **Input is 192×448, not 320×320.** The ROI is 853×1927 — tall and narrow.
   Letterboxing that into a square left the bottle ~142 px wide with 55% of the
   frame black. A tall input fills 97%, so the resolution lands on the meniscus
   and the label print instead of on padding.
4. **ROI is measured, not auto-segmented per frame.** `calibrate.py` finds the
   bottle by edge energy — the body is transparent and nearly as dark as the
   backdrop, so brightness thresholding fails on it. The threshold is relative to
   each profile's own baseline because the backdrop has a lighting gradient and
   the table draws a line across every row. Measured ROI: `[820, 0, 853, 1927]`.
5. **The ROI and input size are baked into each checkpoint**, so rolling back to
   an older model feeds it the crop it was trained on regardless of current config.
6. **The split is by detected scene, not fixed block** — see bug 2 above.
   `data/scenes.json` caches the grouping; delete it to force a rescan.

---

## 1. What the data actually is

1145 JPGs, 2537×1927, in `All Datasets/`. Fixed camera, black background, one
bottle centred in every frame.

| Folder | Images |
|---|---|
| Damaged Bottle | 319 |
| Damaged Label | 229 |
| Skewed Bottle | 167 |
| Tilt Cap | 145 |
| Water Level | 112 |
| Good Bottle | 72 |
| Skewed Label | 71 |
| MIssing Label *(sic)* | 30 |
| ~~Missing Cap~~ | ~~0 — empty~~ **class removed** |

Pass/fail split: **72 good, 1073 defective — a 6% pass rate.** Backwards for a QC
line, where good is normally the overwhelming majority.

### Three things that must be fixed before any training

**a) ~~`Missing Cap` is empty.~~** Resolved by dropping it: the folder and the
column are both gone, so the model no longer carries a defect it has never seen.
To bring it back, make a `Missing Cap` folder with images — the importer picks up
any new folder and names the column after it, no code change.

**b) `Good Bottle` is only 72 images.** For a QC line, "good" is normally 90% of
what the camera sees. 72 is not enough to learn what normal looks like, and the
model will be biased toward calling everything defective. Target ≥300 good frames.

**c) The images are consecutive video frames, not independent samples.**
`snap_1000.jpg` through `snap_1089.jpg` differ by a few hundred bytes each — they
are near-duplicates of the same bottle, same pose. A random train/val split puts
frame 1042 in train and frame 1043 in validation, and the model gets ~99%
validation accuracy that means nothing.

> **The split must be by contiguous frame block**, not random. Reserve whole runs
> of filenames for validation. This single decision is the difference between a
> real accuracy number and a fake one.

---

## 2. Model shape: multi-label, not multi-class

A bottle can be skewed **and** have a damaged label. Folders can only express one
label per image, so folders stop being the source of truth.

**Source of truth = `data/labels.csv`.** One row per image, one column per defect:

```csv
filename,damaged_bottle,damaged_label,skewed_bottle,skewed_label,missing_label,tilt_cap,water_level
snap_1000.jpg,1,0,0,0,0,0,0
snap_1234.jpg,0,1,0,0,0,1,0
snap_0001.jpg,0,0,0,0,0,0,0
```

- **GOOD** = every column is `0`. There is no `good_bottle` column — good is the
  absence of defects, so "good" can never contradict a defect flag.
- **"+ve / -ve" per defect** = the `1` rows and the `0` rows of that column. In
  the dashboard these are two buckets you can browse and move images between.
- **"Add a folder"** = add a defect type = add a column. Every existing image
  starts at `0` for it, and you go label the positives.

A one-time import script reads the 9 existing folders and writes the initial
`labels.csv`. After that the dashboard owns the file; the folders are just an
image store.

Output at inference:

```
snap_1234.jpg
  damaged_bottle  0.02  ✗
  damaged_label   0.91  ✓
  tilt_cap        0.88  ✓
  water_level     0.04  ✗
  ------------------------
  VERDICT: REJECT (2 defects)
```

---

## 3. Preprocessing: crop the bottle first

Black background + fixed camera means the bottle can be isolated with plain
OpenCV — threshold, largest contour, bounding box, pad, crop. ~10 lines, no model
needed.

Why it matters: it removes 60% of every frame that is empty background, and it
normalises bottle scale and position, so the network spends its capacity on the
cap, the label and the meniscus instead of on black pixels.

Resize the crop to **320×320**, not the usual 224. `water_level` is a thin
horizontal meniscus line and `damaged_label` is fine print — both disappear at 224.

Augmentation: rotation ±5°, brightness/contrast jitter, small translate, slight
blur. **No horizontal flip** — it mirrors the label text and inverts skew
direction, which is exactly what two of the classes are about.

---

## 4. Training

- **Backbone:** `efficientnet_b0` or `resnet18`, ImageNet-pretrained (torchvision).
  1145 images is far too few to train from scratch.
- **Head:** one linear layer, N outputs, sigmoid.
- **Loss:** `BCEWithLogitsLoss(pos_weight=...)`, where `pos_weight` is
  `n_negative / n_positive` per column. This is what carries the imbalance —
  30 missing-label images against 1100 negatives will otherwise be ignored.
- **Split:** by contiguous filename block (see §1c), roughly 80/20.
- **Metrics:** per-label precision, recall and PR-AUC. **Not accuracy** — a model
  that predicts "no defect" for everything scores 97% accuracy on
  `missing_label`. Recall is the number that matters; a missed defect ships.
- **Thresholds:** one per defect, tuned on validation, stored in the checkpoint
  and adjustable from the dashboard.
- **Runtime:** seconds per epoch on any GPU; a few minutes per epoch on CPU.
  20–30 epochs is enough for a fine-tune this size.

---

## 5. App shape

One Python process. FastAPI serving one HTML page. No Node, no build step, no
Docker.

```
app.py          FastAPI: routes, static, MJPEG stream
dataset.py      labels.csv read/write, crop, Dataset class
train.py        fine-tune loop, writes models/<timestamp>/
infer.py        camera -> crop -> model -> verdict
index.html      whole dashboard, inline JS

data/images/    all JPGs, flat
data/labels.csv source of truth
models/         versioned checkpoints + thresholds + metrics
```

```
> pip install -r requirements.txt
> python app.py
> open http://localhost:8000
```

---

## 6. Features

### Core — v1

| # | Feature | Notes |
|---|---|---|
| 1 | Image grid with thumbnails | paginated, lazy-loaded; originals are ~870 KB each |
| 2 | Checkbox row per image | one checkbox per defect, saves on click |
| 3 | Bulk label | select many images, apply/clear a defect on all |
| 4 | Keyboard labeling | number keys toggle defects, arrows move — 10× faster than clicking |
| 5 | Add / rename / delete defect type | new column in `labels.csv`, no code change |
| 6 | +ve / -ve browser per defect | two buckets side by side, drag or click to move |
| 7 | Unlabeled inbox | anything not yet reviewed, so nothing gets silently missed |
| 8 | Class count panel | live counts + warning on empty or under-populated classes |
| 9 | **Train** button | streams the training log to the page, disables while running |
| 10 | Metrics after training | per-defect precision/recall table, best/worst class called out |
| 11 | Live view | MJPEG from camera, verdict + per-defect bars overlaid |
| 12 | Threshold sliders | per defect, live effect, saved to the model |

### Small features that earn their keep

| # | Feature | Why |
|---|---|---|
| 13 | **Snapshot from live view straight into a label** | Hold a bottle up, click, it lands labelled — data collection stops being a separate job. Shows a thumbnail of what was saved plus an Undo, because a capture trains immediately and a mis-aimed camera must not add silent garbage. |
| 14 | Near-duplicate warning | Flags consecutive frames so you know 319 "Damaged Bottle" images might be 12 real bottles. |
| 15 | Model version list + rollback | Retraining on bad labels happens. One click back to the model that worked. |
| 16 | Crop preview toggle | See exactly what the model sees. Catches a bad crop before it costs an hour of training. |
| 17 | Reject log CSV | Timestamp, verdict, confidences, frame path. The shift report. |
| 18 | "Show me the mistakes" | After training, the validation images the model got wrong, sorted by confidence. Half of them will be mislabeled — fix and retrain. |
| 19 | Low-confidence capture | Live frames scoring 0.4–0.6 auto-save to the inbox. The model collects its own hard cases. |
| 20 | Dry-run on a video file | ✅ Upload a clip from the PC, it plays at its own frame rate with the verdict overlaid. Uploads stay in the source dropdown for re-runs. |

### Deliberately skipped

- **Bounding boxes / object detection** — one centred bottle, fixed camera. Classification is enough and costs zero annotation.
- **Multi-user, auth, database** — one operator, one PC. CSV and the filesystem.
- **Docker / cloud training** — 1145 images fine-tunes on the local machine.
- **Segmentation masks** — nothing here needs pixel-level output.

Add any of these when the constraint that rules them out stops being true.

---

## 7. Order of work

| Phase | Work | Gate |
|---|---|---|
| **0** | ~~Import folders → `labels.csv`. Crop + visual check.~~ | ✅ done |
| **1** | ~~`train.py`: scene split, pos_weight, per-defect metrics.~~ | ✅ done, macro-F1 0.946 |
| **2** | ~~Dashboard: grid, labeling, defect management, train, metrics.~~ | ✅ done |
| **3** | ~~Live view: camera, overlay, thresholds, snapshot-to-label.~~ | ✅ built, **untested on a real camera** |
| **4** | Small features 14–20, picked by what actually annoys you in phase 3. | open |

**Phase 1 was the honest gate,** and it passed with the caveat above about sample
size. If per-defect recall drops, the answer is more and better-varied *bottles* —
not a bigger model, not more epochs. 1145 images of 112 bottles is 112 bottles.

### What to do next, in order

1. ~~Shoot `Missing Cap` frames.~~ Class dropped instead — folder and column
   both gone. Recreate the folder with images if it is ever wanted back.
2. **Grow the good set.** 72 good against 1073 defective is a 6% pass rate; a
   real line is the other way round, so the model has learned a world where
   defects are normal. This is now the biggest single win, and vary the bottles —
   different fill levels, label rotations, lighting. Distinct scenes, not more
   frames of the same one.
3. **Fix the tilt_cap labels** from the Show me the mistakes panel (see above),
   then retrain. Cheapest accuracy left after that.
4. **Point it at the real camera.** Everything before this is validated on files;
   the camera path is written but has never seen a real device.

---

## 8. Open questions

1. **Camera** — USB webcam, industrial GigE, or an existing capture app? The code
   assumes `cv2.VideoCapture`, which covers webcams and video files but not a GigE
   SDK. This is the one part never run against real hardware.
2. ~~GPU on the inspection PC?~~ Answered on this machine: RTX 4070 Ti SUPER,
   3.4 s/epoch, 9.8 ms/frame. If the *line* PC has no GPU, training moves to
   minutes-per-epoch and live inference to roughly 3–5 FPS.
3. **Is `Water Level` pass/fail, or does it need an actual fill-height number in mm?**
   A number is a different model (regression) and a different plan.
4. **Does the line need a reject signal out** — GPIO, PLC, serial — or is the screen
   the whole output? Right now the only outputs are the screen and `data/rejects.csv`.
5. **Will the camera's framing match the training ROI?** The ROI is measured from
   these files. A different camera or a moved mount needs `python calibrate.py`
   re-run and the model retrained — the ROI is baked into each checkpoint.
