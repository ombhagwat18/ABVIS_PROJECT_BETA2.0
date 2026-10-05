# What each tab is for

The desktop app (`gui.py`) has 14 pages, grouped in the left rail. OPERATOR mode shows only the PRODUCTION
group; ENGINEER mode (header button) shows everything. This answers "what does it do and do we need it" for each.
Sources: the `*Tab` classes in `gui.py`, `hmi.py`, `annotation_studio.py`, and `docs/roadmap/FEATURE_STATUS.md`.
Operator and commissioning workflow: [PRODUCTION_HMI_AND_COMMISSIONING.md](PRODUCTION_HMI_AND_COMMISSIONING.md).

Every page scrolls up/down and left/right when its content is bigger than the window (scroll bars appear only when
needed; Shift + wheel scrolls sideways), so a full panel can never hide a button.

**What an operator sees on a camera:** one word per bottle - GOOD, DEFECT: <defect name>, CHECKING, NO BOTTLE or
FAULT - in a colour, held steady until the bottle leaves. Component boxes and score bars are an engineer's view
(Live page -> "Engineer details").

## PRODUCTION -- run the line (the operator's pages)

| Page | What it does |
|---|---|
| **Production** | The operator HMI: one machine state (READY / RUNNING / FAULT ...), start checklist, both cameras live, current bottle, counters, recent bottles, coded alarms; START INSPECTION / STOP / RESET FAULT; CAMERA TEST and TEST INSPECTION (no PLC). Engineer row: AI task, line cameras, timing, speed calibration, line layout / camera stations, HALT latch, simulator feed |
| **History** | Persistent record (`production.db`): bottles per day or per shift with result, defects, PLC outcome, evidence; counts, yield, defect distribution, bottles per hour, alarms; CSV export |
| **Database** | The production database as tables (bottles / alarms / runs) with range, result and text filters, a plain-words explanation of every column, and export: CSV for Excel or a printable production report. See `DATABASE.md` |
| **Health** | Computer, PLC, cameras, models, storage and alarms as HEALTHY / WARNING / FAULT; searchable event logs (channel / level / text) |

## DATA -- build and clean the dataset (offline, any PC)

| Tab | What it does | Needed on the machine? |
|---|---|---|
| **Label** | Grid of images; tick defects (multi-label), mark reviewed, trash/undo, import a pre-sorted folder, AI pre-labels (suggestions only; never training data until a person accepts) | No -- only when preparing a dataset |
| **Defects** | Add / rename / delete defect classes (adds a `labels.csv` column + `-ve/<Class>/` folder). This is what makes the system product-agnostic | No -- per new product |
| **Data health** | Class balance, scene count, ROI check, "Re-measure ROI" (`calibrate.py`), duplicate/leak warnings | Yes, **after mounting the cameras**: re-measure the ROI |
| **Annotate** (Annotation Studio) | Draw bottle/cap/label **boxes** and **polygons** (label outline), review flag, export YOLO detection/segmentation. **Auto-annotate:** the detector proposes boxes (dashed amber), you accept/reject; active-learning queues: "Next: least sure", "Next: missing part" (bottle without cap/label), "Next: doubtful part" (cap/label at 0.25-0.80, hard negatives) | No -- per new product; feeds the Stage 2 detector and the (untrained) segmenter |

## MODEL -- train and judge

| Tab | What it does |
|---|---|
| **Train** | Train the Stage 1 multi-label classifier (EfficientNet-B0 default; also B1, ResNet18, MobileNetV3, ConvNeXt via `model_bench.py`), live loss/F1 log, "Use" a checkpoint. A training run no longer promotes itself: it is a CANDIDATE until activated on the Models page (or `auto_activate_trained_model` is switched on) |
| **Models** | Model registry: every classifier / detector on disk, status, held-out test, critical-class warnings; Validate (real-camera note) -> Approve -> ACTIVATE, Reject, Roll back |
| **Analysis** | Loss/F1 curves, per-defect precision/recall, confusion matrix per defect, compare checkpoints honestly (flags ones that predate the current val split) |

## ENGINEERING -- cameras and PLC

| Tab | What it does | Use it for |
|---|---|---|
| **Live** | Opens cameras and scores frames **continuously** (Classifier / Classifier + YOLO / YOLO only), draws boxes, shows PASS/REJECT/FAULT per camera and combined | **Tuning and checking views** -- see what the model sees; verify placement, ROI, lighting. Not connected to the PLC. Per-frame verdicts: a development view, not the line |
| **Machine** | PLC connection manager (simulator TCP / real PLC serial, COM scan, station, test link, auto-reconnect), SIMULATOR / REAL PLC badge, PLC RUN/STOP, X / M / Y / T / C live, **operator-armed test** PASS/REJECT (M0/M1), opt-in operator test bits (M10/M11/M2). Auto-connects only to the simulator | **Commissioning**: prove wiring/addresses before running the line |
| **Camera** (BenchTab) | Measures what each camera mode *actually* delivers (real FPS, latency, jitter, **sharpness**); the focus/exposure/WB lock editor (`camera_controls`); CPU/RAM/GPU | **Camera setup**: choose resolution, lock autofocus/exposure. Refuses to run while Live has cameras open |

## SYSTEM

| Tab | What it does |
|---|---|
| **Settings** | Text size, theme (light / dark, next start), evidence policy, auto-activate (off), engineer PIN, camera probe / bench / monitor settings |

## How they connect (one story)

`Label/Defects/Annotate` build data -> `Train/Analysis` make and judge a model -> `Models` validates / approves / activates it ->
`Data health` + `Camera` set ROI and camera lock -> `Live` verifies what the model sees -> `Machine` proves the PLC link ->
**`Production` runs the line** -> `History` / `Health` show what happened.

## Honest assessment (what is weak today)

- **Nothing was run on the real machine** beyond a read-only PLC link. Every green result in this repo is a self-test with fakes.
- **Live vs Production overlap** (both show camera panes): Live is the tuning view (per frame), Production the line (per bottle).
- No user login (the engineer PIN is a convenience lock), no shift reports, no automatic camera reopen during a run.
- `legacy/web_dashboard/` (browser version) is dead code; `CLAUDE.md` says ask before deleting.
