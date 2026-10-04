# What each tab is for

The desktop app (`gui.py`) has 11 tabs, grouped in the left rail. This answers "what does it do and do we need it" for each.
Sources: the `*Tab` classes in `gui.py`, `annotation_studio.py`, and `docs/roadmap/FEATURE_STATUS.md`.

## DATA -- build and clean the dataset (offline, any PC)

| Tab | What it does | Needed on the machine? |
|---|---|---|
| **Label** | Grid of images; tick defects (multi-label), mark reviewed, trash/undo, import a pre-sorted folder, AI pre-labels (suggestions only; never training data until a person accepts) | No -- only when preparing a dataset |
| **Defects** | Add / rename / delete defect classes (adds a `labels.csv` column + `-ve/<Class>/` folder). This is what makes the system product-agnostic | No -- per new product |
| **Data health** | Class balance, scene count, ROI check, "Re-measure ROI" (`calibrate.py`), duplicate/leak warnings | Yes, **after mounting the cameras**: re-measure the ROI |
| **Annotate** (Annotation Studio) | Draw bottle/cap/label **boxes** and **polygons** (label outline), review flag, export YOLO detection/segmentation | No -- per new product; feeds the Stage 2 detector and the (untrained) segmenter |

## MODEL -- train and judge

| Tab | What it does |
|---|---|
| **Train** | Train the Stage 1 multi-label classifier (EfficientNet-B0 default; also B1, ResNet18, MobileNetV3, ConvNeXt via `model_bench.py`), live loss/F1 log, "Use" a checkpoint. Every run promotes itself (writes thresholds + `active_model`) |
| **Analysis** | Loss/F1 curves, per-defect precision/recall, confusion matrix per defect, compare checkpoints honestly (flags ones that predate the current val split) |

## RUNTIME -- cameras, PLC, the line

| Tab | What it does | Use it for |
|---|---|---|
| **Live** | Opens cameras and scores frames **continuously** (Classifier / Classifier + YOLO / YOLO only), draws boxes, shows PASS/REJECT/FAULT per camera and combined | **Tuning and checking views** -- see what the model sees; verify placement, ROI, lighting. Not connected to the PLC. This is your "does it see the bottle properly" screen |
| **Machine** | PLC I/O monitor (X inputs, M bits, Y outputs, link state) + **operator-armed test** PASS/REJECT writes (M0/M1). Auto-connects only to the simulator | **Commissioning**: prove wiring/addresses before running the line. Not an operating screen |
| **Production** | The real run: Start line -> waits for the X0/M2 trigger -> captures frames after it -> decides -> schedules M0/M1. Counters, FIFO of bottles with deadlines, last bottle + evidence thumbnail, alarms, daily CSV | **The operator screen.** This is the tab that runs the machine |
| **Camera** (BenchTab) | Measures what each camera mode *actually* delivers (real FPS, latency, jitter, **sharpness**); the focus/exposure/WB lock editor (`camera_controls`); CPU/RAM/GPU | **Camera setup**: choose resolution, lock autofocus/exposure. Refuses to run while Live has cameras open |

## SYSTEM

| Tab | What it does |
|---|---|
| **Settings** | Text size, grid size, PLC link (TCP simulator / serial), line settings (cameras, travel distance, speed, T0, frames, fault action) |

## How they connect (one story)

`Label/Defects/Annotate` build data -> `Train/Analysis` make and judge a model -> `Data health` + `Camera` set ROI and camera lock ->
`Live` verifies what the model sees -> `Machine` proves the PLC link -> **`Production` runs the line**.

## Honest assessment (what is weak today)

- **The HMI is engineer-oriented.** 11 equal tabs in one window is a developer console, not an operator screen. An operator needs
  one full-screen view: big PASS/REJECT/FAULT, counters, live camera pair, line state, alarms, Start/Stop, E-stop status, nothing else.
  Recommendation: make **Production** the default and (optionally) fullscreen kiosk, and hide Data/Model tabs behind an "Engineer"
  mode / PIN. Not built yet; needs your photos/preferences for the look (the photo you mentioned was not in this session).
- **Live vs Production overlap** (both show camera panes). They are different jobs (tune vs run) but the UI does not say so; the tab
  guide above should become tooltips.
- **No E-stop/fault input, no heartbeat, no fault latching, no alarm history, no user login** (all DEFERRED in FEATURE_STATUS).
- **Nothing was run on the real machine.** Every green result in this repo is a self-test with fakes.
- `app.py` + `index.html` (browser version) are dead code; `CLAUDE.md` says ask before deleting. Recommend deleting them
  together with `Bottle_Defect_Detection_Web_Dashboard_Architecture.docx` if you agree.
