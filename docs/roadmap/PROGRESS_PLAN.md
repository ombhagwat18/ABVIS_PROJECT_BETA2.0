# Progress and flow plan

Status legend: DONE (software-verified) / NEXT (do on or right after the first hardware day) / LATER.
Companion: `FEATURE_STATUS.md` (per-feature truth), `CURRENT_SCOPE.md` (what is in scope now), `docs/hardware/CAMERA_PLACEMENT_AND_LINE_PLAN.md`.

## Flow of the whole system

```
DATA            label / import / annotate  ->  labels.csv, annotations.json
MODEL           train classifier (Stage 1) + detector (Stage 2) + [segmenter, untrained]  ->  checkpoints + metrics + provenance
SETUP           calibrate ROI, lock camera controls, measure belt/distances, set line settings
RUN             X0 -> M2 -> capture after trigger -> models -> decision (vote, fuse) -> FIFO -> M0 / M1 at deadline -> PLC fires Y0
RECORD          production CSV (now) -> evidence images + SQLite (later)
```

## Phase plan

| Phase | Work | Status |
|---|---|---|
| 1 Data + Stage 1 classifier + test evaluation | DONE |
| 2 Stage 2 detector (YOLOv8n/s) + runtime hook | DONE (software) |
| 3 PLC link + trigger/command rules + decision engine + machine cycle | DONE (simulator/fakes) |
| 4 Desktop app, 11 tabs, theme | DONE |
| 5 **Hardware day 1**: measure, fix ladder presets, mount cam A, lock controls, calibrate, collect data from the real enclosure | NEXT |
| 6 Fine-tune classifier on real-enclosure frames; re-measure on a fresh held-out set | NEXT |
| 7 Camera B + per-camera delay + failure injection tests | NEXT |
| 8 Safety: E-stop input, PC<->PLC heartbeat, fault latching, alarm list | NEXT |
| 9 Operator HMI (kiosk Production view, engineer mode) | NEXT |
| 10 Evidence images + SQLite traceability, reports, login | LATER |
| 11 Segmenter: review polygons -> train -> enable label-area rules | LATER (blocked on human review) |

## Universal / data-agnostic system: how a new product works today, and the gap

Today (works): new project -> Import dataset or Label -> Defects (add classes) -> Train -> Data health (ROI) -> Annotate (boxes) ->
train detector -> write the `config.json` `"inspection"` recipe by hand -> Production. No code change.

Gaps to make "just give it data" real:
1. **Auto-annotation** (NEXT after hardware): the classifier already writes *suggestions* (`cache/suggestions.json`, never training data until accepted).
   Extend the same pattern to boxes: run the current YOLO over new images, write boxes as **unreviewed proposals** in the Annotate tab
   (accept / nudge / reject, like seeded polygons). Add active-learning ordering (least-confident first). Rule kept: a model-proposed label
   never enters training until a person accepts it.
2. **Recipe editor in the GUI** (anchor/required parts/zones are hand-edited JSON today).
3. **One "New product wizard"** chaining Import -> classes -> ROI -> train -> recipe, with the checks that each step passed.
4. **Model registry GUI + dataset versioning** (hashes already exist in the provenance files).
5. For products with no defect examples: an anomaly-detection model trained on good samples only (LATER).

## What else we need (hardware + features)

Have: PLC ladder (decoded), simulator link, 2 cameras, LED strips, enclosure, conveyor, Festo cylinder + valve, photo-eye.
Need: belt speed/distance measurements; USB 3 ports or a powered USB 3 hub per camera; DC (non-PWM) LED supply + diffusers; camera mounts with
fine adjustment; lens hoods; hardwired E-stop with a PLC status input; PLC address list from the real machine; known-defect bottles for
every class (especially `missing_cap`, `missing_label` which have ~0 real examples); a sacrificial-bottle test plan.
