# Progress and flow plan

*Status 2026-10-06.* Legend: DONE (software-verified with fakes) / NEXT (on or right after the first hardware day) / LATER.
Companions: [FEATURE_STATUS](FEATURE_STATUS.md), [CURRENT_SCOPE](CURRENT_SCOPE.md),
[CAMERA_PLACEMENT_AND_LINE_PLAN](../hardware/CAMERA_PLACEMENT_AND_LINE_PLAN.md).

## Flow of the whole system

```
DATA      label / import / annotate (model proposals + review queues)  ->  labels.csv, annotations
MODEL     train candidates (never auto-activated) -> held-out test -> validate on a real camera -> approve -> activate / rollback
SETUP     calibrate ROI, lock camera controls, speed calibration, line layout + camera stations, recipe
RUN       X0 -> M2 -> frames after trigger (per camera window) -> AI -> decision (vote, fuse) -> FIFO -> M0 / M1 at deadline -> PLC fires Y0
SHOW      one steady GOOD / DEFECT per bottle; counters, alarms, history
RECORD    SQLite (runs, bottles, alarms) + evidence pictures + event logs + CSV / printable report
```

## Phase plan

| Phase | Work | Status |
|---|---|---|
| 1 | Data + Stage 1 classifier + held-out test | DONE |
| 2 | Stage 2 detector + runtime hook + recipe-driven decision | DONE |
| 3 | PLC link, trigger/command rules, decision engine, machine cycle, time-based tracking, staggered cameras | DONE (simulator / fakes) |
| 4 | Desktop app: 15 pages, operator/engineer modes, light theme, scrolling pages, one machine state | DONE |
| 5 | Traceability: SQLite, evidence, alarms, event logs, History / Database pages, CSV + report export | DONE |
| 6 | Model lifecycle: registry, gated activation, rollback, threshold calibration | DONE |
| 7 | Checks: ladder decoder + ladder simulator + `selfcheck.py` (engineer "Simulation check") | DONE |
| 8 | **Hardware day 1**: fix PLC presets (K15 / K5), add E-stop input / Y0 interlock / answer timeout, USB 3 ports, measure speed and distances, lock camera controls | **NEXT** |
| 9 | Capture real EMEET frames (good + missing-cap bottles, several bottles/poses), retrain classifier and detector, validate on the real camera | **NEXT** |
| 10 | Physical commissioning: PASS, REJECT, line, sequential cameras, multi-bottle, latency p50/p95, E-stop | **NEXT** |
| 11 | PLC protocol v2 (several bottles in flight), PC heartbeat watchdog | LATER (needs ladder + software change together) |
| 12 | Login / roles, shift reports beyond day / shift, anomaly detection, segmenter | LATER |

## New product: how it works today

New project -> Import dataset or Label -> Defects (add classes) -> Train -> Data health (ROI) -> Annotate (boxes, with
proposals and review queues) -> train detector -> **Recipe...** dialog -> Production. No code change.

Remaining gaps: a "new product wizard" chaining those steps with checks; dataset versioning; for products with no
defect examples, an anomaly-detection model trained on good samples only.

## What else we need (hardware + data)

Have: PLC ladder (read and simulated), simulator link, 2 cameras, LED strips, enclosure, conveyor, Festo cylinder +
valve, photo-eye.
Need: belt speed / distance measurements; USB 3 port per camera; DC (non-PWM) LED supply + diffusers; camera mounts;
hardwired E-stop with a PLC status input; the program actually loaded in the real PLC (upload it and compare); known
defective bottles for every class, **especially missing_cap and missing_label**, from several bottles; a
sacrificial-bottle test plan.
