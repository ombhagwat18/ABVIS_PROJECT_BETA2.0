# Current Scope

*Status 2026-10-06. Companion: [FEATURE_STATUS](FEATURE_STATUS.md) (per-feature truth), [PROGRESS_PLAN](PROGRESS_PLAN.md).*

**Goal:** get the core inspection machine working reliably. One bottle, end to end, on the real conveyor.

```
One bottle enters -> photo-eye -> cameras -> AI -> decision -> PLC -> bottle reaches the cylinder -> PLC rejects it
```

**Principle:** AI decides *what* the object/defect is. The PLC decides *how* the physical machine responds.

## Where each step stands

| # | Step | Software | On the machine |
|---|---|---|---|
| 1 | Photo-eye trigger -> inspection id | DONE (self-tested, fake ladder) | not run |
| 2 | Camera acquisition, camera lock, reconnect | DONE (auto-reconnect while running) | EMEETs on one USB 2.0 hub: one 1080p stream; enclosure not commissioned |
| 3 | Frames after the trigger, per-camera window | DONE (`tracking.py`, staggered stations) | speed / distances not measured |
| 4 | AI: classifier + YOLO detector | DONE offline; detector runtime in the line | not validated on EMEET frames; missing cap unreliable |
| 5 | Decision engine (recipe, vote, fusion) | DONE | thresholds are development values |
| 6 | One steady verdict for the operator | DONE (`verdict.py`) | - |
| 7 | PLC link + commands | DONE (simulator); real PLC read-only | write handshake and reject cycle not run |
| 8 | Reject timing (time-based, no encoder) | DONE (deadline FIFO, calibration wizard) | PLC T0/T1 = 15 s / 5 s must be fixed |
| 9 | Safety: halt latch, E-stop input, fault latch | DONE in software | PLC has no E-stop input / interlock / timeout |
| 10 | Record: SQLite, evidence, alarms, logs, export | DONE | - |
| 11 | Physical testing | - | **NOT STARTED** |

## In scope next (physical commissioning)

Fix the ladder presets and add the safety rungs ([PLC_LADDER_REQUIREMENTS](../hardware/PLC_LADDER_REQUIREMENTS.md)); put the
cameras on separate USB 3 ports; measure belt speed and distances; capture real EMEET frames (good bottles and
missing-cap bottles from several bottles); retrain and validate on them; then run the 16-step commissioning order in
[PRODUCTION_HMI_AND_COMMISSIONING](../guides/PRODUCTION_HMI_AND_COMMISSIONING.md).

## Still out of scope until the machine cycle works on the real machine

User login / roles, OCR, barcode/QR, anomaly detection, cloud / SaaS, multi-job platform features, segmentation
models. These are **deferred, not abandoned** -- see [FUTURE_ENHANCEMENTS](FUTURE_ENHANCEMENTS.md).

## Definition of done for this scope

The **first complete machine cycle** has run on the physical machine: a real bottle is sensed, inspected, given a
PASS / REJECT / FAULT, the PLC receives the correct result in time, and the correct bottle is physically rejected --
demonstrated with deliberately defective and good bottles, with the timing budget measured rather than assumed. Until
then, nothing here should be described as working on the machine.
