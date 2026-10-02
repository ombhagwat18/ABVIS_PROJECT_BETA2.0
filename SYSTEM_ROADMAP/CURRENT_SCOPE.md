# Current Scope

**Goal:** get the core inspection machine working reliably. One bottle, end to end, on the real conveyor.

```
One bottle enters
   -> Sensor detects the bottle
   -> Camera captures / inspects
   -> AI processes
   -> Decision generated
   -> PLC receives the result
   -> Bottle reaches the reject location
   -> PLC activates the reject
   -> Bottle is accepted / rejected correctly
```

**Principle:** AI decides *what* the object/defect is. The PLC decides *how* the physical machine responds.

## In scope now (dependency order -- see [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md))

| # | Item | State today |
|---|---|---|
| 1 | YOLO runtime integration | **NEXT.** Detector trained offline; not loaded by `infer.py` |
| 2 | Camera acquisition | PARTIAL. Works on driver defaults; no exposure/gain/focus, reconnect or trigger |
| 3 | Bottle / component detection | PARTIAL. Offline YOLOv8n (bottle, cap, label); classifier is the live path |
| 4 | Inspection window | Not started |
| 5 | Per-bottle association | Not started (a sensor-triggered single capture may replace full tracking -- a design decision for Phase 2) |
| 6 | Multi-camera fusion | PARTIAL. Rule exists (FAULT > REJECT > PASS); no frame alignment across cameras |
| 7 | Decision engine | PARTIAL. Per-frame thresholds only; needs rules for detections, fault latching |
| 8 | PASS / REJECT / FAULT | Foundation done (software-tested) |
| 9 | Inspection trace | Foundation done (in-memory) |
| 10 | Timing model | Not started; needs measured physical values |
| 11 | Mock PLC | Not started |
| 12 | Real PLC communication | Not started; addresses unverified |
| 13 | Conveyor / sensor synchronization | Not started; hardware required |
| 14 | Reject timing | Not started; hardware required |
| 15 | Physical testing | Not started |

## Explicitly out of scope until the machine cycle works

SQLite / any database, production dashboard, user login / roles / permissions, OCR, barcode/QR, auto or
AI-assisted annotation, active learning, anomaly detection, reports, analytics, alarms management,
cloud / SaaS, multi-job platform features. These are **deferred, not abandoned** -- see
[FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md).

Also out of scope right now: further model experiments (YOLOv8s, other backbones). The objective is machine
integration, not endless training.

## Definition of done for this scope

The **first complete machine cycle** has run on the physical machine: a real bottle is sensed, inspected,
given a PASS / REJECT / FAULT, the PLC receives the correct result in time, and the correct bottle is
physically rejected -- demonstrated with deliberately defective and good bottles, with the timing budget
measured rather than assumed. Until then, nothing here should be described as working on the machine.
