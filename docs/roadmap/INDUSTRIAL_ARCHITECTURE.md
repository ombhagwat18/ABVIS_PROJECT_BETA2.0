# Industrial Architecture

*Status 2026-10-06: most layers now exist in software (tags below). Physical validation is still missing for all of them.*
See [CURRENT_SYSTEM.md](CURRENT_SYSTEM.md) for what exists and [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) for what is deferred.

## Layers

```
JOB / RECIPE                 which product, which rules; one versioned configuration     [PARTIAL: job + product per run, recipe editor + hash; no wizard]
      |
PRODUCT CONFIGURATION        classes, dimensions, tolerances, thresholds                  [PARTIAL: per-project config.json]
      |
CAMERA + LIGHTING            devices, exposure/gain/focus, controlled illumination        [PARTIAL: camera lock, rotate, reconnect; lighting not in software]
      |
ACQUISITION / TRIGGER        sensor-triggered capture, frame sequence + timestamps        [IMPLEMENTED in software: X0 -> M2 -> frames after trigger, per-camera window]
      |
PREPROCESSING                ROI crop, resize, normalisation                              [IMPLEMENTED: dataset.model_input]
      |
AI + TRADITIONAL VISION      detection / classification / segmentation; measurement       [PARTIAL: classifier + YOLO in the line; segmentation untrained; no traditional vision]
      |
INSPECTION / DECISION ENGINE per-bottle aggregation, rules, multi-camera fusion,
                             PASS / REJECT / FAULT with reasons                           [IMPLEMENTED in software: decision.py + verdict.py for the screens]
      |
TRACEABILITY                 inspection record, evidence, counters, history               [IMPLEMENTED in software: SQLite, evidence, alarms, logs, export]
      |
PLC INTERFACE                result delivery, handshake, watchdog, timeout                [IMPLEMENTED: M0/M1/M2 handshake; NO watchdog / timeout]
      |
MACHINE CONTROL              deterministic logic: conveyor, interlocks, reject timing     [PLC: ladder read + simulated; presets wrong, no E-stop / interlock / timeout]
      |
REJECT / ACTUATOR            cylinder + valve                                             [HARDWARE]
```

## Responsibility boundary

> **AI decides *what* the object / defect is. The PLC decides *how* the physical machine responds.**

| Software (this repository) | PLC |
|---|---|
| Perception: detect/classify what is in the frame | Deterministic machine control |
| Inspection and the decision (PASS / REJECT / FAULT) with reasons | Conveyor start/stop and speed |
| Evidence and traceability | Sensor inputs and their debouncing |
| Detecting its own failure (FAULT) | Actuator outputs, reject pulse timing |
| Delivering the result to the PLC | Safety interlocks and emergency behaviour |
| | Acting safely if the software stops responding (watchdog / timeout) |

Consequences of this split:

- The PLC must never depend on the PC for safety. A missing, late or FAULT result must lead to a defined safe
  outcome *decided in the PLC* (for example, treat as reject). Which outcome is correct is a machine-design
  decision not yet made.
- The software must never claim PASS when it does not know. That is the purpose of the FAULT state and
  freshness checks already in `infer.py`.
- Timing across the boundary must be measured, not assumed -- see [HARDWARE_INTEGRATION.md](../hardware/HARDWARE_INTEGRATION.md).

## Data flow for one bottle (target)

```
sensor edge -> trigger -> capture N frames (each: camera_id, session, seq, monotonic ts)
   -> detect/classify each frame -> aggregate per bottle (and across cameras)
   -> decision engine -> PASS | REJECT | FAULT (+ reasons, hits, model id)
   -> InspectionRecord (id, timestamps, evidence) -> PLC result -> reject at the right moment
```

The Frame / Inspection / InspectionRecord contracts in `infer.py` and `inspection_trace.py` are the first
pieces of this flow that exist.

## Design rules

1. Fail safe: unknown is FAULT, never PASS.
2. Time is monotonic and explicit; wall-clock time is for humans and reports only.
3. Every result carries enough metadata (camera, session, sequence, model) to trace it back.
4. The PLC owns real-time behaviour; software never tries to be a hard-real-time controller.
5. Add platform features (recipes, registry, dashboards) only after the minimum machine is proven.
