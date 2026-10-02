# Implementation Plan

Dependency-ordered. **A phase starts only when what it depends on exists.** Status reflects the repository
today; nothing past Phase 0 has been started. "Tested" for software phases means software self-tests -- it never
substitutes for the hardware tests in Phases 5-8.

Two design decisions are open and should be settled *inside* the phase that needs them, with evidence:

- **Per-bottle association (Phase 2):** full tracking vs. a sensor-triggered capture window (one trigger = one
  inspection). The latter is simpler and is likely sufficient for the first machine; this is not yet decided.
- **What YOLO boxes decide (Phase 3):** the detector finds bottle/cap/label only. Rules must define which
  defects come from detections (presence, position) and which stay with the Stage 1 classifier.

Physical-parameter gathering (distances, conveyor speed, PLC I/O list, actuator response) does **not** depend
on any software phase and can start immediately in parallel.

---

## Phase 0 -- Git baseline and documentation  *(this task)*

- **Objective:** a truthful, reproducible GitHub baseline.
- **Inputs:** current repository.
- **Outputs:** corrected `.gitignore` (Stage 2 provenance tracked), `MODEL_PROVENANCE.json`, portable
  `stage2_dataset/data.yaml`, README, this folder, corrected stale docs, one baseline commit.
- **Dependencies:** none.
- **Test:** all existing self-tests; no secrets; no images/weights committed.
- **Out of scope:** any new functionality; retraining; hardware.

## Phase 1 -- YOLO runtime integration  *(NEXT)*

- **Objective:** run the trained YOLOv8n inside the live pipeline behind the existing contracts.
- **Inputs:** `models/stage2_yolo/stage2_best.pt` (see its checksum), `Camera` / `Frame` / `Inspection`.
- **Outputs:** a detector wrapper that produces per-frame detections (class, box, confidence) with the same
  freshness/FAULT behavior; detection results attached to the inspection; model id recorded.
- **Dependencies:** Phase 0.
- **Must be tested:** missing/corrupt weights -> FAULT; inference exception -> FAULT; stale result -> FAULT;
  detections on the Stage 2 test images match the offline evaluation; latency measured on the development GPU
  *and* on frames from the real camera.
- **Out of scope:** retraining, YOLOv8s, decision rules, tracking.

## Phase 2 -- Inspection window and per-bottle association

- **Objective:** make one physical bottle produce one inspection.
- **Inputs:** frame stream with `(session, seq, monotonic ts)`; optionally a sensor event (mock first).
- **Outputs:** an inspection window object; per-bottle grouping of frames; temporal aggregation hook.
- **Dependencies:** Phase 1 (something to aggregate).
- **Must be tested:** one bottle -> exactly one result; no duplicate inspection; missed/empty window -> FAULT,
  not PASS.
- **Out of scope:** multi-product tracking, PLC.

## Phase 3 -- Decision engine

- **Objective:** turn detections/classifier outputs into PASS / REJECT / FAULT with explainable reasons.
- **Inputs:** Phase 2 per-bottle data; configurable thresholds/rules.
- **Outputs:** rule evaluation (e.g. cap/label present, positions), multi-camera fusion, fault latching and reset
  semantics, reasons in the `InspectionRecord`.
- **Dependencies:** Phase 2.
- **Must be tested:** each rule on constructed cases; FAULT precedence; latching/reset; regression of existing
  fail-safe behavior.
- **Out of scope:** recipes, a rules-authoring UI.

## Phase 4 -- Mock PLC

- **Objective:** a software PLC stand-in so the result path can be built and tested without hardware.
- **Inputs:** decision output; the (still to be verified) handshake design.
- **Outputs:** a PLC interface abstraction + mock implementation; result delivery, acknowledgement, timeout and
  watchdog behavior.
- **Dependencies:** Phase 3.
- **Must be tested:** result lost / late / unacknowledged -> defined safe behavior; interface independent of a
  specific protocol.
- **Out of scope:** real Delta PLC, real addresses.

## Phase 5 -- Camera and sensor timing

- **Objective:** a measured timing model.
- **Inputs:** measured distances and conveyor speed; measured capture, inference and communication latencies.
- **Outputs:** timing budget calculation and a check that the chain fits the available travel time with margin.
- **Dependencies:** physical measurements; Phase 1 latency data.
- **Must be tested:** budget arithmetic; on the machine, end-to-end latency measurement.
- **Out of scope:** guessing values -- no numbers are recorded in this repository today.

## Phase 6 -- Delta PLC

- **Objective:** real PLC communication.
- **Inputs:** a readable export of the ISPSoft ladder, the verified I/O list, PLC model, protocol settings.
- **Outputs:** a real PLC adapter behind the Phase 4 interface; verified address map.
- **Dependencies:** Phase 4; verified I/O mapping; PLC available.
- **Must be tested:** handshake, watchdog, result timeout, ACK, reset on the real PLC.
- **Out of scope:** the reject mechanism itself.

## Phase 7 -- Conveyor and reject

- **Objective:** a bottle is rejected at the right place.
- **Inputs:** Phase 5 timing, Phase 6 link, the physical actuator.
- **Outputs:** reject timing logic (in the PLC), reject pulse configuration, fault outputs.
- **Dependencies:** Phases 5, 6.
- **Must be tested:** reject position accuracy at the real conveyor speed; behavior on FAULT.
- **Out of scope:** throughput optimisation.

## Phase 8 -- Physical validation

- **Objective:** demonstrate the first complete machine cycle.
- **Inputs:** everything above; a set of known-good and deliberately defective bottles.
- **Outputs:** a recorded validation: counts of correct/incorrect decisions, timing margins, failure modes.
- **Dependencies:** Phase 7.
- **Must be tested:** repeated runs; camera/PLC/model failure injection; recovery after restart.
- **Out of scope:** claiming production readiness from a short demonstration.

## Phase 9 -- Industrial platform enhancements

- **Objective:** re-enter the deferred items from [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md): persistence,
  dashboard, recipes, registry, security, reports, vision tools, annotation automation.
- **Dependencies:** a proven machine (Phase 8).
- **Out of scope until then:** all of it.
