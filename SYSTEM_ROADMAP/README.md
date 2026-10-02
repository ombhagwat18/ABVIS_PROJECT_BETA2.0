# System Roadmap

The map of the whole project: what exists, what is being built now, what is deliberately **not** being built yet,
what comes next, and what the final industrial system could become.

> **Planned architecture is not implemented functionality.** Every document here separates the two. Nothing in
> this repository has been validated on a physical machine.

## Documents

| Document | Answers |
|---|---|
| [CURRENT_SYSTEM.md](CURRENT_SYSTEM.md) | **What we have.** The actual implementation, with honest status labels |
| [CURRENT_SCOPE.md](CURRENT_SCOPE.md) | **What we are building now.** The minimum reliable machine, in scope order |
| [FUTURE_ENHANCEMENTS.md](FUTURE_ENHANCEMENTS.md) | **What we are deliberately not building yet.** Deferred, not abandoned |
| [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) | **What we build next.** Dependency-ordered phases 0-9 |
| [FEATURE_STATUS.md](FEATURE_STATUS.md) | One table: every feature, its factual status, and its horizon |
| [HARDWARE_INTEGRATION.md](HARDWARE_INTEGRATION.md) | The physical system, the timing equation, and what is unknown |
| [TRACEABILITY_PLAN.md](TRACEABILITY_PLAN.md) | Inspection records today, and how they evolve |
| [INDUSTRIAL_ARCHITECTURE.md](INDUSTRIAL_ARCHITECTURE.md) | **What the final industrial system could become** |

## The one-paragraph version

The project is a Python desktop application for AI-based bottle defect inspection. Today it has a labelling and
classifier-training workflow, a verified 594-image detection dataset, a trained and test-evaluated YOLOv8n
detector (held-out test mAP50 0.968, mAP50-95 0.660 on a small 7-scene set), a fail-safe PASS / REJECT / FAULT
layer, frame/session metadata and an in-memory inspection record. It does **not** yet have YOLO in the live
runtime, bottle tracking, a decision engine for detections, a timing model, PLC communication or a reject
controller. The immediate goal is one complete physical cycle: a bottle is sensed, inspected, decided and
correctly rejected.

## Guiding principles

1. **AI decides *what* the object/defect is. The PLC decides *how* the machine responds.**
2. **Unknown is FAULT, never PASS.**
3. **Prove the minimum machine first**, then build the platform around it.
4. **Do not invent facts**: unmeasured timings, unverified PLC addresses and untested hardware are stated as such.
5. Keep this folder honest: update [FEATURE_STATUS.md](FEATURE_STATUS.md) whenever reality changes.

## Next development task

**YOLO runtime integration** (Phase 1 of the [implementation plan](IMPLEMENTATION_PLAN.md)).
