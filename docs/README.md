# Documentation map

> **Planned architecture is not implemented functionality.** Every document separates the two. Nothing in this
> repository has been validated on a physical machine; "tested" means a software self-test with fakes.

| Folder | Contents |
|---|---|
| `hardware/` | `HARDWARE_INTEGRATION.md` (the physical system, timing equation, unknowns), `CAMERA_PLACEMENT_AND_LINE_PLAN.md` (camera/light placement, trigger timing, queue, E-stop, commissioning order) |
| `guides/` | `TAB_GUIDE.md` -- what each of the 11 app tabs is for; `BENCH_TEST_DECISION_ENGINE.md` -- which tab drives the decision engine, and how to bench test it with the real PLC |
| `roadmap/` | `CURRENT_SYSTEM.md` (what exists), `CURRENT_SCOPE.md` (what is in scope now), `FUTURE_ENHANCEMENTS.md` (deferred), `PROGRESS_PLAN.md` (phase plan + universal-system gaps), `FEATURE_STATUS.md` (per-feature truth, keep honest), `PLC_COMMUNICATION.md` (PLC contract + decoded ladder), `VISION_DATASET.md`, `TRACEABILITY_PLAN.md`, `INDUSTRIAL_ARCHITECTURE.md` (target) |
| `audit/` | Dated audits: `SYSTEM_AUDIT_2026-10-03.md`, `SYSTEM_CHECK_2026-10-04.md` |
| `design/` | Original design (`PLAN.md`), multi-project design, `diagrams/` (pipeline.drawio, web-dashboard architecture) |
| `n8n/` | Workflow viewer (experimental) |

Project report material (black book, results tables, viva questions) lives in `../FINAL_YEAR_BLACKBOOK/`.
Entry points: `../README.md`, `../CLAUDE.md`. Earlier web version of the app: `../legacy/web_dashboard/` (unused).

## Guiding principles

1. **AI decides *what* the object/defect is. The PLC decides *how* the machine responds.**
2. **Unknown is FAULT, never PASS.**
3. **Prove the minimum machine first**, then build the platform around it.
4. **Do not invent facts**: unmeasured timings, unverified PLC addresses and untested hardware are stated as such.
5. Update `roadmap/FEATURE_STATUS.md` whenever reality changes.
