# Documentation map

> **Planned architecture is not implemented functionality.** Every document separates the two. Nothing in this
> repository has been validated on a physical machine; "tested" means a software self-test with fakes.

| Folder | Contents |
|---|---|
| *(this folder)* | **`PROJECT_BRIEF_FOR_REVIEW.md`** - one self-contained document explaining the whole project, its status, known problems and review questions (give this to a reviewer or an AI assistant) |
| `hardware/` | `PLC_LADDER_REQUIREMENTS.md` (what the real ladder must do, what `final_year.isp` already does, rungs to add; `python -m plc.ladder_check`), `HARDWARE_INTEGRATION.md` (the physical system, timing equation, unknowns), `CAMERA_PLACEMENT_AND_LINE_PLAN.md` (camera/light placement, trigger timing, queue, E-stop, commissioning order) |
| `guides/` | `DATABASE.md` (what the production database is, tables, export), `AUTO_ANNOTATION.md` (model-assisted labelling and the active-learning queues), `PRODUCTION_HMI_AND_COMMISSIONING.md` -- operator / engineer workflow, timing without an encoder, sequential cameras, model lifecycle, physical commissioning order; `TAB_GUIDE.md` -- what each app page is for; `BENCH_TEST_DECISION_ENGINE.md` -- bench testing the decision engine with the real PLC |
| `roadmap/` | `CURRENT_SYSTEM.md` (what exists), `CURRENT_SCOPE.md` (what is in scope now), `FUTURE_ENHANCEMENTS.md` (deferred), `PROGRESS_PLAN.md` (phase plan + universal-system gaps), `FEATURE_STATUS.md` (per-feature truth, keep honest), `PLC_COMMUNICATION.md` (PLC contract + decoded ladder), `VISION_DATASET.md`, `TRACEABILITY_PLAN.md`, `INDUSTRIAL_ARCHITECTURE.md` (target) |
| `audit/` | Dated audits: `SYSTEM_AUDIT_2026-10-03.md`, `SYSTEM_CHECK_2026-10-04.md`, `GAP_MATRIX_2026-10-05.md` (doc drift + gap matrix + what was implemented) |
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
