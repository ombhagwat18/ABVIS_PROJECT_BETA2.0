# Black Book Content Outline

Suggested chapter structure for the final-year project black book, mapping
each chapter to the source documents already prepared in this workspace.
This outline covers Stage 1 only; later stages (annotation/YOLO, platform
integration) will extend it once that work exists.

## Chapter 1 — Introduction
Source: `01_Project_Overview/PROJECT_OVERVIEW.md`
- Project title, degree context, what has been built, long-term platform
  direction (clearly marked as not-yet-implemented), Stage 1 scope.

## Chapter 2 — Problem Statement and Objectives
Source: `02_Problem_Statement_and_Objectives/PROBLEM_STATEMENT_AND_OBJECTIVES.md`

## Chapter 3 — System Requirements
Source: `03_System_Requirements/SYSTEM_REQUIREMENTS.md`
- Fill remaining "To be added" fields (CPU/RAM, package versions) before
  final submission.

## Chapter 4 — System Architecture
Source: `04_System_Architecture/SYSTEM_ARCHITECTURE.md`
- Consider adding a block diagram in `11_Figures_and_Graphs/` (not yet
  created — architecture diagrams were out of scope for this
  documentation pass, which focused on the numerical benchmark).

## Chapter 5 — Dataset and Data Preparation
Source: `05_Dataset_and_Data_Preparation/DATASET_DOCUMENTATION.md`,
tables in `12_Tables/dataset_split.md` and
`12_Tables/dataset_class_distribution.md`.

## Chapter 6 — AI Model Development
Sources:
- `06_AI_Model_Development/Classification_Methodology/AI_METHODOLOGY.md`
- `06_AI_Model_Development/Training_Configuration/TRAINING_CONFIGURATION.md`
- `06_AI_Model_Development/Architecture_Benchmark/MODEL_ARCHITECTURE_BENCHMARK.md`
- `06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md`
- `06_AI_Model_Development/Failure_Analysis/FAILURE_ANALYSIS.md`

## Chapter 7 — Current System Implementation
Source: `07_Current_System_Implementation/CURRENT_SYSTEM_IMPLEMENTATION.md`

## Chapter 8 — Experimental Results
Source: `08_Experimental_Results/RESULTS_SUMMARY.md`, all of
`11_Figures_and_Graphs/`, all of `12_Tables/`.

## Chapter 9 — Limitations
Source: `09_Limitations/DATASET_LIMITATIONS.md`

## Chapter 10 — Stage 1 Conclusion
Source: `10_Stage_1_Review/STAGE_1_CLOSE_OUT.md`

## Chapter 11 — Future Work
Source: `15_Future_Work/FUTURE_WORK.md`

## Appendices
- Figures: `11_Figures_and_Graphs/`
- Tables: `12_Tables/`
- Screenshots: `13_Screenshots/` (to be added)
- Viva preparation (not for inclusion in the black book itself, but for
  defence preparation): `14_Viva_Preparation/VIVA_QUESTIONS_STAGE_1.md`

## Before final submission — outstanding items

- Populate `03_System_Requirements/SYSTEM_REQUIREMENTS.md`'s "To be added"
  fields (exact package versions via `pip freeze`, CPU/RAM specs).
- Capture application screenshots into `13_Screenshots/`.
- Populate `12_Tables/dataset_class_distribution.md`'s per-class scene and
  per-split breakdowns via a dataset audit, if required for the report.
- Consider adding an architecture block diagram to
  `11_Figures_and_Graphs/`.
