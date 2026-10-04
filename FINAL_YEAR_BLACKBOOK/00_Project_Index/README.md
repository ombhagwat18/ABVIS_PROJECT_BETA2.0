# FINAL_YEAR_BLACKBOOK — Project Index

This folder is a documentation workspace for the B.Tech Robotics & Automation
Engineering final-year project *Bottle Defect Detection System*. It is
maintained separately from the working ML codebase
(`Bottle-train-OpenCV-main_vesion two/`) and does not modify, retrain, or
alter any code, dataset, label, split, checkpoint, or threshold belonging to
that project. Every document in this workspace is READ-ONLY with respect to
the application: it reports on work already completed and recorded in
`projects/om_bottle/models/<stamp>/metrics.json`.

## How this maps to the black book

| Black book section | Folder | Primary document(s) |
|---|---|---|
| Abstract / Introduction | `01_Project_Overview/` | `PROJECT_OVERVIEW.md` |
| Problem statement & objectives | `02_Problem_Statement_and_Objectives/` | `PROBLEM_STATEMENT_AND_OBJECTIVES.md` |
| System requirements | `03_System_Requirements/` | `SYSTEM_REQUIREMENTS.md` |
| System architecture | `04_System_Architecture/` | `SYSTEM_ARCHITECTURE.md` |
| Dataset and data preparation | `05_Dataset_and_Data_Preparation/` | `DATASET_DOCUMENTATION.md` |
| AI model development — methodology | `06_AI_Model_Development/Classification_Methodology/` | `AI_METHODOLOGY.md` |
| AI model development — training setup | `06_AI_Model_Development/Training_Configuration/` | `TRAINING_CONFIGURATION.md` |
| AI model development — architecture comparison | `06_AI_Model_Development/Architecture_Benchmark/` | `MODEL_ARCHITECTURE_BENCHMARK.md` |
| AI model development — held-out evaluation | `06_AI_Model_Development/Test_Evaluation/` | `HELD_OUT_TEST_RESULTS.md` |
| AI model development — failure analysis | `06_AI_Model_Development/Failure_Analysis/` | `FAILURE_ANALYSIS.md` |
| Current system implementation | `07_Current_System_Implementation/` | `CURRENT_SYSTEM_IMPLEMENTATION.md` |
| Experimental results (consolidated) | `08_Experimental_Results/` | `RESULTS_SUMMARY.md` |
| Limitations | `09_Limitations/` | `DATASET_LIMITATIONS.md` |
| Stage 1 review / conclusion | `10_Stage_1_Review/` | `STAGE_1_CLOSE_OUT.md` |
| Figures for the report | `11_Figures_and_Graphs/` | 7 PNG charts, see folder README |
| Tables for the report | `12_Tables/` | Markdown tables, see folder README |
| Screenshots (application UI) | `13_Screenshots/` | placeholder — not yet captured |
| Viva preparation | `14_Viva_Preparation/` | `VIVA_QUESTIONS_STAGE_1.md` |
| Future work | `15_Future_Work/` | `FUTURE_WORK.md` |
| Whole-system analysis (3 Oct 2026) | `16_System_Analysis/` | `SYSTEM_ANALYSIS_2026-10-03.md`, `fig_domain_shift.png` |
| Hardware commissioning (physical tests) | `17_Hardware_Commissioning/` | `README.md` test sheets |
| **Every result, where it is, how strong** | `../RESULTS_INDEX.md` | one row per result |

## Document list

1. `01_Project_Overview/PROJECT_OVERVIEW.md`
2. `02_Problem_Statement_and_Objectives/PROBLEM_STATEMENT_AND_OBJECTIVES.md`
3. `04_System_Architecture/SYSTEM_ARCHITECTURE.md`
4. `05_Dataset_and_Data_Preparation/DATASET_DOCUMENTATION.md`
5. `06_AI_Model_Development/Classification_Methodology/AI_METHODOLOGY.md`
6. `06_AI_Model_Development/Training_Configuration/TRAINING_CONFIGURATION.md`
7. `06_AI_Model_Development/Architecture_Benchmark/MODEL_ARCHITECTURE_BENCHMARK.md`
8. `06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md`
9. `06_AI_Model_Development/Failure_Analysis/FAILURE_ANALYSIS.md`
10. `09_Limitations/DATASET_LIMITATIONS.md`
11. `10_Stage_1_Review/STAGE_1_CLOSE_OUT.md`
12. `15_Future_Work/FUTURE_WORK.md`
13. `14_Viva_Preparation/VIVA_QUESTIONS_STAGE_1.md`
14. `00_Project_Index/BLACKBOOK_CONTENT_OUTLINE.md`

Supporting (created alongside the required 14, to satisfy the requested
folder structure — clearly marked where content is not yet available):

- `03_System_Requirements/SYSTEM_REQUIREMENTS.md`
- `07_Current_System_Implementation/CURRENT_SYSTEM_IMPLEMENTATION.md`
- `08_Experimental_Results/RESULTS_SUMMARY.md`
- `11_Figures_and_Graphs/README.md` + 7 PNG charts
- `12_Tables/` table documents

## Source of truth for all numbers

Every metric quoted anywhere in this workspace originates from one of:

- `projects/om_bottle/models/20260919-164511/metrics.json` (EfficientNet-B0)
- `projects/om_bottle/models/20260919-222031/metrics.json` (EfficientNet-B1)
- `projects/om_bottle/models/20260920-102102/metrics.json` (MobileNetV3-Small)
- `projects/om_bottle/models/20260920-110121/metrics.json` (ResNet18)
- The held-out test evaluations run against each checkpoint's own
  validation-derived thresholds (`train.evaluate_test()` / equivalent
  scoring against `test_paths`), confirmed in the Stage 1 session.
- `dataset.load_labels()` / `dataset.scene_map()` read-only queries against
  `projects/om_bottle/labels.csv`.

No number in this workspace was estimated, assumed, or extrapolated. Where a
figure was not available at documentation time, it is marked **"To be
added"** rather than guessed.
- `18_Model_Training_Summary/` all trained models: epochs, settings, test metrics (generated from `models/MODEL_REGISTRY.json`)
