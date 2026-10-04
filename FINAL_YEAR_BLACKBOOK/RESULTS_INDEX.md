# Results Index

One row per result that can go into the black book: where it is, when it was made, how strong the
evidence is, and how to reproduce it. Results stay where they were produced; this file only points
to them. **Add a row whenever a new result is produced.**

Evidence levels: **FAKE** = software self-test with fake hardware · **OFFLINE** = model scored on
stored images · **REAL-CAMERA** = real EMEET frames, no machine · **SIMULATOR** = ISPSoft simulator ·
**PHYSICAL** = the real machine. Only PHYSICAL proves the machine works.

## Dataset

| Result | File | Date | Level | Reproduce |
|---|---|---|---|---|
| Stage 1 class counts (1,143 images, 8 defects, missing_cap = 0) | `12_Tables/dataset_class_distribution.md`, `projects/om_bottle/labels.csv` | 2026-09 | OFFLINE | `python dataset.py` |
| Stage 1 scene split | `12_Tables/dataset_split.md` | 2026-09 | OFFLINE | `python dataset.py` |
| Stage 2 detection dataset (594 img, 1,994 boxes, 25/7/7 scenes) | `stage2_dataset/` reports, `05_Dataset_and_Data_Preparation/` | 2026-10-02 | OFFLINE | `python stage2_dataset/validate_yolo_export.py` |
| New image audit (606 screenshots, duplicates, scenes) | `06_New_Dataset_Audit/AUDIT_REPORT.md` | 2026-10 | OFFLINE | `06_New_Dataset_Audit/audit_script.py` |
| v1 → v2 cap-box correction (6 class fixes, 709 boxes tightened) | `stage2_dataset/corrections_v2.json`, `stage2_dataset/review_v2/` | 2026-10-03 | OFFLINE (automatic, needs human review) | `python stage2_dataset/annotations_v2.py build` |
| Domain-shift figure (4 image domains) | `16_System_Analysis/fig_domain_shift.png` | 2026-10-03 | — | see `16_System_Analysis/` |

## Classification (Stage 1)

| Result | File | Date | Level | Reproduce |
|---|---|---|---|---|
| 5-architecture held-out comparison (B0 0.935 macro-F1) | `12_Tables/model_comparison.md`, `06_AI_Model_Development/Architecture_Benchmark/` | 2026-09/10 | OFFLINE | `python model_bench.py cls-train --arch <arch>` + `cls-test` |
| Per-class test results, EfficientNet-B0 | `12_Tables/b0_per_class_test_results.md` | 2026-09 | OFFLINE | `train.evaluate_test()` |
| Failure analysis per model | `12_Tables/per_model_failure_summary.md`, `06_AI_Model_Development/Failure_Analysis/` | 2026-09 | OFFLINE | — |
| Charts (F1, FP/FN, FPS, size, train time) | `11_Figures_and_Graphs/*.png` | 2026-09 | OFFLINE | `11_Figures_and_Graphs/_generate_charts.py` |
| Model registry (all checkpoints + metrics) | `models/MODEL_REGISTRY.json` | 2026-10-03 | OFFLINE | `python model_bench.py registry` |
| Classifier false scores on empty desk (0.79 / 0.70 / 0.65) | `captures/NO_BOTTLE_DESK_20261003-134540/result.json` | 2026-10-03 | REAL-CAMERA | `python machine_cycle.py --bench NO_BOTTLE 2 3` |

## Detection (Stage 2)

| Result | File | Date | Level | Reproduce |
|---|---|---|---|---|
| YOLOv8n val/test mAP, provenance | `models/stage2_yolo/MODEL_PROVENANCE.json`, `models/stage2_yolo/test_eval/` | 2026-10-02 | OFFLINE | `python yolo_stage2_train.py` |
| YOLOv8s candidate | `models/stage2_yolo/candidate_yolov8s.json`, `bench_yolov8s_testeval/` | 2026-10-03 | OFFLINE | `python model_bench.py yolo --model yolov8s.pt` |
| **Defect-level score: missing_cap 0/12 (v8n and v8s); missing_label 35/37 (v8n 5 FP, v8s 0 FP)** | `models/stage2_yolo/defects_yolov8n_v1_test.json`, `defects_yolov8s_v1_test.json` | 2026-10-03 | OFFLINE | `python model_bench.py det-defects --weights <pt> --tag <name> --ann v1` |
| Capless test image rejected only for missing_label | `captures/GT_NOCAP_20261003-134650/result.json` | 2026-10-03 | OFFLINE | `python machine_cycle.py --bench GT_NOCAP <img>` |
| YOLOv8n on v2 (tight cap boxes) | `models/stage2_yolo/bench_yolov8n_v2/` | in training 2026-10-03 | OFFLINE | `python model_bench.py yolo --model yolov8n.pt --data v2` |
| Detector latency on EMEET 1080p (~75 ms/frame) | `captures/NO_BOTTLE_DESK_20261003-134540/result.json` | 2026-10-03 | REAL-CAMERA | `python machine_cycle.py --bench` |

## Cameras

| Result | File | Date | Level | Reproduce |
|---|---|---|---|---|
| 2× EMEET survey: 1080p alone OK; together only 640x480 (shared USB 2.0 hub) | `captures/survey_20261003-133658/survey.json` | 2026-10-03 | REAL-CAMERA | `python bench.py --survey 2 3` |

## Decision engine, machine cycle, PLC, GUI

| Result | File | Date | Level | Reproduce |
|---|---|---|---|---|
| Decision rules, recipe-driven detection, frame vote, camera fusion | `decision.py` self-test | 2026-10-03 | FAKE | `python decision.py` |
| Machine cycle: FIFO, deadlines, NOT INSPECTED, FAULT handling, timing check | `machine_cycle.py` self-test | 2026-10-03 | FAKE | `python machine_cycle.py` |
| PLC protocol, write policy, service contract | `plc/test_*.py` | 2026-10-03 | FAKE | `python -m plc.test_simulation`, `python -m plc.test_service` |
| Decoded ladder, simulator measurements, fault matrix | `docs/roadmap/PLC_COMMUNICATION.md` | 2026-10-03 | SIMULATOR / decoded file | `python -m plc.handshake_test --real` (simulator) |
| GUI 11-tab build incl. Production line against fake PLC | `gui.py --selftest` | 2026-10-03 | FAKE | `python gui.py --selftest` |

## System analysis and physical machine

| Result | File | Date | Level |
|---|---|---|---|
| Whole-system analysis, domain shift, timing, universal recipe | `16_System_Analysis/SYSTEM_ANALYSIS_2026-10-03.md` | 2026-10-03 | mixed, labelled per item |
| Hardware commissioning sheets (camera, PLC, bottle runs) | `17_Hardware_Commissioning/` | — | PHYSICAL: **none yet** |
