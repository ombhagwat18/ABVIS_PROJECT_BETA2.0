# System check, 2026-10-04

Run on the development PC (Windows 11, Python 3.8, RTX 3050 4 GB) immediately before the first hardware integration.
Everything below is a **software self-check with fakes**; none of it is a hardware test.

## 1. Self-checks: 17 / 17 pass (exit code 0)

`dataset.py`, `infer.py`, `inspection_trace.py`, `detect.py`, `decision.py`, `segment.py`, `machine_cycle.py`, `vision_data.py`,
`charts.py`, `train.py --demo`, `calibrate.py --demo`, `stage2_dataset/seg_pipeline.py --selftest`, `gui.py --selftest` (builds all 11
tabs incl. Machine/Production against a fake PLC), `plc.test_simulation`, `plc.test_service`, `plc.commissioning --selftest`,
`plc.handshake_test --fake`.
Not run: `python -m plc.handshake_test --real` / `plc.sim_comm_test` (need the ISPSoft simulator) and
`stage2_dataset/review_app.py --selftest` (known stale by design, see CLAUDE.md).

## 2. Models: what exists (from `models/MODEL_REGISTRY.json`)

Stage 1 multi-label classifier, project `om_bottle`. Validation F1 is saturated (~1.0) -- **judge by the held-out test macro-F1**.

| Architecture | Epochs | Val macro-F1 | Held-out test macro-F1 | GPU latency (ms) |
|---|---|---|---|---|
| EfficientNet-B0 (20260919-164511) | 25 | 1.000 | **0.935** | 32.8 |
| EfficientNet-B1 | 25 | 1.000 | 0.845 | 31.3 |
| ConvNeXt-Tiny | 25 | 1.000 | 0.764 | 19.8 |
| ResNet18 | 8 | 0.933 | 0.773 | 12.6 |
| MobileNetV3-Small | 25 | 0.864 | 0.632 | 21.7 |

Five earlier checkpoints (legacy split, no held-out test) are kept for history. Details: `FINAL_YEAR_BLACKBOOK/06_AI_Model_Development/`.

Stage 2 detector (bottle/cap/label, 594 images, 1,994 boxes): YOLOv8n (53 epochs, best 33) and YOLOv8s are both trained; provenance in
`models/stage2_yolo/MODEL_PROVENANCE.json`. Observational only; never produces PASS/REJECT.

**Not trained: segmentation (label outline).** 546 polygons exist but **0 are human-reviewed** (`seg_annotations.json`); seeded polygons are
deliberately never exported, so training now would teach the model the seeding heuristic, not the label. Review them in Annotate first
(`python stage2_dataset/seg_pipeline.py`), then train. I did not train it for that reason.

## 3. Findings that matter before the machine is switched on

| # | Finding | Severity |
|---|---|---|
| 1 | Ladder presets T0 K150 / T1 K50 (15 s / 5 s) vs the stated K15 / K5; with them every REJECT masks triggers for ~15 s | **High** -- fix in ISPSoft (owner's file) |
| 2 | Both EMEETs on one USB 2.0 hub: only one 1080p stream | **High** -- separate USB ports |
| 3 | Classifier trained on other images/lighting; no data from this enclosure | **High** -- collect + fine-tune |
| 4 | `capture_delay_s` is global; staggered cameras need per-camera delay | Medium -- small code change |
| 5 | `inspection_to_reject_mm`, `conveyor_mm_s` = 0 (unmeasured) | Medium -- measure |
| 6 | No E-stop input, heartbeat/watchdog, fault latching, alarm history | Medium -- design + ladder |
| 7 | Camera focus/exposure lock written but never tried on the EMEETs | Medium |
| 8 | HMI is an engineer console; no operator-only view | Medium |
| 9 | `test` split: 238 images / 23 scenes; `missing_cap` has 0 positives; no real multi-defect images | Documented limitation |
| 10 | Dead code: `app.py`, `index.html` (no auth on write endpoints) | Low -- delete on your OK |
