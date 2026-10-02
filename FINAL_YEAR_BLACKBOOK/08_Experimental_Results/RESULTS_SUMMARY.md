# Experimental Results Summary — Stage 1

This is a consolidated pointer document. The authoritative, detailed
results are in:

- `06_AI_Model_Development/Architecture_Benchmark/MODEL_ARCHITECTURE_BENCHMARK.md`
- `06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md`
- `06_AI_Model_Development/Failure_Analysis/FAILURE_ANALYSIS.md`
- `12_Tables/` (machine-readable-style tables)
- `11_Figures_and_Graphs/` (charts)

## Headline table (held-out test set, 238 images / 23 scenes, validation-derived thresholds)

| Model | Val macro-F1 | Best epoch/run | Training time | Test macro-F1 | Test macro-P | Test macro-R | Exact-match | Total FP | Total FN | Test FPS | Model size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 1.000 | 8/25 | 216.2 s | 0.935 | 0.887 | 1.000 | 89.1% | 28 | 0 | 72.0 | 15.57 MB |
| EfficientNet-B1 | 1.000 | 3/25 | 453.9 s | 0.845 | 0.786 | 0.935 | 68.1% | 74 | 10 | 56.5 | 25.24 MB |
| MobileNetV3-Small | 0.864 | 7/25 | 192.2 s | 0.632 | 0.587 | 0.713 | 67.2% | 82 | 36 | 71.2 | 5.93 MB |
| ResNet18 | 0.933 | 3/8 (early-stopped) | 145.3 s | 0.773 | 0.712 | 0.857 | 84.0% | 38 | 12 | 59.8 | 42.72 MB |

## Selected baseline

EfficientNet-B0, checkpoint `20260919-164511` — see
`10_Stage_1_Review/STAGE_1_CLOSE_OUT.md` for the full rationale.
