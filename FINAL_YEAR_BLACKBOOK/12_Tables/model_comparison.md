# Model Comparison — Stage 1

| Model | Val macro-F1 | Best epoch/run | Training time | Test macro-F1 | Test macro-P | Test macro-R | Exact-match | Total FP | Total FN | Test latency | Test FPS | Model size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 1.000 | 8/25 | 216.2 s | 0.935 | 0.887 | 1.000 | 89.1% (212/238) | 28 | 0 | 13.89 ms | 72.0 | 15.57 MB |
| EfficientNet-B1 | 1.000 | 3/25 | 453.9 s | 0.845 | 0.786 | 0.935 | 68.1% (162/238) | 74 | 10 | 17.71 ms | 56.5 | 25.24 MB |
| MobileNetV3-Small | 0.864 | 7/25 | 192.2 s | 0.632 | 0.587 | 0.713 | 67.2% (160/238) | 82 | 36 | 14.04 ms | 71.2 | 5.93 MB |
| ResNet18 | 0.933 | 3/8 (early-stopped) | 145.3 s | 0.773 | 0.712 | 0.857 | 84.0% (200/238) | 38 | 12 | 16.71 ms | 59.8 | 42.72 MB |

Checkpoints: EfficientNet-B0 `20260919-164511`, EfficientNet-B1
`20260919-222031`, MobileNetV3-Small `20260920-102102`, ResNet18
`20260920-110121`. ConvNeXt-Tiny was not evaluated.
