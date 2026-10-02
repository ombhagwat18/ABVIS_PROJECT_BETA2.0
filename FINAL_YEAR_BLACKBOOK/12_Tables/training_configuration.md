# Training Configuration

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 3e-4 |
| Optimizer | AdamW, weight decay 1e-4 |
| LR schedule | Cosine annealing |
| Test fraction | 0.15 |
| Split seed | 0 |
| Epoch budget | 25 |
| Loss | BCEWithLogitsLoss, per-class pos_weight |
| Mixed precision | Enabled (CUDA) |

| Environment | Value |
|---|---|
| Python | 3.8.0 |
| PyTorch | 2.4.1+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM) |

| Model | Patience | Epochs run | Best epoch |
|---|---|---|---|
| EfficientNet-B0 | not used | 25 | 8 |
| EfficientNet-B1 | not used | 25 | 3 |
| MobileNetV3-Small | not used | 25 | 7 |
| ResNet18 | 5 | 8 | 3 |
