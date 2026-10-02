# Training Configuration

## Environment (confirmed at training time)

| Item | Value |
|---|---|
| Python | 3.8.0 |
| PyTorch | 2.4.1+cu124 |
| CUDA | 12.4 |
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB |
| Execution mode | Fresh process per experiment (`python train.py ...`), CUDA-enabled |

## Hyperparameters (identical across all four Stage 1 architecture experiments)

| Hyperparameter | Value |
|---|---|
| Batch size | 32 |
| Learning rate | 3e-4 |
| Optimizer | AdamW, weight decay 1e-4 |
| LR schedule | Cosine annealing over the epoch budget |
| Test fraction | 0.15 |
| Split seed | 0 (deterministic) |
| Epoch budget | 25 |
| Loss | BCEWithLogitsLoss with per-class `pos_weight` |
| Mixed precision | Enabled on CUDA (`torch.amp`) |

## Per-experiment configuration

| Model | Epoch budget | Early-stopping patience | Epochs actually run | Best epoch |
|---|---|---|---|---|
| EfficientNet-B0 | 25 | not used (default full-schedule run) | 25 | 8 |
| EfficientNet-B1 | 25 | not used | 25 | 3 |
| MobileNetV3-Small | 25 | not used | 25 | 7 |
| ResNet18 | 25 | 5 | 8 (stopped early) | 3 |

"Best epoch" is the epoch whose weights were retained as the checkpoint,
selected by highest validation macro-F1 across the run — this checkpoint
selection logic is identical for every model and was not changed for the
early-stopping experiment.

## Early stopping (introduced for the ResNet18 experiment)

An opt-in `patience` parameter was added to `train.py`'s `run()` function
and its CLI (`--patience N`). It is disabled by default (`patience=None`),
so the three earlier experiments (B0, B1, MobileNetV3-Small) ran their full
25-epoch schedule under the unmodified code path. When enabled, training
stops after `N` consecutive epochs with no improvement in validation
macro-F1, and the checkpoint already retained (best validation epoch) is
saved as usual. Early stopping monitors **validation** macro-F1 only; the
held-out test set is never read during training or checkpoint selection,
for any of the four experiments.

## Reproducibility notes

- The scene-based split is derived deterministically (seed 0) from the
  dataset's current labels and folder layout at the time each experiment
  was run. Because the dataset was not modified between experiments, all
  four checkpoints share byte-identical `val_paths` and `test_paths` lists
  (confirmed by comparison during Stage 1).
- Each experiment was launched as an independent fresh Python process
  (`python train.py --epochs 25 --arch <arch> [--patience N]`), not as a
  continuation of a prior in-process run.
