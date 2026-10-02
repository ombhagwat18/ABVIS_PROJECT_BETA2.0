# System Requirements

## Software (as used for Stage 1 training and evaluation)

| Component | Version |
|---|---|
| Python | 3.8.0 |
| PyTorch | 2.4.1+cu124 |
| CUDA (PyTorch build) | 12.4 |
| torchvision | To be added (installed alongside PyTorch; exact version not separately recorded during Stage 1) |
| OpenCV (opencv-python) | To be added |
| CustomTkinter | To be added |
| Pillow | To be added |

These are the versions confirmed present in the training environment at the
time the Stage 1 checkpoints were produced (`python -c "import torch;
print(torch.__version__, torch.version.cuda)"` reported `2.4.1+cu124`,
`12.4`). Package versions marked "To be added" were not queried during this
documentation pass and should be recorded with `pip freeze` before this
section is finalised for submission.

## Hardware (as used for Stage 1 training)

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| VRAM | 4 GB (confirmed via `torch.cuda.get_device_properties`) |
| CPU | To be added |
| RAM | To be added |
| Camera(s) | To be added — not used for Stage 1 (Stage 1 trained/evaluated on a pre-existing labelled image dataset, not live camera capture) |

## Functional requirements (implemented, Stage 1 / current application)

- Import and label bottle images against a configurable set of defect
  classes.
- Train a multi-label defect classifier on labelled images.
- Evaluate a trained checkpoint against a genuinely held-out test set.
- Run live multi-camera inspection with a PASS/FAIL verdict (existing
  application capability, not itself re-verified as part of the Stage 1
  benchmark, which used static labelled images).

## Functional requirements (future stages, not implemented)

See `15_Future_Work/FUTURE_WORK.md` — object detection/segmentation
annotation, model registry, PLC/conveyor/pneumatic integration, production
monitoring and traceability.

## Non-functional observations from Stage 1

- Inference latency for the four benchmarked architectures, measured under
  the same live-scoring methodology on the 238-image held-out test set,
  ranged from 13.89 ms (EfficientNet-B0) to 17.71 ms (EfficientNet-B1) per
  image on the hardware above. See
  `06_AI_Model_Development/Test_Evaluation/HELD_OUT_TEST_RESULTS.md`.
