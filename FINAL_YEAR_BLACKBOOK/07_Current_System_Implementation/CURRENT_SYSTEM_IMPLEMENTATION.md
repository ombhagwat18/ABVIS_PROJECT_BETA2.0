# Current System Implementation

This describes only functionality that exists in the codebase today. See
`04_System_Architecture/SYSTEM_ARCHITECTURE.md` for the component diagram
this summarises.

## Implemented and in use

- Desktop labelling application with a paginated image grid, multi-select,
  keyboard shortcuts, and bulk label apply/clear/delete.
- Defect-class management (add/rename/delete classes; deleting a class
  moves its images to an unlabelled inbox rather than silently marking
  them "good").
- Classifier training from the GUI or from the command line (`train.py`),
  including the Stage 1 architecture-selection capability (`--arch`,
  `--patience`).
- Per-defect metrics table, mistake gallery, and checkpoint version list
  with rollback ("Use" a previous checkpoint).
- Loss/macro-F1 curves, per-defect recall bars, confusion matrix, and a
  "re-test every model on today's labels" comparison view.
- Multi-camera live inspection dashboard with a combined PASS/FAIL verdict,
  a real-time performance panel, and a snapshot-to-label capture panel.
- ROI calibration (`calibrate.py`) and a near-duplicate-frame ("scene")
  health scan.
- Camera benchmarking across resolution/FPS combinations.

## Implemented but not part of the Stage 1 benchmark

Live camera capture and the multi-camera fusion verdict exist in the
application but were not exercised during the Stage 1 benchmark, which
evaluated static, already-labelled images. Their behaviour was not
re-verified as part of this documentation package.

## Not implemented

Object detection, segmentation, a model registry, a decision engine, PLC
communication, conveyor control, pneumatic rejection, production
monitoring, and inspection traceability. See
`15_Future_Work/FUTURE_WORK.md`.

## Screenshots

Application screenshots have not yet been captured for this documentation
package — screenshots still to be captured (Label, Train, Analysis, Live, Production tabs).
