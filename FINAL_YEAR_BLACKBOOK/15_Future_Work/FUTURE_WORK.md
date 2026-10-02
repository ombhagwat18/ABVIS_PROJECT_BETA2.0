# Future Work

None of the items below have been implemented. They are recorded here to
state the project's intended direction, separately and clearly from the
Stage 1 work that has actually been completed and documented elsewhere in
this workspace.

## Dataset improvements

1. Increase `missing_cap` samples — currently zero positive images exist;
   the class cannot be learned or evaluated until examples are collected.
2. Increase `missing_label` samples and scene diversity — currently 30
   images across only 2 scenes, with zero validation positives in the
   current split.
3. Add genuine multi-defect images — currently 0 of 1,143 images carry more
   than one defect flag, so the classifier's multi-label capability has
   not been experimentally validated against a real multi-defect bottle.
4. Improve overall dataset balance and scene count across all classes.

## Platform capability expansion

5. Develop universal annotation support (beyond the current whole-image
   labelling workflow), to support bounding-box and/or segmentation
   labelling for future detection/segmentation models.
6. Add YOLO-based object detection.
7. Add YOLO-based (or equivalent) segmentation.
8. Develop a model registry (structured tracking of models across
   projects/architectures/versions, beyond the current per-project
   checkpoint folder).
9. Develop an inspection pipeline builder (configurable sequencing of
   acquisition, inference, and decision steps).
10. Develop a decision engine (rules or logic that combine multiple model
    outputs, or model output with other sensor data, into a final
    accept/reject decision).

## Industrial integration

11. Integrate camera triggering (synchronised capture, e.g. from a
    conveyor-position sensor) — the current system operates on continuous
    camera streams or static image datasets.
12. Integrate PLC communication, to report inspection verdicts to, and
    receive control signals from, a line PLC.
13. Integrate conveyor control and pneumatic rejection actuation.
14. Develop production monitoring (line-level throughput, reject-rate, and
    system-health dashboards).
15. Add traceability and inspection history (persistent per-unit
    inspection records, beyond the current per-checkpoint metrics/mistake
    logs).
16. Perform physical industrial validation of the complete system on an
    actual production or pilot line, beyond the static-image and
    lab-camera evaluation performed so far.

## Immediate next step (Stage 2, not yet started)

Planning the universal annotation and YOLO object-detection capability
(items 5–6 above), building on the dataset and baseline classifier
established in Stage 1.
