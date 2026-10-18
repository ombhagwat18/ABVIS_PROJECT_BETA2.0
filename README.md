# NexaInspect (ABVIS): Automated Bottle Visual Inspection System

A camera looks at every bottle on a conveyor. A defective bottle is pushed off the line by a pneumatic cylinder.
A bottle the system cannot vouch for is treated the same way: an uninspected bottle never passes as good.

First product: **250 ml water bottles**. Defects detected: missing cap, tilted cap, damaged label, skewed label,
missing label, damaged bottle, skewed bottle and wrong water level.

## What it is

One Windows desktop application that covers the whole inspection workflow:

- **Data**: label images, manage defect classes, annotate boxes and polygons (with model-assisted auto-annotation).
- **Models**: a Stage 1 multi-label classifier (EfficientNet-B0) and a Stage 2 component detector (YOLOv8n:
  bottle, cap, label), a model registry with approval gates and rollback.
- **Live inspection**: two 4K cameras (1080p stream), one PASS / REJECT / FAULT decision per bottle, shown with the
  defect and its location on the picture.
- **Machine control**: a Delta DVP-SS2 PLC over Modbus (serial or simulator) starts the conveyor, triggers the
  inspection and fires the reject cylinder; the application can also operate the PLC inputs and outputs by hand.
- **Record**: every bottle, alarm and evidence picture is stored (SQLite) and reported on the History page.

## Run

```
run.bat
```

Python 3.8, Windows 11. `run.bat` installs what is missing and starts the application. All project documentation
(black book, results, hardware, PLC, guides) is in `FINAL_YEAR_BLACKBOOK/`.
