# Vision dataset readiness (classification / detection / segmentation)

Audit of 2026-10-03, from the working tree. Rebuild every number here with:

```
python vision_data.py build        # audit -> vision_dataset/manifests/{unified_manifest.csv, dataset_report.json}
python vision_data.py export-cls   # derived YOLO-classification tree (hard links; --mode crop for ROI PNG crops)
python vision_data.py validate     # 14 checks, incl. that no checkpoint or source annotation changed
```

`vision_dataset/` is derived output (not in git). Nothing in `projects/`, `stage2_dataset/` or `models/` is written.
No model was trained in this audit.

## 1. What exists

| | Classification | Detection (Stage 2) | Segmentation |
|---|---|---|---|
| Source | `projects/om_bottle/labels.csv` + `images/` | `stage2_dataset/annotations.json`, `split.json`, `yolo_export/` | none |
| Images | 1,143 (2537x1927 JPEG, black backdrop) | 594 approved of 606 (1780x1000 PNG, light backdrop) | 0 |
| Labels | 8 columns; every image has exactly one class | 1,994 boxes: bottle 739, cap 709, label 546 | 0 polygons |
| Scenes | 112 (106 after cross-folder merge) | 39 | - |
| Split | train 688 / val 237 / test 218 (new, scene-based) | train 418 / val 89 / test 87 (existing) | - |
| Ready to train | YES (train + val), with the issues below | YES (already trained: `stage2_best.pt`) | NO |

The app's *active* project is `bottle_detection` (9 unlabelled inbox images, 1 annotated). The labelled classification
data is the `om_bottle` project. `All Datasets/` is the pre-migration copy (1,145 files; the 2 extra are not in the project).

## 2. Classification: findings

* 1,143 rows, all reviewed, 0 malformed, 0 missing files, 0 files without a row.
* Per class: damaged_bottle 319, damaged_label 229, skewed_bottle 166, tilt_cap 145, water_level 112, good 72,
  skewed_label 70, missing_label 30, **missing_cap 0**.
* **missing_cap has no images**, so it cannot be a class in any trained model. It needs captures first.
* **0 multi-label rows.** The CSV is multi-label by design but the data is single-label, which is what makes a YOLO
  classification (one folder per class) export lossless. A future multi-defect image would not be exportable this way;
  `vision_data.py` refuses such rows rather than picking one label.
* 1 byte-identical duplicate pair (`-ve/Skewed Label/snap_099.jpg` = `snap_100.jpg`): flagged, second one excluded from the export.
* **Cross-folder scenes.** `dataset.scene_map` groups frames only inside one class folder. By its own rule (32x32 thumbnail
  mean difference < 4.0), 403 image pairs in *different* folders are the same frame-run: `+ve`<->Skewed Label, Damaged Label<->Tilt
  Cap, Missing Label<->Skewed Bottle, Skewed Label<->Tilt Cap, Tilt Cap<->Water Level. Whether these are the same bottle filed under
  two classes or look-alike bottles cannot be told at that resolution; either way they must not straddle a split. The new split
  merges them (112 -> 106 scenes; 27 images moved). **The existing Stage 1 checkpoints were trained with the per-folder split,
  so their validation scores may include this leak.** They were not retrained.
* **Thin classes.** missing_label has 2 scenes (30 images): it is in train (18) and val (12) but **not in test**, so
  `yolo val split=test` would mis-index classes; test needs a name-mapped evaluation or more missing_label scenes. good has 8
  scenes, skewed_label 8.
* Viewed samples (7 Skewed Bottle, 6 Missing Label): all are label-less bottles and look alike. Worth a
  human check of what distinguishes those two classes before trusting a classifier on them.
* Export: `vision_dataset/classification/{train,val,test}/<class>/`, 1,142 hard links (0 extra disk, bytes identical to the
  source). Frames are full 2537x1927; the bottle is the middle third. Train at imgsz >= 448, or use `--mode crop`
  (lossless PNG of the calibrated ROI, the same crop Stage 1 uses).

## 3. Detection: findings

* 594 images, all reviewed, 1,994 boxes, classes `bottle/cap/label` = ids 0/1/2; split 418/89/87 over 25/7/7 scenes, no scene
  or image in two splits; `yolo_export/` label files equal `annotations.json` and images equal `clean/` byte for byte;
  the existing 19-check validator passes; `annotations.json` and `stage2_best.pt` hashes match `MODEL_PROVENANCE.json`.
* 0 empty images, 0 duplicate boxes, 0 invalid boxes. **91 boxes have an edge up to 4% past the frame** (bottles cut by the
  border; centre inside). Ultralytics clips these at load; not an error, recorded as a warning.
* 140 images have no label box, 51 have more than one bottle.
* The 12 rejected review images are in the manifest with no split.

## 4. How the two datasets relate

**They share nothing.** No byte-identical image; the nearest cross-dataset thumbnail distance is 79 (same-scene threshold 4.0).
Different camera framing, resolution, backdrop and bottles. Consequences:

* No image has both a classification label and a detection annotation. The unified manifest (1,749 rows) carries one row per
  image with the columns of the dataset it came from; the other columns are empty, not guessed.
* The detector was trained on the light-backdrop set; the classifier on the black-backdrop set. **Neither model is validated
  on the other's images**, and the future chain (detect -> crop -> classify) needs both on the *production camera's* domain.
  That is the main remaining data risk, larger than any item above.
* The review manifest's `likely_*` flags on Stage 2 images are heuristics, not labels; they are copied to `notes` only.

## 5. Which defects need which task

| Defect | Classification | Detection | Segmentation | Reason |
|---|---|---|---|---|
| missing_cap | yes | supporting | no | Presence question. A missing cap *box* is not proof (occlusion, miss) - project rule. No images yet |
| tilt_cap | yes (cap crop) | yes: gives the cap crop | optional | An axis-aligned box barely changes with a few degrees of tilt; a classifier on the cap crop is cheaper than polygons |
| missing_label | yes | supporting (label box absent) | no | Presence question; 140 Stage 2 images already have no label box |
| skewed_label | yes | yes: label crop | **yes - label outline** | Skew is an angle. A box cannot express it; a 4-6 point label polygon gives it directly |
| damaged_label | yes | yes: label crop | useful - same label outline | Torn/missing area shows as the outline departing from a clean quadrilateral |
| water_level | yes | no | no | It is one horizontal level, not a region: classification now, a line/keypoint measurement later |
| damaged_bottle | yes | yes: bottle crop | optional - bottle outline | Dents change the silhouette, but the bottle is transparent and the outline is costly to trace |
| skewed_bottle | yes | yes: bottle box aspect/offset | optional - bottle outline | Lean shows in the bottle box already; an outline gives the axis angle more precisely |

Minimum useful polygon class: **label**. Bottle outline is a second, optional phase. Cap polygons are not recommended.

## 6. Segmentation annotation plan (nothing annotated yet)

* Images: the Stage 2 set, where boxes already exist. **454 images contain a label (546 label instances)**: train 328 / val 76 /
  test 50 images. Reuse the existing scene split unchanged.
* Phase 1 - label outline: 546 polygons of 4-8 points. Estimated 15-25 s each -> **about 2.5-4 hours** of annotation.
* Phase 2 (optional) - bottle outline: 739 polygons of 12-20 points, ~40-60 s each -> about 8-12 hours. Decide after phase 1 results.
* Tooling: Annotation Studio (`annotation_studio.py`) already draws and saves polygons, and `annotate.export_yolo_segmentation`
  writes YOLO-seg labels. Missing: a segmentation project holding the Stage 2 images, and a split-aware seg export + validator.
* Keep `stage2_dataset/annotations.json` frozen (its hash is pinned in `MODEL_PROVENANCE.json`): polygons go in a separate
  segmentation annotations file.
* Pre-annotation: the existing label **box** can seed a 4-corner starting polygon to drag into place. A seeded polygon is
  not an annotation: it must be stored unreviewed and never exported until a human has adjusted and confirmed it.
  No automatic mask model is installed; adding one is a separate decision.

## 7. Estimated training time (RTX 3050 Laptop, 4 GB) - estimates except where marked

| Task | Basis | Estimate |
|---|---|---|
| Detection, YOLOv8n | **measured**: 14.8 min for 53 epochs, 418 images, imgsz 640, batch 8 | ~15 min (already done) |
| Classification, YOLOv8n-cls | 688 train images; decoding 2537x1927 JPEGs dominates | ~30-60 min for 50 epochs with hard links; ~10-20 min with `--mode crop` |
| Segmentation, YOLOv8n-seg | 328 train images once annotated; seg head costs ~1.5-2x detection per image | ~20-30 min for 80 epochs |

Pass `workers=` to every Ultralytics `train()`/`val()` on this machine (see CLAUDE.md: the default exhausted the page file).
