# Multi-project inspection trainer — design

**Date:** 2026-08-09
**Diagram:** `docs/design/diagrams/pipeline.drawio`

## Goal

One admin, many inspection projects. Create a named project, upload a `+ve`
folder and a `-ve/<Class>/` tree, get `labels.csv` written automatically, then
label, train and run live — all the existing tabs, scoped to the active project.

## What a project owns

```
projects/<slug>/
  project.json          display name
  images/
    +ve/                good — every column 0
    -ve/<Class>/        one sub-folder per class -> one column
    _inbox/             unlabelled: reviewed = 0
  labels.csv            source of truth after the first import
  config.json           roi, roi_frame, input_wh, cache_wh, thresholds, active_model
  models/<stamp>/       model.pt, metrics.json, labels.csv snapshot
  cache/ thumbs/ scenes.json
projects/active.txt     slug of the project the app opens on
```

Nothing is shared between projects. A project is a directory; deleting one is
deleting a folder.

## Core mechanic

Every path in the app already funnels through ten module constants at the top of
`dataset.py`, read as `D.NAME` at call time. `use_project(slug)` rebinds those
globals and clears the config cache. No other module changes to become
project-aware — `gui.py`, `train.py`, `infer.py` and `calibrate.py` follow for
free.

## Folder → CSV

`_implied(rel)` maps a path to the label its position asserts:

| Path | Column set | `reviewed` |
|---|---|---|
| `+ve/x.jpg` | none | 1 |
| `-ve/Tilt cap/x.jpg` | `tilt_cap` = 1 | 1 |
| `-ve/x.jpg` *(no class)* | none | **0** |
| `_inbox/x.jpg` | none | 0 |

A file dropped straight into `-ve/` with no class folder goes to the inbox, not
to good. Calling an unclassified reject "good" would teach the model that a
defective bottle passes.

Columns are the union of the `-ve/` sub-folder names and the columns already in
`labels.csv`, so an empty class folder still creates its column and a new
sub-folder needs no code change.

### Folders seed once; `labels.csv` is the truth

`sync_new_images` applies the implied label **only to paths not already in the
CSV**. An existing row is never rewritten by a rescan.

This is not a detail. A folder can express exactly one defect per image, but the
model is multi-label. This project already paid for that: 19 `Water Level` frames
were also visibly `tilt_cap`, the folder could not say so, and it cost real
precision (docs/design/PLAN.md §First results). Second defects get ticked in the Label tab,
and a rescan must not wipe them.

Consequences for class edits, so a folder cannot resurrect a deleted column:

- **rename** a class → rename its `-ve/` folder too.
- **delete** a class → drop the column, move its images to `_inbox/`
  (`reviewed = 0`), remove the folder. Those images are not good; they are
  undecided.

## Per-project input size

`input_wh` is already read from config. `cache_wh` becomes config-derived:

```
cache_wh(cfg) = cfg["cache_wh"] if it is >= input_wh, else round(w*1.125), round(h*1.107)
```

The default stays exactly `(216, 496)`, so every existing checkpoint sees the
identical framing. A project with a wide input (a carton at 448×192) would
otherwise `center_crop` a 216-wide cache to 448 and silently get a short frame.

`calibrate.py`'s edge-energy trick assumes a tall object on a dark backdrop, so
non-bottle projects get a manual ROI: the Data health tab already has x/y/w/h
entries and a Save button. No new editor.

## UI

- Header gains a project dropdown and **+ New project** (name → slug → tree created → switched to).
- Label tab gains **Upload** (files or a folder, into `+ve/` or a chosen class) and **Delete** (wires the already-written, path-escape-guarded `delete_images`).
- Switching project calls `use_project` then the existing `App.reload()`. The camera stops first — a running inference thread holds the old project's model.

## Migration

`migrate.py`, run once, guarded, `rename` not copy (987 MB):

```
All Datasets/Good Bottle/  -> projects/om-bottle/images/+ve/
All Datasets/<Other>/      -> projects/om-bottle/images/-ve/<Other>/
All Datasets/_captured/    -> projects/om-bottle/images/_inbox/
data/*, models/*           -> projects/om-bottle/
```

`labels.csv` paths are rewritten to match, preserving every existing label and
`reviewed` flag. `FOLDER_MAP` is deleted — folder position is the mapping now.

## Testing

Each module keeps its `demo()` self-check. New assertions:

- a two-project temp tree stays isolated after `use_project` both ways
- `-ve/Class/` import sets exactly that column; `-ve/loose.jpg` lands unreviewed
- an edited row survives a rescan that adds a new file
- deleting a class moves its images to `_inbox/` and the column does not return
- `cache_wh` for a wide input is >= that input in both axes

## Out of scope

- Auth. One admin, one PC — a project dropdown is not a permission model.
- Deleting a project from the UI. It is a folder; delete it in Explorer.
- Reviving `app.py` / `index.html`. Desktop only; they stay unused.
- A test split. 112 scenes is too few — revisit near 300.
