"""Annotation data layer for detection/segmentation projects.

This module is deliberately independent of dataset.py's global-project
rebinding pattern: every function here takes an explicit `path` (or an
in-memory `data` dict) rather than reading a module-level "active project"
global. That keeps this module safe to exercise in a temporary directory
(see demo()) without any risk of touching a real project's files, and keeps
it decoupled from dataset.py until a later stage deliberately wires the two
together.

Nothing in dataset.py, train.py, infer.py, or labels.csv is read or written
by this module. Classification projects (labels.csv) are untouched by
design -- this file exists only to hold detection/segmentation annotations,
stored per-project in a separate annotations.json.

Schema (annotations.json)
--------------------------
{
  "schema_version": 1,
  "task": "detection" | "segmentation",
  "classes": ["scratch", "dent", ...],        # ordered -- index == YOLO class id
  "images": {
    "images/_inbox/0001.jpg": {
      "reviewed": true,
      "boxes": [                               # detection: YOLO-style, normalized [0,1]
        {"cls": "scratch", "x": 0.41, "y": 0.22, "w": 0.08, "h": 0.05}
      ],
      "polygons": [                            # segmentation: normalized [0,1] point list
        {"cls": "dent", "points": [0.10, 0.10, 0.20, 0.10, 0.20, 0.20]}
      ]
    }
  }
}

`x`/`y` are box centers (YOLO convention), `w`/`h` are box width/height, all
normalized to the image's own width/height. A polygon's `points` list is a
flat `[x1, y1, x2, y2, ..., xn, yn]` sequence, also normalized, with at
least 3 points (6 numbers).

Both `boxes` and `polygons` are always present (possibly empty) on every
image entry regardless of the project's declared `task` -- the task field
records project intent, but the exporters are named explicitly
(`export_yolo_detection` / `export_yolo_segmentation`) and only look at the
list that matches what they export, so a project is never blocked from
holding both kinds of geometry if a future stage wants that.

The schema is intentionally permissive about *extra* keys: an unrecognized
key on the top level, an image entry, a box, or a polygon is left alone by
load/save/validate, so future metadata (confidence scores, an annotator
name, a timestamp, ...) can be added later without breaking files written
by this version of the module.

Standard library only -- no new dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA_VERSION = 1
TASKS = ("detection", "segmentation")


# --------------------------------------------------------------------- errors

class AnnotationError(ValueError):
    """Raised for any schema/coordinate/class violation. Always a ValueError
    subclass so existing `except ValueError` call sites keep working."""


# ---------------------------------------------------------------- create/load/save

def init_annotations(task: str, classes: list[str]) -> dict:
    """A fresh, empty, valid annotations structure. Does not touch disk --
    call save_annotations() to write it."""
    if task not in TASKS:
        raise AnnotationError(f"unknown task {task!r} -- choices: {', '.join(TASKS)}")
    data = {"schema_version": SCHEMA_VERSION, "task": task,
            "classes": list(classes), "images": {}}
    validate_annotations(data)
    return data


def load_annotations(path: Path) -> dict:
    """Read and validate an annotations.json. Raises AnnotationError on any
    schema violation rather than handing back a structure callers must
    re-check themselves."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_annotations(data)
    return data


def save_annotations(path: Path, data: dict) -> None:
    """Validate, then write. A caller can never persist an invalid file
    through this function."""
    validate_annotations(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# -------------------------------------------------------------------- validate

def _num01(v, what: str) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise AnnotationError(f"{what} is not a number: {v!r}")
    if not (0.0 <= f <= 1.0):
        raise AnnotationError(f"{what} out of [0,1] range: {f!r}")
    return f


def _validate_box(box: dict, classes: list[str]) -> None:
    if not isinstance(box, dict):
        raise AnnotationError(f"box is not an object: {box!r}")
    if "cls" not in box:
        raise AnnotationError(f"box missing 'cls': {box!r}")
    if box["cls"] not in classes:
        raise AnnotationError(f"box references unknown class {box['cls']!r} -- "
                               f"known classes: {classes}")
    for key in ("x", "y", "w", "h"):
        if key not in box:
            raise AnnotationError(f"box missing {key!r}: {box!r}")
        _num01(box[key], f"box.{key}")
    if box["w"] <= 0 or box["h"] <= 0:
        raise AnnotationError(f"box has non-positive width/height: {box!r}")


def _validate_polygon(poly: dict, classes: list[str]) -> None:
    if not isinstance(poly, dict):
        raise AnnotationError(f"polygon is not an object: {poly!r}")
    if "cls" not in poly:
        raise AnnotationError(f"polygon missing 'cls': {poly!r}")
    if poly["cls"] not in classes:
        raise AnnotationError(f"polygon references unknown class {poly['cls']!r} -- "
                               f"known classes: {classes}")
    pts = poly.get("points")
    if not isinstance(pts, list) or len(pts) < 6 or len(pts) % 2 != 0:
        raise AnnotationError(
            f"polygon 'points' must be a flat [x1,y1,...,xn,yn] list with "
            f"at least 3 points (6 numbers): {poly!r}")
    for i, v in enumerate(pts):
        _num01(v, f"polygon.points[{i}]")


def validate_annotations(data: dict) -> None:
    """Raises AnnotationError on the first problem found. Unknown/extra keys
    anywhere are ignored on purpose -- see the module docstring."""
    if not isinstance(data, dict):
        raise AnnotationError("annotations must be a JSON object")
    if data.get("task") not in TASKS:
        raise AnnotationError(f"'task' must be one of {TASKS}, got {data.get('task')!r}")
    classes = data.get("classes")
    if not isinstance(classes, list) or not all(isinstance(c, str) and c for c in classes):
        raise AnnotationError("'classes' must be a list of non-empty strings")
    if len(set(classes)) != len(classes):
        raise AnnotationError(f"'classes' contains duplicates: {classes}")
    images = data.get("images")
    if not isinstance(images, dict):
        raise AnnotationError("'images' must be an object keyed by image path")
    for rel, entry in images.items():
        if not isinstance(entry, dict):
            raise AnnotationError(f"image entry for {rel!r} is not an object")
        for box in entry.get("boxes", []):
            _validate_box(box, classes)
        for poly in entry.get("polygons", []):
            _validate_polygon(poly, classes)


# --------------------------------------------------------------- per-image edits

def set_image_annotation(data: dict, rel: str, boxes: list[dict] | None = None,
                          polygons: list[dict] | None = None, reviewed: bool = True) -> dict:
    """Add or replace one image's annotation entry in-memory. `boxes`/
    `polygons` left as None keep whatever that field already held (or an
    empty list, for a new entry) -- pass [] explicitly to clear it."""
    existing = data["images"].get(rel, {"reviewed": False, "boxes": [], "polygons": []})
    entry = {
        **existing,
        "reviewed": reviewed,
        "boxes": existing.get("boxes", []) if boxes is None else list(boxes),
        "polygons": existing.get("polygons", []) if polygons is None else list(polygons),
    }
    data["images"][rel] = entry
    validate_annotations(data)
    return data


def remove_image_annotation(data: dict, rel: str) -> dict:
    data["images"].pop(rel, None)
    return data


# ------------------------------------------------------------ progress tracking

STATUSES = ("pending", "annotated", "reviewed")


def image_status(entry: dict | None) -> str:
    """Three-state progress derived entirely from fields the schema already
    has (`reviewed`, `boxes`, `polygons`) -- no schema change needed.
    "pending": no entry yet, or an entry with no geometry and not reviewed.
    "annotated": has at least one box/polygon but not yet marked reviewed.
    "reviewed": `reviewed` is true (regardless of geometry, matching the
    existing `reviewed` semantics used elsewhere in this module)."""
    entry = entry or {}
    if entry.get("reviewed"):
        return "reviewed"
    if entry.get("boxes") or entry.get("polygons"):
        return "annotated"
    return "pending"


def progress_counts(data: dict, images: list[str]) -> dict[str, int]:
    """Counts of `images` (a caller-supplied ordered list of relative paths,
    e.g. the candidate set for this annotation pass) by image_status()."""
    counts = {"total": len(images), "pending": 0, "annotated": 0, "reviewed": 0}
    for rel in images:
        counts[image_status(data["images"].get(rel))] += 1
    return counts


def next_unannotated(images: list[str], data: dict, after: str | None = None) -> str | None:
    """The next image in `images` (in list order, wrapping around once) whose
    status is "pending" -- i.e. has no boxes/polygons yet and isn't reviewed.
    Starts the search just after `after` (or from the start if `after` is
    None/not found), so repeated calls step through pending images in order.
    Returns None once nothing is pending."""
    if not images:
        return None
    start = images.index(after) + 1 if after in images else 0
    n = len(images)
    for k in range(n):
        rel = images[(start + k) % n]
        if image_status(data["images"].get(rel)) == "pending":
            return rel
    return None


# ------------------------------------------------------------------ YOLO export

def _class_ids(classes: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(classes)}


def _label_stem(rel: str) -> str:
    """image path -> the .txt filename YOLO expects, preserving the
    image-to-label mapping via a matching basename."""
    return Path(rel).stem + ".txt"


def export_yolo_detection(data: dict, out_dir: Path) -> list[str]:
    """One `<image-stem>.txt` per image, each line `class_id x y w h`
    (YOLO detection format). Returns the list of relative image paths
    exported. Refuses (via validate_annotations, called first) to export a
    structure with an unknown class or an out-of-range coordinate -- never
    invents a class id for a name it doesn't recognise."""
    validate_annotations(data)
    if data["task"] != "detection":
        raise AnnotationError(
            f"export_yolo_detection called on a {data['task']!r} project -- "
            f"use export_yolo_segmentation for segmentation annotations")
    ids = _class_ids(data["classes"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rel, entry in data["images"].items():
        lines = []
        for box in entry.get("boxes", []):
            if box["cls"] not in ids:
                raise AnnotationError(f"unknown class {box['cls']!r} in {rel!r}")
            cid = ids[box["cls"]]
            lines.append(f"{cid} {box['x']:.6f} {box['y']:.6f} {box['w']:.6f} {box['h']:.6f}")
        (out_dir / _label_stem(rel)).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                 encoding="utf-8")
        written.append(rel)
    return written


def export_yolo_segmentation(data: dict, out_dir: Path) -> list[str]:
    """One `<image-stem>.txt` per image, each line
    `class_id x1 y1 x2 y2 ... xn yn` (YOLO segmentation format)."""
    validate_annotations(data)
    if data["task"] != "segmentation":
        raise AnnotationError(
            f"export_yolo_segmentation called on a {data['task']!r} project -- "
            f"use export_yolo_detection for detection annotations")
    ids = _class_ids(data["classes"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for rel, entry in data["images"].items():
        lines = []
        for poly in entry.get("polygons", []):
            if poly["cls"] not in ids:
                raise AnnotationError(f"unknown class {poly['cls']!r} in {rel!r}")
            cid = ids[poly["cls"]]
            coords = " ".join(f"{v:.6f}" for v in poly["points"])
            lines.append(f"{cid} {coords}")
        (out_dir / _label_stem(rel)).write_text("\n".join(lines) + ("\n" if lines else ""),
                                                 encoding="utf-8")
        written.append(rel)
    return written


# ------------------------------------------------------------------------- demo

def demo():
    """Self-check, entirely in a temporary directory -- never touches any
    real project's files (om_bottle or otherwise)."""
    import tempfile

    # --- detection: box serialization + save/load round-trip -------------
    det = init_annotations("detection", ["scratch", "dent"])
    set_image_annotation(det, "images/_inbox/a.jpg",
                          boxes=[{"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.1}])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "annotations.json"
        save_annotations(p, det)
        back = load_annotations(p)
        assert back == det, "detection round-trip through disk changed the data"
        assert back["images"]["images/_inbox/a.jpg"]["boxes"][0]["cls"] == "scratch"
    print("ok  detection box serialization + save/load round-trip")

    # --- segmentation: polygon serialization ------------------------------
    seg = init_annotations("segmentation", ["dent"])
    set_image_annotation(seg, "images/_inbox/b.jpg",
                          polygons=[{"cls": "dent",
                                     "points": [0.1, 0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2]}])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "annotations.json"
        save_annotations(p, seg)
        back = load_annotations(p)
        assert back == seg, "segmentation round-trip through disk changed the data"
        assert len(back["images"]["images/_inbox/b.jpg"]["polygons"][0]["points"]) == 8
    print("ok  segmentation polygon serialization + save/load round-trip")

    # --- class ID mapping ---------------------------------------------------
    ids = _class_ids(["scratch", "dent", "chip"])
    assert ids == {"scratch": 0, "dent": 1, "chip": 2}
    print("ok  class ID mapping follows classes[] order")

    # --- normalization validation: in-range values accepted -----------------
    validate_annotations(det)  # no raise
    print("ok  in-range [0,1] coordinates accepted")

    # --- invalid coordinates rejected ---------------------------------------
    bad = init_annotations("detection", ["scratch"])
    bad["images"]["x.jpg"] = {"reviewed": True,
                               "boxes": [{"cls": "scratch", "x": 1.5, "y": 0.5, "w": 0.1, "h": 0.1}],
                               "polygons": []}
    try:
        validate_annotations(bad)
    except AnnotationError as e:
        assert "out of [0,1]" in str(e)
    else:
        raise AssertionError("an out-of-range coordinate was accepted")
    print("ok  out-of-range coordinates rejected")

    bad2 = init_annotations("detection", ["scratch"])
    bad2["images"]["x.jpg"] = {"reviewed": True,
                                "boxes": [{"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0, "h": 0.1}],
                                "polygons": []}
    try:
        validate_annotations(bad2)
    except AnnotationError as e:
        assert "non-positive" in str(e)
    else:
        raise AssertionError("a zero-width box was accepted")
    print("ok  non-positive box width/height rejected")

    # --- unknown classes rejected --------------------------------------------
    unk = init_annotations("detection", ["scratch"])
    unk["images"]["x.jpg"] = {"reviewed": True,
                               "boxes": [{"cls": "not_a_class", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}],
                               "polygons": []}
    try:
        validate_annotations(unk)
    except AnnotationError as e:
        assert "unknown class" in str(e)
    else:
        raise AssertionError("an unknown class name was accepted")
    print("ok  unknown class name rejected")

    # export never silently invents a class id: unknown class caught at export too
    unk_exportable = init_annotations("detection", ["scratch", "dent"])
    unk_exportable["images"]["x.jpg"] = {
        "reviewed": True,
        "boxes": [{"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}],
        "polygons": []}
    # forge a class removal after the box was added, simulating a stale reference
    unk_exportable["classes"] = ["dent"]
    try:
        export_yolo_detection(unk_exportable, Path(tempfile.mkdtemp()))
    except AnnotationError as e:
        assert "unknown class" in str(e)
    else:
        raise AssertionError("export accepted a box whose class is no longer declared")
    print("ok  export rejects a box referencing an undeclared class")

    # --- YOLO detection export ------------------------------------------------
    det2 = init_annotations("detection", ["scratch", "dent"])
    set_image_annotation(det2, "images/_inbox/img001.jpg",
                          boxes=[{"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0.2, "h": 0.1},
                                 {"cls": "dent", "x": 0.25, "y": 0.75, "w": 0.1, "h": 0.1}])
    with tempfile.TemporaryDirectory() as td:
        exported = export_yolo_detection(det2, Path(td))
        assert exported == ["images/_inbox/img001.jpg"]
        txt = (Path(td) / "img001.txt").read_text(encoding="utf-8").strip().splitlines()
        assert txt[0] == "0 0.500000 0.500000 0.200000 0.100000"
        assert txt[1] == "1 0.250000 0.750000 0.100000 0.100000"
    print("ok  YOLO detection export: filenames, class ids, and coordinates correct")

    # calling the segmentation exporter on a detection project must be refused
    try:
        export_yolo_segmentation(det2, Path(tempfile.mkdtemp()))
    except AnnotationError:
        pass
    else:
        raise AssertionError("segmentation exporter accepted a detection-task project")

    # --- YOLO segmentation export --------------------------------------------
    seg2 = init_annotations("segmentation", ["dent"])
    set_image_annotation(seg2, "images/_inbox/img002.jpg",
                          polygons=[{"cls": "dent",
                                     "points": [0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3]}])
    with tempfile.TemporaryDirectory() as td:
        exported = export_yolo_segmentation(seg2, Path(td))
        assert exported == ["images/_inbox/img002.jpg"]
        txt = (Path(td) / "img002.txt").read_text(encoding="utf-8").strip()
        assert txt == "0 0.100000 0.100000 0.300000 0.100000 0.300000 0.300000 0.100000 0.300000"
    print("ok  YOLO segmentation export: filenames, class ids, and points correct")

    # calling the detection exporter on a segmentation project must be refused
    try:
        export_yolo_detection(seg2, Path(tempfile.mkdtemp()))
    except AnnotationError:
        pass
    else:
        raise AssertionError("detection exporter accepted a segmentation-task project")

    # --- image-to-label filename mapping preserved across nested paths ------
    det3 = init_annotations("detection", ["scratch"])
    set_image_annotation(det3, "images/-ve/Scratch/deep/nested_007.jpg",
                          boxes=[{"cls": "scratch", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}])
    with tempfile.TemporaryDirectory() as td:
        export_yolo_detection(det3, Path(td))
        assert (Path(td) / "nested_007.txt").exists()
    print("ok  image-to-label filename mapping preserved for nested source paths")

    # --- unknown top-level task rejected --------------------------------------
    try:
        init_annotations("classification", ["x"])
    except AnnotationError as e:
        assert "unknown task" in str(e)
    else:
        raise AssertionError("init_annotations accepted a non detection/segmentation task")
    print("ok  init_annotations rejects a non detection/segmentation task")

    # --- progress tracking: pending / annotated / reviewed -------------------
    prog = init_annotations("detection", ["bottle"])
    imgs = ["a.jpg", "b.jpg", "c.jpg"]
    assert image_status(prog["images"].get("a.jpg")) == "pending"
    set_image_annotation(prog, "b.jpg",
                          boxes=[{"cls": "bottle", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}],
                          reviewed=False)
    assert image_status(prog["images"].get("b.jpg")) == "annotated"
    set_image_annotation(prog, "c.jpg",
                          boxes=[{"cls": "bottle", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}],
                          reviewed=True)
    assert image_status(prog["images"].get("c.jpg")) == "reviewed"
    counts = progress_counts(prog, imgs)
    assert counts == {"total": 3, "pending": 1, "annotated": 1, "reviewed": 1}, counts
    print("ok  progress_counts: pending/annotated/reviewed derived from existing fields")

    # --- next_unannotated: order, wraparound, exhaustion ----------------------
    assert next_unannotated(imgs, prog) == "a.jpg"
    assert next_unannotated(imgs, prog, after="a.jpg") == "a.jpg"  # wraps: only a.jpg is pending
    set_image_annotation(prog, "a.jpg",
                          boxes=[{"cls": "bottle", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}],
                          reviewed=False)
    assert next_unannotated(imgs, prog) is None, "no pending image left, must return None"
    print("ok  next_unannotated: finds pending images in order, wraps, returns None when exhausted")

    print("ok  annotate.py self-test complete -- no real project files were touched")


if __name__ == "__main__":
    demo()
