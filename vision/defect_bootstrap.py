"""Bootstrap DEFECT boxes (where the defect is) from what already exists: image-level labels + the component detector.

The Stage 2 detector finds bottle / cap / label. The classification project already says, per image, which defect it
shows (labels.csv). Together they give a first defect box for the defects that are a STATE of a component:

    tilt_cap      -> the image's cap box becomes   cap_tilted
    skewed_label  -> the image's label box becomes label_skewed
    damaged_label -> the image's label box becomes label_damaged

Defects that are a REGION, not a whole component, cannot be guessed and are listed for drawing by hand:
    damaged_bottle -> bottle_damaged (the dent / crack)     missing_cap -> neck_bare (the bare neck)
    water_level    -> fill_line (the meniscus)              skewed_bottle -> bottle_skewed (or an angle measurement)

Everything is written as PROPOSALS (vision/autoannotate.py): never into "boxes", never marked reviewed, so nothing
reaches a training set until a person accepts it in the Annotate page. The frozen Stage 2 annotations
(stage2_dataset/annotations*.json, hash-pinned in MODEL_PROVENANCE.json) are never touched.

    python -m vision.defect_bootstrap                    # self-test (fake detector, temp folder)
    python -m vision.defect_bootstrap --plan             # what it would do, on the real project (runs the detector)
    python -m vision.defect_bootstrap --write [--limit N]   # create / update projects/<active>/annotations.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from vision import annotate as A
from vision import autoannotate as AA
from vision import dataset as D

# Detector v4 classes: the three components (kept, so presence checks still work) + defect states / regions.
V4_CLASSES = ["bottle", "cap", "label", "cap_tilted", "neck_bare", "label_damaged", "label_skewed",
              "bottle_damaged", "bottle_skewed", "fill_line"]

# image defect -> {component class: defect-state class}; the most confident component box is relabelled
STATE_MAP = {"tilt_cap": {"cap": "cap_tilted"},
             "skewed_label": {"label": "label_skewed"},
             "damaged_label": {"label": "label_damaged"}}

HAND_DRAWN = {"damaged_bottle": "bottle_damaged: draw the dent / crack / crush region",
              "missing_cap": "neck_bare: draw the bare neck (tamper ring visible, no cap)",
              "water_level": "fill_line: draw a thin box on the meniscus",
              "skewed_bottle": "bottle_skewed: box the bottle (or measure its axis angle instead)",
              "missing_label": "nothing to draw: the defect is the ABSENCE of a label box (recipe rule)"}


def recipe_v4() -> dict:
    """The inspection recipe to use with a detector trained on V4_CLASSES (config.json "inspection"). A component in
    a defective state still counts as present (any_of), and the defect classes report what and where."""
    return {"anchor": "bottle",
            "parts": [{"name": "cap", "search": [-0.25, 1.0], "zone": [-0.25, 0.5], "misplaced": "cap_misplaced",
                       "any_of": ["cap_tilted"]},
                      {"name": "label", "search": [0.0, 1.0], "any_of": ["label_damaged", "label_skewed"]}],
            "defects": {"cap_tilted": {"finding": "tilt_cap", "min_conf": 0.6},
                        "neck_bare": {"finding": "missing_cap", "min_conf": 0.6},
                        "label_damaged": {"finding": "damaged_label", "min_conf": 0.6},
                        "label_skewed": {"finding": "skewed_label", "min_conf": 0.6},
                        "bottle_damaged": {"finding": "damaged_bottle", "min_conf": 0.6},
                        "bottle_skewed": {"finding": "skewed_bottle", "min_conf": 0.6}}}


def relabel(dets, defects) -> list:
    """dets: [(class, conf, x1, y1, x2, y2)]; defects: the image's labelled defects. The most confident box of each
    mapped component becomes its defect-state class; every other box is returned unchanged."""
    out = list(dets)
    for d in defects:
        for comp, state in STATE_MAP.get(d, {}).items():
            idx = [i for i, b in enumerate(out) if b[0] == comp]
            if idx:
                i = max(idx, key=lambda k: out[k][1])
                out[i] = (state,) + tuple(out[i][1:])
    return out


def bootstrap(data: dict, items, detect_fn, read_wh, min_conf: float = 0.25) -> dict:
    """items: [(rel path, [defects])]. Adds proposals to `data` (an annotations dict with V4_CLASSES).
    Returns a summary: proposals per class, images relabelled, and the hand-drawing list per defect."""
    summary = {"images": 0, "proposals": {}, "relabelled": {}, "skipped_reviewed": 0, "no_detection": 0,
               "draw_by_hand": {}}
    for rel, defects in items:
        entry = data["images"].get(rel)
        if entry and entry.get("reviewed"):
            summary["skipped_reviewed"] += 1
            continue
        dets = detect_fn(rel)
        if not dets:
            summary["no_detection"] += 1
        new = relabel(dets, defects)
        for b0, b1 in zip(dets, new):
            if b0[0] != b1[0]:
                summary["relabelled"][b1[0]] = summary["relabelled"].get(b1[0], 0) + 1
        AA.propose(data, rel, new, read_wh(rel), min_conf=min_conf)
        for p in data["images"][rel].get("proposals", []):
            summary["proposals"][p["cls"]] = summary["proposals"].get(p["cls"], 0) + 1
        for d in defects:
            if d in HAND_DRAWN:
                summary["draw_by_hand"].setdefault(d, []).append(rel)
        summary["images"] += 1
    return summary


def project_items() -> list:
    """(rel, defects) for every REVIEWED image of the active project (labels.csv is the truth)."""
    defects, labels = D.load_labels()
    return [(rel, [d for d in defects if row.get(d)]) for rel, row in sorted(labels.items()) if row.get("reviewed", 1)]


def _real_detector():
    import cv2
    from vision import detect
    det = detect.YoloDetector(require=("bottle", "cap", "label"), conf=0.25)
    cache = {}

    def detect_fn(rel):
        img = cv2.imread(str(D.IMAGE_ROOT / rel))
        if img is None:
            return []
        cache[rel] = (img.shape[1], img.shape[0])
        r = det.detect(img, camera_id="bootstrap", frame_seq=0, frame_ts=0.0)
        return [(d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2) for d in r.detections]

    def read_wh(rel):
        return cache.get(rel) or (1, 1)
    return detect_fn, read_wh, det.model_id


def main(argv) -> None:
    limit = int(argv[argv.index("--limit") + 1]) if "--limit" in argv else None
    path = D.PROJECT_DIR / "annotations.json"
    if path.exists():
        data = A.load_annotations(path)
        if data["classes"] != V4_CLASSES:
            raise SystemExit(f"{path} exists with classes {data['classes']}: not overwritten. Move it aside first.")
    else:
        data = A.init_annotations("detection", V4_CLASSES)
    items = project_items()[:limit] if limit else project_items()
    detect_fn, read_wh, model_id = _real_detector()
    print(f"project {D.PROJECT}: {len(items)} reviewed image(s), detector {model_id}")
    s = bootstrap(data, items, detect_fn, read_wh)
    print(json.dumps({k: v for k, v in s.items() if k != "draw_by_hand"}, indent=1))
    for d, rels in s["draw_by_hand"].items():
        print(f"  draw by hand  {d:<15} {len(rels):>4} image(s)   {HAND_DRAWN[d]}")
    if "--write" in argv:
        A.save_annotations(path, data)
        todo = D.PROJECT_DIR / "cache" / "defect_draw_by_hand.json"
        todo.parent.mkdir(parents=True, exist_ok=True)
        todo.write_text(json.dumps({d: {"what": HAND_DRAWN[d], "images": r} for d, r in s["draw_by_hand"].items()},
                                   indent=1))
        print(f"wrote proposals to {path} (none accepted: review them in the Annotate page)\n"
              f"hand-drawing list: {todo}\nrecipe for a v4 detector: vision.defect_bootstrap.recipe_v4()")
    else:
        print("(plan only: nothing written; add --write)")


def demo():
    import tempfile
    from line import decision as DEC
    # relabelling: the most confident cap of a tilt_cap image becomes cap_tilted; others untouched
    dets = [("bottle", 0.9, 10, 10, 60, 200), ("cap", 0.8, 20, 0, 50, 20), ("cap", 0.4, 0, 0, 5, 5),
            ("label", 0.7, 12, 80, 58, 140)]
    r = relabel(dets, ["tilt_cap", "damaged_label"])
    assert [b[0] for b in r] == ["bottle", "cap_tilted", "cap", "label_damaged"], r
    assert relabel(dets, ["damaged_bottle"]) == dets                     # region defects are not guessed
    # the v4 recipe is valid and a v4 detector's tilted cap is "tilt_cap", not "missing_cap"
    assert DEC.recipe_problems(recipe_v4()) == []
    assert set(DEC.recipe_classes(recipe_v4())) <= set(V4_CLASSES)
    # bootstrap writes PROPOSALS only: nothing in boxes, nothing reviewed, reviewed images left alone
    data = A.init_annotations("detection", V4_CLASSES)
    data["images"]["done.jpg"] = {"reviewed": True, "boxes": [{"cls": "bottle", "x": .5, "y": .5, "w": .2, "h": .8}],
                                  "polygons": []}
    items = [("a.jpg", ["tilt_cap"]), ("b.jpg", ["water_level"]), ("c.jpg", []), ("done.jpg", ["tilt_cap"])]
    fake = {"a.jpg": dets, "b.jpg": dets[:1], "c.jpg": [], "done.jpg": dets}
    s = bootstrap(data, items, lambda rel: fake[rel], lambda rel: (100, 220))
    a = data["images"]["a.jpg"]
    assert a["boxes"] == [] and not a["reviewed"] and {p["cls"] for p in a["proposals"]} == {"bottle", "cap_tilted",
                                                                                              "cap", "label"}, a
    assert data["images"]["done.jpg"]["boxes"] and "proposals" not in data["images"]["done.jpg"]
    assert s["skipped_reviewed"] == 1 and s["no_detection"] == 1 and s["relabelled"] == {"cap_tilted": 1}, s
    assert s["draw_by_hand"] == {"water_level": ["b.jpg"]}, s
    # the exporter cannot see proposals: an unreviewed bootstrap never reaches a training set
    with tempfile.TemporaryDirectory() as td:
        A.save_annotations(Path(td) / "annotations.json", data)
        back = A.load_annotations(Path(td) / "annotations.json")
        assert back["images"]["a.jpg"]["proposals"]
        exported = A.export_yolo_detection(back, Path(td) / "out")
        assert "a.jpg" not in " ".join(map(str, exported)), exported
    print("ok  defect bootstrap: component -> defect-state relabel, region defects listed for hand drawing, "
          "proposals only (never boxes / reviewed / exported), v4 recipe valid")


if __name__ == "__main__":
    if "--plan" in sys.argv or "--write" in sys.argv:
        main(sys.argv)
    else:
        demo()
