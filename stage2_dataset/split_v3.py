"""Stage 2 detection data, version 3: the full-bottle missing-cap scene moves from TEST to TRAIN.

Why (2026-10-05): the 22 missing-cap images the user captured (All Datasets/Missing Cap/snap001-022)
are already in Stage 2:
  * snap013-022 = scene 38 (img_635-644): neck CLOSE-UPS, in train;
  * snap001-012 = scene 22 (img_328-341, 12 frames of ONE bottle): full-bottle view, in TEST only.
So no detector ever trained on a bare neck at full-bottle scale, and every candidate (v1 n, v1 s,
v2 n) calls the green tamper ring a "cap" (conf 0.69-0.71) and passes all 12 (0/12 recall,
models/stage2_yolo/defects_*_test.json). A confidence threshold does not fix that.

v3 = v2 boxes (annotations_v2.json, cap = cap only) + split_v3.json, which is split.json with scene 22
moved test -> train. Nothing else changes: same images, same boxes, the other 38 scenes stay where
they were. annotations.json / annotations_v2.json / split.json are never written.

THE COST, stated rather than hidden: the test split then has NO missing-cap bottle, so a v3 model's
missing-cap recall cannot be measured offline. It must be validated on the real machine with real
missing-cap bottles (model_registry: VALIDATED needs a written real-camera validation) before any
activation. Scene 22 is the whole white-background session (35 frames: the 12 uncapped ones plus the
same bottles capped); scenes are moved whole (dataset audit rule), so test drops from 87 to 52 images /
6 scenes and its mAP is not directly comparable with v1/v2.

    python stage2_dataset/split_v3.py build      # split_v3.json
    python stage2_dataset/split_v3.py export     # yolo_export_v3/ (v2 boxes, v3 split)
    python stage2_dataset/split_v3.py validate
    python stage2_dataset/split_v3.py --selftest
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "split.json"
SPLIT_V3 = HERE / "split_v3.json"
ANN_V2 = HERE / "annotations_v2.json"
EXPORT_V3 = HERE / "yolo_export_v3"
CLEAN = HERE / "clean"
MOVE = {"22": ("test", "train", "white-background session incl. the 12 full-bottle missing-cap frames "
                               "(img_328-341 = user snap001-012), moved whole")}


def build_split(split: dict, move=MOVE) -> dict:
    out = copy.deepcopy(split)
    for sid, (src, dst, why) in move.items():
        s, d = out["splits"][src], out["splits"][dst]
        if sid not in s["files_by_scene"]:
            raise KeyError(f"scene {sid} is not in {src}")
        files = s["files_by_scene"].pop(sid)
        d["files_by_scene"][sid] = files
        for part, sign in ((s, -1), (d, 1)):
            part["scene_ids"] = sorted({int(x) for x in part["files_by_scene"]})
            part["n_scenes"] = len(part["files_by_scene"])
            part["filenames"] = sorted(f for fs in part["files_by_scene"].values() for f in fs)
            part["n_images"] = len(part["filenames"])
    total = sum(p["n_images"] for p in out["splits"].values())
    for p in out["splits"].values():
        p["percentage"] = round(100 * p["n_images"] / total, 2)
    out["version"] = "v3"
    out["derived_from"] = "split.json"
    out["moved_scenes"] = {sid: {"from": a, "to": b, "why": w} for sid, (a, b, w) in move.items()}
    out["caveat"] = ("test has no missing-cap bottle: validate missing-cap on the real machine before activation")
    return out


def export(split_path=SPLIT_V3, ann=ANN_V2, out=EXPORT_V3):
    """The existing exporter, pointed at v2 boxes + the v3 split (its paths are module globals)."""
    sys.path.insert(0, str(HERE))
    import export_yolo as E
    sp = json.loads(Path(split_path).read_text(encoding="utf-8"))
    E.ANNOT_PATH, E.SPLIT_PATH, E.EXPORT_DIR = str(ann), str(split_path), str(out)
    E.EXPECTED = dict(E.EXPECTED, **{k: v["n_images"] for k, v in sp["splits"].items()})
    E.main()


def validate(split_path=SPLIT_V3, ann=ANN_V2, out=EXPORT_V3) -> list:
    problems = []
    sp = json.loads(Path(split_path).read_text(encoding="utf-8"))
    base = json.loads(SPLIT.read_text(encoding="utf-8"))
    a = json.loads(Path(ann).read_text(encoding="utf-8"))["images"]
    seen = {}
    for name, part in sp["splits"].items():
        for f in part["filenames"]:
            if f in seen:
                problems.append(f"{f} in {seen[f]} and {name}")
            seen[f] = name
    base_files = {f for p in base["splits"].values() for f in p["filenames"]}
    if set(seen) != base_files:
        problems.append("v3 does not contain exactly the v2 images")
    for sid, (src, dst, _) in MOVE.items():
        if sid in sp["splits"][src]["files_by_scene"] or sid not in sp["splits"][dst]["files_by_scene"]:
            problems.append(f"scene {sid} not moved {src} -> {dst}")
    for name in ("train", "val", "test"):
        unchanged = {k: v for k, v in base["splits"][name]["files_by_scene"].items() if k not in MOVE}
        mine = {k: v for k, v in sp["splits"][name]["files_by_scene"].items() if k not in MOVE}
        if unchanged != mine:
            problems.append(f"{name}: scenes other than the moved one changed")
    if out.exists():
        for f, name in seen.items():
            img = out / "images" / name / f
            lbl = out / "labels" / name / (Path(f).stem + ".txt")
            if not img.exists() or not lbl.exists():
                problems.append(f"export missing {name}/{f}")
                continue
            n = len([x for x in lbl.read_text().splitlines() if x.strip()])
            if n != len(a[f]["boxes"]):
                problems.append(f"{f}: {n} label lines, {len(a[f]['boxes'])} boxes in annotations_v2")
    return problems


def selftest():
    base = json.loads(SPLIT.read_text(encoding="utf-8"))
    v3 = build_split(base)
    assert "22" in v3["splits"]["train"]["files_by_scene"] and "22" not in v3["splits"]["test"]["files_by_scene"]
    n22 = len(base["splits"]["test"]["files_by_scene"]["22"])
    assert v3["splits"]["test"]["n_images"] == base["splits"]["test"]["n_images"] - n22
    assert sum(p["n_images"] for p in v3["splits"].values()) == base["totals"]["total_keep_images"]
    assert "img_328.png" in v3["splits"]["train"]["filenames"]
    assert base["splits"]["test"]["files_by_scene"]["22"], "build_split must not modify its input"
    try:
        build_split(v3)                                             # already moved: refused, not repeated
        raise AssertionError("moving a scene twice was accepted")
    except KeyError:
        pass
    print("ok  split_v3: scene 22 test -> train, other scenes unchanged, totals kept, input untouched")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "--selftest"
    if cmd == "build":
        SPLIT_V3.write_text(json.dumps(build_split(json.loads(SPLIT.read_text(encoding="utf-8"))), indent=2),
                            encoding="utf-8")
        print("wrote", SPLIT_V3)
    elif cmd == "export":
        export()
    elif cmd == "validate":
        p = validate()
        print("ok  v3 export valid" if not p else "PROBLEMS:\n" + "\n".join(p[:50]))
        sys.exit(1 if p else 0)
    else:
        selftest()
