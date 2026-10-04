"""Stage 2 segmentation (label outline): seed -> annotate -> export -> validate -> train.

Minimum pipeline from docs/roadmap/VISION_DATASET.md section 6. Polygons live in
their own file, stage2_dataset/seg_annotations.json; annotations.json (hash pinned in
MODEL_PROVENANCE.json) is only read. The existing scene split (split.json) is reused
unchanged.

    python stage2_dataset/seg_pipeline.py seed       # seg_annotations.json, label box -> 4-corner polygon, UNREVIEWED
    python stage2_dataset/seg_pipeline.py studio     # Annotation Studio: delete the box seed, draw the outline, mark reviewed
    python stage2_dataset/seg_pipeline.py export     # reviewed images only -> stage2_dataset/yolo_seg_export/
    python stage2_dataset/seg_pipeline.py validate   # split / polygon / file checks; non-zero exit on failure
    python stage2_dataset/seg_pipeline.py train --model yolov8n-seg.pt   # needs the weights locally
    python stage2_dataset/seg_pipeline.py --selftest # synthetic data in a temp dir; touches nothing real

A seeded polygon is NOT an annotation: it is the label's bounding box. Only an image a
human has marked reviewed in the studio is exported; an unreviewed image is left out
entirely (not exported as an empty label file, which would teach "no label here").
Images with no label box are seeded with zero polygons, so once reviewed they become
genuine negatives.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import annotate as A  # noqa: E402

CLASSES = ["label"]
SPLITS = ("train", "val", "test")
DET_JSON = HERE / "annotations.json"
SEG_JSON = HERE / "seg_annotations.json"
SPLIT_JSON = HERE / "split.json"
IMAGE_ROOT = HERE / "clean"
EXPORT = HERE / "yolo_seg_export"
OUT = REPO_ROOT / "models" / "stage2_seg"


def split_map(split_json: Path) -> dict[str, tuple[str, str]]:
    """filename -> (split, scene_id), refusing a file listed twice."""
    data = json.loads(Path(split_json).read_text(encoding="utf-8"))
    out = {}
    for s, info in data["splits"].items():
        for scene, files in info["files_by_scene"].items():
            for f in files:
                if f in out:
                    raise ValueError(f"{f} is in both {out[f][0]} and {s}")
                out[f] = (s, str(scene))
    return out


def box_polygon(b: dict) -> list[float]:
    x0, y0 = b["x"] - b["w"] / 2, b["y"] - b["h"] / 2
    x1, y1 = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    c = lambda v: min(1.0, max(0.0, v))
    return [c(x0), c(y0), c(x1), c(y0), c(x1), c(y1), c(x0), c(y1)]


def seed(det_json=DET_JSON, seg_json=SEG_JSON, split_json=SPLIT_JSON) -> dict:
    """Create seg_annotations.json. Never overwrites: adds only images it lacks."""
    det = A.load_annotations(det_json)
    smap = split_map(split_json)
    seg = A.load_annotations(seg_json) if Path(seg_json).exists() else A.init_annotations("segmentation", CLASSES)
    added = 0
    for rel, entry in sorted(det["images"].items()):
        if rel not in smap or rel in seg["images"]:
            continue
        polys = [{"cls": "label", "points": box_polygon(b), "seeded": True}
                 for b in entry.get("boxes", []) if b["cls"] == "label"]
        A.set_image_annotation(seg, rel, polygons=polys, reviewed=False)
        added += 1
    A.save_annotations(seg_json, seg)
    return {"added": added, "total": len(seg["images"])}


def export(seg_json=SEG_JSON, split_json=SPLIT_JSON, image_root=IMAGE_ROOT, out=EXPORT) -> dict:
    seg = A.load_annotations(seg_json)
    if seg["task"] != "segmentation":
        raise A.AnnotationError(f"{seg_json} is a {seg['task']!r} file")
    smap = split_map(split_json)
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)            # generated output only; rebuilt from the json every time
    counts = {s: {"images": 0, "polygons": 0} for s in SPLITS}
    skipped = {"unreviewed": 0, "not_in_split": 0}
    reviewed = {rel: e for rel, e in seg["images"].items() if e.get("reviewed")}
    skipped["unreviewed"] = len(seg["images"]) - len(reviewed)
    for s in SPLITS:
        (out / "images" / s).mkdir(parents=True)
        (out / "labels" / s).mkdir(parents=True)
    for rel, entry in sorted(reviewed.items()):
        if rel not in smap:
            skipped["not_in_split"] += 1
            continue
        s = smap[rel][0]
        sub = {"classes": seg["classes"], "task": "segmentation", "schema_version": seg["schema_version"],
               "images": {rel: entry}}
        A.export_yolo_segmentation(sub, out / "labels" / s)
        dst = out / "images" / s / Path(rel).name
        try:
            os.link(Path(image_root) / rel, dst)        # hard link: 0 extra disk, identical bytes
        except OSError:
            shutil.copy2(Path(image_root) / rel, dst)
        counts[s]["images"] += 1
        counts[s]["polygons"] += len(entry.get("polygons", []))
    (out / "data.yaml").write_text(
        "# generated by stage2_dataset/seg_pipeline.py export -- pass this file as an ABSOLUTE path\n"
        + "".join(f"{s}: images/{s}\n" for s in SPLITS)
        + f"nc: {len(seg['classes'])}\nnames: {json.dumps(seg['classes'])}\n", encoding="utf-8")
    report = {"counts": counts, "skipped": skipped}
    (out / "export_info.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def validate(seg_json=SEG_JSON, split_json=SPLIT_JSON, image_root=IMAGE_ROOT, out=EXPORT) -> dict:
    """Returns {"errors": [...], "warnings": [...], "counts": ...}; errors mean do not train."""
    errors, warnings = [], []
    seg = A.load_annotations(seg_json)
    try:
        A.validate_annotations(seg)
    except A.AnnotationError as e:
        errors.append(f"annotations invalid: {e}")
    smap = split_map(split_json)
    scene_split = {}
    for f, (s, sc) in smap.items():
        if scene_split.setdefault(sc, s) != s:
            errors.append(f"scene {sc} is in {scene_split[sc]} and {s}")
    out = Path(out)
    if not (out / "data.yaml").exists():
        return {"errors": errors + [f"no export at {out} - run export"], "warnings": warnings}
    seen = set()
    counts = {s: 0 for s in SPLITS}
    for s in SPLITS:
        for img in (out / "images" / s).iterdir():
            rel = img.name
            seen.add(rel)
            counts[s] += 1
            e = seg["images"].get(rel)
            if e is None:
                errors.append(f"{s}/{rel}: exported but not in {Path(seg_json).name}")
                continue
            if not e.get("reviewed"):
                errors.append(f"{s}/{rel}: exported but not reviewed")
            if smap.get(rel, (None,))[0] != s:
                errors.append(f"{s}/{rel}: split.json puts it in {smap.get(rel, ('nowhere',))[0]}")
            if img.stat().st_size != (Path(image_root) / rel).stat().st_size:
                errors.append(f"{s}/{rel}: image differs from {image_root.name}/")
            lbl = out / "labels" / s / (img.stem + ".txt")
            if not lbl.exists():
                errors.append(f"{s}/{rel}: no label file")
                continue
            lines = [ln.split() for ln in lbl.read_text().splitlines() if ln.strip()]
            if len(lines) != len(e.get("polygons", [])):
                errors.append(f"{s}/{rel}: {len(lines)} label lines vs {len(e.get('polygons', []))} polygons")
            for ln in lines:
                cid, pts = int(ln[0]), [float(v) for v in ln[1:]]
                if not 0 <= cid < len(seg["classes"]):
                    errors.append(f"{s}/{rel}: class id {cid}")
                if len(pts) < 6 or len(pts) % 2:
                    errors.append(f"{s}/{rel}: polygon with {len(pts)} coords")
                if any(v < 0 or v > 1 for v in pts):
                    errors.append(f"{s}/{rel}: coordinate outside [0,1]")
            for p in e.get("polygons", []):
                if p.get("seeded"):
                    warnings.append(f"{s}/{rel}: reviewed, but a polygon still carries the unedited box seed")
    for rel, e in seg["images"].items():
        if e.get("reviewed") and rel in smap and rel not in seen:
            errors.append(f"{rel}: reviewed but missing from the export")
    for s in SPLITS:
        if counts[s] == 0:
            errors.append(f"split {s} has no reviewed images yet")
    return {"errors": errors, "warnings": warnings, "counts": counts,
            "reviewed": sum(1 for e in seg["images"].values() if e.get("reviewed")),
            "total": len(seg["images"])}


def studio():
    import customtkinter as ctk
    from annotation_studio import AnnotationTab
    seg = A.load_annotations(SEG_JSON)
    smap = split_map(SPLIT_JSON)
    # images that have a label first, train before val before test
    images = sorted(seg["images"], key=lambda r: (not seg["images"][r]["polygons"],
                                                  SPLITS.index(smap[r][0]), r))
    meta = {r: {"scene_id": smap[r][1], "split": smap[r][0]} for r in images}
    root = ctk.CTk()
    root.title("Stage 2 segmentation -- label outline")
    root.geometry("1300x760")
    tab = AnnotationTab(app=None, parent=root)
    tab.load_external(IMAGE_ROOT, SEG_JSON, "segmentation", CLASSES, images, meta)
    root.mainloop()


def train(model: str, epochs=80, imgsz=640, batch=8, patience=20, workers=0):
    import hashlib
    import time
    import torch
    from ultralytics import YOLO
    v = validate()
    if v["errors"]:
        raise SystemExit("validate failed:\n  " + "\n  ".join(v["errors"][:20]))
    if not Path(model).exists():
        raise SystemExit(f"{model} not found locally. Pretrained seg weights are not downloaded "
                         f"automatically; place yolov8n-seg.pt in the repo root first.")
    dev = 0 if torch.cuda.is_available() else "cpu"
    data = str((EXPORT / "data.yaml").resolve())
    stem = Path(model).stem
    m = YOLO(model)
    t0 = time.time()
    m.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch, patience=patience, device=dev, seed=0,
            deterministic=True, workers=workers, project=str(OUT), name=f"run_{stem}", exist_ok=True)
    best = Path(m.trainer.save_dir) / "weights" / "best.pt"
    b = YOLO(str(best))
    kw = dict(data=data, imgsz=imgsz, batch=batch, device=dev, workers=workers, verbose=False,
              project=str(OUT), exist_ok=True)
    res = {}
    for split in ("val", "test"):
        r = b.val(split=split, name=f"run_{stem}_{split}eval", **kw)
        res[split] = {"box_map50": float(r.box.map50), "box_map50_95": float(r.box.map),
                      "mask_map50": float(r.seg.map50), "mask_map50_95": float(r.seg.map),
                      "mask_precision": float(r.seg.mp), "mask_recall": float(r.seg.mr)}
    meta = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "model": model, "checkpoint": str(best),
            "sha256": hashlib.sha256(best.read_bytes()).hexdigest(), "size_mb": best.stat().st_size / 1e6,
            "train_minutes": (time.time() - t0) / 60, "counts": v["counts"],
            "seg_annotations_sha256": hashlib.sha256(SEG_JSON.read_bytes()).hexdigest(), **res}
    (OUT / f"run_{stem}_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


# ------------------------------------------------------------------ self-test

def selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "clean").mkdir()
        names = [f"img_{i:03d}.png" for i in range(6)]
        for n in names:
            (t / "clean" / n).write_bytes(b"\x89PNG fake " + n.encode())
        det = A.init_annotations("detection", ["bottle", "cap", "label"])
        for i, n in enumerate(names):
            boxes = [{"cls": "bottle", "x": .5, "y": .5, "w": .2, "h": .8}]
            if i != 5:
                boxes.append({"cls": "label", "x": .5, "y": .6, "w": .17, "h": .18})
            A.set_image_annotation(det, n, boxes=boxes, reviewed=True)
        A.save_annotations(t / "det.json", det)
        split = {"splits": {"train": {"files_by_scene": {"1": names[:2], "2": [names[5]]}},
                            "val": {"files_by_scene": {"3": names[2:4]}},
                            "test": {"files_by_scene": {"4": names[4:5]}}}}
        (t / "split.json").write_text(json.dumps(split))
        P = dict(seg_json=t / "seg.json", split_json=t / "split.json")

        r = seed(det_json=t / "det.json", **P)
        assert r == {"added": 6, "total": 6}, r
        assert seed(det_json=t / "det.json", **P)["added"] == 0, "re-seed must not touch existing images"
        seg = A.load_annotations(t / "seg.json")
        assert not any(e["reviewed"] for e in seg["images"].values()), "seeds must start unreviewed"
        assert seg["images"][names[5]]["polygons"] == [], "no label box -> no seeded polygon"
        assert len(seg["images"][names[0]]["polygons"][0]["points"]) == 8
        print("ok  seed: box -> unreviewed 4-corner polygon, idempotent, negatives seeded empty")

        rep = export(image_root=t / "clean", out=t / "exp", **P)
        assert sum(c["images"] for c in rep["counts"].values()) == 0 and rep["skipped"]["unreviewed"] == 6
        v = validate(image_root=t / "clean", out=t / "exp", **P)
        assert any("no reviewed images" in e for e in v["errors"]), v
        print("ok  export: unreviewed seeds are never exported; validate refuses an empty split")

        # a human edits 4 images and marks 5 reviewed (img_005 = reviewed negative)
        for n in names[:5]:
            pts = [.42, .5, .58, .52, .59, .7, .41, .69, .40, .6]
            A.set_image_annotation(seg, n, polygons=[{"cls": "label", "points": pts}], reviewed=True)
        A.set_image_annotation(seg, names[5], polygons=[], reviewed=True)
        seg["images"][names[4]]["polygons"][0]["seeded"] = True       # one left unedited
        A.save_annotations(t / "seg.json", seg)
        rep = export(image_root=t / "clean", out=t / "exp", **P)
        assert rep["counts"]["train"] == {"images": 3, "polygons": 2}, rep
        assert rep["counts"]["val"]["images"] == 2 and rep["counts"]["test"]["images"] == 1
        assert (t / "exp" / "labels" / "train" / "img_005.txt").read_text() == ""
        v = validate(image_root=t / "clean", out=t / "exp", **P)
        assert v["errors"] == [], v["errors"]
        assert len(v["warnings"]) == 1 and "img_004" in v["warnings"][0], v["warnings"]
        print("ok  export+validate: reviewed only, split.json respected, negatives empty, seed warning")

        (t / "exp" / "labels" / "val" / "img_002.txt").write_text("0 0.1 0.1 1.4 0.2 0.3 0.3\n")
        os.replace(t / "exp" / "images" / "test" / "img_004.png", t / "exp" / "images" / "val" / "img_004.png")
        v = validate(image_root=t / "clean", out=t / "exp", **P)
        assert any("outside [0,1]" in e for e in v["errors"]), v["errors"]
        assert any("split.json puts it in test" in e for e in v["errors"]), v["errors"]
        print("ok  validate: catches out-of-range coordinates and a wrong-split image")
    print("ok")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=["seed", "studio", "export", "validate", "train"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == "seed":
        print(seed())
    elif a.cmd == "studio":
        studio()
    elif a.cmd == "export":
        print(json.dumps(export(), indent=2))
    elif a.cmd == "validate":
        v = validate()
        print(json.dumps({k: v[k] for k in v if k not in ("errors", "warnings")}, indent=2))
        for e in v["errors"][:30]:
            print("ERROR  ", e)
        for w in v["warnings"][:30]:
            print("warn   ", w)
        sys.exit(1 if v["errors"] else 0)
    elif a.cmd == "train":
        train(a.model, epochs=a.epochs, workers=a.workers)
    else:
        ap.print_help()
